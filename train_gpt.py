import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import glob
import json
import yaml
import argparse
import subprocess
import contextlib
from dataclasses import dataclass
import wandb
from utils import print_model_parameters, get_model_size_stats
import pickle
from pathlib import Path

import torch
torch.empty(1, device='cuda', requires_grad=True).backward()
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from galore_torch import GaLoreAdamW
from momentum_factorized_sgd import MomentumFactorizedSGD
from lr_schedulers import create_scheduler  # Add this import

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        param_groups = [dict(params=[p for p in params if p.numel() == size],
                             update_buffer=[torch.empty(size, device='cuda', dtype=torch.bfloat16) for _ in range(self.world_size)])
                        for size in sizes]
        super().__init__(param_groups, defaults)

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            # generate weight updates in distributed fashion
            params = group['params']
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffers):
                    p_world.data.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffers[rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):

    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum('i,j -> ij', t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x):
        cos, sin = self.cos[None, :x.size(-3), None, :], self.sin[None, :x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, ve, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, 'Must use batch size = 1 for FlexAttention'
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, model_dim, num_heads, use_attn=True):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads) if use_attn else None
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        ve = [emb(inputs).bfloat16() for emb in self.embed]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        return ve

# -----------------------------------------------------------------------------
# The main GPT-2 model

class GPT(nn.Module):

    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, use_attn=(i != 7))
                                     for i in range(num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    def forward(self, inputs, targets, sliding_window_num_blocks):
        BLOCK_SIZE = 128
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        total_num_blocks = seq_len // BLOCK_SIZE
        assert inputs.ndim == 1
        docs = (inputs == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks):
            kv_idx = block_idx = torch.arange(total_num_blocks, dtype=torch.int32, device='cuda')
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm # block-wise sliding window by @YouJiacheng
            # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm  = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        x0 = norm(self.embed(inputs[None]).bfloat16()) # use of norm here by @Grad62304977
        x = x0
        ve = self.value_embeds(inputs)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15) # @Grad62304977 added tanh softcapping, @KoszarskyB reduced it from 30 to 15
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets)
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(path, False, 256, dtype=torch.int32)
    assert header[0] == 20240520, 'magic number mismatch in the data .bin file'
    assert header[1] == 1, 'unsupported version'
    num_tokens = int(header[2]) # number of tokens (claimed)
    with open(path, 'rb', buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, 'number of tokens read does not match header'
    return tokens

class DistributedDataLoader:

    def __init__(self, filename_pattern):
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, batch_size):
        assert batch_size % self.world_size == 0
        device_batch_size = batch_size // self.world_size
        # load next shard if necessary
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        pos = self.current_position + self.rank * device_batch_size
        device_batch_tokens = self.tokens[pos:pos+device_batch_size+1]
        # advance current position
        self.current_position += batch_size
        inputs = device_batch_tokens[:-1].to(device='cuda', dtype=torch.int32, non_blocking=True)
        targets = device_batch_tokens[1:].to(device='cuda', dtype=torch.int64, non_blocking=True)
        return inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_bin = '/scratch/pxm5426/datasets/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    val_bin = '/scratch/pxm5426/datasets/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization
    batch_size = 8*64*1024 # batch size in tokens
    max_device_batch_size = 64*1024 # batch size per device in tokens
    num_iterations = 1390 # number of iterations to run
    bf16_embeds = True
    optimizer_name = 'muon'  # Options: 'muon', 'adam', 'adamw', 'sgd'
    optimizer_config_path = None  # Path to optimizer config YAML file
    use_momentum_warmup = True  # Whether to use momentum warmup for Muon/MFSGD
    # evaluation and logging
    val_loss_every = 20 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # implementation
    save_checkpoint = False
    # wandb config
    wandb_project = "nanogpt-training"
    wandb_entity = None  # set to your wandb username or team name, or leave as None

    def load_optimizer_config(self):
        """Load optimizer configuration from YAML file."""
        if self.optimizer_config_path:
            config_path = self.optimizer_config_path
        else:
            config_path = f'configs/optimizers/{self.optimizer_name}.yaml'
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default configuration.")
            return {}

    def update_from_args(self, args):
        """Update hyperparameters from command line arguments."""
        # 1. Set basic parameters from command line
        if args.optimizer:
            self.optimizer_name = args.optimizer
        if args.config_path:
            self.optimizer_config_path = args.config_path
        if args.no_momentum_warmup:
            self.use_momentum_warmup = False
        
        # 2. Load base config from YAML
        base_config = self.load_optimizer_config()
        
        # 3. Apply JSON config overrides if any
        if args.config:
            try:
                config_override = json.loads(args.config)
                base_config.update(config_override)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in config override: {args.config}")
                
        # 4. Store the final merged config
        self.optimizer_config = base_config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train GPT model with different optimizers and configurations')
parser.add_argument('--optimizer', type=str, help='Optimizer to use (muon, adam, adamw, sgd)')
parser.add_argument('--config', type=str, help='JSON string with optimizer configuration overrides')
parser.add_argument('--config-path', type=str, help='Path to optimizer config YAML file')
parser.add_argument('--run-name', type=str, help='Name for the Wandb run')
parser.add_argument('--no-momentum-warmup', action='store_true', help='Disable momentum warmup for Muon/MFSGD')
parser.add_argument('--mem-prof', action='store_true', help='Record full CUDA memory history and dump a snapshot.')
parser.add_argument('--mem-snap-step', type=int, default=None, help='Which training step to capture an early memory snapshot (requires --mem-prof)')
parser.add_argument('--mem-start-step', type=int, default=None, help='Step number to start memory recording (requires --mem-prof)')
parser.add_argument('--mem-end-step', type=int, default=None, help='Step number to stop memory recording and snapshot (requires --mem-start-step)')
parser.add_argument('--is-resuming', action='store_true', help='Flag to indicate if the run is resuming')
cmd_args = parser.parse_args()

args = Hyperparameters()
args.update_from_args(cmd_args)

micro_bs = args.max_device_batch_size

# set up DDP (distributed data parallel). torchrun sets this env variable
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
assert torch.cuda.is_available()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', device_id=torch.device(local_rank))
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# Helper run id for logs/snapshots
run_id = os.getenv('RUN_ID', str(uuid.uuid4()))  # used for log/snapshot folder

# Flag to track if recording has begun when using start/end range
started_recording = False

# begin logging
logfile = None
if master_process:
    os.makedirs('logs', exist_ok=True)
    logfile = f'logs/{run_id}.txt'
    print(logfile)
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={**vars(args), 'optimizer': args.optimizer_name},
        name=cmd_args.run_name if cmd_args.run_name else f"run_{run_id}_{args.optimizer_name}",
    )

def print0(s, console=False):
    if master_process:
        with open(logfile, 'a') as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0('='*100)
# log information about the hardware/software environment this is running on
print0(f'Running Python {sys.version}')
print0(f'Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}')
print0(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
print0('='*100)

# load data
train_loader = DistributedDataLoader(args.train_bin)
val_loader = DistributedDataLoader(args.val_bin)
print0(f'Training dataloader files: {train_loader.files}')
print0(f'Validation dataloader files: {val_loader.files}')
print0('='*100)

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
model = GPT(vocab_size=50304, num_layers=12, num_heads=6, model_dim=768)
model = model.cuda()

if master_process:
    print_model_parameters(model, print0)
    # Log model statistics to wandb
    if args.wandb_project:
        wandb.config.update(get_model_size_stats(model))

if args.bf16_embeds:
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
model = torch.compile(model)
ddp_model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)

def get_hidden_matrix_optimizer(params, optimizer_name):
    """Initialize optimizer for hidden matrix parameters based on configuration."""
    config = args.optimizer_config
    
    if optimizer_name.lower() == 'muon':
        # Filter out non-Muon parameters to avoid unexpected keyword errors
        muon_config = {k: v for k, v in config.items() 
                      if k in ['lr', 'momentum', 'nesterov', 'ns_steps']}
        return Muon(params, **muon_config)
    elif optimizer_name.lower() == 'adam':
        # Filter out non-Adam parameters
        adam_config = {k: v for k, v in config.items()
                     if k in ['lr', 'betas', 'eps', 'weight_decay', 'fused']}
        return torch.optim.Adam(params, **adam_config)
    elif optimizer_name.lower() == 'adamw':
        # Filter out non-AdamW parameters
        adamw_config = {k: v for k, v in config.items()
                      if k in ['lr', 'betas', 'eps', 'weight_decay', 'fused']}
        return torch.optim.AdamW(params, **adamw_config)
    elif optimizer_name.lower() == 'sgd':
        # Filter out non-SGD parameters
        sgd_config = {k: v for k, v in config.items()
                    if k in ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov']}
        return torch.optim.SGD(params, **sgd_config)
    elif optimizer_name.lower() == 'galore':
        # Extract GaLore-specific parameters for param groups
        group_params = config.get('group_params', {})
        galore_group = {
            'params': params,
            'rank': group_params.get('rank', 128),
            'update_proj_gap': group_params.get('update_proj_gap', 200),
            'scale': group_params.get('scale', 0.25),
            'proj_type': group_params.get('proj_type', 'std'),
        }
        
        # Standard optimizer parameters
        optimizer_params = {
            'lr': config.get('lr', 0.02),
            'betas': config.get('betas', (0.8, 0.95)),
            'weight_decay': config.get('weight_decay', 0.0),
            'eps': config.get('eps', 1e-8),
            'no_deprecation_warning': True
        }
        
        # Add fused parameter if it exists in the config
        if 'fused' in config:
            # Pass the fused parameter to both the optimizer params and the group params
            fused_value = config.get('fused', False)
            optimizer_params['fused'] = fused_value
            galore_group['fused'] = fused_value
            print0(f"Using GaLoreAdamW with fused={fused_value}")
            # Add additional debug prints that will definitely show up
            print(f"\n{'*' * 80}")
            print(f"GALORE OPTIMIZER DEBUG: fused parameter is set to {fused_value}")
            print(f"optimizer_params={optimizer_params}")
            print(f"galore_group parameters: rank={galore_group['rank']}, update_proj_gap={galore_group['update_proj_gap']}, scale={galore_group['scale']}, fused={galore_group.get('fused', False)}")
            print(f"{'*' * 80}\n")
        else:
            # Add debug print for when fused is NOT in the config
            print(f"\n{'*' * 80}")
            print(f"GALORE OPTIMIZER DEBUG: fused parameter is NOT SET in the config!")
            print(f"{'*' * 80}\n")
        
        return GaLoreAdamW([galore_group], **optimizer_params)
    elif optimizer_name.lower() == 'mfsgd':
        return MomentumFactorizedSGD(params, 
                                   lr=config.get('lr', 0.01),
                                   rank=config.get('rank', 2),
                                   beta=config.get('beta', 0.9),
                                   eta1=config.get('eta1', 1.0),
                                   eta2=config.get('eta2', 1.0),
                                   use_current_projection=config.get('use_current_projection', False),
                                   use_ones_for_nonzero_s=config.get('use_ones_for_nonzero_s', False),
                                   eps=config.get('eps', 1e-4),
                                   nesterov=config.get('nesterov', False),
                                   max_value=config.get('max_value', 10000))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# collect the parameters to optimize
hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim == 2]
embed_params = [model.embed.weight, *model.value_embeds.parameters()]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
optimizer1 = torch.optim.Adam([dict(params=embed_params, lr=0.6),
                               dict(params=head_params, lr=0.008),
                               dict(params=scalar_params, lr=0.04)],
                              betas=(0.8, 0.95), fused=True)
optimizer2 = get_hidden_matrix_optimizer(hidden_matrix_params, args.optimizer_name)
optimizers = [optimizer1, optimizer2]

# Create schedulers using the factory
schedulers = [
    create_scheduler('original', optimizer1, args.num_iterations, args.optimizer_config),  # Always use original for optimizer1
    create_scheduler(args.optimizer_config.get('scheduler_name', 'original'), optimizer2, args.num_iterations, args.optimizer_config)  # Use scheduler from config
]

if master_process:
    print0(f"Using scheduler '{args.optimizer_config.get('scheduler_name', 'original')}' for optimizer2", console=True)

# sliding window size schedule: linear increase over training in chunks of 128 from 128 -> 1792. By @fernbear.bsky.social
def get_sliding_window_blocks(it):
    x = it / args.num_iterations # training progress
    assert 0 <= x <= 1
    return int(((1 - x) * 128 + x * 1856) // 128)
sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device='cuda')

# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    sliding_window_num_blocks.copy_(get_sliding_window_blocks(step))

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        # calculate the number of steps to take in the val loop.
        val_batch_size = world_size * micro_bs
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        for _ in range(val_steps):
            with torch.no_grad():
                inputs_val, targets_val = val_loader.next_batch(val_batch_size)
                val_loss += ddp_model(inputs_val, targets_val, sliding_window_num_blocks)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # logging
        print0(f'step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms', console=True)
        # Log to wandb
        if master_process:
            wandb.log({
                "val_loss": val_loss,
                "step": step,
                "train_time_ms": training_time_ms,
                "step_avg_ms": training_time_ms/(timed_steps-1)
            })
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f'logs/{run_id}', exist_ok=True)
            torch.save(log, f'logs/{run_id}/state_step{step:06d}.pt')
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    model.train()
    batch_size = args.batch_size
    assert batch_size % world_size == 0
    inputs_train, targets_train = train_loader.next_batch(batch_size)
    assert len(inputs_train) <= micro_bs or len(inputs_train) % micro_bs == 0
    for micro_inputs_train, micro_targets_train in zip(inputs_train.split(micro_bs), targets_train.split(micro_bs)):
        ddp_model(micro_inputs_train, micro_targets_train, sliding_window_num_blocks).backward()
    # momentum warmup for Muon/MFSGD
    if args.use_momentum_warmup:
        if args.optimizer_name.lower() == 'muon':
            frac = min(step/300, 1)
            for group in optimizer2.param_groups:
                group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
        elif args.optimizer_name.lower() == 'mfsgd':
            # Get warmup parameters from config
            warmup_steps = args.optimizer_config.get('warmup_steps', 600)  # default to original value
            beta_start = args.optimizer_config.get('beta_start', 0.15)    # default to original value
            beta_end = args.optimizer_config.get('beta_end', 0.95)        # default to original value
            frac = min(step/warmup_steps, 1)
            for group in optimizer2.param_groups:
                group['beta'] = (1 - frac) * beta_start + frac * beta_end
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        if step != train_steps-1:
            sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f'step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms', console=True)

    # Add learning rate logging
    if master_process:
        wandb.log({
            "lr_optimizer1": schedulers[0].get_last_lr()[0],
            "lr_optimizer2": schedulers[1].get_last_lr()[0],
            # ... existing logging ...
        })

    # Optionally capture an early memory snapshot at a user-specified step
    if master_process and cmd_args.mem_prof and cmd_args.mem_snap_step is not None and step == cmd_args.mem_snap_step:
        snap = torch.cuda.memory._snapshot()  # freeze history so far
        out_dir = Path('logs') / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_file = out_dir / f"mem_snapshot_step{step:06d}.pkl"
        with open(snap_file, 'wb') as f:
            pickle.dump(snap, f)
        print(f"Early memory snapshot saved to {snap_file} — load it at https://pytorch.org/memory_viz", flush=True)
        # stop recording to save memory; no final snapshot
        torch.cuda.memory._record_memory_history(enabled=None)
        cmd_args.mem_prof = False  # prevent later snapshot logic

    # Deferred start of recording
    if master_process and cmd_args.mem_prof and not started_recording and cmd_args.mem_start_step is not None and step == cmd_args.mem_start_step:
        torch.cuda.memory._record_memory_history(enabled='all')
        started_recording = True
        print(f'CUDA-memory history recording ENABLED ⏺️ at step {step}', flush=True)

    # Stop recording at mem_end_step
    if master_process and started_recording and cmd_args.mem_end_step is not None and step == cmd_args.mem_end_step:
        snap = torch.cuda.memory._snapshot()
        out_dir = Path('logs') / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_file = out_dir / f"mem_snapshot_range_{cmd_args.mem_start_step:06d}_{cmd_args.mem_end_step:06d}.pkl"
        with open(snap_file, 'wb') as f:
            pickle.dump(snap, f)
        print(f"Range memory snapshot saved to {snap_file} — load it at https://pytorch.org/memory_viz", flush=True)
        torch.cuda.memory._record_memory_history(enabled=None)
        started_recording = False
        cmd_args.mem_prof = False  # disable to skip final snapshot logic

print0(f'peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')
if master_process and cmd_args.mem_prof and cmd_args.mem_snap_step is None and cmd_args.mem_end_step is None:
    try:
        snap = torch.cuda.memory._snapshot()  # ⏸️ freeze history
        out_dir = Path('logs') / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"step{step:06d}" if 'step' in locals() else 'final'
        snap_file = out_dir / f"mem_snapshot_{tag}.pkl"
        with open(snap_file, 'wb') as f:
            pickle.dump(snap, f)
        print(f"Memory snapshot saved to {snap_file} — load it at https://pytorch.org/memory_viz", flush=True)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)  # always turn off
if master_process:
    wandb.finish()
dist.destroy_process_group()