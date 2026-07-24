"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import math
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        distributed_barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def distributed_barrier():
    if not is_dist_avail_and_initialized():
        return
    if dist.get_backend() == 'nccl' and torch.cuda.is_available():
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return
        except TypeError:
            pass
    dist.barrier()


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def load_checkpoint(*args, **kwargs):
    try:
        return torch.load(*args, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(*args, **kwargs)

class CFSContrastiveMLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_features, hidden_features),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def use_cfs_sampling(args):
    return bool(getattr(args, 'cfs_sampling', False))


def train_cfs_model(features, args, device):
    if not use_cfs_sampling(args) or features.shape[0] < 2:
        return None

    with torch.enable_grad():
        features = features.detach().float().to(device)
        max_samples = int(getattr(args, 'cfs_train_max_samples', 1024))
        if features.shape[0] > max_samples:
            sample_ids = torch.randperm(features.shape[0], device=device)[:max_samples]
            features = features[sample_ids]

        in_features = features.shape[1]
        hidden_features = int(getattr(args, 'cfs_hidden_dim', 512))
        hidden_features = max(1, min(hidden_features, in_features))
        model = CFSContrastiveMLP(in_features, hidden_features).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(getattr(args, 'cfs_lr', 0.01)),
            momentum=float(getattr(args, 'cfs_momentum', 0.9)),
        )
        epochs = int(getattr(args, 'cfs_epochs', 50))
        batch_size = min(int(getattr(args, 'cfs_batch_size', 256)), features.shape[0])
        tau = float(getattr(args, 'cfs_tau', 1.0))

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(features.shape[0], device=device)
            for start in range(0, features.shape[0], batch_size):
                batch = features[perm[start:start + batch_size]]
                if batch.shape[0] < 2:
                    continue
                out = model(batch)
                sim = torch.mm(out, out.t()) / tau
                sim.fill_diagonal_(float('-inf'))
                loss = torch.logsumexp(sim, dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model


@torch.no_grad()
def sample_cfs_features(mean, cov, num_samples, args, device, cfs_model=None):
    distribution = torch.distributions.MultivariateNormal(mean.float(), cov.float())
    if not use_cfs_sampling(args) or cfs_model is None or num_samples <= 1:
        return distribution.sample(sample_shape=(num_samples,))

    multiplier = max(1, int(getattr(args, 'cfs_candidate_multiplier', 3)))
    candidate_count = max(num_samples, num_samples * multiplier)
    candidates = distribution.sample(sample_shape=(candidate_count,)).to(device)

    cfs_model = cfs_model.to(device)
    embeddings = cfs_model(candidates.float())
    tau = float(getattr(args, 'cfs_tau', 1.0))
    sim = torch.exp(torch.mm(embeddings, embeddings.t()) / tau)
    sim.fill_diagonal_(0)

    selected = [torch.randint(candidate_count, (1,), device=device).item()]
    score_sum = sim[:, selected[0]].clone()
    available = torch.ones(candidate_count, dtype=torch.bool, device=device)
    available[selected[0]] = False

    for _ in range(1, num_samples):
        scores = score_sum / len(selected)
        scores = scores.masked_fill(~available, float('inf'))
        next_id = torch.argmin(scores).item()
        selected.append(next_id)
        available[next_id] = False
        score_sum += sim[:, next_id]

    return candidates[torch.tensor(selected, device=device)]


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    distributed_barrier()
    setup_for_distributed(args.rank == 0)


def task_inference_accuracy(prompt_idx, target, target_task_map,filtered_index_tensor=None,corr_id=None):
    target_2_task = torch.tensor([target_task_map[v.item()] for v in target]).to(prompt_idx.device)
    # if filtered_index_tensor!=None and corr_id!=None:
    #     target_2_task[filtered_index_tensor] = corr_id
    batch_size = target.size(0)
    prompt_idx = prompt_idx.t()
    correct = prompt_idx.eq(target_2_task.reshape(1, -1).expand_as(prompt_idx))
    return correct.reshape(-1).float().sum(0) * 100. / batch_size

