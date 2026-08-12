import torch
import utils
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import time, datetime, os, sys, random, numpy as np
from datasets import build_continual_dataloader
from engines.hrm_lora_wtp_and_tap_engine import (
    _compute_mean, _compute_shared_feature_memory, evaluate_till_now,
    get_real_feature_memory, get_shared_feature_memory,
    reset_replay_statistics, restore_real_feature_memory,
    set_replay_task_router, train_and_evaluate,
)
from engines.replay_logit_calibration import calibrate_task_logits
from engines.replay_task_router import train_replay_task_router
from engines.distilled_task_router import train_distilled_task_router
import vits.hrm_lora_vision_transformer as hide_lora_vision_transformer
import torch.nn as nn
import torch.nn.init as init


        


def train(args):
    device = torch.device(args.device)
    data_loader, data_loader_per_cls, class_mask, target_task_map = build_continual_dataloader(args)
    print(f"Creating original model: {args.original_model}")
    original_model = create_model(
            args.original_model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            mlp_structure=args.original_model_mlp_structure,
        )
    print(f"Creating model: {args.model}")
    model = create_model(args.model,
                         pretrained=args.pretrained,
                         num_classes=args.nb_classes,
                         drop_rate=args.drop,
                         drop_path_rate=args.drop_path,
                         drop_block_rate=None,
                         lora=True, 
                         lora_type=args.lora_type,
                         rank=args.lora_rank, 
                         lora_pool_size=args.size,
                         )
    original_model.to(device)
    model.to(device)
    # custom_mlp.to(device)

    # all backbobe parameters are frozen for original vit model
    for n, p in original_model.named_parameters():
        p.requires_grad = False
    if args.freeze:
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        prototype_enabled = bool(getattr(args, 'prototype_rematching', False))
        local_prototype_enabled = (
            float(getattr(args, 'exhaustive_local_prototype_weight', 0.0)) > 0.0
        )
        shared_router_enabled = bool(getattr(args, 'shared_prototype_router', False))
        calibration_enabled = bool(getattr(args, 'replay_logit_calibration', False))
        learned_router_enabled = bool(getattr(args, 'replay_task_router', False))
        distilled_router_enabled = bool(
            getattr(args, 'distilled_router_rematching', False))
        if (prototype_enabled or local_prototype_enabled
                or shared_router_enabled or calibration_enabled
                or learned_router_enabled or distilled_router_enabled):
            reset_replay_statistics()
        if prototype_enabled:
            args.crct_real_feature_replay = True
            print('Prototype evaluation will restore or reconstruct real-feature memory.')
        if local_prototype_enabled:
            args.crct_real_feature_replay = True
            print(
                'Task-local prototype fusion will restore real-feature memory; '
                'task routing remains exhaustive.')
        if shared_router_enabled:
            print('Shared prototype evaluation will reconstruct backbone feature memory.')

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = utils.load_checkpoint(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = utils.load_checkpoint(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'])
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
            if distilled_router_enabled:
                print('Training exhaustive-teacher top-k router for task', task_id + 1)
                distilled_router, router_stats = train_distilled_task_router(
                    model=model,
                    original_model=original_model,
                    data_loader_per_cls=data_loader_per_cls,
                    class_mask=class_mask,
                    seen_task_count=task_id + 1,
                    args=args,
                    device=device,
                )
                set_replay_task_router(distilled_router)
                if router_stats is not None:
                    print(
                        'Distilled top-k router:',
                        'accepted=', router_stats['accepted'],
                        'samples=', router_stats['samples'],
                        'validation_samples=', router_stats['validation_samples'],
                        'TII top1/topk=', router_stats['baseline_top1'],
                        router_stats['baseline_topk'],
                        'router top1/topk=', router_stats['router_top1'],
                        router_stats['router_topk'],
                    )
            elif learned_router_enabled:
                print('Reconstructing validated task-router memory for task', task_id + 1)
                _compute_shared_feature_memory(
                    original_model=original_model, data_loader=data_loader_per_cls,
                    device=device, class_ids=class_mask[task_id], args=args,
                )
                learned_router, router_stats = train_replay_task_router(
                    original_model=original_model,
                    feature_memory=get_shared_feature_memory(),
                    class_mask=class_mask, seen_task_count=task_id + 1,
                    args=args, device=device,
                )
                set_replay_task_router(learned_router)
                if router_stats is not None:
                    print(
                        'Validated replay task router:',
                        'accepted=', router_stats['accepted'],
                        'samples=', router_stats['samples'],
                        'validation_samples=', router_stats['validation_samples'],
                        'baseline_val_acc=', router_stats['baseline_validation_accuracy'],
                        'router_val_acc=', router_stats['router_validation_accuracy'],
                    )
            elif calibration_enabled:
                saved_memory = checkpoint.get('real_feature_memory')
                if saved_memory:
                    restore_real_feature_memory(saved_memory)
                    calibration_memory = saved_memory
                    print(
                        'Replay calibration restored checkpoint memory:',
                        len(saved_memory), 'classes')
                else:
                    args.crct_real_feature_replay = True
                    print(
                        'EXPLORATORY replay calibration: reconstructing training '
                        'feature memory for task', task_id + 1)
                    _compute_mean(
                        model=model, data_loader=data_loader_per_cls,
                        device=device, task_id=task_id,
                        class_mask=class_mask[task_id], args=args,
                    )
                    calibration_memory = get_real_feature_memory()
                calibration_stats = calibrate_task_logits(
                    model=model, feature_memory=calibration_memory,
                    class_mask=class_mask, seen_task_count=task_id + 1,
                    args=args, device=device,
                )
                if calibration_stats is not None:
                    print(
                        'Replay logit calibration:',
                        'accepted=', calibration_stats['accepted'],
                        'samples=', calibration_stats['samples'],
                        'before=', calibration_stats['before'],
                        'after=', calibration_stats['after'],
                        'scale=', calibration_stats['scale'],
                        'bias=', calibration_stats['bias'],
                    )
            elif shared_router_enabled:
                print('Reconstructing shared feature memory for task', task_id + 1)
                _compute_shared_feature_memory(
                    original_model=original_model, data_loader=data_loader_per_cls,
                    device=device, class_ids=class_mask[task_id], args=args,
                )
            elif prototype_enabled or local_prototype_enabled:
                saved_memory = checkpoint.get('real_feature_memory')
                if saved_memory:
                    restore_real_feature_memory(saved_memory)
                    print('Restored real-feature memory from checkpoint:', len(saved_memory), 'classes')
                else:
                    print('Reconstructing real-feature memory for task', task_id + 1)
                    _compute_mean(
                        model=model, data_loader=data_loader_per_cls,
                        device=device, task_id=task_id,
                        class_mask=class_mask[task_id], args=args,
                    )
            _ = evaluate_till_now(model, original_model, data_loader, device,
                                  task_id, class_mask, target_task_map, acc_matrix, args, )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0


    # base_params = [p for name, p in model_without_ddp.named_parameters() if p.requires_grad == True]
    # base_fc_params = [p for name, p in custom_mlp.named_parameters() ]
    # base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
    # base_fc_params = {'params': base_fc_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
    # network_params = [base_params, base_fc_params]

    optimizer = create_optimizer(args, model_without_ddp)
    #optimizer = create_optimizer(args, network_params)
    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                       criterion, data_loader, data_loader_per_cls,
                       optimizer, lr_scheduler, device, class_mask, target_task_map, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    

