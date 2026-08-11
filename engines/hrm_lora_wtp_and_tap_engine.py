import copy
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from engines.prototype_rematching import (
    build_prototype_bank, prototype_assisted_rematching)
from engines.shared_prototype_router import (
    build_shared_prototype_bank, shared_space_prototype_routing)
from torch.distributions.multivariate_normal import MultivariateNormal



def generalized_entropy(softmax_id_val, gamma, M):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
           
        return -scores 

def compute_task_energy_scores(logits, class_mask, num_tasks, temperature=0.1):
    temperature = max(float(temperature), 1e-6)
    task_scores = []
    for task_idx in range(num_tasks):
        class_ids = torch.tensor(class_mask[task_idx], dtype=torch.long, device=logits.device)
        task_logits = logits.index_select(1, class_ids)
        task_scores.append(temperature * torch.logsumexp(task_logits / temperature, dim=1))
    return torch.stack(task_scores, dim=1)

def compute_ctird_rank_weights(top_values, args):
    temperature = max(float(getattr(args, 'ctird_weight_temperature', 1.0)), 1e-6)
    weights = F.softmax(top_values.float() / temperature, dim=1)
    floor = float(getattr(args, 'ctird_weight_floor', 0.2))
    floor = max(0.0, min(1.0, floor))
    uniform = torch.full_like(weights, 1.0 / weights.shape[1])
    weights = (1.0 - floor) * weights + floor * uniform
    return weights.mean(dim=0) * weights.shape[1]

def get_old_features(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.eval()
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    all_res = []
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        ctird_rank_weights = None
        
        if task_id>0:

            probabilities = temp[:,:task_id*len(class_mask[0])]
            m = min(task_id, args.K)
            ctird_selection = str(getattr(args, 'ctird_task_selection', 'legacy')).lower()
            if ctird_selection == 'task_energy':
                task_scores = compute_task_energy_scores(
                    temp, class_mask, task_id,
                    temperature=getattr(args, 'ctird_task_temperature', 0.1))
                top_values, top5_id = torch.topk(task_scores, k=m, dim=1, largest=True)
                if str(getattr(args, 'ctird_task_weighting', 'uniform')).lower() == 'energy':
                    ctird_rank_weights = compute_ctird_rank_weights(top_values, args)
            else:
                _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
                top5_id = []
                for i in range(top_indices.shape[0]):
                    top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
                top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            # robust_logits = robust_loss(model, input, output['features'], target,device,task_id,class_mask,top5_id)
            all_old_logits = []
            for k in range(top5_id.shape[1]):
                prompt_id = top5_id[:,k]
                with torch.no_grad():
                    output = model(input, task_id=prompt_id)
                    old_logits = output['features']
                    old_norm_features = F.normalize(output['features'], p=2, dim=1)
                    old_similarity_matrix = torch.mm(old_norm_features, old_norm_features.t())
                    old_similarity_matrix = torch.exp(old_similarity_matrix)
                    old_similarity_matrix = old_similarity_matrix / old_similarity_matrix.sum(1, keepdim=True)
                    all_old_logits.append(old_similarity_matrix)

            if ctird_rank_weights is None:
                all_res.append(all_old_logits)
            else:
                all_res.append((all_old_logits, ctird_rank_weights.detach()))
                
    return all_res



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, old_features = None):
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    global_index = 0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            
        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']
        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        ctird_rank_weights = None
        
        if task_id>0:

            probabilities = temp[:,:task_id*len(class_mask[0])]
            m = min(task_id, args.K)
            ctird_selection = str(getattr(args, 'ctird_task_selection', 'legacy')).lower()
            if ctird_selection == 'task_energy':
                task_scores = compute_task_energy_scores(
                    temp, class_mask, task_id,
                    temperature=getattr(args, 'ctird_task_temperature', 0.1))
                top_values, top5_id = torch.topk(task_scores, k=m, dim=1, largest=True)
                if str(getattr(args, 'ctird_task_weighting', 'uniform')).lower() == 'energy':
                    ctird_rank_weights = compute_ctird_rank_weights(top_values, args)
            else:
                _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
                top5_id = []
                for i in range(top_indices.shape[0]):
                    top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
                top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            #robust_logits = robust_loss(model, input, output['features'], target,device,task_id,class_mask,top5_id)
            robust_bundle = old_features[global_index]
            ctird_rank_weights = None
            if isinstance(robust_bundle, tuple):
                robust_logits, ctird_rank_weights = robust_bundle
            else:
                robust_logits = robust_bundle
        
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
 
        
        loss_ctird = 0
        if task_id>0:

            loss = criterion(logits, target)
            for k in range(len(robust_logits)):
                norm_features = F.normalize(output['features'], p=2, dim=1)
                
                # 计算相似度矩阵
                similarity_matrix = torch.mm(norm_features, norm_features.t())
                similarity_matrix = torch.exp(similarity_matrix)
                similarity_matrix = similarity_matrix / similarity_matrix.sum(1, keepdim=True)
                pos_mask = (similarity_matrix > 0.038).float()
                relation_target = utils.apply_semantic_relation_distillation(robust_logits[k], target, args, device)
                loss_ctird = F.kl_div(torch.log(similarity_matrix.clamp_min(1e-12)), relation_target, reduction='batchmean')
                
                ctird_weight = 1.0 if ctird_rank_weights is None else ctird_rank_weights[k]
                loss = loss + args.con * ctird_weight * loss_ctird
        else:
            loss = criterion(logits, target)+args.con*loss_ctird
            
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        global_index = global_index+1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

con_num=0
incon_num=0
con_all=0
incon_all=0
max_p = []
cls_mean = {}
cls_cov = {}
cls_cfs_model = {}
cls_real_features = {}
cls_shared_features = {}


def reset_replay_statistics():
    global cls_mean, cls_cov, cls_cfs_model, cls_real_features, cls_shared_features
    cls_mean = {}
    cls_cov = {}
    cls_cfs_model = {}
    cls_real_features = {}
    cls_shared_features = {}


def restore_real_feature_memory(feature_memory):
    global cls_real_features
    cls_real_features = {
        int(class_id): features.detach().cpu().half()
        for class_id, features in feature_memory.items()
    }


def get_real_feature_memory():
    return cls_real_features

@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    global con_num, incon_num ,max_p, con_all, incon_all
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()
    prototype_bank = None
    if bool(getattr(args, 'prototype_rematching', False)):
        seen_classes = [
            int(class_id)
            for seen_task in range(task_id + 1)
            for class_id in class_mask[seen_task]
        ]
        prototype_bank = build_prototype_bank(
            model, cls_real_features, seen_classes, device)
        if utils.is_main_process():
            print(
                'Prototype rematching:',
                'classes=', len(prototype_bank),
                'candidate_tasks=', int(getattr(args, 'prototype_candidate_tasks', 2)),
                'temperature=', float(getattr(args, 'prototype_temperature', 0.07)),
            )
    shared_prototype_bank = None
    if bool(getattr(args, 'shared_prototype_router', False)):
        seen_classes = [
            int(class_id)
            for seen_task in range(task_id + 1)
            for class_id in class_mask[seen_task]
        ]
        shared_prototype_bank = build_shared_prototype_bank(
            cls_shared_features, seen_classes, device)
        if utils.is_main_process():
            print(
                'Shared prototype router:',
                'classes=', len(shared_prototype_bank),
                'temperature=', float(getattr(args, 'shared_prototype_temperature', 0.07)),
            )

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    shared_features = output.get('pre_logits')
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            lora_id = torch.max(old_logits, dim=1)[1]
            lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
            
            output = model(input, task_id=lora_id)
            logits = output['logits']
            

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                #print(logits[0])
            id_logits = logits
            routing_mode = str(getattr(args, 'task_routing_mode', 'class')).lower()
            if routing_mode == 'task_energy':
                task_scores = compute_task_energy_scores(
                    id_logits, class_mask, task_id + 1,
                    temperature=getattr(args, 'task_routing_temperature', 0.1))
                candidate_count = min(2, task_scores.shape[1])
                top5_id = torch.topk(task_scores, k=candidate_count, dim=1, largest=True).indices
                prompt_id = top5_id[:, 0]
            else:
                prompt_class = torch.max(id_logits, dim=1)[1]
                _, top_5_indices = torch.topk(id_logits, 2, dim=1)
                task_map_keys = list(target_task_map.keys())
                task_map_values = torch.tensor([target_task_map[k] for k in task_map_keys], device=device)
                task_map_tensor = torch.zeros((max(task_map_keys) + 1,), dtype=torch.long, device=device)
                task_map_tensor[task_map_keys] = task_map_values
                top5_id = torch.index_select(task_map_tensor, 0, top_5_indices.view(-1)).view_as(top_5_indices)
                prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_class], device=device)
            ##############
            # target_id = torch.tensor([target_task_map[v.item()] for v in target], device=device)

            
            ###########
            id_logits = F.softmax(id_logits,dim=1)
            values, _ = torch.topk(id_logits, 1, dim=1)
            if args.En == 'msp':
                low_confidence_indices = torch.where(values <args.tau)[0]
            else:
                softmax_ood = id_logits
                softmax_ood = softmax_ood.cpu().numpy()
                #energy = (0.1*torch.logsumexp(inp / 0.1, dim=1))
                en = generalized_entropy(softmax_ood, 0.1, 20)
                
                en = torch.tensor(en)
                en = en.to(device, non_blocking=True)
                low_confidence_indices = torch.where(en <args.tau)[0]


            filtered_index_tensor = low_confidence_indices    

            
            equal_drm = torch.nonzero(prompt_id != lora_id).flatten()
            output_drm = model(input[equal_drm], task_id=prompt_id[equal_drm])
            output_drm_logits = output_drm['logits']
            if routing_mode == 'task_energy' and args.train_mask and class_mask is not None:
                output_drm_logits = output_drm_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
            logits[equal_drm] = output_drm_logits
            #promtp_idx = output['prompt_idx']  # tensor B x topk
            
            corr_id=None
            re_id = None
            if task_id>0:
                    error_input = input[filtered_index_tensor]
                    # error_target = target[filtered_index_tensor]
                    ensemble_id = top5_id[filtered_index_tensor,:2]
                    #ensemble_id = second_largest_classes
                    error_output = []
                    error_output.append(logits[filtered_index_tensor,:])
                    # for i in range(ensemble_id.shape[1]):
                    out = model(error_input, task_id=ensemble_id[:,1])
                    alternative_logits = out['logits']
                    if routing_mode == 'task_energy' and args.train_mask and class_mask is not None:
                        alternative_logits = alternative_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    error_output.append(alternative_logits)
                    error_output = torch.stack(error_output,dim=1)
                   
                    entropy = (0.1*torch.logsumexp( error_output/ 0.1, dim=2))
                    corr_id = torch.max(entropy, dim=1)[1]
                    # print(corr_id)
                    corr = corr_id.unsqueeze(1).unsqueeze(2)
                    result = torch.gather(error_output, 1, corr.expand(-1, 1, len(class_mask[0])*len(class_mask)))

                    result = result.squeeze(1)
                    logits[filtered_index_tensor]=result
                    if routing_mode == 'task_energy':
                        selected_tasks = torch.gather(ensemble_id, 1, corr_id.unsqueeze(1)).squeeze(1)
                        prompt_id[filtered_index_tensor] = selected_tasks
                        re_id = selected_tasks
                    # re_id = []
                    # for k in range(len(corr_id)):
                    #     #print(corr_id[k].item())
                    #     re_id.append(ensemble_id[k,corr_id[k].item()].item())
                    # re_id = torch.tensor(re_id)
                    # re_id = re_id.to(device, non_blocking=True)
                    # try:
                    #     prompt_id[filtered_index_tensor] = re_id
                    # except:
                    #     pass
                    
            
            if shared_prototype_bank is not None:
                logits, prompt_id = shared_space_prototype_routing(
                    model,
                    input,
                    shared_features,
                    old_logits,
                    class_mask,
                    task_id + 1,
                    shared_prototype_bank,
                    args,
                )
                filtered_index_tensor = torch.empty(0, dtype=torch.long, device=device)
                re_id = None
            elif prototype_bank is not None:
                logits, prompt_id = prototype_assisted_rematching(
                    model,
                    input,
                    old_logits,
                    class_mask,
                    task_id + 1,
                    prototype_bank,
                    args,
                )
                filtered_index_tensor = torch.empty(0, dtype=torch.long, device=device)
                re_id = None

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            task_inference_acc = utils.task_inference_accuracy(prompt_id.unsqueeze(-1), target, target_task_map, filtered_index_tensor,re_id)

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    global con_num, incon_num ,con_all, incon_all
    
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        con_num=0
        incon_num=0
        con_all = 0
        incon_all=0
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    global cls_cfs_model
    global cls_real_features
    cls_mean = dict()
    cls_cov = dict()
    cls_cfs_model = dict()
    cls_real_features = dict()
    norm_blend_enabled = bool(getattr(args, 'continual_norm_blend', False))
    norm_update_ratio = min(
        1.0, max(0.0, float(getattr(args, 'continual_norm_update_ratio', 0.25))))

    task_count = args.num_tasks
    max_train_tasks = int(getattr(args, 'max_train_tasks', 0))
    if max_train_tasks > 0:
        task_count = min(task_count, max_train_tasks)
        if utils.is_main_process():
            print('Limiting run to', task_count, 'of', args.num_tasks, 'tasks')
    for task_id in range(task_count):
        previous_fc_norm = None
        if norm_blend_enabled and task_id > 0:
            previous_fc_norm = {
                name: parameter.detach().clone()
                for name, parameter in model_without_ddp.fc_norm.named_parameters()
            }
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        if args.trained_original_model:
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = utils.load_checkpoint(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'], strict=False)
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
       
        if task_id > 0:
            with torch.no_grad():
                if args.distributed:
                    model.module.lora_layer.k_lora_A.grad.zero_()
                    model.module.lora_layer.k_lora_A[task_id] = model.module.lora_layer.k_lora_A[task_id-1]
                    model.module.lora_layer.k_lora_B.grad.zero_()
                    model.module.lora_layer.k_lora_B[task_id] = model.module.lora_layer.k_lora_B[task_id-1]
                    model.module.lora_layer.v_lora_A.grad.zero_()
                    model.module.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.module.lora_layer.v_lora_B.grad.zero_()
                    model.module.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]
                else:
                    model.lora_layer.k_lora_A.grad.zero_()
                    model.lora_layer.k_lora_A[task_id] = model.lora_layer.k_lora_A[task_id-1]
                    model.lora_layer.k_lora_B.grad.zero_()
                    model.lora_layer.k_lora_B[task_id] = model.lora_layer.k_lora_B[task_id-1]
                    model.lora_layer.v_lora_A.grad.zero_()
                    model.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.lora_layer.v_lora_B.grad.zero_()
                    model.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]

        if task_id>0:
            old_features = get_old_features(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=0, max_norm=args.clip_grad,
                                            set_training_mode=False, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, )
        else:
            old_features = None

        for epoch in range(args.epochs):
            # model.module.init_weights_proj()
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, old_features=old_features)

            if lr_scheduler:
                lr_scheduler.step(epoch)
        model_without_ddp.after_task(task_id=task_id, device=device)

        if previous_fc_norm is not None:
            squared_update_before = 0.0
            squared_update_after = 0.0
            with torch.no_grad():
                for name, parameter in model_without_ddp.fc_norm.named_parameters():
                    previous = previous_fc_norm[name].to(parameter.device)
                    update = parameter - previous
                    squared_update_before += update.float().pow(2).sum().item()
                    parameter.mul_(norm_update_ratio).add_(
                        previous, alpha=1.0 - norm_update_ratio)
                    retained_update = parameter - previous
                    squared_update_after += retained_update.float().pow(2).sum().item()
            if utils.is_main_process():
                print(
                    'Continual norm blend:',
                    'update_ratio=', norm_update_ratio,
                    'delta_before=', math.sqrt(squared_update_before),
                    'delta_after=', math.sqrt(squared_update_after),
                )

        if args.lora_momentum > 0 and task_id > 0:
            with torch.no_grad():
                model.module.lora_layer.k_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.k_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_B[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_B[0:task_id].detach().clone().mean(dim=0))


        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)
        if bool(getattr(args, 'crct_real_feature_replay', False)) and utils.is_main_process():
            memory_samples = sum(memory.shape[0] for memory in cls_real_features.values())
            memory_bytes = sum(memory.numel() * memory.element_size() for memory in cls_real_features.values())
            print(
                'Real feature memory:',
                'classes=', len(cls_real_features),
                'samples=', memory_samples,
                'MiB=', round(memory_bytes / (1024.0 * 1024.0), 3),
            )
        if utils.use_semantic_feature_adapter(args):
            seen_classes = []
            for seen_task in range(task_id + 1):
                seen_classes.extend(class_mask[seen_task])
            utils.update_semantic_feature_adapter(args, cls_mean, device, available_classes=seen_classes)


        if task_id > 0 and not args.not_train_ca:
            pre_ca_test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                                  device=device,
                                                  task_id=task_id, class_mask=class_mask,
                                                  target_task_map=target_task_map,
                                                  acc_matrix=pre_ca_acc_matrix, args=args)
            #train_dis(model, args, device,data_loader[task_id]['train'], class_mask,target_task_map, task_id)
            train_task_adaptive_prediction(model, args, device, class_mask, task_id)
            
        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if bool(getattr(args, 'crct_real_feature_replay', False)):
                state_dict['real_feature_memory'] = {
                    int(class_id): memory.cpu()
                    for class_id, memory in cls_real_features.items()}
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                    'a') as f:
                f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
def _select_real_feature_memory(features, args):
    """Keep central but diverse real features without retaining input images."""
    features = features.detach().float()
    budget = max(1, int(getattr(args, 'crct_real_memory_per_class', 48)))
    if features.shape[0] <= budget:
        return features.cpu().half()

    normalized = F.normalize(features, dim=1)
    normalized_center = F.normalize(features.mean(dim=0), dim=0)
    center_distance = 1.0 - normalized.matmul(normalized_center)
    outlier_quantile = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_outlier_quantile', 0.9))))
    distance_limit = torch.quantile(center_distance, outlier_quantile)
    eligible = torch.nonzero(center_distance <= distance_limit, as_tuple=False).flatten()
    if eligible.numel() < budget:
        eligible = torch.argsort(center_distance)[:budget]

    diversity_weight = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_diversity_weight', 0.7))))
    first_index = eligible[torch.argmin(center_distance.index_select(0, eligible))]
    selected = [int(first_index.item())]
    available = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    available[eligible] = True
    available[first_index] = False
    min_distance = 1.0 - normalized.matmul(normalized[first_index])

    while len(selected) < budget and bool(available.any()):
        score = (
            diversity_weight * min_distance
            - (1.0 - diversity_weight) * center_distance
        )
        score = score.masked_fill(~available, float('-inf'))
        next_index = torch.argmax(score)
        selected.append(int(next_index.item()))
        available[next_index] = False
        distance_to_next = 1.0 - normalized.matmul(normalized[next_index])
        min_distance = torch.minimum(min_distance, distance_to_next)

    selected_index = torch.as_tensor(selected, dtype=torch.long, device=features.device)
    return features.index_select(0, selected_index).cpu().half()


@torch.no_grad()
def _compute_shared_feature_memory(original_model, data_loader, device,
                                   class_ids, args):
    original_model.eval()
    for class_id in class_ids:
        features_per_class = []
        for inputs, _ in data_loader[class_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            output = original_model(inputs)
            features_per_class.append(output['pre_logits'])
        features_per_class = torch.cat(features_per_class, dim=0)
        gathered = [
            torch.zeros_like(features_per_class, device=device)
            for _ in range(args.world_size)
        ]
        utils.distributed_barrier()
        dist.all_gather(gathered, features_per_class)
        gathered = torch.cat(gathered, dim=0)
        cls_shared_features[int(class_id)] = _select_real_feature_memory(
            gathered, args)


@torch.no_grad()
def _sample_real_feature_memory(class_id, sample_count, model, seen_classes, args, device):
    """Mix hard real features with random representatives from one class memory."""
    memory = cls_real_features.get(int(class_id))
    if memory is None or memory.numel() == 0 or sample_count <= 0:
        return None

    feature_bank = memory.to(device=device, dtype=torch.float32, non_blocking=True)
    bank_size = feature_bank.shape[0]
    hard_ratio = min(
        1.0, max(0.0, float(getattr(args, 'crct_real_hard_ratio', 0.5))))
    hard_count = min(bank_size, sample_count, int(round(sample_count * hard_ratio)))
    selected = []

    if hard_count > 0:
        logits = model(feature_bank, fc_only=True)['logits']
        seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
        seen_logits = logits.index_select(1, seen_index)
        target_position = torch.nonzero(
            seen_index == int(class_id), as_tuple=False).flatten()
        if target_position.numel() == 1 and seen_index.numel() > 1:
            target_position = int(target_position.item())
            target_logit = seen_logits[:, target_position]
            competitor_logits = seen_logits.clone()
            competitor_logits[:, target_position] = float('-inf')
            margin = target_logit - competitor_logits.max(dim=1).values
            selected.extend(torch.argsort(margin)[:hard_count].tolist())

    selected_mask = torch.zeros(bank_size, dtype=torch.bool, device=device)
    if selected:
        selected_mask[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
    remaining_count = sample_count - len(selected)
    available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
    if remaining_count > 0 and available.numel() > 0:
        take_count = min(remaining_count, available.numel())
        permutation = torch.randperm(available.numel(), device=device)[:take_count]
        selected.extend(available.index_select(0, permutation).tolist())
        remaining_count -= take_count
    if remaining_count > 0:
        selected.extend(torch.randint(bank_size, (remaining_count,), device=device).tolist())

    selected_index = torch.as_tensor(selected, dtype=torch.long, device=device)
    return feature_bank.index_select(0, selected_index)


@torch.no_grad()
def _sample_hybrid_class_replay(class_id, total_count, model, seen_classes,
                                old_classes, args, device):
    """Use one fixed per-class budget shared by real memory and CFS replay."""
    default_ratio = float(getattr(args, 'crct_real_replay_ratio', 0.25))
    if int(class_id) in old_classes:
        real_ratio = float(getattr(args, 'crct_real_old_replay_ratio', default_ratio))
    else:
        real_ratio = float(getattr(args, 'crct_real_new_replay_ratio', default_ratio))
    real_ratio = min(1.0, max(0.0, real_ratio))
    has_memory = int(class_id) in cls_real_features
    real_count = int(round(total_count * real_ratio)) if has_memory else 0
    real_count = min(total_count, max(0, real_count))
    synthetic_count = total_count - real_count
    synthetic_chunks = []

    means = cls_mean[class_id]
    covariances = cls_cov[class_id]
    if not isinstance(means, list):
        means = [means]
        covariances = [covariances]
    valid_components = []
    for mean, covariance in zip(means, covariances):
        covariance = torch.as_tensor(covariance, device=device)
        if covariance.numel() > 0 and float(covariance.float().abs().mean()) > 0.0:
            valid_components.append((mean, covariance))

    if synthetic_count > 0 and valid_components:
        base_count, remainder = divmod(synthetic_count, len(valid_components))
        for component_id, (mean, covariance) in enumerate(valid_components):
            component_count = base_count + int(component_id < remainder)
            if component_count <= 0:
                continue
            mean = torch.as_tensor(mean, device=device).float()
            covariance = covariance.float()
            if covariance.dim() == 1:
                covariance = torch.diag(covariance)
            covariance = covariance + torch.eye(
                mean.numel(), device=device, dtype=mean.dtype) * 1e-4
            synthetic_chunks.append(utils.sample_boundary_aware_cfs_features(
                mean, covariance, component_count, args, device, model,
                class_id, seen_classes, cfs_model=cls_cfs_model.get(class_id)))

    real_features = _sample_real_feature_memory(
        class_id, real_count, model, seen_classes, args, device)
    chunks = synthetic_chunks
    if real_features is not None:
        chunks.append(real_features)
    if not chunks:
        raise RuntimeError('No replay features available for class {}'.format(class_id))
    replay = torch.cat(chunks, dim=0)
    if replay.shape[0] != total_count:
        raise RuntimeError(
            'Hybrid replay generated {} instead of {} samples for class {}'.format(
                replay.shape[0], total_count, class_id))
    return replay, 0 if real_features is None else real_features.shape[0]


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        utils.distributed_barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)
        gathered_features_per_cls = torch.cat(features_per_cls_list, dim=0)
        if bool(getattr(args, 'crct_real_feature_replay', False)):
            cls_real_features[int(cls_id)] = _select_real_feature_memory(
                gathered_features_per_cls, args)
        if utils.use_cfs_sampling(args):
            cls_cfs_model[cls_id] = utils.train_cfs_model(gathered_features_per_cls, args, device)

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = gathered_features_per_cls.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


def _crct_old_class_distillation_loss(student_logits, teacher_logits, targets,
                                      seen_classes, old_classes, temperature=2.0,
                                      confidence_threshold=0.6):
    """Distill confident old-class replay samples while leaving boundary samples plastic."""
    zero = student_logits.new_zeros(())
    if student_logits.numel() == 0 or not old_classes or not seen_classes:
        return zero, 0

    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=student_logits.device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=student_logits.device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    if not bool(old_sample_mask.any()):
        return zero, 0

    student_seen = student_logits[old_sample_mask].index_select(1, seen_index)
    teacher_seen = teacher_logits[old_sample_mask].index_select(1, seen_index)
    temperature = max(float(temperature), 1e-6)

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_seen / temperature, dim=1)
        teacher_confidence_probs = F.softmax(teacher_seen, dim=1)
        class_positions = torch.full(
            (student_logits.shape[1],), -1, dtype=torch.long, device=student_logits.device)
        class_positions[seen_index] = torch.arange(
            seen_index.numel(), dtype=torch.long, device=student_logits.device)
        target_positions = class_positions[targets[old_sample_mask]]
        target_confidence = teacher_confidence_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)
        keep_mask = target_confidence >= float(confidence_threshold)

    kept_samples = int(keep_mask.sum().item())
    if kept_samples == 0:
        return zero, 0

    per_sample_kl = F.kl_div(
        F.log_softmax(student_seen[keep_mask] / temperature, dim=1),
        teacher_probs[keep_mask],
        reduction='none',
    ).sum(dim=1)
    return per_sample_kl.mean() * (temperature ** 2), kept_samples


def _crct_replay_reliability_weights(teacher_logits, targets, seen_classes,
                                     old_classes, floor=0.25, power=1.0,
                                     preserve_mass=False,
                                     preserve_class_mass=False):
    """Weight uncertain old-class synthetic samples without weakening new-class learning."""
    weights = teacher_logits.new_ones(targets.shape, dtype=torch.float32)
    if teacher_logits.numel() == 0 or not old_classes or not seen_classes:
        return weights, weights.new_zeros(()), 0

    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=targets.device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=targets.device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    old_sample_count = int(old_sample_mask.sum().item())
    if old_sample_count == 0:
        return weights, weights.new_zeros(()), 0

    with torch.no_grad():
        teacher_seen = teacher_logits[old_sample_mask].index_select(1, seen_index)
        teacher_probs = F.softmax(teacher_seen, dim=1)
        class_positions = torch.full(
            (teacher_logits.shape[1],), -1, dtype=torch.long, device=targets.device)
        class_positions[seen_index] = torch.arange(
            seen_index.numel(), dtype=torch.long, device=targets.device)
        target_positions = class_positions[targets[old_sample_mask]]
        target_confidence = teacher_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)

        floor = min(1.0, max(0.0, float(floor)))
        power = max(0.0, float(power))
        old_weights = floor + (1.0 - floor) * target_confidence.pow(power)
        if bool(preserve_class_mass):
            old_targets = targets[old_sample_mask]
            normalized_weights = old_weights.clone()
            for class_id in torch.unique(old_targets):
                class_sample_mask = old_targets == class_id
                class_weights = old_weights[class_sample_mask]
                normalized_weights[class_sample_mask] = (
                    class_weights / class_weights.mean().clamp_min(1e-12)
                )
            old_weights = normalized_weights
        elif bool(preserve_mass):
            old_weights = old_weights / old_weights.mean().clamp_min(1e-12)
        weights[old_sample_mask] = old_weights

    return weights, target_confidence.mean(), old_sample_count


def _scale_old_classifier_row_gradients(head, old_classes, scale):
    """Apply a smaller effective learning rate to consolidated old classifier rows."""
    if not old_classes:
        return
    scale = min(1.0, max(0.0, float(scale)))
    if scale >= 1.0:
        return

    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=head.weight.device)
    if head.weight.grad is not None:
        old_weight_grad = head.weight.grad.index_select(0, old_index) * scale
        head.weight.grad.index_copy_(0, old_index, old_weight_grad)
    if head.bias is not None and head.bias.grad is not None:
        old_bias_grad = head.bias.grad.index_select(0, old_index) * scale
        head.bias.grad.index_copy_(0, old_index, old_bias_grad)


@torch.no_grad()
def _build_crct_trust_anchors(class_means, class_covariances, old_classes,
                              args, device):
    """Draw independent high-density anchors from stored real-feature statistics."""
    samples_per_component = max(
        1, int(getattr(args, 'crct_trust_samples_per_component', 4)))
    covariance_scale = max(
        0.0, float(getattr(args, 'crct_trust_cov_scale', 0.25)))
    anchor_chunks = []
    anchor_labels = []

    for class_id in old_classes:
        stored_means = class_means.get(class_id)
        stored_covariances = class_covariances.get(class_id)
        if stored_means is None or stored_covariances is None:
            continue

        means = stored_means if isinstance(stored_means, list) else [stored_means]
        covariances = (
            stored_covariances
            if isinstance(stored_covariances, list)
            else [stored_covariances]
        )
        for component_mean, component_covariance in zip(means, covariances):
            mean = torch.as_tensor(component_mean, device=device).float()
            covariance = torch.as_tensor(component_covariance, device=device).float()
            component_anchors = [mean.unsqueeze(0)]
            random_count = samples_per_component - 1
            if random_count > 0 and covariance_scale > 0.0:
                if covariance.dim() == 1:
                    std = (covariance.clamp_min(1e-6) * covariance_scale).sqrt()
                    random_anchors = (
                        mean.unsqueeze(0)
                        + torch.randn(
                            random_count, mean.numel(), device=device,
                            dtype=mean.dtype) * std.unsqueeze(0)
                    )
                else:
                    scaled_covariance = covariance * covariance_scale
                    scaled_covariance = scaled_covariance + torch.eye(
                        mean.numel(), device=device, dtype=mean.dtype) * 1e-5
                    distribution = torch.distributions.MultivariateNormal(
                        mean, scaled_covariance)
                    random_anchors = distribution.sample((random_count,))
                component_anchors.append(random_anchors)

            component_anchors = torch.cat(component_anchors, dim=0)
            anchor_chunks.append(component_anchors)
            anchor_labels.extend([int(class_id)] * component_anchors.shape[0])

    if not anchor_chunks:
        return None, None
    return (
        torch.cat(anchor_chunks, dim=0),
        torch.as_tensor(anchor_labels, dtype=torch.long, device=device),
    )


@torch.no_grad()
def _select_crct_adaptive_trust_alpha(base_model, teacher_fc_norm, teacher_head,
                                      anchors, targets, seen_classes,
                                      old_classes, args):
    """Interpolate the whole classifier to satisfy worst-class logit-drift limits."""
    if anchors is None or targets is None or anchors.numel() == 0:
        return 1.0, None

    device = anchors.device
    seen_index = torch.as_tensor(seen_classes, dtype=torch.long, device=device)
    old_index = torch.as_tensor(old_classes, dtype=torch.long, device=device)
    old_sample_mask = (targets.unsqueeze(1) == old_index.unsqueeze(0)).any(dim=1)
    if not bool(old_sample_mask.any()):
        return 1.0, None

    anchors = anchors[old_sample_mask]
    targets = targets[old_sample_mask]
    teacher_logits = teacher_head(teacher_fc_norm(anchors)).index_select(1, seen_index)
    student_logits = base_model.head(base_model.fc_norm(anchors)).index_select(1, seen_index)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=1)
    teacher_probs = teacher_log_probs.exp()

    class_positions = torch.full(
        (base_model.head.out_features,), -1, dtype=torch.long, device=device)
    class_positions[seen_index] = torch.arange(
        seen_index.numel(), dtype=torch.long, device=device)
    target_positions = class_positions[targets]
    teacher_target_confidence = teacher_probs.gather(
        1, target_positions.unsqueeze(1)).squeeze(1)
    teacher_competitors = teacher_logits.clone()
    teacher_competitors.scatter_(
        1, target_positions.unsqueeze(1), float('-inf'))
    teacher_margin = (
        teacher_logits.gather(1, target_positions.unsqueeze(1)).squeeze(1)
        - teacher_competitors.max(dim=1).values
    )

    steps = max(1, int(getattr(args, 'crct_trust_steps', 10)))
    quantile = min(
        1.0, max(0.0, float(getattr(args, 'crct_trust_quantile', 0.9))))
    max_kl = max(0.0, float(getattr(args, 'crct_trust_max_kl', 0.02)))
    max_conf_drop = max(
        0.0, float(getattr(args, 'crct_trust_max_conf_drop', 0.02)))
    max_margin_drop = max(
        0.0, float(getattr(args, 'crct_trust_max_margin_drop', 0.10)))

    best_alpha = 0.0
    best_metrics = {'kl': 0.0, 'conf_drop': 0.0, 'margin_drop': 0.0}
    class_ids = torch.unique(targets, sorted=True)
    for step in range(steps + 1):
        alpha = step / float(steps)
        candidate_logits = teacher_logits.lerp(student_logits, alpha)
        candidate_log_probs = F.log_softmax(candidate_logits, dim=1)
        candidate_probs = candidate_log_probs.exp()
        sample_kl = (
            teacher_probs * (teacher_log_probs - candidate_log_probs)
        ).sum(dim=1)
        candidate_target_confidence = candidate_probs.gather(
            1, target_positions.unsqueeze(1)).squeeze(1)
        confidence_drop = (
            teacher_target_confidence - candidate_target_confidence
        ).clamp_min(0.0)
        candidate_competitors = candidate_logits.clone()
        candidate_competitors.scatter_(
            1, target_positions.unsqueeze(1), float('-inf'))
        candidate_margin = (
            candidate_logits.gather(1, target_positions.unsqueeze(1)).squeeze(1)
            - candidate_competitors.max(dim=1).values
        )
        margin_drop = (teacher_margin - candidate_margin).clamp_min(0.0)

        class_kl = []
        class_conf_drop = []
        class_margin_drop = []
        for class_id in class_ids:
            class_mask = targets == class_id
            class_kl.append(sample_kl[class_mask].mean())
            class_conf_drop.append(confidence_drop[class_mask].mean())
            class_margin_drop.append(margin_drop[class_mask].mean())
        kl_value = torch.quantile(torch.stack(class_kl), quantile).item()
        conf_drop_value = torch.quantile(
            torch.stack(class_conf_drop), quantile).item()
        margin_drop_value = torch.quantile(
            torch.stack(class_margin_drop), quantile).item()
        if (
                kl_value <= max_kl
                and conf_drop_value <= max_conf_drop
                and margin_drop_value <= max_margin_drop):
            best_alpha = alpha
            best_metrics = {
                'kl': kl_value,
                'conf_drop': conf_drop_value,
                'margin_drop': margin_drop_value,
            }

    return best_alpha, best_metrics


@torch.no_grad()
def _blend_crct_classifier(base_model, teacher_head, alpha):
    """Apply one shared interpolation factor to every classifier parameter."""
    for student_parameter, teacher_parameter in zip(
            base_model.head.parameters(), teacher_head.parameters()):
        student_parameter.copy_(teacher_parameter.lerp(student_parameter, alpha))


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    base_model = model.module if hasattr(model, 'module') else model
    head_only = bool(getattr(args, 'crct_head_only', False))
    fc_norm_grad_states = []
    if head_only:
        if not hasattr(base_model, 'head') or not hasattr(base_model, 'fc_norm'):
            raise AttributeError('Head-only CRCT requires model.head and model.fc_norm')
        fc_norm_grad_states = [
            (parameter, parameter.requires_grad)
            for parameter in base_model.fc_norm.parameters()
        ]
        for parameter, _ in fc_norm_grad_states:
            parameter.requires_grad_(False)
        param_list = [
            parameter for parameter in base_model.head.parameters()
            if parameter.requires_grad
        ]
    else:
        param_list = [
            p for n, p in model.named_parameters()
            if p.requires_grad and 'lora' not in n
        ]
    if not param_list:
        raise ValueError('CRCT has no trainable parameters')
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    old_classes = [int(cls_id) for seen_task in range(task_id) for cls_id in class_mask[seen_task]]
    seen_classes = [
        int(cls_id) for seen_task in range(task_id + 1) for cls_id in class_mask[seen_task]]
    crct_num = len(old_classes)

    distill_weight = max(0.0, float(getattr(args, 'crct_distill_weight', 0.25)))
    distill_temperature = max(
        1e-6, float(getattr(args, 'crct_distill_temperature', 2.0)))
    distill_confidence = min(
        1.0, max(0.0, float(getattr(args, 'crct_distill_confidence', 0.6))))
    distill_enabled = (
        bool(getattr(args, 'crct_old_class_distill', False))
        and distill_weight > 0.0
        and len(old_classes) > 0
    )
    anchor_weight = max(0.0, float(getattr(args, 'crct_anchor_weight', 0.0)))
    reliability_enabled = (
        bool(getattr(args, 'crct_reliability_weighting', False))
        and len(old_classes) > 0
    )
    reliability_floor = min(
        1.0, max(0.0, float(getattr(args, 'crct_reliability_floor', 0.25))))
    reliability_power = max(
        0.0, float(getattr(args, 'crct_reliability_power', 1.0)))
    reliability_preserve_mass = bool(
        getattr(args, 'crct_reliability_preserve_mass', False))
    reliability_preserve_class_mass = bool(
        getattr(args, 'crct_reliability_preserve_class_mass', False))
    hybrid_real_replay = bool(getattr(args, 'crct_real_feature_replay', False))
    if hybrid_real_replay and utils.use_semantic_projection(args):
        raise ValueError(
            'Hybrid real-feature replay and semantic projection must be evaluated separately')
    if hybrid_real_replay and not cls_real_features:
        raise RuntimeError('Real-feature replay is enabled but the feature memory is empty')
    hybrid_samples_per_class = int(
        getattr(args, 'crct_hybrid_samples_per_class', 0))
    trust_region_enabled = (
        bool(getattr(args, 'crct_adaptive_trust_region', False))
        and len(old_classes) > 0
    )
    if trust_region_enabled and not head_only:
        raise ValueError(
            'Adaptive CRCT trust region requires --crct_head_only')
    old_row_lr_scale = min(
        1.0, max(0.0, float(getattr(args, 'crct_old_row_lr_scale', 1.0))))

    teacher_fc_norm = None
    teacher_head = None
    if (distill_enabled or anchor_weight > 0.0 or reliability_enabled
            or trust_region_enabled):
        if not hasattr(base_model, 'head') or not hasattr(base_model, 'fc_norm'):
            raise AttributeError('Stability-aware CRCT requires model.head and model.fc_norm')
        teacher_fc_norm = copy.deepcopy(base_model.fc_norm).to(device).eval()
        teacher_head = copy.deepcopy(base_model.head).to(device).eval()
        for teacher_parameter in list(teacher_fc_norm.parameters()) + list(teacher_head.parameters()):
            teacher_parameter.requires_grad_(False)

    if utils.is_main_process() and (
            head_only or distill_enabled or anchor_weight > 0.0
            or reliability_enabled or trust_region_enabled or hybrid_real_replay
            or old_row_lr_scale < 1.0):
        print(
            'CRCT stability:',
            'head_only=', head_only,
            'old_classes=', len(old_classes),
            'distill_weight=', distill_weight if distill_enabled else 0.0,
            'temperature=', distill_temperature,
            'confidence=', distill_confidence,
            'anchor_weight=', anchor_weight,
            'reliability=', reliability_enabled,
            'reliability_floor=', reliability_floor,
            'reliability_power=', reliability_power,
            'reliability_preserve_mass=', reliability_preserve_mass,
            'reliability_preserve_class_mass=', reliability_preserve_class_mass,
            'adaptive_trust_region=', trust_region_enabled,
            'real_feature_replay=', hybrid_real_replay,
            'real_replay_ratio=', float(getattr(args, 'crct_real_replay_ratio', 0.25)),
            'real_old_new_ratio=', (float(getattr(args, 'crct_real_old_replay_ratio', 0.35)), float(getattr(args, 'crct_real_new_replay_ratio', 0.10))),
            'real_hard_ratio=', float(getattr(args, 'crct_real_hard_ratio', 0.5)),
            'hybrid_samples_per_class=', hybrid_samples_per_class,
            'old_row_lr_scale=', old_row_lr_scale,
        )


    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5
        if hybrid_real_replay and hybrid_samples_per_class > 0:
            num_sampled_pcls = hybrid_samples_per_class
        real_replay_total = 0

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if hybrid_real_replay:
            for seen_task in range(task_id + 1):
                for c_id in class_mask[seen_task]:
                    sampled_data_single, real_count = _sample_hybrid_class_replay(
                        c_id, num_sampled_pcls, model, seen_classes, old_classes, args, device)
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([c_id] * sampled_data_single.shape[0])
                    real_replay_total += real_count
        elif args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    projected_count = 0
                    if utils.use_semantic_projection(args):
                        projected_count = int(num_sampled_pcls * float(getattr(args, 'semantic_projection_ratio', 0.25)))
                        projected_count = max(0, min(num_sampled_pcls - 1, projected_count))
                    base_count = num_sampled_pcls - projected_count
                    sampled_data_single = utils.sample_boundary_aware_cfs_features(
                        mean.float(), cov.float(), base_count, args, device, model,
                        c_id, seen_classes, cfs_model=cls_cfs_model.get(c_id))
                    if projected_count > 0:
                        available_classes = []
                        for seen_task in range(task_id + 1):
                            available_classes.extend(class_mask[seen_task])
                        projected_data = utils.sample_semantic_projected_features(
                            c_id, mean.float(), projected_count, args, device,
                            cls_mean, cls_cov, cls_cfs_model=cls_cfs_model,
                            available_classes=available_classes)
                        if projected_data is not None:
                            sampled_data_single = torch.cat([sampled_data_single, projected_data], dim=0)
                        elif base_count < num_sampled_pcls:
                            fallback_data = utils.sample_cfs_features(
                                mean.float(), cov.float(), num_sampled_pcls - base_count, args, device,
                                cfs_model=cls_cfs_model.get(c_id))
                            sampled_data_single = torch.cat([sampled_data_single, fallback_data], dim=0)
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * sampled_data_single.shape[0])

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       cov = (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float()
                       projected_count = 0
                       if utils.use_semantic_projection(args):
                           projected_count = int(num_sampled_pcls * float(getattr(args, 'semantic_projection_ratio', 0.25)))
                           projected_count = max(0, min(num_sampled_pcls - 1, projected_count))
                       base_count = num_sampled_pcls - projected_count
                       sampled_data_single = utils.sample_boundary_aware_cfs_features(
                           mean.float(), cov, base_count, args, device, model,
                           c_id, seen_classes, cfs_model=cls_cfs_model.get(c_id))
                       if projected_count > 0:
                           available_classes = []
                           for seen_task in range(task_id + 1):
                               available_classes.extend(class_mask[seen_task])
                           projected_data = utils.sample_semantic_projected_features(
                               c_id, mean.float(), projected_count, args, device,
                               cls_mean, cls_cov, cls_cfs_model=cls_cfs_model,
                               available_classes=available_classes)
                           if projected_data is not None:
                               sampled_data_single = torch.cat([sampled_data_single, projected_data], dim=0)
                           elif base_count < num_sampled_pcls:
                               fallback_data = utils.sample_cfs_features(
                                   mean.float(), cov, num_sampled_pcls - base_count, args, device,
                                   cfs_model=cls_cfs_model.get(c_id))
                               sampled_data_single = torch.cat([sampled_data_single, fallback_data], dim=0)
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * sampled_data_single.shape[0])
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)
        if hybrid_real_replay and utils.is_main_process():
            print(
                'CRCT hybrid replay:',
                'epoch=', epoch + 1,
                'samples_per_class=', num_sampled_pcls,
                'real=', real_replay_total,
                'synthetic=', sampled_data.shape[0] - real_replay_total,
                'classes=', len(seen_classes),
            )

        inputs = sampled_data
        targets = sampled_label

        if bool(getattr(args, 'crct_balanced_batches', False)):
            sf_indexes = utils.class_balanced_replay_order(targets)
        else:
            sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        replay_iterations = crct_num
        if bool(getattr(args, 'crct_use_all_samples', False)):
            replay_iterations = int(math.ceil(inputs.size(0) / float(num_sampled_pcls)))
        print('CRCT replay iterations:', replay_iterations, 'of', int(math.ceil(inputs.size(0) / float(num_sampled_pcls))))

        for _iter in range(replay_iterations):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            teacher_logits = None
            if distill_enabled or reliability_enabled:
                with torch.no_grad():
                    teacher_logits = teacher_head(teacher_fc_norm(inp))

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            replay_weight = logits.new_ones(tgt.shape, dtype=torch.float32)
            old_teacher_confidence = logits.new_zeros(())
            old_replay_count = 0
            if reliability_enabled:
                replay_weight, old_teacher_confidence, old_replay_count = (
                    _crct_replay_reliability_weights(
                        teacher_logits,
                        tgt,
                        seen_classes,
                        old_classes,
                        floor=reliability_floor,
                        power=reliability_power,
                        preserve_mass=reliability_preserve_mass,
                        preserve_class_mass=reliability_preserve_class_mass,
                    )
                )
                per_sample_ce = F.cross_entropy(logits, tgt, reduction='none')
                loss_ce = (
                    per_sample_ce * replay_weight
                ).sum() / replay_weight.sum().clamp_min(1e-12)
            else:
                loss_ce = criterion(logits, tgt)
            loss_kd = logits.new_zeros(())
            kd_kept = 0
            if distill_enabled:
                loss_kd, kd_kept = _crct_old_class_distillation_loss(
                    logits,
                    teacher_logits,
                    tgt,
                    seen_classes,
                    old_classes,
                    temperature=distill_temperature,
                    confidence_threshold=distill_confidence,
                )

            loss_anchor = logits.new_zeros(())
            if anchor_weight > 0.0:
                old_class_index = torch.as_tensor(
                    old_classes, dtype=torch.long, device=base_model.head.weight.device)
                weight_delta = (
                    base_model.head.weight.index_select(0, old_class_index)
                    - teacher_head.weight.index_select(0, old_class_index)
                )
                loss_anchor = weight_delta.pow(2).sum(dim=1).mean()
                if base_model.head.bias is not None and teacher_head.bias is not None:
                    bias_delta = (
                        base_model.head.bias.index_select(0, old_class_index)
                        - teacher_head.bias.index_select(0, old_class_index)
                    )
                    loss_anchor = loss_anchor + bias_delta.pow(2).mean()

            loss = (
                loss_ce
                + distill_weight * loss_kd
                + anchor_weight * loss_anchor
            )
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            _scale_old_classifier_row_gradients(
                base_model.head, old_classes, old_row_lr_scale)
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(
                Loss=loss.item(),
                CE=loss_ce.item(),
                KD=loss_kd.item(),
                Anchor=loss_anchor.item(),
                KDKeep=kd_kept,
                ReplayW=replay_weight.mean().item(),
                OldConf=old_teacher_confidence.item(),
                OldReplay=old_replay_count,
            )
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()

    if trust_region_enabled:
        selected_alpha = 1.0
        trust_metrics = None
        trust_anchor_count = 0
        if not dist.is_initialized() or utils.is_main_process():
            trust_anchors, trust_targets = _build_crct_trust_anchors(
                cls_mean, cls_cov, old_classes, args, device)
            trust_anchor_count = 0 if trust_targets is None else trust_targets.numel()
            selected_alpha, trust_metrics = _select_crct_adaptive_trust_alpha(
                base_model, teacher_fc_norm, teacher_head,
                trust_anchors, trust_targets, seen_classes, old_classes, args)
        if dist.is_initialized():
            alpha_tensor = torch.tensor(selected_alpha, dtype=torch.float32, device=device)
            dist.broadcast(alpha_tensor, src=0)
            selected_alpha = float(alpha_tensor.item())
        _blend_crct_classifier(base_model, teacher_head, selected_alpha)
        if utils.is_main_process():
            print(
                'CRCT adaptive trust region:',
                'alpha=', selected_alpha,
                'anchors=', trust_anchor_count,
                'kl=', None if trust_metrics is None else trust_metrics['kl'],
                'conf_drop=', None if trust_metrics is None else trust_metrics['conf_drop'],
                'margin_drop=', None if trust_metrics is None else trust_metrics['margin_drop'],
            )

    for parameter, requires_grad in fc_norm_grad_states:
        parameter.requires_grad_(requires_grad)


def orth_loss(features, targets, device, args):
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        return args.reg * loss
        # return 0.

def add_gaussian_noise(tensor, mean=0., std=1.):
    noise = torch.randn_like(tensor) * std + mean
    tensor_noisy = tensor + 0.01*noise
    return tensor_noisy


def robust_loss(model, inputs, features, targets, devices, task_id, class_mask,index):
    all_old_logits = []
    bs = inputs.shape[0]
    mask = []
    for i in range(task_id+1):
        mask.extend(class_mask[i])

    
    for k in range(index.shape[1]):
        prompt_id = index[:,k]
        with torch.no_grad():

            output = model(inputs, task_id=prompt_id)
            
            old_logits = output['features']
        old_logits = 1*old_logits+0.0*features


        old_norm_features = F.normalize(output['features'], p=2, dim=1)
        old_similarity_matrix = torch.mm(old_norm_features, old_norm_features.t())
        # old_similarity_matrix.fill_diagonal_(1)
        old_similarity_matrix = torch.exp(old_similarity_matrix)
        old_similarity_matrix = old_similarity_matrix / old_similarity_matrix.sum(1, keepdim=True)
        #old_logits = F.softmax(old_logits,dim=1)
        #all_old_logits.append(old_logits)
        all_old_logits.append(old_similarity_matrix)

    
    output = all_old_logits
    return output
