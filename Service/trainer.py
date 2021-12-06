# coding=utf-8
import os
import shutil
import copy
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from functions_utils import load_model_and_parallel
from evaluator import evaluation


logger = logging.getLogger(__name__)


def save_model(opt, model, global_step=None):
    if global_step is None:
        output_dir = os.path.join(opt.output_dir, 'checkpoint-best')
    else:
        output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        shutil.rmtree(output_dir)
        # os.rmdir(os.path.join(output_dir, 'model.pt'))
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    logger.info(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def train_bacth(opt, service_model, batch_data, dev_info, info_dict):
    service_model, device = load_model_and_parallel(service_model, opt.gpu_ids)

    use_n_gpus = False
    if hasattr(service_model, "module"):
        use_n_gpus = True

    service_model.zero_grad()

    service_model.train()
    for key in batch_data.keys():
        batch_data[key] = batch_data[key].to(device)
    batch_data['inputs_embeds'].requires_grad_(True)
    # try:
    output, loss = service_model(**batch_data)[0]
    # except:
    #     print(batch_data)
    #     continue

    if use_n_gpus:
        loss = loss.mean()

    loss.backward()
    gradient = batch_data['inputs_embeds'].weight.grad
    torch.nn.utils.clip_grad_norm_(service_model.parameters(), opt.max_grad_norm)

    service_model.step()
    service_model.step()
    service_model.zero_grad()

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()

    logger.info('Train done')


def train_best(opt, service_model, train_dataset, dev_info, info_dict):
    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,
                              num_workers=8)

    service_model, device = load_model_and_parallel(service_model, opt.gpu_ids)

    use_n_gpus = False
    if hasattr(service_model, "module"):
        use_n_gpus = True

    t_total = len(train_loader) * opt.train_epochs

    service_optimizer, service_scheduler = build_optimizer_and_scheduler(opt, service_model, t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {opt.train_epochs}")
    logger.info(f"  Total training batch size = {opt.train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    service_model.zero_grad()

    save_steps = t_total // opt.train_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 20

    avg_loss = 0.
    max_f1 = 0.
    max_f1_step = 0
    metric_str = ''
    for epoch in range(opt.train_epochs):
        for step, batch_data in enumerate(train_loader):
            service_model.train()
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            # try:
            output, loss = service_model(**batch_data)[0]
            # except:
            #     print(batch_data)
            #     continue

            if use_n_gpus:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(service_model.parameters(), opt.max_grad_norm)

            service_model.step()
            service_model.step()
            service_model.zero_grad()

            global_step += 1

            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()

        tmp_metric_str, tmp_f1 = evaluation(service_model, dev_info, device,
                                            start_threshold=opt.start_threshold,
                                            end_threshold=opt.end_threshold)

        logger.info(f'In epoch {epoch}: {tmp_metric_str}')

        metric_str += f'In epoch {epoch}: {tmp_metric_str}' + '\n\n'

        if tmp_f1 > max_f1:
            max_f1 = tmp_f1
            max_f1_step = epoch
            save_model(opt, service_model)

    max_metric_str = f'Max f1 is: {max_f1}, in epoch {max_f1_step}'
    logger.info(max_metric_str)
    metric_str += max_metric_str + '\n'
    eval_save_path = os.path.join(opt.output_dir, 'eval_metric.txt')

    with open(eval_save_path, 'a', encoding='utf-8') as f1:
        f1.write(metric_str)

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()

    logger.info('Train done')