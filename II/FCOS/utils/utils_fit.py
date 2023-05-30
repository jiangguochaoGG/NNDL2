import os
import torch
from tqdm import tqdm

from utils.utils import get_lr

debug = False

def fit_one_epoch(model_train, model, fcos_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss  = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if debug:
            if iteration >= 3:
                break
        if iteration >= epoch_step:
            break
        images, bboxes, classes = batch[0], batch[1],  batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                bboxes  = bboxes.cuda(local_rank)
                classes = classes.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #-------------------#
            #   获得预测结果
            #-------------------#
            outputs = model_train(images)
            #-------------------#
            #   计算损失
            #-------------------#
            loss = fcos_loss(outputs, bboxes, classes)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #-------------------#
                #   获得预测结果
                #-------------------#
                outputs = model_train(images)
                #-------------------#
                #   计算损失
                #-------------------#
                loss = fcos_loss(outputs, bboxes, classes)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : total_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if debug:
            if iteration >= 3:
                break
        if iteration >= epoch_step_val:
            break
        images, bboxes, classes = batch[0], batch[1],  batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                bboxes  = bboxes.cuda(local_rank)
                classes = classes.cuda(local_rank)

            optimizer.zero_grad()
            outputs     = model_train(images)
            loss        = fcos_loss(outputs, bboxes, classes)
            val_loss    += loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        loss_history.append_map(epoch + 1, eval_callback.maps[-1])
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))