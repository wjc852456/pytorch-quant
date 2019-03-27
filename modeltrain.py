from __future__ import print_function, division

import torch
import torch.nn as nn
import time
import os
import argparse
#import piexif
from utee import quant, misc
from imagenet import dataset
#from read_ImageNetData import ImageNetData
import torch.optim as optim
from torch.optim import lr_scheduler
ds_fetcher = dataset.get


def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes, dataloders, device_ids):
    since = time.time()
    resumed = False

    best_model_wts = model.state_dict()

    val_ds_tmp = ds_fetcher(batch_size=8, 
                            data_root=args.data_root, 
                            train=False, 
                            val = True,
                            shuffle=args.shuffle,
                            input_size=args.input_size)

    for epoch in range(args.start_epoch, num_epochs + 1):
        print("qauntize activation")
        model = model.module
        model = torch.nn.DataParallel(model.cuda(), device_ids=[device_ids[0]])
        quant.add_counter(model, args.n_sample)
        misc.eval_model(model, val_ds_tmp, device_ids=device_ids[0], n_sample=args.n_sample)

        model = model.module
        model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

        for phase in ['train','val']:
            if phase == 'train':
                print("train phase")
                scheduler.step(epoch)
                model.train(True)  # Set model to training mode

                running_loss = 0.0
                running_corrects = 0

                tic_batch = time.time()
                # Iterate over data for 1 epoch
                for i, (inputs, labels) in enumerate(dataloders[phase]):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                    batch_loss = running_loss / ((i+1)*args.batch_size)
                    batch_acc = float(running_corrects) / ((i+1)*args.batch_size)
                    if i % args.print_freq == 0:
                        print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.8f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                            epoch, num_epochs, 
                            i, round(dataset_sizes[phase])-1, 
                            scheduler.get_lr()[0], phase, batch_loss, batch_acc,
                            args.print_freq/(time.time()-tic_batch)))
                        tic_batch = time.time()
                    #if i>=3:
                    #    break

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float(running_corrects) / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            else:
                print("val phase")
                model.eval()  # Set model to evaluate mode
                acc1, acc5 = misc.eval_model(model, dataloders[phase], device_ids=device_ids)
                res_str = "epoch={}, type={}, quant_method={}, \n \
                            param_bits={}, fwd_bits={},\n \
                            acc1={:.4f}, acc5={:.4f}".format(
                            epoch, args.type, args.quant_method, 
                            args.param_bits,args.fwd_bits, acc1, acc5)
                print(res_str)
                with open(str(args.param_bits)+"-"+str(args.fwd_bits)+'bits_quant_acc1_acc5.txt','a') as f:
                    f.write(res_str + '\n')

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str, default="/home/jcwang/dataset/imagenet")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_class', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_epoch_freq', type=int, default=1)
    parser.add_argument('--save_path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start_epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    # read data
    # dataloders, dataset_sizes = ImageNetData(args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    #model = mobilenetv2_19(num_classes = args.num_class)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00004)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.98)

    model = train_model(args=args,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer_ft,
                        scheduler=exp_lr_scheduler,
                        num_epochs=args.num_epochs,
                        dataset_sizes=dataset_sizes)
