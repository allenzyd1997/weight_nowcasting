import numpy as np
import os
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.measure import compare_ssim as ssim
import random
import time
from models.models import EncoderRNN
# from data.moving_mnist import MovingMNIST
import argparse
import cv2
import sys
from torch.nn.parallel import DistributedDataParallel as DDP
devices_list = [i for i in range(torch.cuda.device_count())]

#os.environ["CUDA_VISIBLE_DEVICES"] = '2,5,6,7'


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--n_epochs', type=int, default=20, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=1, help='')
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--gen_frm_dir', type=str, default='results')
parser.add_argument('--checkpoint_path', type=str, default='', help='folder for checkpoint')
parser.add_argument('--train_data_paths', type=str, default='/data1/shuliang/Radar_900/train/')
# parser.add_argument('--train_data_paths', type=str, default='/data4/shuliang/Dataset/Radar_900/train')
#parser.add_argument('--valid_data_paths', type=str, default='/data_set_medical/shuliang/Weather_Model_Update/Data_Weather')
parser.add_argument('--valid_data_paths', type=str, default='/data1/shuliang/Radar_900/test/')
# parser.add_argument('--valid_data_paths', type=str, default='/data4/shuliang/Dataset/Radar_900/test')
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=80000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=1 / 80000)

parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=30)
parser.add_argument('--img_width', type=int, default=896)
parser.add_argument('--img_channel', type=int, default=1)

args = parser.parse_args()

args.device = torch.cuda.device('cuda:'+str(args.local_rank))
dist.init_process_group(backend='nccl')
torch.cuda.set_device('cuda:'+str(args.local_rank))

train_Datase = TrainDataset(args.train_data_paths)
train_loader = DataLoader(dataset=train_Datase, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=False, drop_last=True)
test_Dataset = TestDataset(args.valid_data_paths)
test_loader = DataLoader(dataset=test_Dataset, batch_size=1, shuffle=False,
                         num_workers=4, pin_memory=False, drop_last=True)
   

def get_weight_symbol(target):
    weights=torch.ones(target.shape).to(target.device)
    balancing_weights=(torch.Tensor([5,10,15,20,25,30,35,40,45,150,155,160,165,170,175])).to(target.device)
    # balancing_weights=(torch.Tensor([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75])).to(target.device)
    threshold= (torch.Tensor([5,10,15,20,25,30,35,40,45,50,55,60,65,70])/80.0).to(target.device)
    for i, threshold in enumerate(threshold):
        weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) *(target >= threshold)
    return weights.float()
    
# def get_weight_symbol(target):
#     weights=torch.ones(target.shape).to(target.device)
#     balancing_weights=(torch.Tensor([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75])).to(target.device)
#     threshold= (torch.Tensor([5,10,15,20,25,30,35,40,45,50,55,60,65,70])/80.0).to(target.device)
#     for i, threshold in enumerate(threshold):
#         weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) *(target >= threshold)
#     return weights.float()

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length,
                      args.img_width, args.img_width,
                      args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width, args.img_width, args.img_channel))
    zeros = np.zeros((args.img_width, args.img_width, args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length,
                                  args.img_width, args.img_width,
                                  args.img_channel))
    return eta, real_input_flag


def train_on_batch(input_tensor, target_tensor, mask, encoder, encoder_optimizer, criterion):
    mask = torch.FloatTensor(mask).permute(0, 1, 4, 2, 3).contiguous().to('cuda:'+str(args.local_rank))
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0.0
    for ei in range(input_length - 1):
        target = input_tensor[:, ei, :, :, :]
        output_image = encoder(target)
        weight  = get_weight_symbol(input_tensor[:, ei + 1, :, :, :])
        # loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])
        loss += torch.sum(weight*((input_tensor[:, ei + 1, :, :, :] - output_image) ** 2)) + \
                torch.sum(weight*(torch.abs(input_tensor[:, ei + 1, :, :, :] - output_image)))

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence

    for di in range(target_length):
        output_image = encoder(decoder_input)
        target = target_tensor[:, di, :, :, :]
        weight  = get_weight_symbol(target)
        loss += torch.sum(weight*((target - output_image) ** 2)) + \
                torch.sum(weight*(torch.abs(target - output_image)))
        decoder_input = target * mask[:, di] + output_image * (1 - mask[:, di])

    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, n_epochs, print_every, eval_every):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    criterion = nn.MSELoss()
    itr = 0
    loss_epoch = 0
    t0 = time.time()
    eta = args.sampling_start_value
    for epoch in range(0, n_epochs):
        for i, out in enumerate(train_loader, 0):
            itr += 1
            # input_batch =  torch.Size([8, 20, 1, 64, 64])
            # input_tensor = out[1].to(device)
            # target_tensor = out[2].to(device)
            input_tensor = out[:, 0:10].permute(0, 1, 4, 2, 3).contiguous().float().to('cuda:'+str(args.local_rank))
            target_tensor = out[:, 10:].permute(0, 1, 4, 2, 3).contiguous().float().to('cuda:'+str(args.local_rank))
            eta, real_input_flag = schedule_sampling(eta, itr)
          
            encoder.module.set_initial(args.batch_size,input_tensor.device)
            loss = train_on_batch(input_tensor, target_tensor, real_input_flag, encoder, encoder_optimizer, criterion)
            loss_epoch += loss

            if itr % 500 == 0 and torch.distributed.get_rank() == 0:
                print('epoch ', epoch, ' loss ', loss_epoch, ' epoch time ', time.time() - t0)
                t0 = time.time()
                loss_epoch = 0

            if itr % 2500 == 0 and  torch.distributed.get_rank() == 0:
                mse, mae, ssim = evaluate(encoder, test_loader, itr)
                scheduler_enc.step(mse)

                stats = {}
                stats['net_param'] = encoder.state_dict()
                save_dir = os.path.join(args.save_dir, 'iter-' + str(itr))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                checkpoint_path = os.path.join(save_dir, 'model.ckpt' + '-' + str(itr))
                torch.save(stats, checkpoint_path)

    stats = {}
    stats['net_param'] = encoder.state_dict()
    save_dir = os.path.join(args.save_dir, 'iter-' + str(itr))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'model.ckpt' + '-' + str(itr))
    torch.save(stats, checkpoint_path)

    return train_losses


    
 
def evaluate(encoder, loader, itr):
    total_mse, total_mae, total_ssim, total_bce = 0, 0, 0, 0
    with torch.no_grad():
        for id, out in enumerate(loader, 0):
            # input_batch = torch.Size([8, 20, 1, 64, 64])
            # input_tensor = out[1].to(device)
            # target_tensor = out[2].to(device)
            input_tensor = out[:, 0:10].permute(0, 1, 4, 2, 3).contiguous().float().to('cuda:'+str(args.local_rank))
            target_tensor = out[:, 10:].permute(0, 1, 4, 2, 3).contiguous().float().to('cuda:'+str(args.local_rank))

            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length - 1):
                output_image = encoder(input_tensor[:, ei, :, :, :])

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            
            for di in range(target_length):
                output_image = encoder(decoder_input)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)  # for MM: (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0, 1)  # (batch_size,10, 1, 64, 64)

            

            # save prediction examples
#            if id < 200 and id%10==0:
            if id<10:
                
                path = os.path.join(args.gen_frm_dir, str(itr), str(id))
                if not os.path.exists(path):
                    os.makedirs(path)
                for i in range(10):
                    name = 'gt' + str(i + 1).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(input[0, i, :, :, :] * 80)
                    mask = img_gt < 1.0
                    img_gt = 255 * mask + (1 - mask) * img_gt
                    img_gt = np.transpose(img_gt, [1, 2, 0])
                    # print('img:', img_gt.shape)
                    # img_gt = cv2.resize(img_gt, (900, 900), interpolation=cv2.INTER_NEAREST)
                    # img_gt = img_gt[100:800, :]
                    cv2.imwrite(file_name, img_gt)

                for i in range(20):
                    name = 'gt' + str(i + 11).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(target[0, i, :, :, :] * 80)
                    mask = img_gt < 1.0
                    img_gt = 255 * mask + (1 - mask) * img_gt
                    img_gt = np.transpose(img_gt, [1, 2, 0])

                    # img_gt = cv2.resize(img_gt, (900, 900), interpolation=cv2.INTER_NEAREST)
                    # img_gt = img_gt[100:800, :]
                    cv2.imwrite(file_name, img_gt)

                for i in range(20):
                    name = 'pd' + str(i + 11).zfill(2) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = predictions[0, i, :, :, :]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 80)
                    mask = img_pd < 5.0
                    img_pd = 255 * mask + (1 - mask) * img_pd
                    img_pd = np.transpose(img_pd, [1, 2, 0])
                    # img_pd = cv2.resize(img_pd, (900, 900), interpolation=cv2.INTER_NEAREST)
                    # img_pd = img_pd[100:800, :]

                    cv2.imwrite(file_name, img_pd)
            
            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0], predictions[a, b, 0]) / (target.shape[0] * target.shape[1])

            cross_entropy = -target * np.log(predictions) - (1 - target) * np.log(1 - predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size * target_length)
            total_bce += cross_entropy
            if id > 199:
                break
    # print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ',
    #       total_ssim / len(loader), ' eval bce ', total_bce / len(loader))
    print('eval mse ', total_mse / 200, ' eval mae ', total_mae / 200, ' eval ssim ',
          total_ssim / 200, ' eval bce ', total_bce / 200)
    return total_mse / 200, total_mae / 200, total_ssim / 200


print('BEGIN TRAIN')
# encoder = EncoderRNN('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = EncoderRNN().cuda()
encoder = DDP(encoder, device_ids=['cuda:'+str(args.local_rank)], output_device='cuda:'+str(args.local_rank))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('encoder ', count_parameters(encoder))

if args.checkpoint_path != '':
    print('load model:', args.checkpoint_path)
    stats = torch.load(args.checkpoint_path)
    encoder.load_state_dict(stats['net_param'])
    mse, mae, ssim = evaluate(encoder, test_loader,25000)
#     plot_losses = trainIters(encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)

else:
    plot_losses = trainIters(encoder, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
    print(plot_losses)