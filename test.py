import argparse
import typing
import io
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS

from datasets import build_dataset
import seaborn as sns

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import math
import time


def get_args_parser():
    parser = argparse.ArgumentParser('ViT evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    # parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def token_rank(input_arr, name, path, block_num):
    ranked_token_value_idx = np.argsort(-input_arr) # ranked_avg_token_value_idx: each element M(i) indicates the token with the ranking (i)
    np.savetxt(path + '/{}_{}.txt'.format(name, block_num), np.argsort(-input_arr), fmt='%d')

def get_rank(avg_attention, path, block_num):
    token_value_1 = np.sum(avg_attention, axis = 1)[1:]
    print('token_value_1 is {}'.format(token_value_1))
    token_value_2 = np.sum(avg_attention, axis = 0)[1:]
    token_value_sum = token_value_1 + token_value_2
    token_value_1minus2 = token_value_1 - token_value_2
    token_value_2minus1 = token_value_2 - token_value_1
    if not os.path.isdir(path + '/token_value_1'):
        os.mkdir(path + '/token_value_1')
    if not os.path.isdir(path + '/token_value_2'):
        os.mkdir(path + '/token_value_2')  
    if not os.path.isdir(path + '/token_value_sum'):
        os.mkdir(path + '/token_value_sum')  
    if not os.path.isdir(path + '/token_value_2minus1'):
        os.mkdir(path + '/token_value_2minus1')      
    token_rank(token_value_1, 'token_value_1', path + '/token_value_1', block_num)
    token_rank(token_value_2, 'token_value_2', path + '/token_value_2', block_num)
    token_rank(token_value_sum, 'token_value_sum', path + '/token_value_sum', block_num)
    # token_rank(token_value_1minus2, 'token_value_1minus2', path, block_num)
    token_rank(token_value_2minus1, 'token_value_2minus1', path + '/token_value_2minus1', block_num)
    return [token_value_1, token_value_2, token_value_sum, token_value_2minus1]


def plot_chunk_img(im, name, chunk_num, path):
    im_chunk = torch.chunk(im, chunk_num, dim = -2)

    chunk_list = []
    for item in im_chunk:
        im_chunk_chunk = torch.chunk(item, chunk_num, dim = -1)
        chunk_list.append(im_chunk_chunk)
    chunk_list = [chunk_chunk for chunk in chunk_list for chunk_chunk in chunk]

    fig, axes = plt.subplots(nrows = chunk_num, ncols = chunk_num, figsize=(20, 20))
    plt.setp(axes, xticks=[], yticks=[])
    for row_idx, ax_row in enumerate(axes):
        for col_idx, ax in enumerate(ax_row):
            ax.set_title('{}'.format(row_idx * chunk_num + col_idx) ) 
            im = transforms.ToPILImage()(chunk_list[row_idx * chunk_num + col_idx]).convert('RGB')
            ax.imshow(im)
    plt.savefig(path + '/{}.jpg'.format(name))
    plt.close(fig)    

def plot_chunk_img_arr(im_arr, name, chunk_num, path):
    im = Image.fromarray(im_arr.astype('uint8'))
    im = np.array(im)
    print('im shape {}'.format(im.shape))
    im_chunk = np.split(im, chunk_num, axis = 0)
    chunk_list = []
    for item in im_chunk:
        im_chunk_chunk = np.split(item, chunk_num, axis = 1)
        chunk_list.append(im_chunk_chunk)
    chunk_list = [chunk_chunk for chunk in chunk_list for chunk_chunk in chunk]

    fig, axes = plt.subplots(nrows = chunk_num, ncols = chunk_num, figsize=(20, 20))
    plt.setp(axes, xticks=[], yticks=[])
    for row_idx, ax_row in enumerate(axes):
        for col_idx, ax in enumerate(ax_row):
            ax.set_title('{}'.format(row_idx * chunk_num + col_idx) ) 
            ax.imshow(chunk_list[row_idx * chunk_num + col_idx])
    plt.savefig(path + '/{}.jpg'.format(name))
    plt.close(fig) 

def plot_token_value(avg_attention, ori_path, save_path, chunk_num):
    name_list = ['token_value_1', 'token_value_2', 'token_value_sum', 'token_value_2minus1']
    for mask, name in zip(avg_attention, name_list):
        mask = mask.reshape(chunk_num, chunk_num)
        mask = cv2.resize(mask / mask.max(), (224,224))
        mask = np.uint8(255 * mask)

        plt.imsave(ori_path + '/{}.jpg'.format(name), mask)
        plt.close()
        mask = Image.open(ori_path + '/{}.jpg'.format(name))
        mask = np.array(mask)
    
        plot_chunk_img_arr(mask, 'mask_{}'.format(name), chunk_num, save_path) 

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct[:1]

def validate(val_loader, model, criterion, batch_size):
    imagenet_labels = dict(enumerate(open('/home/yifan/github/DynamicViT/ilsvrc2012_wordnet_lemmas.txt')))
    if not os.path.isdir('/data/yifan/ViT'):
        os.mkdir('/data/yifan/ViT')

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            print('the value of i is {}'.format(i))
            # print('the shape of images is {}'.format(images.shape))
            images = images.cuda()
            target = target.cuda()
            handle_list = []
                    
            # compute output
            output, att_mat_list = model(images)

            probs = torch.nn.Softmax(dim=-1)(output)
            top5_idx = torch.argsort(probs, dim=-1, descending=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, correct = accuracy(output, target, topk=(1, 5))
            acc1 = acc[0]
            acc5 = acc[1]
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            correct_arr = np.array(correct[0].cpu().detach())


            att_mat_list = torch.stack(att_mat_list).squeeze(1) 
            

            plot_count = 0
            image_label_list = []
            for image_idx in range(images.shape[0]):
                if correct_arr[image_idx] == True: # Only save the attention, pred_score, heatmap for correct predicted images
                    image_label = imagenet_labels[top5_idx[image_idx, 0].item()].replace('\n','')
                    if image_label not in image_label_list:
                        image_label_list.append(image_label)
                        plot_count = 0
                    att_mat = att_mat_list[:, image_idx] # take the pic image_idx from the batch
                    print('the shaoe of att mat is {}'.format(att_mat.shape))
                    print('the shape of att_mat_list is {}'.format(att_mat_list.shape)) # the shape of att_mat_list is torch.Size([12, 64, 12, 197, 197]) [layer, image_idx, attention_heads, token #, token #]
                    att_mat = att_mat.squeeze(1) 

                    att_mat = torch.mean(att_mat, dim=1) # avg the attn across the attention heads
                    for block_num in range(att_mat_list.shape[0]):
                        temp = np.array(att_mat_list[block_num].cpu())

                        #-----create the dir for saving attention_value, pred_score, and heatmap--------
                        if not os.path.isdir('/data/yifan/ViT/{}'.format(image_label)):
                            os.mkdir('/data/yifan/ViT/{}'.format(image_label))
                        if not os.path.isdir('/data/yifan/ViT/{}/{}'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}'.format(image_label, batch_size * i + image_idx))

                        if not os.path.isdir('/data/yifan/ViT/{}/{}/attention_value'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}/attention_value'.format(image_label, batch_size * i + image_idx))

                        if not os.path.isdir('/data/yifan/ViT/{}/{}/token_rank'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}/token_rank'.format(image_label, batch_size * i + image_idx))

                        if not os.path.isdir('/data/yifan/ViT/{}/{}/attn_fig'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}/attn_fig'.format(image_label, batch_size * i + image_idx))

                        if not os.path.isdir('/data/yifan/ViT/{}/{}/ori_fig'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}/ori_fig'.format(image_label, batch_size * i + image_idx))

                        if not os.path.isdir('/data/yifan/ViT/{}/{}/token_fig'.format(image_label, batch_size * i + image_idx)):
                            os.mkdir('/data/yifan/ViT/{}/{}/token_fig'.format(image_label, batch_size * i + image_idx))

                        # # save the attention value
                        # np.save('/data/yifan/ViT/{}/{}/attention_value/{}.npy'.format(image_label, batch_size * i + image_idx, block_num), temp[image_idx])

                    #     #----------------plot the heatmap for every block---------------
                    #     if plot_count < 10:
                    #         heatmap_fig = sns.heatmap(att_mat[block_num].cpu().numpy()[1:, 1:], cmap='YlGnBu')
                    #         heatmap_fig = heatmap_fig.get_figure()
                    #         heatmap_fig.savefig('/data/yifan/ViT/{}/{}/attn_fig/heatmap_block{}.pdf'.format(image_label, batch_size * i + image_idx, block_num))
                    #         plt.close(heatmap_fig)
                    
                        if plot_count < 10:
                            avg_attention = np.mean(temp[image_idx], axis = 0) 
                            token_value_path = '/data/yifan/ViT/{}/{}/token_rank'.format(image_label, batch_size * i + image_idx, block_num)
                            get_rank(avg_attention, token_value_path, block_num)

                    #     #------------plot the avg heatmap---------------
                    #     avg_attn_map = torch.mean(att_mat, dim = 0) # average the attn map across the 12 blocks
                    #     avg_attn_map_array = avg_attn_map.cpu().numpy()[1:, 1:] # take away the cls_token
                    #     heatmap_fig = sns.heatmap(avg_attn_map_array, cmap='YlGnBu')
                    #     heatmap_fig = heatmap_fig.get_figure()
                    #     heatmap_fig.savefig('/data/yifan/ViT/{}/{}/attn_fig/heatmap.pdf'.format(image_label, batch_size * i + image_idx))
                    #     plt.close(heatmap_fig)



                        #------------cal the attention------------------
                        # To account for residual connections, we add an identity matrix to the
                        # attention matrix and re-normalize the weights.
                        residual_att = torch.eye(att_mat.size(1)).cuda()
                        aug_att_mat = att_mat + residual_att
                        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

                        # Recursively multiply the weight matrices
                        joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
                        joint_attentions[0] = aug_att_mat[0]
                        
                        print('The size of aug_att_mat is {}'.format(aug_att_mat.shape))
                        for n in range(1, aug_att_mat.size(0)):
                            # print('the shape of aug_attn_mat[n] is {} and joint_attention[n-1] is {}'.format(aug_att_mat[n].shape, joint_attentions[n-1].shape))
                            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

                        #------------plot the attention------------------
                        for k, v in enumerate(joint_attentions):
                            grid_size = int(math.sqrt(att_mat_list.shape[-1] - 1))
                            # Attention from the output token to the input space.
                            print('the shape of v is {}'.format(v.shape))
                            mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
                            mask = cv2.resize(mask / mask.max(), (224,224))
                            mask = np.uint8(255 * mask)
                            
                            plt.imsave('/data/yifan/ViT/{}/{}/ori_fig/{}.jpg'.format(image_label, batch_size * i + image_idx, (k+1)), mask)
                            plt.close()
                            mask = Image.open('/data/yifan/ViT/{}/{}/ori_fig/{}.jpg'.format(image_label, batch_size * i + image_idx, (k+1)))
                            mask = np.array(mask)
                            # print('the shape of mask is {}'.format(mask.shape))
                            plot_chunk_img_arr(mask, 'mask_{}'.format(k), grid_size, '/data/yifan/ViT/{}/{}/token_fig'.format(image_label, batch_size * i + image_idx)) 
                        #     result = (mask * im).astype("uint8")

                        #     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
                        #     ax1.set_title('Original')
                        #     ax2.set_title('Attention Map_%d Layer' % (i+1))
                        #     _ = ax1.imshow(im)
                        #     _ = ax2.imshow(result)
                        #     _ = ax3.imshow(mask)
                        #     plt.savefig('/data/yifan/ViT/{}/{}/attn_fig/{}.jpg'.format(image_label, batch_size * i + image_idx, (k+1)))
                        #     plt.close(fig)                       


                    avg_attn_map_arr = np.mean(np.array(att_mat.cpu()), axis = 0)
                    avg_attention = get_rank(avg_attn_map_arr, token_value_path, 'avg')

                    plot_token_value(avg_attention, '/data/yifan/ViT/{}/{}/ori_fig'.format(image_label, batch_size * i + image_idx), '/data/yifan/ViT/{}/{}/token_fig'.format(image_label, batch_size * i + image_idx), grid_size)

                    #-----------------------plot the split img-------------------------------
                    
                    unorm = UnNormalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                    im = unorm(images[image_idx])
                    plot_chunk_img(im, 'ori', grid_size, '/data/yifan/ViT/{}/{}/token_fig'.format(image_label, batch_size * i + image_idx))   
                    plot_count = plot_count + 1

                    # avg_token_value_1 = torch.sum(avg_attn_map, dim = 1).reshape(-1)[1:] # remove the cls_token
                    # avg_token_value_2 = torch.sum(avg_attn_map, dim = 0).reshape(-1)[1:]
                    # avg_token_value = avg_token_value_1 + avg_token_value_2

                    # ranked_avg_token_value, ranked_avg_token_value_idx = avg_token_value.sort(dim=0, descending=True) # ranked_avg_token_value_idx: each element M(i) indicates the token with the ranking (i)
                    # # print(ranked_avg_token_value_idx)
                    # _, avg_token_value_ranking = ranked_avg_token_value_idx.sort(dim=0) # each element M(i) indicates the ranking of the token (i)
                    # # print(avg_token_value_ranking)

                    # max_attn_map = np.max(avg_attn_map_array)
                    # min_attn_map = np.min(avg_attn_map_array)
                    # diag_avg_attn_map_array = avg_attn_map_array.diagonal()
                    
                    
                    # avg_attn_map_array = avg_attn_map_array - np.diag(diag_avg_attn_map_array)
                    
                    # print('the max attn map is {} and the min attn map is {}'.format(max_attn_map, min_attn_map))

                    im = unorm(images[image_idx])
                    im_PIL = transforms.ToPILImage()(im).convert('RGB')
                    im_PIL.save('/data/yifan/ViT/{}/{}/ori_fig/ori.jpg'.format(image_label, batch_size * i + image_idx))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)
            
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


    return top1.avg

def main(args):

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    os.makedirs("attention_data", exist_ok=True)
    if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
        urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")
    if not os.path.isfile("attention_data/ViT-B_16-224.npz"):
        urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")

    imagenet_labels = dict(enumerate(open('/home/yifan/github/DynamicViT/ilsvrc2012_wordnet_lemmas.txt')))



    # Prepare Model
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
    model.load_from(np.load("/data/yifan/models/imagenet21k+imagenet2012/ViT-B_16-224.npz"))
    model = model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(data_loader_val, model, criterion, args.batch_size)

# logits, att_mat = model(x.unsqueeze(0))

# att_mat = torch.stack(att_mat).squeeze(1)

# # Average the attention weights across all heads.
# att_mat = torch.mean(att_mat, dim=1)

# # To account for residual connections, we add an identity matrix to the
# # attention matrix and re-normalize the weights.
# residual_att = torch.eye(att_mat.size(1))
# aug_att_mat = att_mat + residual_att
# aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# # Recursively multiply the weight matrices
# joint_attentions = torch.zeros(aug_att_mat.size())
# joint_attentions[0] = aug_att_mat[0]

# for n in range(1, aug_att_mat.size(0)):
#     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
# # Attention from the output token to the input space.
# v = joint_attentions[-1]
# grid_size = int(np.sqrt(aug_att_mat.size(-1)))
# mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
# mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
# result = (mask * im).astype("uint8")

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# _ = ax1.imshow(im)
# _ = ax2.imshow(result)

# probs = torch.nn.Softmax(dim=-1)(logits)
# top5 = torch.argsort(probs, dim=-1, descending=True)
# print("Prediction Label and Attention Map!\n")
# for idx in top5[0, :5]:
#     print(f'{probs[0, idx.item()]:.5f} : {imagenet_labels[idx.item()]}', end='')

# for i, v in enumerate(joint_attentions):
#     # Attention from the output token to the input space.
#     mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
#     mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
#     result = (mask * im).astype("uint8")

#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
#     ax1.set_title('Original')
#     ax2.set_title('Attention Map_%d Layer' % (i+1))
#     _ = ax1.imshow(im)
#     _ = ax2.imshow(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)