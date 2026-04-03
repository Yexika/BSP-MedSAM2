import argparse
#cfg.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='Cervicalcancerdata_MedSAM2', type=str, help='experiment name')

    parser.add_argument('-vis', type=bool, default=1, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=True, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default='/root/autodl-tmp/Medical-SAM2-main/pretrained weight/MedSAM2_pretrain.pth', help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=1,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=640, help='image_size')
    parser.add_argument('-out_size', type=int, default=640, help='output_size')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='Cervicalcancerdata' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default='./checkpoints/sam2.1_hiera_small.pt', help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_s' , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=2, help='sam checkpoint address')
    parser.add_argument('-b', type=int, default=1
                        , help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('-weights', type=str, default ='/root/autodl-tmp/Medical-SAM2-main/pretrained weight/MedSAM2_pretrain.pth', help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')

    # ddp改动
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    # function.py里面添加的bce交叉损失和paper损失的动态选择，默认BCE
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'paper'],
                        help='Loss type: bce or paper')

    #parser = argparse.ArgumentParser(description='MedSAM2 Testing on Liver Dataset')

    parser.add_argument(
    '-data_path',
    type=str,
    default='/root/autodl-tmp/Medical-SAM2-main/data/Cervicalcancerdata/Output(no crop)/t2_sag',
    help='The path of segmentation data')

    # 可视化输出目录
    parser.add_argument('-vis_output_dir', type=str, default='./visualization_results',
                        help='Directory to save visualization results')
    # 可视化频率控制
    parser.add_argument('-vis_freq', type=int, default=10,
                        help='Frequency of saving visualizations (every N batches)')
    # 每轮最大可视化样本数
    parser.add_argument('-vis_max_samples', type=int, default=5,
                        help='Maximum number of samples to visualize per epoch')

    opt = parser.parse_args()
    return opt