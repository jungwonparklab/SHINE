import os
import datetime
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from Nets.Blindspot_Net import *
from Nets.UDVD import *
from Nets.UDVD_double import *
from Nets.UNet import *
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from Trainer.TEM_denoiser_patch_main import TEM_denoiser_main
from Utils.patch_generator_5frame import *
from Utils.Dataloader_mrc import *
from Utils.Dataloader_N2V import *
from Utils.Dataloader_plain import *
from sys import platform
import torch
if platform == "Windows":
    os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
    windows = True
if platform == "Darwin":
    macos = True

torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('medium')
def maybe_str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "bf16":
        return arg
    raise argparse.ArgumentTypeError("x must be an int or 'bf16'")

def cli_main():
    pl.seed_everything(2222)

    # ------------
    # path_args
    # ------------

    parser = argparse.ArgumentParser()

    parser.add_argument('--common_path', type=str, default='./Experiment/Au_3x3_denoising')
    parser.add_argument('--training_path', type=str, default='./Datasets/Au')
    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--data_path_test', type=str, default='./Datasets/Au')
    parser.add_argument('--save_folder_name', type=str,
                        default='experiment')
    parser.add_argument('--version_folder_name', type=str,
                        default='3x3_blind_spot')

    # -------------
    # training_args
    # -------------

    parser.add_argument('--file_type', type=str, default='Image')  # Image or mrc, dm4, large, single
    parser.add_argument('--recursive_factor', type=float, default=1)    # how many times each image is trained within a single epoch
    parser.add_argument('--loss_function', type=str, default='L2')  # L2, L1
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--frame_num', type=int, default=5)
    parser.add_argument('--filter', type=int, default=64)
    parser.add_argument('--blocks', type=int, default=14)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--precision', type=maybe_str_or_int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str, default=None)

    # ------------
    # mrc_file and large only arguments
    # ------------
    
    parser.add_argument('--prepare_patch', type=int, default=0)
    parser.add_argument('--subset_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=1024)
    parser.add_argument('--patch_stride', type=int, default=768)
    parser.add_argument('--processor_num', type=int, default=20)
    parser.add_argument('--patches_folder', type=str, default=None)
    parser.add_argument('--gain_path', type=str, default=None)
    parser.add_argument('--patch_ratio', type=float, default=0.01)

    # ------------
    # select_model
    # ------------

    # 3x3_blind(default), 1x1_blind, 5x5_blind, N2V, UDVD
    parser.add_argument('--model', type=str, default='3x3_blind')

    # ------------
    # generated_folder_name
    # ------------

    time_stamp = datetime.datetime.now().strftime('%Y%m%d')
    parser.add_argument('--time_stamp', type=str, default=time_stamp)
    args = parser.parse_args()

    # -------------
    # prepare_patch
    # -------------

    if args.prepare_patch == True:
        os.makedirs(args.patches_folder, exist_ok=True)
        if args.file_type == 'mrc':
            generate_patch_memory_eficient_gainfix(args.training_path, args.gain_path, args.patches_folder, args.patch_size, args.
                                                   patch_stride, 1, processor_num=args.processor_num, ratio=args.patch_ratio, frame_num=args.frame_num)
            print('------generate_patch_finished------')
        if args.file_type == 'dm4':
            generate_patch_memory_eficient_dm4(args.training_path, args.gain_path, args.patches_folder, args.patch_size, args.
                                                   patch_stride, 1, processor_num=args.processor_num, ratio=args.patch_ratio, frame_num=args.frame_num)
            print('------generate_patch_finished------')
        if args.file_type == 'single_mrc':
            generate_patch_memory_eficient_gainfix(args.training_path, args.gain_path, args.patches_folder, args.patch_size, args.
                                                   patch_stride, 1, processor_num=args.processor_num, ratio=args.patch_ratio, frame_num=1)
            print('------generate_patch_finished------')
        if args.file_type == 'large':
            generate_patch_img(args.training_path, args.gain_path, args.patches_folder, args.patch_size, args.
                               patch_stride, 1, frames=args.frame_num, processor_num=args.processor_num, ratio=args.patch_ratio)
            print('------generate_patch_finished------')
        if args.file_type == 'single':
            generate_patch_img(args.training_path, args.gain_path, args.patches_folder, args.patch_size, args.
                               patch_stride, 1, frames=1, processor_num=args.processor_num, ratio=args.patch_ratio)
            print('------generate_patch_finished------')
    
    # ------------
    # model
    # ------------

    print('-----train_with_model_type-------', args.model)
    additional_dilation_i = 0
    additional_dilation_j = 0
    if args.model == 'UDVD' or args.file_type == 'UDVD_e':
        network = BlindVideoNet(channels_per_frame=args.in_channels, out_channels=args.out_channels, bias=False, blind=True, sigma_known=False)
    elif args.model == 'UDVD_e':
        network = BlindVideoNet_e(channels_per_frame=args.in_channels, out_channels=args.out_channels, bias=False, blind=True, sigma_known=False)
    elif args.model == 'N2V':
        network = UNet(in_channels=args.in_channels, out_channels=args.out_channels)
    else:
        additional_dilation_i = int(args.model.split('x')[0])//2
        additional_dilation_j = int(args.model.split('x')[1].split('_')[0])//2
        network = SHINE(args.in_channels, args.out_channels, add_dilation=(additional_dilation_i,additional_dilation_j), frame_num=args.frame_num, filter=args.filter, blocks=args.blocks, Bias=False)
    
    # ------------
    # dataloader_train_val
    # ------------

    train_loader = None
    validation_loader = None
    test_loader = None
    Trainset = None
    Validationset = None
    Testset = None
    mean_train = None
    std_train = None
    maximum_train = None
    if args.train:
        if args.file_type == 'mrc' or args.file_type == 'large' or args.file_type == 'single' or args.file_type == 'UDVD_mrc' or args.file_type == 'dm4':
            Trainset, Validationset = Sequentialloader(args.patches_folder, args.img_size, gt_path=args.gt_path,
                                                validation_length=2*args.batch_size, recursive_factor=args.recursive_factor, frame_num=args.frame_num)
            train_loader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                        num_workers=args.processor_num, drop_last=True, persistent_workers=True)
            validation_loader = DataLoader(Validationset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=args.processor_num, drop_last=False, persistent_workers=True)
        elif args.model == 'N2V':
            Trainset, Validationset = Sequentialloader_N2V(args.training_path, args.img_size, gt_path=args.gt_path,
                                                validation_length=2*args.batch_size, recursive_factor=args.recursive_factor)
            train_loader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                        num_workers=args.processor_num, drop_last=True, persistent_workers=True)
            validation_loader = DataLoader(Validationset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=args.processor_num, drop_last=False, persistent_workers=True)
        else:
            Trainset, Validationset = Sequentialloader_plain(args.training_path, args.img_size, gt_path=args.gt_path,
                                        validation_length=2, recursive_factor=args.recursive_factor, frame_num=args.frame_num)  
            train_loader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                        num_workers=args.processor_num, drop_last=True, persistent_workers=True)
            validation_loader = DataLoader(Validationset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=args.processor_num, drop_last=False, persistent_workers=True)
        #get mean and std of training set
        mean_train, std_train, maximum_train = Trainset.get_mean_std()
        
    # ------------
    # dataloader_test
    # ------------

    if args.test:
        if args.file_type == 'mrc':
            Testset = TestLoader_mrc(args.data_path_test, subset=args.subset_size, gain_dir=args.gain_path)
        elif args.file_type == 'dm4':
            Testset = TestLoader_dm4(args.data_path_test, subset=args.subset_size, gain_dir=args.gain_path, frames=args.frame_num)
        elif args.file_type == 'large':
            Testset = TestLoader_large(args.data_path_test, subset=args.subset_size, frame_num=args.frame_num)
        elif args.file_type == 'single':
            Testset = TestLoader_single(args.data_path_test, subset=args.subset_size)
        elif args.file_type == 'UDVD_mrc':
            Testset = TestLoader_mrc(args.data_path_test, subset=args.subset_size, gain_dir=args.gain_path)
        else:
            Testset = TestLoader_plain(args.data_path_test, frame_num=args.frame_num)

        
        # ------------
        # define_datamodule
        # ------------
        
        class Dataloader_denoiser(pl.LightningDataModule):        
            def __init__(self):
                super().__init__() 
            def train_dataloader(self):
                return train_loader
            def val_dataloader(self):
                return validation_loader
            

        # ------------
        # define_model
        # ------------
        
        model = TEM_denoiser_main(
                network=network,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                frame_num=args.frame_num,
                img_size=args.img_size,
                training_path=args.training_path,
                save_folder=args.common_path+'/'+args.save_folder_name,
                time_stamp=args.time_stamp,
                model_type=args.model,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                lossF=args.loss_function,
                beta1=args.beta1,
                beta2=args.beta2,
                eps=args.eps,
                weight_decay=args.weight_decay,
                total_epochs=args.max_epochs,
                trainset=Trainset,
                validationset=Validationset,
                testset=Testset,
                mean_train=mean_train,
                std_train=std_train,
                maximum_train=maximum_train,
                additional_dilation_i=additional_dilation_i,
                additional_dilation_j=additional_dilation_j,
            )
    # ------------
    # training
    # ------------    
    #     
    full_folder = args.save_folder_name+args.time_stamp
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.common_path, full_folder) + '/model', save_last=False, every_n_epochs=1,
                                          filename='{epoch}', monitor='val_loss', auto_insert_metric_name=True, 
                                          save_top_k=3, verbose=True)
    logger = TensorBoardLogger(
        args.common_path, name=full_folder, version=args.version_folder_name, max_queue=100)
    if platform == "Darwin":
        trainer = pl.Trainer(default_root_dir=full_folder+'/checkpoint', precision=args.precision, accelerator='mps',
                            callbacks=checkpoint_callback, num_sanity_val_steps=2,
                            max_epochs=args.max_epochs, logger=logger)
    elif platform == "Windows":
        trainer = pl.Trainer(default_root_dir=full_folder+'/checkpoint', precision=args.precision, accelerator='auto', devices=args.gpus,
                        callbacks=checkpoint_callback, num_sanity_val_steps=2,
                        max_epochs=args.max_epochs, logger=logger, strategy="dp")
        print("-----------windows is not supporting ddp strategy, using dp instead---------")
    else:
        strategy = DDPStrategy(find_unused_parameters=True)
        trainer = pl.Trainer(default_root_dir=full_folder+'/checkpoint', precision=args.precision, accelerator='auto', devices=args.gpus,
                        callbacks=checkpoint_callback, num_sanity_val_steps=2, 
                        max_epochs=args.max_epochs, logger=logger, strategy='auto')
        
    if args.train:

        # ------------
        # define_test_datamodule
        # ------------

        if args.ckpt_path is None:
            if args.file_type == 'mrc' or args.file_type == 'dm4' or args.file_type == 'large' or args.file_type == 'single' or args.file_type == 'UDVD_mrc':
                dm = Dataloader_denoiser()
                trainer.fit(model)
            else:
                dm = Dataloader_denoiser()
                trainer.fit(model)
        else:
            if args.file_type == 'mrc' or args.file_type == 'dm4' or args.file_type == 'large' or args.file_type == 'single' or args.file_type == 'UDVD_mrc':
                dm = Dataloader_denoiser()
                trainer.fit(model,  ckpt_path=args.ckpt_path)
            else:
                trainer.fit(model, ckpt_path=args.ckpt_path)

    # ------------
    # testing
    # ------------

    if args.test:

        class Dataloader_denoiser_test(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
            def test_dataloader(self):
                return DataLoader(Testset, batch_size=1, shuffle=False, pin_memory=False)
            def predict_dataloader(self):
                return DataLoader(Testset, batch_size=1, shuffle=False, pin_memory=False)
    

        if args.ckpt_path is not None:
            if args.file_type == 'mrc' or args.file_type == 'UDVD_mrc':
                dm = Dataloader_denoiser_test()
                trainer.test(model, ckpt_path=args.ckpt_path)
            else:
                dm = Dataloader_denoiser_test()
                trainer.predict(model, ckpt_path=args.ckpt_path)
        else:
            if args.file_type == 'mrc' or args.file_type == 'UDVD_mrc':
                dm = Dataloader_denoiser_test()
                trainer.test(model, ckpt_path="best")
            else:
                dm = Dataloader_denoiser_test()
                trainer.predict(model, ckpt_path="best")


if __name__ == '__main__':
    cli_main()
