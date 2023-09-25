import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util

# from . import loss
from torch.autograd import Variable
import itertools

from .imagepool import ImagePool


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True,lev include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        
        ##########################################################################################
        parser.add_argument('--num_outcomes', type=int, default=30, help='Number of outcomes of D.')
        parser.add_argument('--positive_skew', type=float, default=1.0, help='Skewness of anchor1 when computing loss.')
        parser.add_argument('--negative_skew', type=float, default=-1.0, help='Skewness of anchor0 when computing loss.')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'edge_A_B']

        #self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'perpetual']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_B_in', 'attn_real_A']#,'attn_fake_B'
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            # self.model_names = ['G', 'F', 'D', 'A_A']
            self.model_names = ['G', 'F', 'D', 'A_A', 'A_B']
        else:  # during test time, only load G
            # self.model_names = ['G', 'A_A']
            self.model_names = ['G', 'A_A', 'A_B']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        #####################5.13
        self.netA_A = networks.define_A(opt.input_nc, 1, opt.ngf, opt.netA, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netA_B = networks.define_A(opt.input_nc, 1, opt.ngf, opt.netA, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        if self.isTrain:
            self.fake_B_pool = ImagePool(50)#########5.23
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionPer = networks.PerpetualLoss(self.gpu_ids)

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # self.Triplet_Loss = loss.CategoricalLoss(opt.num_outcomes, opt.positive_skew, opt.negative_skew).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            ####################5.13
            # self.optimizer_A = torch.optim.Adam(self.netA_A.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_A = torch.optim.Adam(itertools.chain(self.netA_A.parameters(), self.netA_B.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            ####################5.13
            self.optimizers.append(self.optimizer_A)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.zeros = torch.Tensor(self.opt.batch_size, 1, self.opt.crop_size, self.opt.crop_size).to(self.device)

        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
        

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_A.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_A.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        bc, c, h, w = self.real_A.size()
        self.zeros.resize_((bc, 1, h, w)).fill_(0.0)

    ######################################5.13
    def mask_layer(self, foreground, background, mask):
        img = foreground * mask + background * (1- mask)
        return img

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.real_A.shape)
        # print(self.real_B.shape)
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        # print(self.real.shape)
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B_in = self.fake[:self.real_A.size(0)]
        # print(self.fake.shape)
        # print(self.real_A.size(0))
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        self.zeros_attn = Variable(self.zeros, requires_grad = False)

        self.attn_real_A = self.netA_A(self.real_A)
        self.fake_B = self.mask_layer(self.fake_B_in, self.real_A,self.attn_real_A)
        # ###########################################2021.5.12
        # self.gauss = np.random.normal(0, 0.1, 1000)
        # self.count, self.bins = np.histogram(self.gauss, self.opt.num_outcomes)
        # self.anchor0 = self.count / sum(self.count)

        # self.unif = np.random.uniform(-1, 1, 1000)
        # self.count, self.bins = np.histogram(self.unif, self.opt.num_outcomes)
        # self.anchor1 = self.count / sum(self.count)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # fake = self.fake_B.detach()

        #########5.23
        fake_B = self.fake_B_pool.query(self.fake_B)
        pred_fake = self.netD(fake_B.detach())

        # Fake; stop backprop to the generator by detaching fake_B
        # pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        ##############


        # pred_fake = self.netD(self.fake_B.detach())

        # # Fake; stop backprop to the generator by detaching fake_B
        # # pred_fake = self.netD(fake)
        # self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # # Real
        # self.pred_real = self.netD(self.real_B)
        # loss_D_real = self.criterionGAN(self.pred_real, True)
        # self.loss_D_real = loss_D_real.mean()

        # # combine loss and calculate gradients
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # fake = self.fake_B

        

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:

            pred_fake = self.netD(self.fake_B)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            # pred_fake = self.netD(fake)
            # self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        ####################h2z_edge_loss_atten_3
        # self.attn_fake_B = self.netA_B(self.fake_B)
        # self.attn_real_B = self.netA_B(self.real_B)
        # self.loss_attnsparse_A = self.criterionIdt(self.attn_real_A, self.zeros_attn) #* self.opt.loss_attn_A
        # self.loss_attnsparse_B = self.criterionIdt(self.attn_real_B.detach(), self.zeros_attn) #* self.opt.loss_attn_B
        # self.loss_attnconst_A_B = self.criterionIdt(self.attn_real_A.detach(), self.attn_fake_B) #* self.opt.loss_attn_A


        import cv2
        img1 = self.attn_real_A.mul(255).byte()
        img1_ = img1.cpu().numpy().squeeze(0).transpose((1, 2, 0))

        img2 = self.attn_fake_B.detach().mul(255).byte()
        img2_ = img2.cpu().numpy().squeeze(0).transpose((1, 2, 0))

        self.attn_real_A_edge = cv2.Canny(img1_,0,120)
        self.attn_fake_B_edge = cv2.Canny(img2_,0,120)

        self.loss_edge_A_B = np.sqrt(np.sum(np.square(self.attn_real_A_edge-self.attn_fake_B_edge)))

        #Perpetual loss
        #self.loss_perpetual = self.criterionPer(self.real_A, self.real_B, self.fake_B)

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_attnsparse_A +self.loss_attnsparse_B+ self.loss_attnconst_A_B + 10*self.loss_edge_A_B
        #self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_perpetual

        # self.loss_G = self.loss_G_GAN + loss_NCE_both

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

