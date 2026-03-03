import torch.nn as nn
from .reconstruct import ReconLoss
from .perceptual import PerceptualLoss
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class SetCriterion(nn.Module):
    def __init__(self,opts):
        super().__init__()
        # 외곽 영역만 비교하는 재구성 손실
        self.recon_loss=ReconLoss()
        # VGG feature space에서 비교하는 perceptual loss        
        self.perceptual_loss=PerceptualLoss()
        # Generator 손실 가중치
        self.gen_weight_dict={'loss_g_recon':5, 'loss_g_adversarial':1, 'loss_g_perceptual':10}
        # Discriminator 손실 가중치
        self.dis_weight_dict = {'loss_d_adversarial': 1}
        self.imagenet_normalize=transforms.Normalize( mean=torch.tensor(IMAGENET_DEFAULT_MEAN),  std=torch.tensor(IMAGENET_DEFAULT_STD))
        self.patch_mean=opts.patch_mean
        self.patch_std=opts.patch_std

    def renorm(self,tensor):
        """
        모델 출력은 patch_mean, patch_std 기준으로 정규화된 상태.
        이를 원래 픽셀 스케일로 되돌린 후, ImageNet 기준 정규화로 다시 변환한다.
        (Perceptual Loss 계산을 위함)
        """        
        tensor = tensor * self.patch_std + self.patch_mean
        return self.imagenet_normalize(tensor)

    def get_dis_loss(self,  input_fake, input_real, discriminator=None):
        """
        Discriminator 손실 계산 
        : real과 fake를 잘 구분하도록 학습, 이때 Generator는 업데이트되면 안됨.
        : input_fake는 detach()를 통해 Generator gradient 차단.
        """
        assert discriminator is not None
        return {'loss_d_adversarial': discriminator.calc_dis_loss(input_fake.detach(), input_real)}

    def get_gen_loss(self, input_fake, input_real, discriminator=None, warmup=False):
        """
        Generator 손실 계산
        """
        # Warmup 단계가 아닌 경우
        if not warmup:
            assert discriminator is not None
            g_loss_dict={'loss_g_adversarial': discriminator.calc_gen_loss(input_fake, input_real)}
            g_loss_dict['loss_g_recon']=self.recon_loss(input_fake, input_real)
            g_loss_dict['loss_g_perceptual']=self.perceptual_loss(self.renorm(input_fake), self.renorm(input_real))
            return g_loss_dict

        # Warmup 단계인 경우
        else:
            return {'loss_g_recon':self.recon_loss(input_fake, input_real)}






