import torch
import torch.nn as nn
from einops import rearrange

class ReconLoss(nn.Module):
    def __init__(self,image_size=192,crop_width=32,loss_type='mse'):
        super(ReconLoss, self).__init__()
        assert loss_type in ['l1','mse']
        
        # ---------------------------------------------
        # 1. 외곽 영역 마스크 생성
        # ---------------------------------------------
        # image_size x image_size 크기의 1로 채워진 마스크 생성
        mask = torch.ones((image_size, image_size))
        # 중앙 영역을 0으로 설정
        # crop_width 만큼 테두리를 제외한 내부 영역을 0으로 바꿈
        # 즉, 외곽은 1 / 중앙은 0        
        mask[crop_width:-crop_width, crop_width:-crop_width] = 0

        # (H, W) → (H*W)
        self.mask=mask.view(-1).long().cuda()
        # outer 영역(=1인 부분)에 해당하는 boolean index 생성
        self.outer_index=self.mask==1
        
        if loss_type=='l1':
            self.loss=nn.L1Loss()
        else:
            self.loss=nn.MSELoss()

    def forward(self,input_fake, input_real):
        """
        input_fake: (B, C, W, H)
        input_real: (B, C, W, H)
        """        
        # (B, C, W, H) → (B, W*H, C) 공간 차원을 펼쳐서 pixel 단위로 정렬
        input_fake = rearrange(input_fake, 'b c w h -> b (w h) c')
        input_real = rearrange(input_real, 'b c w h -> b (w h) c')
        
        # outer 영역만 선택
        input_fake = input_fake[:, self.outer_index]
        input_real = input_real[:, self.outer_index]

        # 선택된 영역에 대해 손실 계산
        return self.loss(input_fake,input_real)


