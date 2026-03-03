# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F

def DiffAugment(x, policy='', channels_first=True):
    """
    x: 입력 이미지 텐서
       shape = (B, C, H, W)  (channels_first=True일 때)
            또는 (B, H, W, C)

    policy: 적용할 augmentation 종류
            예: 'color,translation,cutout'

    channels_first:
        True  -> (B, C, H, W)
        False -> (B, H, W, C)

    반환:
        동일한 shape의 augmentation 적용 텐서
    """    
    if policy:
        if not channels_first:
            # (B, H, W, C) 형태라면 (B, C, H, W)로 변경
            x = x.permute(0, 3, 1, 2)
        # policy를 ',' 기준으로 분리            
        for p in policy.split(','):
            # 각 policy에 해당하는 augmentation 함수 리스트 실
            for f in AUGMENT_FNS[p]:
                x = f(x)
        # 원래 형식으로 되돌림                
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    """
    각 이미지마다 서로 다른 밝기 offset을 추가

    x shape: (B, C, H, W)

    torch.rand(B,1,1,1) → 이미지마다 하나의 스칼라값 생성
    범위: [0,1) → -0.5 ~ 0.5 로 이동

    즉, 각 이미지에 일정한 밝기 값을 더함
    """    
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    """
    채도 조절

    1) 채널 평균 계산 (색상 평균)
       x_mean shape: (B, 1, H, W)

    2) 평균을 기준으로 편차 계산
    3) 편차에 랜덤 스케일 곱
       스케일 범위: [0, 2)

    결과:
        색 대비(채도)가 증가 또는 감소
    """    
    # 채널 평균 (RGB 평균)
    x_mean = x.mean(dim=1, keepdim=True)
    # x = (x - x_mean) * scale + x_mean
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    """
    명암 대비 조절

    1) 이미지 1장당 평균 밝기 계산
       x_mean shape: (B,1,1,1)

    2) 평균을 기준으로 스케일링

    스케일 범위: [0.5, 1.5)
    """    
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    # x = (x - x_mean) * s + x_mean
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    """
    이미지 평행 이동

    ratio:
        이미지 크기의 몇 %까지 이동 허용할지 결정

    예:
        H=32, ratio=0.125 → 최대 4픽셀 이동

    구현 방식:
        1) 이동 범위 계산
        2) 각 이미지마다 랜덤 이동값 생성
        3) padding 후 indexing으로 위치 이동
    """    
    # 이동 가능한 최대 픽셀 계산 [-shift_x, +shift_x], [-shift_y, +shift_y]
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    # 각 배치별 랜덤 이동값: (B,1,1) (각 이미지마다 서로 다른 이동값을 가짐)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    # indexing을 위한 grid 생성: (B,H,W)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    # 이동 적용
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    # 가장자리 처리를 위해 padding
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    # indexing으로 실제 이동 수행 (B, C, H+2, W+2) → (B, H+2, W+2, C)
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    """
    이미지 일부를 0으로 마스킹

    ratio:
        잘라낼 영역의 비율

    예:
        H=32, ratio=0.5 → 약 16x16 영역 제거
    """    
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    # 잘라낼 영역 중심 좌표
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    # cutout 영역 grid
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    # 중심 기준 위치 계산
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    # 전체 1인 mask 생성
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    # cutout 영역을 0으로 설정
    mask[grid_batch, grid_x, grid_y] = 0
    # 채널 차원 확장 후 적용
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],

}

