import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d, deform_conv2d


class DeformConv(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=None):
        super(DeformConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        channels_ = groups * 3 * self.kernel_size[0] * self.kernel_size[1]  # 1 * 3 * 3 * 3 = 27
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        # 채널을 3등분 (offset_x, offset_y, mask)
        # - o1: (B, 9, H_out, W_out)
        # - o2: (B, 9, H_out, W_out)
        # - mask: (B, 9, H_out, W_out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # offset은 x,y를 concat하여 구성 (B, 2 * 9, H_out, W_out)
        offset = torch.cat((o1, o2), dim=1)
        # mask는 0~1 범위로 제한
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)


class ResidualBlock(nn.Module):
    def __init__(self, planes, ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.conv2 = DeformConv(planes, planes, 3, 1, 1)
        # 정규화 (style 제거 목적, 생성 모델에서 자주 사용)        
        self.norm1 = nn.InstanceNorm2d(planes, affine=True)
        self.norm2 = nn.InstanceNorm2d(planes, affine=True)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x_sc = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + x_sc

# ------------------------------------------------------------
# Query Expansion Module (QEM)
# ------------------------------------------------------------
# 8×8 query grid (64개) → 12×12 grid (144개)로 확장
#   1) 144개 전체를 noise 기반으로 초기화
#   2) 중앙 8×8 위치에 기존 query 삽입
#   3) Conv/DeformConv로 spatial refinement
#   4) 중앙 query는 마지막에 다시 원본으로 복원
# ------------------------------------------------------------
class QueryExpansionModule(nn.Module):
    def __init__(self, hidden_num=768, n_block=8, input_size=128, outout_size=192, patch_size=16):
        super(QueryExpansionModule, self).__init__()

        self.hidden_num = hidden_num
        # 입력 query grid 한 변 (128/16=8)
        self.input_query_width = input_size // patch_size
        # 출력 query grid 한 변 (192/16=12)
        self.output_query_width = outout_size // patch_size

        # 12×12 공간에서 refinement 수행할 residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_num) for _ in range(n_block)])
        
        # noise (C/8 차원) → hidden_num 차원으로 확장
        self.noise_mlp = nn.Sequential(
            nn.Linear(hidden_num // 8, hidden_num // 4),
            nn.LayerNorm(hidden_num // 4),
            nn.ReLU(),
            nn.Linear(hidden_num // 4, hidden_num // 2),
            nn.LayerNorm(hidden_num // 2),
            nn.ReLU(),
            nn.Linear(hidden_num // 2, hidden_num)
        )

        self.norm = nn.LayerNorm(hidden_num)
        self.embed = nn.Linear(hidden_num, hidden_num)
        self.inner_query_index, self.outer_query_index = self.get_index()

    def get_index(self):
        # 12×12 마스크 생성
        mask = torch.ones(size=[self.output_query_width, self.output_query_width]).long()
        # 중앙 8×8 영역 계산
        pad_width = (self.output_query_width - self.input_query_width) // 2
        # 중앙은 0 (inner), 외곽은 1 (outer)
        mask[pad_width:-pad_width, pad_width:-pad_width] = 0
        mask = mask.view(-1)
        return mask == 0, mask == 1

    def forward(self, src_query):
        # src_query: (B, 64, 768)
        b, n, c = src_query.size()

        ori_src_query = src_query

        # 입력 query 개수 확인
        assert n == self.input_query_width ** 2, \
            f'QEM input spatial dimension is wrong, {n} and {self.input_query_width ** 2}'

        # 144개 위치에 대해 noise 생성
        # 각 위치마다 C/8 차원
        noise = torch.randn(size=(b, self.output_query_width ** 2, c // 8), dtype=torch.float32).to(src_query.device)
        
        # noise → 768차원 query로 확장
        initial_query = self.noise_mlp(noise)  # (B,144,768)
        
        # 중앙 64개 위치에 기존 query 삽입
        initial_query[:, self.inner_query_index] = src_query

        # (B,144,768) → (B,768,144) → (B,768,12,12)
        x = initial_query.permute(0, 2, 1)
        x = x.reshape(b, c, self.output_query_width, self.output_query_width).contiguous()

        # spatial refinement (Conv + DeformConv)
        for layer in self.res_blocks:
            x = layer(x)  #  (B,768,12,12)

        # (B,768,12,12) → (B,768,144) → (B,144,768)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        # 중앙 query는 원본으로 다시 복구 (정보 보존 목적)
        x[:, self.inner_query_index, :] = ori_src_query

        # 최종 선형 변환
        x = self.embed(x)
        
        return x

if __name__ == '__main__':
    m1 = QueryExpansionModule()
    x1 = torch.randn([1, 64, 768])
    y1 = m1(x1)
    print(y1.size())

