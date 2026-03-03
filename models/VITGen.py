import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import get_sinusoid_encoding_table, CorssAttnBlock
from .VIT import *
from .QEM import QueryExpansionModule
from .PSM import PatchSmoothingModule

class TransGen(nn.Module):
    def __init__(self,opts, enc_ckpt_path=None):
        super(TransGen, self).__init__()
        self.output_size = opts.output_size  # 예: 192
        self.input_size = opts.input_size    # 예: 128
        
        # ViT-B/16과 동일한 설정을 가정
        self.patch_size = 16                 # 패치 한 변 크기
        hidden_num = 768                     # 임베딩 차원 (ViT-B)

        # initialize the weight of decoder, psm and qem
        self.qem=QueryExpansionModule(
            hidden_num=hidden_num,
            input_size=self.input_size,
            outout_size=self.output_size,
            patch_size=self.patch_size
        )
        self.transformer_decoder=nn.ModuleList([
            CorssAttnBlock(
                dim=hidden_num, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                init_values=0., window_size= None)
            for _ in range(opts.dec_depth)])
        self.psm=PatchSmoothingModule(patch_size=16,out_chans=3,embed_dim=hidden_num)
        self.apply(self._init_weights)

        # initialize the weight of encoder using pretrain checkpoint

        self.transformer_encoder = vit_base_patch16(pretrained=True, img_size=224, init_ckpt=enc_ckpt_path)
        #vit_base_patch16(pretrain=True,  init_ckpt=enc_ckpt_path, img_size=self.input_size)

        self.enc_image_size=224  # 인코더 입력 이미지 크기

        # initialize the weight of encoder using pretrain checkpoint
        # 12**2 = 144 토큰에 대한 positional encoding을 생성
        #   output_size=192, patch=16 => 12x12 = 144 토큰
        #   즉, output_size가 192일 때만 정확히 맞음.        
        self.pos_embed = get_sinusoid_encoding_table(12**2, hidden_num)  # (144, 768)
        # inner/outer 토큰 인덱스 마스크를 미리 계산
        self.inner_index, self.outer_index=self.get_index()

    def get_index(self):
        """
        - input_size=128, patch=16 => input_query_width=8
        - output_size=192, patch=16 => output_query_width=12
        - pad_width=(12-8)//2=2
        - 토큰 인덱스 마스크에서 중앙 8x8 영역을 0(inner), 그 외를 1(outer)로 설정        
        """
        input_query_width=self.input_size//self.patch_size
        output_query_width=self.output_size//self.patch_size
        mask=torch.ones(size=[output_query_width,output_query_width]).long()
        pad_width=(output_query_width-input_query_width)//2
        mask[pad_width:-pad_width,pad_width:-pad_width] = 0
        mask=mask.view(-1)
        return mask==0, mask==1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, samples):
        if type(samples) is not dict:
            samples={'input':samples, 'gt_inner':F.pad(samples,(32,32,32,32))}
            
        x = samples['input']        # (B,C,128,128) 가정
        gt_inner = samples['gt_inner']  # (B,C,192,192) 가정

        b,c,w,h=x.size()
        assert w==128 and h==128
        
        # -------------------------
        # 인코더 입력 만들기: reflect pad로 224x224 구성
        # -------------------------
        # 128에 좌우/상하 48씩 패딩 => 128 + 48*2 = 224
        # reflect 모드는 경계 반사로 자연스럽게 확장
        padded_x = F.pad(x, (48, 48, 48, 48), mode='reflect')  # (B,C,224,224)

        # -------------------------
        # ViT 마스크 생성: 중앙 8x8 토큰만 "가려지고", 바깥은 사용
        # -------------------------
        # 224/16 = 14 => ViT 토큰 격자 14x14        
        vit_mask = torch.ones(size=(14, 14)).long()
        
        # 중앙 8x8(= [3:-3, 3:-3])을 0으로 설정
        # 즉 중앙(입력 중심부)은 mask=0, 바깥은 1
        # 이후 bool로 바꾸면 1(True)=마스크 적용, 0(False)=사용        
        vit_mask[3:-3, 3:-3] = 0

        # (14,14) -> (196,) -> (B,196)
        vit_mask = vit_mask.view(-1).expand(b, -1).contiguous().bool()

        # -------------------------
        # 인코더 특징 추출
        # -------------------------
        # src: (B, N, C) = (B, 196, 768)        
        src = self.transformer_encoder.forward_features(padded_x, vit_mask)  # b n c

        # -------------------------
        # QEM으로 출력 토큰(query_embed) 생성
        # -------------------------
        # query_embed: (B, output_tokens, C) = (B, 12*12, 768)         
        query_embed=self.qem(src)

        # self.pos_embed: (144,768) -> (B,144,768) 로 확장
        full_pos=self.pos_embed.type_as(x).to(x.device).clone().detach().expand(x.size(0),-1,-1)

        # -------------------------
        # outer 토큰만 추출하여 positional encoding을 더함
        # -------------------------
        tgt_outer = query_embed[:,self.outer_index,:] + full_pos[:,self.outer_index,:]

        # outer 토큰을 src(인코더 특징)와 cross-attn으로 업데이트
        for i,dec in enumerate(self.transformer_decoder):
            tgt_outer = dec(tgt_outer, src)

        # 전체 tgt(144 토큰) 텐서 구성
        # - inner는 0으로 두고, outer만 디코더 결과로 채움
        tgt = torch.zeros_like(query_embed,dtype=torch.float32)
        tgt[:, self.outer_index] = tgt_outer

        # -------------------------
        # PSM으로 이미지 복원
        # - tgt: outer에 대한 예측 토큰 포함
        # - gt_inner: inner에 해당하는 GT/기반 정보(192x192)로 보임
        # fake: 최종 출력 이미지
        # -------------------------        
        fake=self.psm(tgt,gt_inner)
        return fake











