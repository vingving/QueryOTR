
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF','npy','mat'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    """
    주어진 디렉토리에서 이미지 파일 경로를 모두 수집

    Args:
        dir (str): 데이터 루트 경로
        max_dataset_size (int): 최대 데이터 개수 제한

    Returns:
        list: 이미지 파일 경로 리스트
    """    
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
from copy import deepcopy

class ImageDataset(data.Dataset):
    def __init__(self, opts):
        self.img_paths = sorted(make_dataset(opts.data_root))
        self.is_train=not opts.eval
        input_size=opts.input_size
        output_size=opts.output_size
        # 중앙 crop 영역 계산
        # output_size 안에서 input_size만큼 중앙 영역을 사용할 것이므로
        # 양쪽 가장자리 패딩 크기 계산        
        per_edge_pad=(output_size-input_size)//2
        normlize_target=opts.normlize_target
        patch_mean=opts.patch_mean
        patch_std=opts.patch_std

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(output_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
            ])

        self.input_image_normalize=transforms.Normalize( mean=torch.tensor(IMAGENET_DEFAULT_MEAN),  std=torch.tensor(IMAGENET_DEFAULT_STD))
        if normlize_target:
            self.output_patch_normalize=transforms.Normalize( mean=torch.tensor((patch_mean,patch_mean,patch_mean)),  std=torch.tensor((patch_std,patch_std,patch_std)))
        else:
            self.output_patch_normalize=self.input_image_normalize

        self._mean=torch.tensor((patch_mean,patch_mean,patch_mean))
        self._std=torch.tensor((patch_std,patch_std,patch_std))

        self.mask=torch.zeros([1,output_size, output_size])
        self.mask[:,per_edge_pad:-per_edge_pad,per_edge_pad:-per_edge_pad]=1

        self.per_edge_pad=per_edge_pad

    def __getitem__(self, index):
        """
        Returns:
            dict {
                'input': 
                'ground_truth':
                'gt_inner':
                'name': 
            }
        """
        # 파일 이름 (확장자 제거)
        name= os.path.splitext(os.path.split(self.img_paths[index])[-1])[0]

        # 이미지 로드 (RGB 변환)
        im=Image.open(self.img_paths[index]).convert('RGB')
        
        # transform 적용
        im=self.transform(im)
        
        # =========================
        # 입력 이미지 생성
        # =========================
        # ImageNet 정규화 후, 중앙 영역만 crop
        input_img = self.input_image_normalize(
            deepcopy(im)
        )[:, self.per_edge_pad:-self.per_edge_pad,
           self.per_edge_pad:-self.per_edge_pad]
        
        # =========================
        # Ground Truth 생성
        # =========================        
        gt=self.output_patch_normalize(deepcopy(im))
        
        # 중앙 영역만 남긴 gt
        gt_inner=deepcopy(gt)*self.mask
        
        return {'input':input_img,'ground_truth':gt,'gt_inner':gt_inner,'name':name}

    def __len__(self):
        return len(self.img_paths)


