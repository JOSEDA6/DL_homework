"""Utility functions used for viewing and processing images."""

import urllib.request, urllib.error, urllib.parse, os, tempfile
import requests  # 新增导入

import numpy as np
from imageio import imread
from PIL import Image


def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    from .fast_layers import conv_forward_fast

    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {"stride": 1, "pad": 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]], dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """Preprocess an image for squeezenet.

    Subtracts the pixel mean and divides by the standard deviation.
    """
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    img = img * SQUEEZENET_STD + SQUEEZENET_MEAN
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    Windows系统安全的临时文件处理方案[1,2](@ref)
    """
    temp_path = None  # 记录临时文件路径
    try:
        # 创建不自动删除的临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # 关键：立即关闭释放文件锁[1](@ref)
        
        # 下载图片到临时路径
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # 使用PIL加载图像（兼容性更好）
            img = Image.open(temp_path)
            return np.array(img)
        else:
            print(f"HTTP错误({response.status_code}): {url}")
            return None
    except Exception as e:
        print(f"图片下载失败: {url}, 错误: {str(e)}")
        return None
    finally:
        # 确保清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError as e:
                print(f"临时文件清理失败: {temp_path}, 错误: {str(e)}")


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        # 使用BILINEAR插值以获得更好的缩放效果[3](@ref)
        img = np.array(Image.fromarray(img).resize(new_shape, resample=Image.BILINEAR))
    return img