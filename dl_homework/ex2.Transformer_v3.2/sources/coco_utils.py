import os, json
import numpy as np
import h5py
import tempfile  # 新增导入
import requests  # 新增导入
from PIL import Image  # 新增导入

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(dir_path, "../../datasets_coco/coco_captioning")

def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    print('base dir ', base_dir)
    data = {}
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])

    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])

    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f])
    data["train_urls"] = train_urls

    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]
    return data

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def sample_coco_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls

# ====================== 新增函数：安全的图像下载方法 ======================
def image_from_url(url):
    """安全的图像下载方法（解决Windows临时文件权限问题）[1,2](@ref)
    
    采用最小修改原则：
    1. 创建临时文件时设置 delete=False 防止自动删除
    2. 先关闭文件再访问
    3. 使用 try/finally 确保临时文件清理
    """
    temp_path = None
    try:
        # 创建临时文件（不自动删除）
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # 立即关闭释放文件锁
        
        # 下载图片到临时路径
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # 使用PIL加载图像
            img = Image.open(temp_path)
            return np.array(img)
        else:
            print(f"URL错误: {url}")
            return None
    except Exception as e:
        print(f"图片下载失败: {url}, 错误: {str(e)}")
        return None
    finally:
        # 确保清理临时文件
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)