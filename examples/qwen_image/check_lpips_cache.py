import os
import shutil


def ensure_vgg_cache():
    """确保 VGG16 权重在缓存中，如果不在则从指定路径复制"""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    cache_file = os.path.join(cache_dir, 'vgg16-397923af.pth')

    # 如果缓存文件已存在且大小合理，跳过
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 500 * 1024 * 1024:  # > 500MB
        return

    # 你的本地下载路径
    local_src = '/mnt/workspace/yihu/ckpt/vgg16-397923af.pth'
    if not os.path.exists(local_src):
        local_src = '/data/oss_bucket_0/Users/yihu/ckpts/vgg16-397923af.pth'

    if not os.path.exists(local_src):
        raise FileNotFoundError(f"未找到本地 VGG 权重文件: {local_src}，请先下载或修正路径")

    print(f"⚠️ 检测到 VGG 缓存缺失或损坏，正在从 {local_src} 复制到缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(local_src, cache_file)
    os.chmod(cache_file, 0o644)
    print("✅ 缓存准备就绪")

def ensure_dists_cache():
    """确保 dists 权重在缓存中，如果不在则从指定路径复制"""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/pyiqa')
    cache_file = os.path.join(cache_dir, 'DISTS_weights-f5e65c96.pth')

    # 如果缓存文件已存在且大小合理，跳过
    if os.path.exists(cache_file):  # > 500MB
        return

    # 你的本地下载路径
    local_src = '/data/oss_bucket_0/Users/yuqifan/ckpt/hub/pyiqa/DISTS_weights-f5e65c96.pth'

    if not os.path.exists(local_src):
        raise FileNotFoundError(f"未找到本地 dists 权重文件: {local_src}，请先下载或修正路径")

    print(f"⚠️ 检测到 dists 缓存缺失或损坏，正在从 {local_src} 复制到缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(local_src, cache_file)
    os.chmod(cache_file, 0o644)
    print("✅ 缓存准备就绪")

def ensure_musiq_cache():
    """确保 MUSIQ 权重在缓存中，如果不在则从指定路径复制"""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/pyiqa')
    cache_file = os.path.join(cache_dir, 'musiq_koniq_ckpt-e95806b9.pth')

    # 如果缓存文件已存在且大小合理，跳过
    if os.path.exists(cache_file):  # > 500MB
        return

    # 你的本地下载路径
    local_src = '/data/oss_bucket_0/Users/yuqifan/ckpt/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth'

    if not os.path.exists(local_src):
        raise FileNotFoundError(f"未找到本地 dists 权重文件: {local_src}，请先下载或修正路径")

    print(f"⚠️ 检测到 dists 缓存缺失或损坏，正在从 {local_src} 复制到缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(local_src, cache_file)
    os.chmod(cache_file, 0o644)
    print("✅ 缓存准备就绪")
    return 

def ensure_clipiqa_cache():
    """确保 rn50 权重在缓存中，如果不在则从指定路径复制"""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/clip')
    cache_file = os.path.join(cache_dir, 'RN50.pt')

    # 如果缓存文件已存在且大小合理，跳过
    if os.path.exists(cache_file):  # > 500MB
        return

    # 你的本地下载路径
    local_src = '/data/oss_bucket_0/Users/yuqifan/ckpt/hub/clip/RN50.pt'

    if not os.path.exists(local_src):
        raise FileNotFoundError(f"未找到本地 clipiqa 权重文件: {local_src}，请先下载或修正路径")

    print(f"⚠️ 检测到 clipiqa 缓存缺失或损坏，正在从 {local_src} 复制到缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(local_src, cache_file)
    os.chmod(cache_file, 0o644)
    print("✅ 缓存准备就绪")
    return 

def ensure_alexnet_cache():
    """确保 AlexNet 权重在缓存中（LPIPS 依赖），如果不在则从指定路径复制"""
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    cache_file = os.path.join(cache_dir, 'alexnet-owt-7be5be79.pth')

    # 如果缓存文件已存在且大小合理，跳过（AlexNet ~233MB）
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 200 * 1024 * 1024:  # > 200MB
        return

    # 你的本地下载路径（需要提前下载好放到这里）
    local_src = '/mnt/workspace/yihu/ckpt/alexnet-owt-7be5be79.pth'
    if not os.path.exists(local_src):
        local_src = '/data/oss_bucket_0/Users/yihu/ckpts/alexnet-owt-7be5be79.pth'

    if not os.path.exists(local_src):
        raise FileNotFoundError(f"未找到本地 AlexNet 权重文件: {local_src}，请先下载或修正路径")

    print(f"⚠️ 检测到 AlexNet 缓存缺失或损坏，正在从 {local_src} 复制到缓存...")
    os.makedirs(cache_dir, exist_ok=True)
    shutil.copy2(local_src, cache_file)
    os.chmod(cache_file, 0o644)
    print("✅ AlexNet 缓存准备就绪")