import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("✅ CUDA 可用！")
    print(f"🔢 CUDA 版本: {torch.version.cuda}")
    print(f"📊 GPU 数量: {torch.cuda.device_count()}")
    print(f"🏷️ 当前 GPU 名称: {torch.cuda.get_device_name(0)}")

    # 创建一个张量并移动到 GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"🧪 测试张量是否在 GPU 上: {x.is_cuda}")
else:
    print("❌ CUDA 不可用，请检查驱动、PyTorch 版本或安装是否正确。")