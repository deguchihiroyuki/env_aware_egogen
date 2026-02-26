import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# 作成済みのモデルをimport
from model.voxel_autoencoder import VoxelViTAutoEncoder
from functools import lru_cache

@lru_cache(maxsize=200) # ★ここの数値をメモリ容量に合わせて調整します（後述）
def load_npz_cached(file_path):
    # with文を使って開き、メモリ上に完全にコピーしてからファイルを閉じます
    with np.load(file_path) as data:
        return data['voxels'].copy()

class VoxelDataset(Dataset):
    def __init__(self, datadir, split="train"):
        self.datadir = datadir
        self.split = split
        self.samples = []
        
        search_pattern = os.path.join(datadir, split, "*", "voxel_frames_head.npz")
        npz_files = glob.glob(search_pattern)
        
        if len(npz_files) == 0:
            print(f"Warning: No .npz files found for split '{split}' in {datadir}")

        # 初期化時はフレーム数(T)のカウントだけ行います
        for file_path in npz_files:
            try:
                with np.load(file_path) as data:
                    T = data['voxels'].shape[0]

                for t in range(T):
                    self.samples.append({
                        "file_path": file_path,
                        "frame_idx": t
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info["file_path"]
        frame_idx = sample_info["frame_idx"]
        
        # 修正: キャッシュ関数を経由してデータを取得（2回目以降はオンメモリから爆速で返ってきます）
        voxel_data = load_npz_cached(file_path)
        voxel = voxel_data[frame_idx] # Shape: (10, 10, 10)
        
        voxel_tensor = torch.from_numpy(voxel).float().unsqueeze(0)
        
        # 入力(0〜0.1)をモデルの出力(0〜1)のスケールに合わせる
        voxel_tensor = voxel_tensor * 10.0
        
        return voxel_tensor

# ==========================================
# 2. Training Loop の定義
# ==========================================
def main():
    # --- Config ---
    cfg_path = "config/pcd.yaml"
    argv = sys.argv.copy()
    if len(argv) > 2 and argv[1] == "CONFIG":
        cfg_path = argv[2]

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})

    DATADIR = data_cfg.get("data_dir", "data/nymeria_voxel_head_100")
    BATCH_SIZE = data_cfg.get("batch_size", 32)
    NUM_WORKERS = data_cfg.get("num_workers", 4)
    PIN_MEMORY = data_cfg.get("pin_memory", True)

    EPOCHS = train_cfg.get("epochs", 10)
    LR = train_cfg.get("lr", 1e-3)
    SAVE_DIR = train_cfg.get("save_dir", "checkpoints")
    DEVICE_TYPE = train_cfg.get("device", "auto")
    DEVICE = torch.device("cuda" if DEVICE_TYPE == "auto" and torch.cuda.is_available() else DEVICE_TYPE)

    LOG_DIR = log_cfg.get("log_dir", "runs/voxel_ae_experiment")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"Using device: {DEVICE}")

    # --- Dataset & DataLoader ---
    train_dataset = VoxelDataset(DATADIR, split="train")
    val_dataset = VoxelDataset(DATADIR, split="val")
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # --- Model, Loss, Optimizer ---
    model = VoxelViTAutoEncoder(
        volume_size=model_cfg.get("volume_size", 10),
        patch_size=model_cfg.get("patch_size", 2),
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        embed_dim=model_cfg.get("embed_dim", 128),
        depth=model_cfg.get("depth", 4),
        heads=model_cfg.get("heads", 4),
    )
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss() # L2 Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- TensorBoard Writer ---
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    best_val_loss = float('inf')
    global_step = 0
    val_step = 0

    # --- 学習ループ ---
    for epoch in range(1, EPOCHS + 1):
        # [Train Phase]
        model.train()
        train_loss = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in pbar_train:
            inputs = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 順伝播
            reconstructed, _ = model(inputs)
            
            # Loss計算 (入力自身をターゲットとして再構成誤差を計算)
            loss = criterion(reconstructed, inputs)
            
            # 逆伝播・最適化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix({"loss": f"{loss.item():.4f}"})
            writer.add_scalar("Loss/TrainIter", loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # [Validation Phase]
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]")
            for batch in pbar_val:
                inputs = batch.to(DEVICE)
                reconstructed, _ = model(inputs)
                loss = criterion(reconstructed, inputs)
                val_loss += loss.item() * inputs.size(0)
                writer.add_scalar("Loss/ValIter", loss.item(), val_step)
                val_step += 1
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # --- ログ記録とモデル保存 ---
        print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        
        # 検証ロスが更新されたらモデルを保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" -> Best model saved to {save_path}")

    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()
