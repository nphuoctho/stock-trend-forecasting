# %% [markdown]
# # Fine-tune PhoBERT cho phân loại cảm xúc tin tài chính (Colab / Kaggle)
#
# Notebook này fine-tune `vinai/phobert-base` thành mô hình phân loại cảm xúc 3 lớp
# (NEGATIVE / NEUTRAL / POSITIVE) cho tin tài chính tiếng Việt, dùng cho đồ án
# "Dự báo xu hướng giá cổ phiếu từ cảm xúc tin tức".
#
# **Chạy trên GPU** (Colab: Runtime > Change runtime type > T4 GPU; Kaggle: Settings > Accelerator > GPU).
#
# Notebook tự cài đúng phiên bản thư viện đã kiểm thử để tránh lỗi tương thích API.

# %% [markdown]
# ## 1. Kiểm tra môi trường và GPU

# %%
import sys
import platform
print("Python:", sys.version.split()[0], "|", platform.platform())

# Kiểm tra GPU trước khi cài (để biết có nên train hay không).
import subprocess
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    print("GPU:", out.decode().strip())
except Exception:
    print("CẢNH BÁO: Không phát hiện GPU. Fine-tune trên CPU sẽ rất chậm.")
    print("Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU.")

# %% [markdown]
# ## 2. Cài đặt thư viện (pin phiên bản để tránh lỗi API)
#
# QUAN TRỌNG: notebook này ĐỘC LẬP, không phụ thuộc code `stf`. Ta pin `transformers`
# 4.46.3 vì bản này ổn định trên Colab/Kaggle và có đủ các tham số dùng bên dưới
# (`eval_strategy`, `warmup_ratio`, `fp16`). KHÔNG cài lại `torch` để giữ nguyên bản
# torch+CUDA sẵn có của Colab/Kaggle (cài đè dễ vỡ CUDA).
#
# Ghi chú Kaggle: nếu gặp xung đột phụ thuộc khi cài đè, thêm cờ
# `--no-deps` cho `transformers`/`accelerate` hoặc bật Internet trong Settings.

# %%
import subprocess, sys

# Ghim numpy < 2 để tương thích chắc chắn với transformers 4.46 và torch của Colab/Kaggle.
PKGS = [
    "transformers==4.46.3",   # ổn định, có eval_strategy/warmup_ratio/fp16
    "tokenizers>=0.20,<0.22",
    "accelerate==1.2.1",
    "sentencepiece==0.2.0",
    "protobuf<5",             # một số môi trường cần cho tokenizer/sentencepiece
    "scikit-learn>=1.3",
    "pandas>=2.0,<2.3",
    "numpy<2",                # tránh xung đột ABI với torch/transformers bản này
    "matplotlib>=3.5",        # vẽ confusion matrix
    "openpyxl>=3.1",          # đọc raw_data.xlsx
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], check=True)
print("Đã cài xong. Nếu Colab/Kaggle báo cần KHỞI ĐỘNG LẠI runtime, hãy restart")
print("rồi chạy tiếp TỪ MỤC 3 (không chạy lại mục 2).")

# %% [markdown]
# ## 3. Xác nhận phiên bản sau khi cài

# %%
import torch, transformers, sklearn, pandas, numpy
print("torch        :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("transformers :", transformers.__version__)
print("sklearn      :", sklearn.__version__)
print("pandas       :", pandas.__version__)
print("numpy        :", numpy.__version__)
assert torch.cuda.is_available(), "Cần GPU. Bật accelerator rồi chạy lại."

# %% [markdown]
# ## 4. Lấy code `stf` và dữ liệu nhãn
#
# Có hai cách:
# - **A. Clone repo** (nếu repo public hoặc đã cấu hình quyền truy cập).
# - **B. Tải thủ công** file `cafef_seed.csv` và (tuỳ chọn) nhãn in-domain, rồi upload.
#
# Mặc định dùng cách B (không phụ thuộc quyền repo). Bỏ chú thích cách A nếu muốn clone.

# %%
# --- Cách A: clone repo (bỏ chú thích nếu dùng) ---
# !git clone -b feature/data-pipeline-and-phobert https://github.com/nphuoctho/stock-trend-forecasting.git
# %cd stock-trend-forecasting
# import sys; sys.path.insert(0, "src")

# --- Cách B: tải seed CafeF công khai trực tiếp (không cần repo) ---
import subprocess, sys
subprocess.run([
    "wget", "-q", "-O", "cafef_seed_raw.xlsx",
    "https://raw.githubusercontent.com/209sontung/Vietnamese-stock-article-classification/main/Dataset/raw_data.xlsx"
], check=True)
print("Đã tải seed CafeF (raw_data.xlsx).")

# %% [markdown]
# ## 5. Chuẩn bị dữ liệu nhãn
#
# Mapping nhãn gốc: 1=NEGATIVE, 2=NEUTRAL, 3=POSITIVE. Nếu có tập in-domain đã gán nhãn
# (cột `text`, `label`), gộp vào đây để tăng chất lượng in-domain.

# %%
import pandas as pd

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
RAW2LABEL = {1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE"}

seed = pd.read_excel("cafef_seed_raw.xlsx").rename(columns={"title": "text"})
seed["label"] = seed["label"].map(RAW2LABEL)
seed = seed.dropna(subset=["text", "label"]).drop_duplicates("text")[["text", "label"]]

# (Tuỳ chọn) gộp nhãn in-domain nếu đã upload file to_label_done.csv
import os
if os.path.exists("indomain_labeled.csv"):
    ind = pd.read_csv("indomain_labeled.csv")
    ind = ind.dropna(subset=["text", "label"])[["text", "label"]]
    df = pd.concat([seed, ind], ignore_index=True).drop_duplicates("text")
    print(f"Gộp seed ({len(seed)}) + in-domain ({len(ind)}) = {len(df)}")
else:
    df = seed
    print(f"Chỉ dùng seed CafeF: {len(df)} mẫu")

df["label_id"] = df["label"].astype(str).str.upper().str.strip().map(LABEL2ID)
assert df["label_id"].notna().all(), "Có nhãn lạ ngoài NEGATIVE/NEUTRAL/POSITIVE"
df["label_id"] = df["label_id"].astype(int)
print("Phân bố lớp:", df["label_id"].value_counts().sort_index().to_dict())

# %% [markdown]
# ## 6. Chia train / val / test (phân tầng theo lớp, seed cố định)

# %%
from sklearn.model_selection import train_test_split

SEED = 42
train_df, tmp = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=SEED)
val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label_id"], random_state=SEED)
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# %% [markdown]
# ## 7. Tokenize với PhoBERT (tối đa 256 token)
#
# Lưu ý: PhoBERT huấn luyện trên văn bản đã tách từ (word segmentation bằng VnCoreNLP).
# Ở đây ta tokenize text thô cho đơn giản và tái lập. Nếu muốn tăng chất lượng, có thể
# thêm bước tách từ (py_vncorenlp) trước khi đưa vào tokenizer.

# %%
import torch  # đảm bảo có torch kể cả khi chạy lẻ cell này
from transformers import AutoTokenizer

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=False)
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item

ds_train = SentimentDataset(train_df["text"], train_df["label_id"])
ds_val   = SentimentDataset(val_df["text"], val_df["label_id"])
ds_test  = SentimentDataset(test_df["text"], test_df["label_id"])
print("Đã tokenize:", len(ds_train), "train /", len(ds_val), "val /", len(ds_test), "test")

# %% [markdown]
# ## 8. Nạp mô hình và cấu hình huấn luyện
#
# Mọi artifact của lần chạy này (checkpoint, log, metric, hình) được lưu vào một thư mục
# RUN riêng có timestamp, để tái lập và trích số vào báo cáo. Nếu chạy Colab, mount Drive
# trước (ô đầu mục 8) thì RUN_DIR nằm luôn trên Drive, KHÔNG mất khi hết session.

# %%
import os, json, datetime

# (Colab) Mount Drive để lưu bền vững. Bỏ chú thích nếu muốn.
# from google.colab import drive; drive.mount("/content/drive")
# BASE_OUT = "/content/drive/MyDrive/phobert-runs"
BASE_OUT = "runs"   # mặc định lưu tại chỗ; nhớ tải về ở mục 12

RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# ĐỂ RESUME khi bị ngắt: dán đường dẫn thư mục RUN cũ vào đây (vd "runs/phobert-sentiment-20260826-101500").
# Để trống ("") = tạo RUN mới. Nếu resume, phải giữ nguyên seed/siêu tham số/split như lần trước.
RESUME_RUN_DIR = ""

if RESUME_RUN_DIR:
    RUN_DIR = RESUME_RUN_DIR
    assert os.path.isdir(RUN_DIR), f"Không thấy thư mục RUN để resume: {RUN_DIR}"
    print("Sẽ RESUME vào RUN_DIR cũ:", RUN_DIR)
else:
    RUN_DIR = os.path.join(BASE_OUT, f"phobert-sentiment-{RUN_ID}")
    os.makedirs(RUN_DIR, exist_ok=True)
    print("Tạo RUN_DIR mới:", RUN_DIR)

# %%
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

ID2LABEL = {i: n for i, n in enumerate(LABELS)}
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
    }

args = TrainingArguments(
    output_dir=os.path.join(RUN_DIR, "checkpoints"),
    logging_dir=os.path.join(RUN_DIR, "logs"),   # TensorBoard log
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,           # giữ 3 checkpoint gần nhất để RESUME được nếu bị ngắt
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    seed=SEED,
    logging_steps=20,
    report_to="none",             # đổi thành "tensorboard" nếu muốn xem trực quan
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds_train, eval_dataset=ds_val,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 9. Huấn luyện (có RESUME + log an toàn từng bước)
#
# - **RESUME:** nếu Colab/Kaggle ngắt giữa chừng, chạy lại notebook TỪ ĐẦU nhưng đặt
#   `RESUME_RUN_DIR` (ở mục 8) trỏ đúng thư mục RUN cũ, rồi chạy ô này. Trainer sẽ nạp
#   checkpoint gần nhất và train tiếp, KHÔNG mất số epoch đã chạy. Để trống = train mới.
# - **Log an toàn:** một callback ghi từng dòng log (loss/lr/metric) ra `train_log.jsonl`
#   NGAY khi phát sinh, nên dù crash giữa chừng vẫn còn số liệu để debug + viết báo cáo.

# %%
import json
from transformers import TrainerCallback

class JsonlLoggerCallback(TrainerCallback):
    """Ghi từng dòng log ra jsonl ngay lập tức (an toàn nếu crash giữa train)."""
    def __init__(self, path):
        self.path = path
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

trainer.add_callback(JsonlLoggerCallback(os.path.join(RUN_DIR, "train_log.jsonl")))

# Tìm checkpoint để resume: ưu tiên RESUME_RUN_DIR (mục 8), nếu trống thì tự dò trong RUN_DIR.
from transformers.trainer_utils import get_last_checkpoint
ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
resume_ckpt = None
if os.path.isdir(ckpt_dir):
    resume_ckpt = get_last_checkpoint(ckpt_dir)   # None nếu chưa có checkpoint nào
if resume_ckpt:
    print("RESUME từ checkpoint:", resume_ckpt)
else:
    print("Train mới (không tìm thấy checkpoint để resume).")

train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

# Lưu ngay log lịch sử huấn luyện (loss/metric mỗi epoch) ra file, KHÔNG mất khi hết session.
import pandas as pd
log_hist = pd.DataFrame(trainer.state.log_history)
log_hist.to_csv(os.path.join(RUN_DIR, "train_log_history.csv"), index=False)

# Vẽ đường cong học (train loss + val macro-F1) để chèn báo cáo và debug hội tụ.
try:
    import matplotlib.pyplot as plt
    lh = log_hist.copy()
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    tr = lh.dropna(subset=["loss"]) if "loss" in lh else lh.iloc[0:0]
    if len(tr):
        ax1.plot(tr["step"], tr["loss"], color="tab:red", label="train loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss", color="tab:red")
    if "eval_macro_f1" in lh:
        ev = lh.dropna(subset=["eval_macro_f1"])
        ax2 = ax1.twinx()
        ax2.plot(ev["step"], ev["eval_macro_f1"], color="tab:blue",
                 marker="o", label="val macro-F1")
        ax2.set_ylabel("val macro-F1", color="tab:blue")
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "learning_curve.png"), dpi=150)
    plt.show()
except Exception as e:
    print("Bỏ qua vẽ learning curve:", e)

print("Đã lưu log huấn luyện:", os.path.join(RUN_DIR, "train_log_history.csv"))
print("Log an toàn từng bước:", os.path.join(RUN_DIR, "train_log.jsonl"))
print(log_hist.tail(6))

# %% [markdown]
# ## 10. Đánh giá trên tập test + LƯU báo cáo, confusion matrix
#
# Mọi chỉ số được ghi ra file để trích vào báo cáo (không chỉ in màn hình).

# %%
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

pred = trainer.predict(ds_test)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = pred.label_ids

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
report_dict = classification_report(y_true, y_pred, target_names=LABELS,
                                    zero_division=0, output_dict=True)
report_txt = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
print("Macro-F1 test:", round(macro_f1, 4))
print(report_txt)

# Lưu classification report (txt + json)
with open(os.path.join(RUN_DIR, "test_classification_report.txt"), "w") as f:
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n{report_txt}")
with open(os.path.join(RUN_DIR, "test_report.json"), "w") as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# Confusion matrix (lưu cả số liệu csv và hình png để chèn báo cáo)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
    os.path.join(RUN_DIR, "confusion_matrix.csv"))
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
ax.set_title(f"Confusion matrix (macro-F1={macro_f1:.3f})")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Đã lưu report + confusion matrix vào", RUN_DIR)

# LƯU BẢN DỰ ĐOÁN SAI (misclassified) để phân tích lỗi định tính vào chương báo cáo.
# Gồm cả xác suất từng lớp và độ "tự tin" của dự đoán sai (sắp giảm dần để soi ca sai nặng nhất).
probs_test = np.exp(pred.predictions - pred.predictions.max(axis=1, keepdims=True))
probs_test = probs_test / probs_test.sum(axis=1, keepdims=True)
err_df = pd.DataFrame({
    "text": list(test_df["text"]),
    "true": [LABELS[i] for i in y_true],
    "pred": [LABELS[i] for i in y_pred],
    "p_NEGATIVE": probs_test[:, 0],
    "p_NEUTRAL":  probs_test[:, 1],
    "p_POSITIVE": probs_test[:, 2],
})
err_df["pred_conf"] = probs_test[np.arange(len(y_pred)), y_pred]
err_df["correct"] = (y_true == y_pred)
mis = err_df[~err_df["correct"]].sort_values("pred_conf", ascending=False)
mis.to_csv(os.path.join(RUN_DIR, "misclassified.csv"), index=False)
# Lưu luôn TOÀN BỘ dự đoán test (để tái phân tích sau này mà không cần chạy lại model).
err_df.to_csv(os.path.join(RUN_DIR, "test_predictions.csv"), index=False)
print(f"Số ca sai: {len(mis)}/{len(err_df)}. Đã lưu misclassified.csv + test_predictions.csv")
print(mis.head(10)[["text", "true", "pred", "pred_conf"]].to_string(index=False))

# %% [markdown]
# ## 11. Lưu mô hình + MANIFEST tái lập
#
# Manifest ghi lại mọi thứ cần để tái lập và trích số vào báo cáo: version thư viện,
# seed, siêu tham số, kích thước split, và metric cuối. Đây là bằng chứng cho acceptance
# #8 (thí nghiệm tái lập) trong đề cương.

# %%
SAVE_DIR = os.path.join(RUN_DIR, "best")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

manifest = {
    "run_id": RUN_ID,
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "seed": SEED,
    "labels": LABELS,
    "hyperparams": {
        "epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
    },
    "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "class_distribution": df["label_id"].value_counts().sort_index().to_dict(),
    "test_metrics": {
        "macro_f1": macro_f1,
        "accuracy": report_dict["accuracy"],
        "per_class": {lab: report_dict[lab] for lab in LABELS},
    },
    "versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "timestamp": datetime.datetime.now().isoformat(),
}
with open(os.path.join(RUN_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("Đã lưu mô hình + manifest.json vào", RUN_DIR)
print(json.dumps(manifest["test_metrics"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 12. Đóng gói toàn bộ RUN để tải về / lưu Drive
#
# Gói cả thư mục RUN (model + log + report + manifest + hình) thành 1 zip.

# %%
import shutil
zip_path = shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
print("Đã tạo:", zip_path)

# Colab: tải về máy
# from google.colab import files; files.download(zip_path)

# Kaggle: file nằm trong /kaggle/working, tự xuất hiện ở tab Output để tải.
# Hoặc lưu Drive (nếu đã mount ở mục 8):
# import shutil; shutil.copytree(RUN_DIR, f"/content/drive/MyDrive/{os.path.basename(RUN_DIR)}")

# %% [markdown]
# ## 13. Sinh xác suất cảm xúc cho corpus tin (inference)
#
# Sau khi có mô hình, dùng để sinh xác suất 3 lớp cho toàn bộ tin Vietstock đã crawl
# (upload articles.parquet lên, hoặc clone repo và đọc data/raw/news/articles.parquet).

# %%
def predict_proba(texts, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    texts = list(texts)  # tránh lỗi slice theo index nếu texts là pandas Series
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)

# Ví dụ:
demo = ["Lợi nhuận quý 3 tăng mạnh vượt kỳ vọng", "Khối ngoại bán ròng liên tục"]
print(predict_proba(demo))
