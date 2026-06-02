import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

from modeling import BertConfig, BertForSequenceClassification
from tokenizer import Tokenizer
from dataloader import get_dataloaders


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.weight
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_with_threshold(logits, neutral_threshold=0.65):
    """
    کلاس 2 = خنثی

    فقط وقتی کلاس خنثی را پیش‌بینی می‌کنیم که احتمال آن
    از neutral_threshold بیشتر یا مساوی باشد.

    در غیر این صورت فقط بین کلاس‌های 0 و 1 تصمیم می‌گیریم.
    """
    probs = torch.softmax(logits, dim=1)

    neutral_mask = probs[:, 2] >= neutral_threshold
    non_neutral_preds = torch.argmax(probs[:, :2], dim=1)

    preds = torch.where(
        neutral_mask,
        torch.full_like(non_neutral_preds, 2),
        non_neutral_preds
    )

    return preds


def evaluate(model, data_loader, device, neutral_threshold=None):
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask)

            if neutral_threshold is None:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = predict_with_threshold(
                    logits,
                    neutral_threshold=neutral_threshold
                )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return acc, macro_f1, all_labels, all_preds, all_logits


def find_best_threshold(model, data_loader, device):
    """
    پیدا کردن بهترین threshold برای کلاس خنثی بر اساس Macro-F1
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask)

            all_logits.append(logits.cpu())
            all_labels.extend(labels.cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.array(all_labels)

    best_threshold = 0.50
    best_macro_f1 = 0.0

    print("\n--- Threshold Search For Neutral Class class=2 ---")

    for threshold in np.arange(0.50, 0.91, 0.05):
        preds = predict_with_threshold(
            all_logits,
            neutral_threshold=float(threshold)
        ).numpy()

        macro_f1 = f1_score(
            all_labels,
            preds,
            average="macro",
            zero_division=0
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Macro-F1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)

    print(
        f"\n>>> Best neutral threshold: {best_threshold:.2f} | "
        f"Best Macro-F1: {best_macro_f1:.4f}"
    )

    return float(best_threshold), float(best_macro_f1)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_classification_report(labels, preds, path):
    report = classification_report(
        labels,
        preds,
        digits=4,
        zero_division=0
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def save_confusion_matrix(labels, preds, path, title):
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive", "Neutral"],
        yticklabels=["Negative", "Positive", "Neutral"]
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train():
    # -----------------------------
    # تنظیمات اصلی پروژه
    # -----------------------------
    set_seed(42)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = 10000
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-4

    DATA_DIR = r"F:\ALL_CODE\ZIBA\bert\bert_sens\data"
    SAVE_DIR = r"F:\ALL_CODE\ZIBA\bert\bert_sens\models"

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    # -----------------------------
    # 1. بارگذاری دیتا و توکنایزر
    # -----------------------------
    train_df = pd.read_csv(train_path)

    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(train_df["text"].astype(str).tolist())

    tokenizer_vocab_path = os.path.join(SAVE_DIR, "tokenizer_vocab.json")
    tokenizer.save_vocab(tokenizer_vocab_path)

    train_loader, test_loader = get_dataloaders(
        train_path,
        test_path,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LEN
    )

    # -----------------------------
    # 2. ساخت config و مدل
    # -----------------------------
    model_config = {
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": MAX_LEN,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "num_labels": 3
    }

    model_config_path = os.path.join(SAVE_DIR, "model_config.json")
    save_json(model_config, model_config_path)

    config = BertConfig(**model_config)
    model = BertForSequenceClassification(config).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01
    )

    scheduler = StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    criterion = FocalLoss(
        alpha=1.0,
        gamma=1.5,
        weight=None
    )

    # -----------------------------
    # 3. حلقه آموزش
    # -----------------------------
    best_macro_f1 = 0.0
    best_epoch = 0

    history = []

    best_model_path = os.path.join(SAVE_DIR, "best_model.pt")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        acc, macro_f1, _, _, _ = evaluate(
            model,
            test_loader,
            DEVICE,
            neutral_threshold=None
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "accuracy_argmax": float(acc),
            "macro_f1_argmax": float(macro_f1),
            "lr": float(current_lr)
        }

        history.append(epoch_log)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Macro-F1: {macro_f1:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = float(macro_f1)
            best_epoch = epoch + 1

            torch.save(model.state_dict(), best_model_path)

            print(">>> Best model saved Macro-F1 improved.")

    history_path = os.path.join(SAVE_DIR, "training_history.json")
    save_json(history, history_path)

    print(
        f"\nTraining finished. "
        f"Best Epoch: {best_epoch} | "
        f"Best Macro-F1: {best_macro_f1:.4f}"
    )

    # -----------------------------
    # 4. بارگذاری بهترین مدل
    # -----------------------------
    model.load_state_dict(
        torch.load(best_model_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    # -----------------------------
    # 5. جستجوی بهترین threshold
    # -----------------------------
    best_threshold, best_threshold_f1 = find_best_threshold(
        model,
        test_loader,
        DEVICE
    )

    threshold_path = os.path.join(SAVE_DIR, "threshold.json")
    save_json(
        {
            "neutral_threshold": float(best_threshold),
            "macro_f1_with_best_threshold": float(best_threshold_f1),
            "note": "class 2 is neutral"
        },
        threshold_path
    )

    # -----------------------------
    # 6. ارزیابی نهایی بدون threshold
    # -----------------------------
    print("\n=== Final Report Without Threshold ===")

    acc_raw, macro_f1_raw, labels_raw, preds_raw, _ = evaluate(
        model,
        test_loader,
        DEVICE,
        neutral_threshold=None
    )

    raw_report_path = os.path.join(
        SAVE_DIR,
        "classification_report_without_threshold.txt"
    )

    raw_report = save_classification_report(
        labels_raw,
        preds_raw,
        raw_report_path
    )

    raw_cm_path = os.path.join(
        SAVE_DIR,
        "confusion_matrix_without_threshold.png"
    )

    save_confusion_matrix(
        labels_raw,
        preds_raw,
        raw_cm_path,
        title="Confusion Matrix - Without Threshold"
    )

    print(f"Accuracy: {acc_raw:.4f}")
    print(f"Macro-F1: {macro_f1_raw:.4f}")
    print(raw_report)

    # -----------------------------
    # 7. ارزیابی نهایی با threshold
    # -----------------------------
    print("\n=== Final Report With Neutral Threshold ===")

    acc_thr, macro_f1_thr, labels_thr, preds_thr, _ = evaluate(
        model,
        test_loader,
        DEVICE,
        neutral_threshold=best_threshold
    )

    thr_report_path = os.path.join(
        SAVE_DIR,
        "classification_report_with_threshold.txt"
    )

    thr_report = save_classification_report(
        labels_thr,
        preds_thr,
        thr_report_path
    )

    thr_cm_path = os.path.join(
        SAVE_DIR,
        "confusion_matrix_with_threshold.png"
    )

    save_confusion_matrix(
        labels_thr,
        preds_thr,
        thr_cm_path,
        title=f"Confusion Matrix - With Neutral Threshold {best_threshold:.2f}"
    )

    final_metrics = {
        "best_epoch": int(best_epoch),
        "best_macro_f1_during_training_argmax": float(best_macro_f1),

        "without_threshold": {
            "accuracy": float(acc_raw),
            "macro_f1": float(macro_f1_raw)
        },

        "with_threshold": {
            "accuracy": float(acc_thr),
            "macro_f1": float(macro_f1_thr),
            "neutral_threshold": float(best_threshold)
        }
    }

    metrics_path = os.path.join(SAVE_DIR, "final_metrics.json")
    save_json(final_metrics, metrics_path)

    print(f"Accuracy: {acc_thr:.4f}")
    print(f"Macro-F1: {macro_f1_thr:.4f}")
    print(f"Neutral Threshold: {best_threshold:.2f}")
    print(thr_report)

    print("\nSaved files:")
    print(f"- {best_model_path}")
    print(f"- {tokenizer_vocab_path}")
    print(f"- {model_config_path}")
    print(f"- {threshold_path}")
    print(f"- {history_path}")
    print(f"- {metrics_path}")
    print(f"- {raw_report_path}")
    print(f"- {thr_report_path}")
    print(f"- {raw_cm_path}")
    print(f"- {thr_cm_path}")


if __name__ == "__main__":
    train()
