import re
import json
from collections import Counter
import torch


class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self._init_vocab()

    def _init_vocab(self):
        self.stoi = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3
        }
        self.itos = {
            0: "[PAD]",
            1: "[UNK]",
            2: "[CLS]",
            3: "[SEP]"
        }

    def normalize(self, text):
        """
        اصلاح نویسه‌های فارسی و حذف کاراکترهای مزاحم
        """
        if not isinstance(text, str):
            text = str(text)

        # یکسان‌سازی ی و ک عربی/فارسی
        text = text.replace("ي", "ی").replace("ك", "ک")

        # تبدیل نیم‌فاصله به فاصله
        text = text.replace("\u200c", " ")

        # حذف اعراب و حرکات عربی
        text = re.sub(r"[\u064B-\u065F]", "", text)

        # lowercase برای انگلیسی‌ها
        text = text.lower().strip()

        # حذف فاصله‌های تکراری
        text = re.sub(r"\s+", " ", text)

        return text

    def _tokenize(self, text):
        """
        جدا کردن کلمات بر اساس Regex
        """
        text = self.normalize(text)

        tokens = re.findall(
            r'[a-zA-Z0-9]+|[\u0600-\u06FF]+|[^\w\s]',
            text
        )

        return tokens

    def tokenize(self, text):
        """
        Alias عمومی برای جلوگیری از خطای tokenizer.tokenize
        """
        return self._tokenize(text)

    def fit(self, texts):
        """
        ساختن لغت‌نامه بر اساس پراکندگی کلمات در کل دیتاست
        """
        self._init_vocab()

        counter = Counter()

        print("Fitting tokenizer on texts...")

        for text in texts:
            counter.update(self._tokenize(text))

        most_common = counter.most_common(self.vocab_size - 4)

        for i, (token, _) in enumerate(most_common):
            idx = i + 4
            self.stoi[token] = idx
            self.itos[idx] = token

        print(f"Vocab built with {len(self.stoi)} tokens.")

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ):
        tokens = self._tokenize(text)

        if truncation:
            if add_special_tokens:
                tokens = tokens[:max_length - 2]
            else:
                tokens = tokens[:max_length]

        token_ids = [
            self.stoi.get(token, self.stoi["[UNK]"])
            for token in tokens
        ]

        if add_special_tokens:
            token_ids = (
                [self.stoi["[CLS]"]]
                + token_ids
                + [self.stoi["[SEP]"]]
            )

        attention_mask = [1] * len(token_ids)

        if padding == "max_length":
            pad_len = max_length - len(token_ids)

            if pad_len > 0:
                token_ids += [self.stoi["[PAD]"]] * pad_len
                attention_mask += [0] * pad_len
            else:
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }

    def encode(self, text, max_length=128):
        """
        خروجی ساده فقط input_ids.
        برای تست‌های سریع.
        """
        encoded = self.encode_plus(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        return encoded["input_ids"]

    def decode(self, ids):
        """
        تبدیل IDها به کلمات
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return " ".join([
            self.itos.get(int(idx), "[UNK]")
            for idx in ids
            if int(idx) != self.stoi["[PAD]"]
        ])

    def save_vocab(self, path):
        """
        ذخیره vocabulary برای inference مستقل از train.csv
        """
        data = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer vocab saved to: {path}")

    def load_vocab(self, path):
        """
        لود vocabulary ذخیره‌شده
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = int(data["vocab_size"])
        self.stoi = {str(k): int(v) for k, v in data["stoi"].items()}
        self.itos = {int(v): str(k) for k, v in self.stoi.items()}

        print(f"Tokenizer vocab loaded from: {path}")
        print(f"Loaded vocab size: {len(self.stoi)}")
