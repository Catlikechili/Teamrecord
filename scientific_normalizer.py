r"""
Bộ công cụ ngôn ngữ cho Ghi Nhận Nguyên Văn Văn Phong Khoa Học (`d:\Gravity\Dich\scientific_normalizer.py`)
==========================================================================================================
1. `DOMAIN_PROFILES`        : Câu mồi ngữ cảnh (initial prompt) + từ khóa ưu tiên (hotwords) theo lĩnh vực.
2. `build_initial_prompt`   : Ghép câu mồi lĩnh vực + từ khóa riêng của buổi họp + ngữ cảnh câu vừa nói.
3. `ScientificTermNormalizer`: Chuẩn hóa cách viết thuật ngữ/ký hiệu khoa học (p-value, R-squared, et al., CO2...).
4. `deduplicate_overlap_text`: Khử từ bị lặp ở ranh giới hai đoạn âm thanh gối đầu nhau.
5. `EnsembleArbitrator`     : Hợp nhất kết quả Whisper (có xác suất từng từ) với các phương án của Google.

Toàn bộ module chỉ dùng thư viện chuẩn Python, không gọi mạng, không tốn CPU đáng kể.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================================
# 1. HỒ SƠ LĨNH VỰC KHOA HỌC (PROMPT + HOTWORDS)
# ============================================================================
# Lưu ý kỹ thuật: Whisper chỉ nhận tối đa ~224 token cho phần prompt (faster-whisper tự cắt
# phần đầu nếu dài hơn), vì vậy câu mồi được giữ ngắn gọn và ngữ cảnh câu trước được đặt ở CUỐI
# (phần được giữ lại khi bị cắt).

_BASE_ACADEMIC_PROMPT = (
    "Transcript of a scientific research meeting. The speakers use precise academic language, "
    "complete sentences and correct punctuation."
)

DOMAIN_PROFILES: Dict[str, Dict[str, str]] = {
    "general_science": {
        "label": "🔬 Khoa học tổng hợp",
        "prompt": _BASE_ACADEMIC_PROMPT,
        "hotwords": (
            "hypothesis, methodology, empirical, statistically significant, p-value, standard deviation, "
            "confidence interval, correlation, regression, dataset, experiment, et al."
        ),
    },
    "ai_cs_math": {
        "label": "🤖 AI / Khoa học máy tính / Toán",
        "prompt": _BASE_ACADEMIC_PROMPT + " Topics: machine learning, algorithms and mathematics.",
        "hotwords": (
            "neural network, transformer, gradient descent, backpropagation, hyperparameter, overfitting, "
            "loss function, convolutional, embedding, inference, benchmark, eigenvalue, Bayesian, Monte Carlo"
        ),
    },
    "bio_med_chem": {
        "label": "🧬 Y sinh / Dược / Hóa học",
        "prompt": _BASE_ACADEMIC_PROMPT + " Topics: biology, medicine and chemistry.",
        "hotwords": (
            "in vitro, in vivo, assay, protein, enzyme, gene expression, cohort, placebo, randomized controlled trial, "
            "concentration, molecule, catalyst, pathway, clinical, dosage"
        ),
    },
    "physics_engineering": {
        "label": "⚙️ Vật lý / Kỹ thuật / Vật liệu",
        "prompt": _BASE_ACADEMIC_PROMPT + " Topics: physics, engineering and materials science.",
        "hotwords": (
            "thermodynamics, finite element, Navier-Stokes, boundary condition, tensile strength, wavelength, "
            "frequency, nanoparticle, semiconductor, simulation, Reynolds number, eigenmode"
        ),
    },
    "econ_social_stats": {
        "label": "📈 Kinh tế lượng / Xã hội / Thống kê",
        "prompt": _BASE_ACADEMIC_PROMPT + " Topics: econometrics, social science and statistics.",
        "hotwords": (
            "econometric, heteroscedasticity, endogeneity, instrumental variable, panel data, fixed effects, "
            "ordinary least squares, coefficient, elasticity, survey, sample size, ANOVA, chi-square"
        ),
    },
}

DEFAULT_DOMAIN = "general_science"


def build_initial_prompt(
    domain_key: str,
    user_hotwords: str = "",
    previous_context: str = "",
    max_context_chars: int = 250,
) -> Optional[str]:
    """Ghép câu mồi cho Whisper: [câu mồi lĩnh vực] + [từ khóa riêng] + [ngữ cảnh câu trước] (tiếng Anh)."""
    profile = DOMAIN_PROFILES.get(domain_key, DOMAIN_PROFILES[DEFAULT_DOMAIN])
    parts: List[str] = [profile["prompt"]]
    kw = (user_hotwords or "").strip()
    if kw:
        parts.append(f"Key terms: {kw}.")
    ctx = (previous_context or "").strip()
    if ctx:
        parts.append(ctx[-max_context_chars:])
    return " ".join(parts).strip() or None


def build_hotwords(domain_key: str, user_hotwords: str = "") -> Optional[str]:
    """Gộp từ khóa riêng do người dùng nhập (ưu tiên trước) với từ khóa của lĩnh vực."""
    profile = DOMAIN_PROFILES.get(domain_key, DOMAIN_PROFILES[DEFAULT_DOMAIN])
    items: List[str] = []
    if (user_hotwords or "").strip():
        items.append(user_hotwords.strip())
    items.append(profile["hotwords"])
    joined = ", ".join(items).strip(", ")
    return joined or None


def collapse_repetitions(text: str) -> str:
    """
    Bỏ các câu lặp lại liên tiếp y hệt nhau — dấu hiệu Whisper bị "kẹt vòng lặp"
    (vd: "We'll see you in a little bit." x10). Câu đầu tiên được giữ lại.
    """
    return " ".join(w.text for w in collapse_repeated_sentences([WordInfo(t, 1.0) for t in (text or "").split()]))


def collapse_repeated_sentences(words: List["WordInfo"]) -> List["WordInfo"]:
    """Như `collapse_repetitions` nhưng làm trên danh sách từ (giữ nguyên xác suất từng từ)."""
    sentences: List[List[WordInfo]] = [[]]
    for w in words:
        sentences[-1].append(w)
        if w.text[-1:] in ".!?":
            sentences.append([])
    out: List[WordInfo] = []
    prev_plain = None
    for sent in sentences:
        if not sent:
            continue
        plain = " ".join(_plain_token(w.text) for w in sent)
        if plain and plain == prev_plain:
            continue
        out.extend(sent)
        prev_plain = plain
    return out


# ============================================================================
# 2. CHUẨN HÓA THUẬT NGỮ & KÝ HIỆU KHOA HỌC
# ============================================================================

class ScientificTermNormalizer:
    """
    Chuẩn hóa cách viết các thuật ngữ khoa học hay bị ASR viết rời/viết sai.
    Chỉ sửa HÌNH THỨC VIẾT (gạch nối, viết hoa, dấu chấm viết tắt), không thay đổi nội dung lời nói.
    """

    _RULES: List[Tuple[re.Pattern, str]] = [
        # Thống kê
        (re.compile(r"\bp[\s\-]?values?\b", re.I), "p-value"),
        (re.compile(r"\br[\s\-]?squared\b", re.I), "R-squared"),
        (re.compile(r"\bchi[\s\-]?squared?\b", re.I), "chi-square"),
        (re.compile(r"\bt[\s\-]tests?\b", re.I), "t-test"),
        (re.compile(r"\bz[\s\-]scores?\b", re.I), "z-score"),
        (re.compile(r"\bf[\s\-]?1[\s\-]scores?\b", re.I), "F1-score"),
        (re.compile(r"\bf[\s\-]?one[\s\-]scores?\b", re.I), "F1-score"),
        (re.compile(r"\banova\b", re.I), "ANOVA"),
        (re.compile(r"\bmonte[\s\-]carlo\b", re.I), "Monte Carlo"),
        (re.compile(r"\bbayesian\b", re.I), "Bayesian"),
        (re.compile(r"\bnavier[\s\-]stokes\b", re.I), "Navier-Stokes"),
        (re.compile(r"\breynolds number\b", re.I), "Reynolds number"),
        # Cụm Latin học thuật
        (re.compile(r"\bet\s+al\b\.?", re.I), "et al."),
        (re.compile(r"\bin\s+vitro\b", re.I), "in vitro"),
        (re.compile(r"\bin\s+vivo\b", re.I), "in vivo"),
        (re.compile(r"\bin\s+silico\b", re.I), "in silico"),
        (re.compile(r"\ba\s+priori\b", re.I), "a priori"),
        (re.compile(r"\ba\s+posteriori\b", re.I), "a posteriori"),
        (re.compile(r"\bpost[\s\-]hoc\b", re.I), "post hoc"),
        (re.compile(r"\bi\.?\s?e\.(?=\s)", re.I), "i.e."),
        (re.compile(r"\be\.?\s?g\.(?=\s)", re.I), "e.g."),
        # Hóa học & đơn vị thường gặp
        (re.compile(r"\bco2\b", re.I), "CO2"),
        (re.compile(r"\bh2o\b", re.I), "H2O"),
        (re.compile(r"\bdna\b", re.I), "DNA"),
        (re.compile(r"\brna\b", re.I), "RNA"),
        (re.compile(r"\bmrna\b", re.I), "mRNA"),
        (re.compile(r"\bpcr\b", re.I), "PCR"),
        # Viết tắt kỹ thuật phổ biến
        (re.compile(r"\bgpu(s?)\b", re.I), r"GPU\1"),
        (re.compile(r"\bcpu(s?)\b", re.I), r"CPU\1"),
        (re.compile(r"\bapi(s?)\b", re.I), r"API\1"),
        (re.compile(r"\bcnn(s?)\b", re.I), r"CNN\1"),
        (re.compile(r"\brnn(s?)\b", re.I), r"RNN\1"),
        (re.compile(r"\bllm(s?)\b", re.I), r"LLM\1"),
    ]

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
        out = text
        for pattern, repl in cls._RULES:
            out = pattern.sub(repl, out)
        # Tránh "et al.." khi câu kết thúc ngay sau et al.
        out = out.replace("et al..", "et al.")
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"\s{2,}", " ", out)
        return out.strip()


def normalize_sentence(text: str) -> str:
    """Chuẩn hóa khoảng trắng, viết hoa chữ đầu câu, thêm dấu chấm cuối nếu thiếu."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    # Không viết hoa nếu câu mở đầu bằng ký hiệu kiểu p-value, mRNA, t-test
    if not re.match(r"^(p-value|mRNA|t-test|z-score|in vitro|in vivo|in silico|et al\.)", cleaned):
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned[-1] not in ".!?…":
        cleaned += "."
    return cleaned


# ============================================================================
# 3. KHỬ LẶP TỪ Ở RANH GIỚI ĐOẠN GỐI ĐẦU
# ============================================================================

def _plain_token(tok: str) -> str:
    return re.sub(r"[^\w'\-]", "", tok).lower()


def deduplicate_overlap_text(prev_text: str, new_text: str, max_overlap_words: int = 8) -> str:
    """
    Xóa phần đầu của `new_text` nếu trùng với phần đuôi của `prev_text` (do âm thanh gối đầu nhau).
    Ví dụ: prev="... migrate the production database", new="the production database to PostgreSQL."
           => "to PostgreSQL."
    """
    if not prev_text or not new_text:
        return (new_text or "").strip()

    prev_words = [w for w in (_plain_token(t) for t in prev_text.split()) if w]
    new_tokens = new_text.strip().split()
    new_words = [_plain_token(t) for t in new_tokens]
    if not prev_words or not new_words:
        return new_text.strip()

    max_k = min(max_overlap_words, len(prev_words), len(new_words))
    for k in range(max_k, 0, -1):
        if prev_words[-k:] == new_words[:k]:
            # Chỉ cắt 1 từ khi từ đó đủ dài, tránh cắt nhầm "the", "a" lặp tự nhiên
            if k == 1 and len(new_words[0]) <= 3:
                break
            return " ".join(new_tokens[k:]).strip()
    return new_text.strip()


# ============================================================================
# 4. HỢP NHẤT KẾT QUẢ HAI ĐỘNG CƠ (WHISPER + GOOGLE)
# ============================================================================

@dataclass
class WordInfo:
    text: str
    probability: float


def group_whisper_words(raw_words: Sequence[Tuple[str, float]]) -> List[WordInfo]:
    """
    Gom các mảnh từ của Whisper thành từ hiển thị theo khoảng trắng.
    Whisper trả về mảnh có dấu cách đầu cho từ mới (" squared") và mảnh dính liền cho phần nối
    ("R", "-squared") -> gom thành "R-squared" với xác suất = xác suất thấp nhất của các mảnh.
    """
    grouped: List[WordInfo] = []
    for piece, prob in raw_words:
        if not piece:
            continue
        starts_new = piece[:1].isspace() or not grouped
        token = piece.strip()
        if not token:
            continue
        if starts_new:
            grouped.append(WordInfo(text=token, probability=float(prob)))
        else:
            last = grouped[-1]
            grouped[-1] = WordInfo(text=last.text + token, probability=min(last.probability, float(prob)))
    return grouped


@dataclass
class EnsembleResult:
    text: str
    confidence: float
    low_confidence_words: List[str] = field(default_factory=list)
    substitutions: List[Tuple[str, str]] = field(default_factory=list)   # [(whisper_cũ, google_mới)]
    source: str = ""


class EnsembleArbitrator:
    """
    Nguyên tắc hợp nhất (thận trọng, ưu tiên KHÔNG làm sai lời nói):
    1. Whisper là khung chính vì có dấu câu, viết hoa và xác suất từng từ.
    2. Căn chỉnh từ của Whisper với phương án tốt nhất của Google bằng `difflib.SequenceMatcher`.
    3. Chỉ thay một cụm từ của Whisper bằng cụm tương ứng của Google khi ĐỒNG THỜI:
       - Mọi từ Whisper trong cụm có xác suất < `replace_threshold` (mặc định 0.45),
       - Google tự tin (confidence >= 0.80, hoặc nhiều phương án Google cùng thống nhất cụm đó),
       - Cụm thay thế ngắn (<= 3 từ mỗi bên) để tránh viết lại cả câu.
    4. Các từ Whisper có xác suất < `flag_threshold` (0.65) còn lại được đánh dấu để người dùng rà soát.
    """

    def __init__(self, replace_threshold: float = 0.45, flag_threshold: float = 0.65) -> None:
        self.replace_threshold = replace_threshold
        self.flag_threshold = flag_threshold

    def merge(
        self,
        whisper_words: Sequence[WordInfo],
        whisper_text: str,
        google_alts: Sequence[Tuple[str, float]],
    ) -> EnsembleResult:
        google_alts = [(t.strip(), c) for t, c in google_alts if t and t.strip()]
        w_words = [w for w in whisper_words if w.text.strip()]

        # Trường hợp chỉ có 1 động cơ trả kết quả
        if not w_words and not whisper_text.strip():
            if not google_alts:
                return EnsembleResult(text="", confidence=0.0, source="none")
            g_text, g_conf = google_alts[0]
            return EnsembleResult(text=g_text, confidence=g_conf or 0.80, source="google")
        if not google_alts:
            conf = self._mean_prob(w_words)
            low = [w.text.strip() for w in w_words if w.probability < self.flag_threshold]
            return EnsembleResult(text=whisper_text.strip(), confidence=conf, low_confidence_words=low, source="whisper")

        g_text, g_conf = google_alts[0]
        g_tokens = g_text.split()
        g_plain = [_plain_token(t) for t in g_tokens]
        w_tokens = [w.text.strip() for w in w_words]
        w_plain = [_plain_token(t) for t in w_tokens]
        alt_plain_texts = [" ".join(_plain_token(t) for t in a.split()) for a, _ in google_alts[1:]]

        matcher = difflib.SequenceMatcher(a=w_plain, b=g_plain, autojunk=False)
        out_tokens: List[str] = []
        out_probs: List[float] = []
        substitutions: List[Tuple[str, str]] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                out_tokens.extend(w_tokens[i1:i2])
                # Hai động cơ độc lập cùng ra một từ -> nâng độ tin cậy của từ đó
                out_probs.extend(max(w_words[k].probability, 0.90) for k in range(i1, i2))
                continue

            w_seg = w_words[i1:i2]
            g_seg_tokens = g_tokens[j1:j2]
            all_low = bool(w_seg) and all(w.probability < self.replace_threshold for w in w_seg)
            short = (i2 - i1) <= 3 and (j2 - j1) <= 3
            g_seg_plain = " ".join(g_plain[j1:j2])
            alt_agree = sum(1 for a in alt_plain_texts if g_seg_plain and g_seg_plain in a)
            google_confident = (g_conf >= 0.80) or alt_agree >= 1

            if tag == "replace" and all_low and short and google_confident and g_seg_tokens:
                replacement = list(g_seg_tokens)
                # Giữ dấu câu dính ở cuối cụm Whisper (vd "dataset," -> "data set,")
                tail_punct = re.findall(r"[,.;:!?]+$", w_tokens[i2 - 1])
                if tail_punct:
                    replacement[-1] = replacement[-1] + tail_punct[0]
                if i1 == 0 and w_tokens[0][:1].isupper():
                    replacement[0] = replacement[0][:1].upper() + replacement[0][1:]
                substitutions.append((" ".join(w_tokens[i1:i2]), " ".join(replacement)))
                out_tokens.extend(replacement)
                out_probs.extend([0.75] * len(replacement))
            elif tag == "insert":
                # Google nghe thêm từ mà Whisper bỏ sót: không tự chèn (tránh thêm chữ không có thật)
                continue
            else:
                # "delete" hoặc "replace" không đủ điều kiện -> giữ nguyên Whisper
                out_tokens.extend(w_tokens[i1:i2])
                out_probs.extend(w.probability for w in w_seg)

        final_text = " ".join(out_tokens).strip()
        low = [tok for tok, p in zip(out_tokens, out_probs) if p < self.flag_threshold]
        conf = (sum(out_probs) / len(out_probs)) if out_probs else 0.0
        return EnsembleResult(
            text=final_text,
            confidence=conf,
            low_confidence_words=low,
            substitutions=substitutions,
            source="ensemble",
        )

    @staticmethod
    def _mean_prob(words: Sequence[WordInfo]) -> float:
        return (sum(w.probability for w in words) / len(words)) if words else 0.0


def prompt_echo_ratio(text: str, prompt: Optional[str]) -> float:
    """
    Tỉ lệ từ của `text` xuất hiện trong prompt. Whisper đôi khi "đọc lại" prompt khi đoạn âm thanh
    gần như im lặng; tỉ lệ cao + xác suất thấp là dấu hiệu ảo giác cần loại bỏ.
    """
    if not text or not prompt:
        return 0.0
    t = [w for w in (_plain_token(x) for x in text.split()) if len(w) > 2]
    if not t:
        return 0.0
    p = set(_plain_token(x) for x in prompt.split())
    return sum(1 for w in t if w in p) / float(len(t))
