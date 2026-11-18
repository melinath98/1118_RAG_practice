"""多策略 RAG 文件問答系統 —— Groq + ChromaDB + Gradio(FastAPI)
單檔版:支援 PWA 與 RWD,不需要 static 資料夾,可直接部署到 Render。"""
import csv
import io
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import chromadb
import gradio as gr
import numpy as np
import uvicorn
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from groq import Groq
from PIL import Image, ImageDraw
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定(可用環境變數覆蓋)──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")  # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()  # 伺服器預設 Key(選填);不會傳到瀏覽器
THEME_COLOR = "#059669"

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf); buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p; fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


# ═══ 共用核心:嵌入模型 + 向量庫(全站只載入一次,背景載入以免卡住啟動)═══
class Core:
    def __init__(self):
        import torch  # 延後匯入,讓伺服器能先開 port
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"載入嵌入模型 {EMBED_MODEL}({device})…", flush=True)
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        self.embedder.max_seq_length = min(512, self.embedder.max_seq_length or 512)
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體向量庫;各使用者以 sid 區隔
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40]
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        print("嵌入模型就緒", flush=True)


_CORE, _CORE_LOCK = None, threading.Lock()


def get_core():
    global _CORE
    with _CORE_LOCK:
        if _CORE is None:
            _CORE = Core()
    return _CORE


def _warm():
    try:
        get_core()
    except Exception as e:  # noqa
        print("嵌入模型載入失敗:", e, flush=True)


# ═══ 每位使用者獨立的 RAG 工作階段 ═══
class RAG:
    def __init__(self):
        self.client, self.model, self.sid, self.name = None, MODELS[0], None, ""
        self.chunks, self.labels, self.n_child, self.tfidf, self.tm = [], [], 0, None, None
        self.set_key(GROQ_KEY)

    @property
    def core(self):
        return get_core()

    def cleanup(self):
        if self.sid and _CORE is not None:
            for c in (_CORE.col, _CORE.ccol):
                try: c.delete(where={"sid": self.sid})
                except Exception: pass
        self.sid = None

    # ── Groq ──
    def set_key(self, key):
        self.client = Groq(api_key=key.strip()) if key and key.strip() else None

    def list_models(self):
        if not self.client: return []
        try:
            ids = [m.id for m in self.client.models.list().data if getattr(m, "active", True) is not False]
        except Exception: return []
        return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))

    def llm(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client: return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try: r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower(): raise
                kw.pop("reasoning_effort", None); kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err: raise
            return None

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"): return "✗ 僅支援 PDF / DOCX"
        try:
            core = self.core
            segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
            pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
            if not pairs: return "✗ 無可擷取文字（可能是掃描檔，需先 OCR）"
            self.cleanup()
            self.sid, self.name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}", path.name
            self.chunks, self.labels = [p[0] for p in pairs], [p[1] for p in pairs]
            self._add(core.col, [f"{self.sid}_{i}" for i in range(len(pairs))], self.chunks,
                      [{"sid": self.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{self.sid}_{p}_{c}", t, p) for p, par in enumerate(self.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            self.n_child = len(kids)
            self._add(core.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": self.sid, "idx": k[2]} for k in kids])
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            self.tm = self.tfidf.fit_transform(self.chunks)
            return f"✓ {self.name}\n{len(self.chunks)} 片段 · {time.time() - t0:.1f}s"
        except Exception as e:
            return f"✗ 載入失敗：{type(e).__name__}: {e}"

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc); txt = para.text.strip()
                if not txt: continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush(); head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1; rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t): cells.append(t)
                    if cells: rows.append(" | ".join(cells))
                if rows: flush(); segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        core = self.core
        if core.e5: texts = [("query: " if query else "passage: ") + t for t in texts]
        return core.embedder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=2000):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, qs, k, child=False):
        total = self.n_child if child else len(self.chunks)
        if not total: return [[] for _ in qs]
        core = self.core
        res = (core.ccol if child else core.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": self.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    def kw(self, q, k):
        s = (self.tm @ self.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)): sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, q, strategy, k):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec([q], n * 2)[0], self.kw(q, n * 2)], n)
        if strategy == "semantic": idx = self.vec([q], k)[0]
        elif strategy == "tfidf": idx = self.kw(q, k)
        elif strategy == "parent_child": idx = list(dict.fromkeys(self.vec([q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = self.llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec([q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = self.llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec([q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12)); idx = cands[:k]
            if self.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {self.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = self.llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                             15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else: idx = hybrid(k)
        hits = [(i, self.chunks[i]) for i in idx]
        if strategy == "compress" and self.client and hits:
            def comp(h):
                r = self.llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex: hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model):
        if not self.chunks: return "請先上傳 PDF 或 DOCX 文件。", ""
        self.model, t0 = model or MODELS[0], time.time()
        hits = self.retrieve(q, strategy, int(k)); t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{self.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX: break
            blocks.append(b); used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {self.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits: return "找不到相關片段，請換個問法或策略。", src
        if not self.client: return "⚠ 尚未設定 Groq API Key（檢索結果請看下方片段）", src
        try:
            ans = self.llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                           f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                           1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        except Exception as e:
            return f"✗ 生成失敗：{type(e).__name__}: {e}", src
        return f"{ans or '（模型未回傳內容）'}\n\n<sub>{self.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s</sub>", src


# ═══ PWA:<head> ═══
HEAD = f"""
<meta name="theme-color" content="{THEME_COLOR}">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="文件問答">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" type="image/png" sizes="192x192" href="/pwa/icons/icon-192.png">
<link rel="apple-touch-icon" href="/pwa/icons/apple-touch-icon.png">
<script>
  if ("serviceWorker" in navigator) {{
    window.addEventListener("load", function () {{
      navigator.serviceWorker.register("/sw.js", {{ scope: "/" }})
        .catch(function (err) {{ console.warn("SW 註冊失敗:", err); }});
    }});
  }}
</script>
"""

# ═══ RWD:CSS ═══
CSS = """
:root { --tap-size: 44px; }
.gradio-container {
  max-width: 1200px !important; margin: 0 auto !important;
  padding-left: max(12px, env(safe-area-inset-left)) !important;
  padding-right: max(12px, env(safe-area-inset-right)) !important;
  padding-bottom: max(12px, env(safe-area-inset-bottom)) !important;
}
#hdr { padding: 18px 4px 6px; }
#hdr h1 { font-size: 26px; margin: 0; font-weight: 700; }
#hdr p { color: #6b7280; margin: 4px 0 0; font-size: 14px; }
.side { border-radius: 14px !important; }
#send { min-width: 90px; }
.tap { min-height: var(--tap-size) !important; }
footer { display: none !important; }

/* 平板與桌機:主區塊可換行 */
.main-row { flex-wrap: wrap !important; gap: 16px !important; }
.side { min-width: 280px; }

/* 手機:直向堆疊、滿版、觸控友善 */
@media (max-width: 768px) {
  .gradio-container { padding-top: 6px !important; }
  .main-row { flex-direction: column !important; }
  .main-row > * { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
  #hdr { padding: 10px 2px 2px; }
  #hdr h1 { font-size: 20px; }
  #hdr p { font-size: 12.5px; }
  input, textarea, select { font-size: 16px !important; }  /* 避免 iOS 自動放大 */
  .export-row { flex-wrap: wrap !important; }
  .export-row > * { min-width: 140px !important; }
}
@media (max-width: 480px) {
  #hdr h1 { font-size: 18px; }
  #send { min-width: 72px; }
}
"""
THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Noto Sans TC"), "sans-serif"])


# ═══ Gradio 事件 ═══
def make_rag():
    return RAG()


def on_key(key, cur, rag):
    effective = (key or "").strip() or GROQ_KEY
    rag.set_key(effective)
    if not effective: return "⚠ 未設定 Key", gr.update()
    ms = rag.list_models()
    if not ms: return "✗ Key 無效或無法取得模型", gr.update()
    src = "" if (key or "").strip() else "（伺服器預設）"
    return f"✓ Key 有效{src} · {len(ms)} 個模型", gr.update(
        choices=ms, value=cur if cur in ms else (MODELS[0] if MODELS[0] in ms else ms[0]))


def on_upload(f, rag):
    if f is None: return "⚠ 請選擇檔案"
    if _CORE is None: gr.Info("首次使用需載入嵌入模型，請稍候…")
    return rag.load(f if isinstance(f, str) else f.name)


def on_ask(q, history, records, strategy, k, model, rag):
    q = (q or "").strip()
    if not q: return "", history, gr.update(), records
    ans, src = rag.answer(q, strategy, k, model)
    records = records + [{"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": rag.name, "問題": q,
                          "回答": re.sub(r"\n*<sub>.*?</sub>", "", ans, flags=re.S).strip(),
                          "策略": STRATEGIES[strategy][0], "模型": rag.model,
                          "參考片段": src.split("─" * 40, 1)[-1].strip()}]
    return "", history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}], src, records


# ── 匯出：Word / CSV ──
def _out_path(ext):
    return os.path.join(tempfile.gettempdir(), f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}.{ext}")


def export_csv(records):
    if not records: gr.Warning("目前沒有問答紀錄可匯出"); return None
    path = _out_path("csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    return path


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part: par.add_run(part).bold = bool(i % 2)


def export_docx(records, with_src=True):
    if not records: gr.Warning("目前沒有問答紀錄可匯出"); return None
    doc = Document()
    st = doc.styles["Normal"]; st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip(): continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line): doc.add_heading(m.group(1).replace("**", ""), level=2); continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"); _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number"); _md_runs(par, m.group(1))
            else: _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph(); run = h.add_run("參考片段"); run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip()); p.paragraph_format.left_indent = Pt(12)
                for run in p.runs: run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    path = _out_path("docx"); doc.save(path)
    return path


# ═══ 介面 ═══
_boot = RAG()  # 僅用來在啟動時取得模型清單(有伺服器預設 Key 時)
init_models = _boot.list_models() or MODELS

with gr.Blocks(title="RAG 文件問答", theme=THEME, css=CSS, head=HEAD) as demo:
    rag_state = gr.State(make_rag, delete_callback=lambda r: r.cleanup(), time_to_live=6 * 3600)
    records = gr.State([])

    gr.HTML("<div id='hdr'><h1>📚 RAG 文件問答</h1><p>上傳 PDF / DOCX，選擇檢索策略，直接提問 · Groq × ChromaDB</p></div>")
    with gr.Row(equal_height=False, elem_classes="main-row"):
        with gr.Column(scale=1, min_width=280, elem_classes="side"):
            with gr.Accordion("🔑 Groq API Key", open=not GROQ_KEY):
                key = gr.Textbox(value="", type="password", show_label=False, placeholder="gsk_…（Enter 套用）")
                key_status = gr.Markdown("✓ 使用伺服器預設 Key（可自行輸入覆蓋）" if GROQ_KEY else "⚠ 尚未設定")
            model = gr.Dropdown(init_models, value=MODELS[0] if MODELS[0] in init_models else init_models[0],
                                allow_custom_value=True, label="模型")
            file = gr.File(label="上傳文件", file_types=[".pdf", ".docx"], type="filepath", height=120)
            status = gr.Textbox(show_label=False, interactive=False, lines=2, placeholder="尚未載入文件")
            strategy = gr.Radio([(v[0], sk) for sk, v in STRATEGIES.items()], value="hybrid", label="檢索策略")
            desc = gr.Markdown(f"<small>{STRATEGIES['hybrid'][1]}</small>")
            topk = gr.Slider(1, 10, 4, step=1, label="片段數量 Top-K")
        with gr.Column(scale=2, min_width=300):
            chat = gr.Chatbot(height="60vh", show_label=False, type="messages",
                              placeholder="**上傳文件後開始提問**<br>例如：這份文件的重點是什麼？")
            with gr.Row():
                q = gr.Textbox(show_label=False, placeholder="輸入問題，Enter 送出", scale=6, container=False)
                send = gr.Button("送出", variant="primary", scale=1, elem_id="send", elem_classes="tap")
            with gr.Row():
                gr.Examples([["這份文件的重點是什麼？"], ["文件提到哪些規定或限制？"], ["有哪些風險或注意事項？"]], q, label=None)
                clear = gr.Button("🗑 清除對話", size="sm", variant="secondary", elem_classes="tap")
            with gr.Accordion("🔎 檢索到的片段", open=False):
                src = gr.Textbox(show_label=False, lines=12, interactive=False)
            with gr.Row(elem_classes="export-row"):
                with_src = gr.Checkbox(True, label="Word 包含參考片段", scale=2)
                btn_docx = gr.Button("📄 匯出 Word", size="sm", scale=1, elem_classes="tap")
                btn_csv = gr.Button("📊 匯出 CSV", size="sm", scale=1, elem_classes="tap")
            out_file = gr.File(label="下載檔案", visible=False)

    key.submit(on_key, [key, model, rag_state], [key_status, model])
    key.blur(on_key, [key, model, rag_state], [key_status, model])
    file.upload(on_upload, [file, rag_state], status)
    strategy.change(lambda s: f"<small>{STRATEGIES[s][1]}</small>", strategy, desc)
    for ev in (q.submit, send.click):
        ev(on_ask, [q, chat, records, strategy, topk, model, rag_state], [q, chat, src, records])
    clear.click(lambda: ([], "", [], gr.update(value=None, visible=False)), None, [chat, src, records, out_file])
    show = lambda p: gr.update(value=p, visible=bool(p))
    btn_docx.click(lambda r, w: show(export_docx(r, w)), [records, with_src], out_file)
    btn_csv.click(lambda r: show(export_csv(r)), records, out_file)

demo.queue(max_size=20, default_concurrency_limit=2)


# ═══ PWA 資源(全部內嵌,不需要 static 資料夾)═══
MANIFEST = {
    "id": "/",
    "name": "RAG 文件問答",
    "short_name": "文件問答",
    "description": "上傳 PDF / DOCX，以多種檢索策略向文件提問",
    "lang": "zh-TW",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "orientation": "any",
    "background_color": "#ffffff",
    "theme_color": THEME_COLOR,
    "icons": [
        {"src": "/pwa/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/pwa/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        {"src": "/pwa/icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}

# 更新快取內容時,請調高 VERSION
SW_JS = """
const VERSION = "v1";
const CACHE = "rag-qa-" + VERSION;
const OFFLINE_URL = "/pwa/offline.html";
const PRECACHE = [OFFLINE_URL, "/pwa/icons/icon-192.png", "/pwa/icons/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // 頁面導覽:優先走網路,離線時顯示離線頁
  if (req.mode === "navigate") {
    event.respondWith(fetch(req).catch(() => caches.match(OFFLINE_URL)));
    return;
  }
  // /pwa/ 靜態資源:快取優先;Gradio 的 API、上傳與檔案請求一律不攔截
  if (url.pathname.startsWith("/pwa/")) {
    event.respondWith(
      caches.match(req).then((hit) =>
        hit || fetch(req).then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
          return res;
        })
      )
    );
  }
});
"""

OFFLINE_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#059669">
  <title>目前離線</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 24px;
      font-family: system-ui, -apple-system, "Noto Sans TC", "Microsoft JhengHei", sans-serif;
      background: #ecfdf5; color: #1f2937; text-align: center; }
    @media (prefers-color-scheme: dark) { body { background: #111827; color: #e5e7eb; } }
    .card { max-width: 420px; width: 100%; }
    img { width: 96px; height: 96px; border-radius: 22px; }
    h1 { font-size: clamp(1.25rem, 5vw, 1.6rem); margin: 16px 0 8px; }
    p { margin: 0 0 24px; line-height: 1.6; opacity: .8; }
    button { min-height: 44px; padding: 0 28px; border: 0; border-radius: 10px;
      background: #059669; color: #fff; font-size: 1rem; cursor: pointer; }
  </style>
</head>
<body>
  <div class="card">
    <img src="/pwa/icons/icon-192.png" alt="">
    <h1>目前沒有網路連線</h1>
    <p>文件問答需要連線才能使用。請確認網路後再試一次。</p>
    <button onclick="location.reload()">重新連線</button>
  </div>
</body>
</html>
"""


def _make_icon(size, rounded=True, content_scale=0.62):
    bg, fg, accent = THEME_COLOR, "#ffffff", "#a7f3d0"
    k = 4  # 超取樣,邊緣更平滑
    s = size * k
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if rounded:
        d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=bg)
    else:
        d.rectangle([0, 0, s, s], fill=bg)

    c = s * content_scale
    pw, ph = c * 0.74, c
    px, py = (s - pw) / 2, (s - ph) / 2
    fold = pw * 0.3
    # 文件外框(右上角折角)
    d.polygon([(px, py), (px + pw - fold, py), (px + pw, py + fold), (px + pw, py + ph), (px, py + ph)], fill=fg)
    d.polygon([(px + pw - fold, py), (px + pw, py + fold), (px + pw - fold, py + fold)], fill=accent)
    # 文字行
    lx, lw = px + pw * 0.15, pw * 0.7
    y = py + ph * 0.42
    for wf in (1.0, 1.0, 0.6):
        d.rounded_rectangle([lx, y, lx + lw * wf, y + ph * 0.06], radius=ph * 0.03, fill=bg)
        y += ph * 0.15
    return img.resize((size, size), Image.LANCZOS)


def _png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


ICONS = {
    "icon-192.png": _png(_make_icon(192)),
    "icon-512.png": _png(_make_icon(512)),
    "icon-maskable-512.png": _png(_make_icon(512, rounded=False, content_scale=0.5)),  # 安全區內縮
    "apple-touch-icon.png": _png(_make_icon(180, rounded=False, content_scale=0.6).convert("RGB")),
}

# ═══ FastAPI:PWA 路由 + 掛載 Gradio ═══
app = FastAPI(title="RAG QA PWA")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return JSONResponse(MANIFEST, media_type="application/manifest+json", headers={"Cache-Control": "no-cache"})


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    # Service Worker 必須從根路徑提供,才能控制整個網站範圍
    return Response(SW_JS, media_type="application/javascript",
                    headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


@app.get("/pwa/offline.html", include_in_schema=False)
def offline_page():
    return HTMLResponse(OFFLINE_HTML)


@app.get("/pwa/icons/{name}", include_in_schema=False)
def icon(name: str):
    data = ICONS.get(name)
    if data is None:
        return Response(status_code=404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "embedding_ready": _CORE is not None}


# 必須在所有 PWA 路由註冊之後再掛載 Gradio
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    threading.Thread(target=_warm, daemon=True).start()  # 背景載入嵌入模型,不阻擋開 port
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
