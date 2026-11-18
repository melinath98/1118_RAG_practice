# 📚 RAG 文件問答(PWA + RWD)

上傳 PDF / DOCX,選擇檢索策略後直接向文件提問。Groq × ChromaDB × Gradio,
支援 **PWA**(可安裝到主畫面)與 **RWD**(手機、平板、桌機自適應)。

## 功能

- 8 種檢索策略:混合、語意、關鍵字、重新排序、多查詢、壓縮、父子文檔、HyDE
- PDF / DOCX(含表格與標題結構)
- 回答附 `[片段n]` 依據,可檢視檢索到的原文片段
- 匯出問答紀錄為 Word / CSV
- 每位使用者的文件與 API Key 互相獨立
- PWA:Manifest、Service Worker、離線提示頁
- RWD:手機直向堆疊、觸控友善、安全區域適配

## 檔案

```
app.py            # 全部程式(PWA 資源與圖示皆內嵌,不需要 static 資料夾)
requirements.txt
render.yaml       # Render Blueprint(選用)
.python-version   # 隱藏檔,上傳時請確認有帶上
README.md
```

## 本機執行

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...      # 選填,也可在網頁內輸入
python app.py
```

開啟 http://localhost:7860

## 部署到 Render

1. 把整個專案推上 GitHub(`git add -A` 才會包含 `.python-version` 等隱藏檔)。
2. Render → **New +** → **Web Service** → 選擇 repo。
3. 設定:

   | 欄位 | 值 |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `python app.py` |
   | Health Check Path | `/healthz` |
   | Instance Type | **Standard(2GB)以上** |

4. Environment 新增:

   | 名稱 | 值 |
   |---|---|
   | `PYTHON_VERSION` | `3.11.9` |
   | `GROQ_API_KEY` | 你的 Groq Key(選填) |
   | `EMBED_MODEL` | 預設 `intfloat/multilingual-e5-small`(選填) |

   (或選 **Blueprint**,自動讀取 `render.yaml`。)

## 重要注意事項

- **記憶體**:torch + 嵌入模型 + ChromaDB 需要 1GB 以上。Free / Starter 只有 512MB,會因記憶體不足被終止。
- **首次啟動較慢**:嵌入模型(約 470MB)在背景下載載入,網站會先開啟;載入完成前上傳文件會稍等。Render 磁碟是暫時性的,每次重新部署或重啟都會重新下載。
- **API Key**:設定 `GROQ_API_KEY` 後,所有訪客都會使用你的額度,且不會顯示在網頁上;若不想分享額度,請不要設定,讓使用者自行輸入。
- **資料不保存**:向量庫放在記憶體,使用者離開後其文件資料會被清除。
- PWA 需要 HTTPS,Render 預設提供;更新快取內容時請調高 `app.py` 內 `SW_JS` 的 `VERSION`。
- 離線時無法問答,只會顯示離線提示頁。

## 安裝成 App

- Android / Chrome:選單 → 安裝應用程式
- iPhone / Safari:分享 → 加入主畫面
- 桌機 Chrome / Edge:網址列右側的安裝圖示
