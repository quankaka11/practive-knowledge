# Tổng hợp kiến thức ôn phỏng vấn AI Engineer

15 chủ đề · 712 câu hỏi trong web quiz. File này được sinh tự động từ `data/src/*.json`.

## Mục lục

- 📝 NLP & Transformer
- 🧠 LLM: Kiến trúc, Training & Alignment
- ⚡ LLM Inference & Model Optimization
- 🔎 RAG, Retrieval & Vector DB
- 🤖 AI Agents, Tool Calling & MCP
- 🧩 Agent Engineering: Skills, Subagents, Frameworks
- 📊 LLM Evaluation, Observability & LLMOps
- 📈 Machine Learning & Deep Learning
- 👁️ Computer Vision, Speech, OCR & RecSys
- 🛠️ Python, Backend & MLOps cho AI
- 🏗️ ML System Design, Ranking & Experimentation
- ∑ Nền tảng: Toán & Thống kê cho AI
- 🧱 Nền tảng: ML & Deep Learning cốt lõi
- 📘 Nền tảng: NLP, LLM, RAG & Agent
- 🎓 Nền tảng: Lý thuyết AI tổng quát

---

## 📝 NLP & Transformer

### Tokenization: BPE, WordPiece, Unigram, byte-level

| Thuật toán | Train | Inference | Dùng ở |
|---|---|---|---|
| BPE | Bắt đầu từ ký tự, lặp: gộp cặp **tần suất cao nhất** | Áp merge rules theo thứ tự | GPT-2/3/4 (byte-level), RoBERTa, LLaMA (SentencePiece-BPE) |
| WordPiece | Gộp cặp có điểm `freq(ab)/(freq(a)·freq(b))` (tăng likelihood nhiều nhất) | Greedy longest-match-first, tiền tố `##` | BERT, DistilBERT |
| Unigram LM | Bắt đầu từ vocab **lớn**, loại dần token làm likelihood giảm ít nhất | Viterbi; hỗ trợ subword regularization (sampling) | T5, ALBERT, XLNet (qua SentencePiece) |

- **SentencePiece** là *thư viện* (không phải thuật toán thứ 3): train thẳng trên raw text, coi khoảng trắng là ký hiệu `▁` → detokenize lossless; cài đặt cả BPE và Unigram.
- **Byte-level BPE**: vocab cơ sở 256 byte → không bao giờ có `[UNK]`. GPT-2: 50,257 = 256 byte + 50,000 merges + `<|endoftext|>`. Một token có thể là *nửa ký tự* UTF-8 → cẩn thận khi stream/decode từng token.
- **Tiếng Việt**: ký tự có dấu chiếm 2–3 byte UTF-8 (`ă`=2, `ệ`=3); tokenizer thiên tiếng Anh tốn nhiều token hơn → tốn tiền, tốn context. Phải chuẩn hóa Unicode **NFC** (NFD tách dấu thành combining char → chuỗi byte khác → token khác).
- **PhoBERT** yêu cầu input đã **word-segment** (VnCoreNLP RDRSegmenter: `sản_phẩm`), max 256 token.
- Số token ≠ số ký tự ≠ số từ: tiếng Anh ~0.75 từ/token (~4 ký tự/token); so sánh perplexity giữa 2 tokenizer khác nhau là **không hợp lệ** nếu không quy về cùng đơn vị (word/byte).

### Biểu diễn từ: TF-IDF, n-gram LM, word2vec, GloVe, fastText

- **TF-IDF** cổ điển: `tf·log(N/df)` → từ xuất hiện ở mọi doc có idf = 0. **sklearn** (`smooth_idf=True`): `idf = ln((1+N)/(1+df)) + 1`, tf = raw count, sau đó chuẩn hóa L2 từng vector.
- **n-gram LM**: Markov bậc n−1; MLE cho xác suất 0 với n-gram chưa thấy → smoothing: Laplace (add-1, kém), Good-Turing, interpolation/back-off, **Kneser-Ney** (absolute discount d≈0.75 + *continuation probability* ∝ số ngữ cảnh khác nhau mà từ đứng sau → 'Francisco' thấp dù tần suất cao).
- **word2vec**: skip-gram (center → context, tốt cho từ hiếm), CBOW (trung bình context → center, nhanh hơn). 2 ma trận W_in, W_out. Tránh softmax O(V) bằng hierarchical softmax hoặc **negative sampling**: `log σ(u_o·v_c) + Σ_k log σ(−u_k·v_c)`, noise ∝ `count^0.75`, k = 5–20 (data nhỏ), 2–5 (data lớn). Subsampling từ phổ biến (t ≈ 1e-5).
- Số cặp skip-gram câu n từ, window w: mỗi vị trí i có `min(i,w) + min(n−1−i,w)` context.
- **GloVe**: weighted least squares trên log co-occurrence: `Σ f(X_ij)(w_i·w̃_j + b_i + b̃_j − log X_ij)²`, `f(x) = (x/x_max)^0.75` nếu x < x_max=100, ngược lại 1; f(0)=0.
- **fastText**: vector từ = tổng vector character n-gram (3–6, có biên `<`, `>`, hashing vào bucket) + vector từ → tạo được vector cho OOV, tốt cho ngôn ngữ giàu hình thái.
- Tất cả là **static embedding**: 1 vector / từ, không phân biệt đa nghĩa ('bank'). Contextual (ELMo, BERT) giải quyết.

### RNN, LSTM, GRU, seq2seq & attention cổ điển

- Vanilla RNN: `h_t = tanh(W_hh h_{t−1} + W_xh x_t + b)`. Gradient = tích nhiều Jacobian → **vanishing/exploding** theo hàm mũ.
- **LSTM**: 3 cổng (input, forget, output) + candidate g; `c_t = f⊙c_{t−1} + i⊙g` (cập nhật **cộng**) → gradient đi qua cell state ít bị suy giảm khi f≈1. Không chống exploding → vẫn cần **gradient clipping**. Mẹo: forget-gate bias = 1.
- **GRU**: 2 cổng (update z, reset r), không có cell state riêng, 3 khối trọng số (≈3/4 tham số LSTM).

| Công thức tham số (x = input, h = hidden) | |
|---|---|
| LSTM Keras (1 bias) | `4·(h·(x+h) + h)` |
| LSTM PyTorch (b_ih + b_hh) | `4·(h·(x+h) + 2h)` |
| GRU PyTorch | `3·(h·(x+h) + 2h)` |
| Bidirectional | ×2 |

- **Seq2seq**: encoder nén cả câu vào 1 vector cố định → nút cổ chai với câu dài → **attention**.
- **Bahdanau** (additive): `v^T tanh(W1 s_{t−1} + W2 h_j)`, dùng trạng thái decoder *trước*, encoder BiRNN. **Luong** (multiplicative): `dot` hoặc `general h_tᵀ W h̄_s`, dùng trạng thái decoder *hiện tại* h_t.

### Transformer: attention, multi-head, mask, độ phức tạp

- `Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)·V`. Chia **√d_k** vì nếu q,k có thành phần độc lập mean 0 var 1 thì `Var(q·k) = d_k` → softmax bão hòa, gradient ≈ 0.
- **Multi-head**: h head, mỗi head d_k = d/h → tổng tham số chiếu vẫn `4·d²` (+4d bias), bằng single-head d; nhưng có **h ma trận score n×n** → bộ nhớ attention weights tăng theo h.
- **Mask**: cộng −∞ *trước* softmax. Causal: j > i. Padding: theo chiều **key**. Cross-attention: padding mask của nguồn, không causal.
- **FLOPs 1 layer** (d_ff = 4d): chiếu QKVO `8nd²`, FFN `16nd²`, QKᵀ + AV `4n²d`. Phần n² vượt FFN khi n > 4d → với LLM d=4096, n=512 thì FFN chiếm ưu thế (~32×).
- **Bộ nhớ attention weights**: `B·H·n²·bytes` mỗi layer (B=8, H=16, n=4096, fp16 → 4 GiB). FlashAttention không vật chất hóa ma trận n×n (tiled + online softmax) → bộ nhớ O(n).
- Không có positional info, self-attention là **permutation-equivariant**.
- Câu hay hỏi: vì sao √d_k? MHA có tốn tham số hơn không? Độ phức tạp O(n²d) vs O(nd²)? Mask đặt ở đâu?

### Thành phần Transformer & đếm tham số

| Thành phần | Ghi nhớ |
|---|---|
| Post-LN (gốc) | `LN(x + F(x))`; gradient lớn ở layer gần output lúc init → cần LR warmup, khó train sâu |
| Pre-LN | `x + F(LN(x))`; ổn định, train được không warmup (Xiong 2020), cần LN cuối |
| LayerNorm | trừ mean, chia std, có γ và β (2d tham số) |
| RMSNorm | chỉ chia RMS, không trừ mean, chỉ γ (d tham số), rẻ hơn — LLaMA, T5 |
| FFN chuẩn | `W2·act(W1 x)`, d_ff = 4d, `8d²` tham số |
| SwiGLU | `W2(SiLU(W1 x) ⊙ W3 x)`, 3 ma trận, hidden ≈ 2/3·4d (LLaMA-7B: 11008) |
| Sinusoidal PE | **cộng** vào embedding, không tham số |
| Learned PE | BERT/GPT-2, không ngoại suy quá max_position |
| RoPE | xoay Q, K (không V) → tích chỉ phụ thuộc m−n |
| ALiBi | không PE, cộng bias âm tỷ lệ với khoảng cách vào score, slope riêng mỗi head |
| Weight tying | output head dùng chung ma trận embedding (tiết kiệm V·d) |

**Đếm tham số 1 layer encoder (có bias)**: attention `4(d²+d)` + FFN `2·d·d_ff + d_ff + d` + 2 LN `4d`. BERT-base (768, 3072): 2,362,368 + 4,722,432 + 3,072 = **7,087,872** ≈ 7.09M; ×12 = 85M + embedding 23.8M ≈ 110M.

Xấp xỉ nhanh decoder không bias: `12·L·d²` (FFN 4d).

### Kiến trúc & mục tiêu pre-train

| Model | Kiến trúc | Objective |
|---|---|---|
| BERT | Encoder | MLM 15% (80% [MASK] / 10% random / 10% giữ) + NSP |
| RoBERTa | Encoder | MLM **dynamic masking**, **bỏ NSP**, data 160GB, batch lớn, byte-level BPE 50K |
| ELECTRA | Encoder | Replaced Token Detection: generator nhỏ (MLM) + discriminator phân loại **mọi** token |
| GPT | Decoder | Causal LM (next token) |
| T5 | Enc-Dec | Span corruption 15%, mean span 3, sentinel `<extra_id_i>`, target chỉ chứa các span bị xóa |
| BART | Enc-Dec | Denoising (text infilling, sentence permutation), target = toàn câu gốc |

- Loss MLM chỉ tính trên ~15% vị trí được chọn → kém hiệu quả mẫu → ELECTRA cải thiện.
- 10% random + 10% giữ nguyên: giảm lệch pha pre-train/fine-tune (fine-tune không có [MASK]).
- Chọn kiến trúc: encoder → phân loại/NER/embedding; decoder → sinh; enc-dec → dịch/tóm tắt (nhưng T5 làm được cả phân loại dạng text-to-text, GPT dịch được qua prompting).

### Tác vụ NLP: NER, CRF, QA, tóm tắt, MT

- **NER BIO (IOB2)**: B- bắt đầu, I- bên trong, O ngoài. `O I-PER` không hợp lệ. BIOES/BILOU thêm E/L (end) và S/U (single).
- Subword: gán nhãn subword đầu, các subword còn lại `-100` (bị CrossEntropy bỏ qua) — hoặc gán I-.
- **Linear-chain CRF**: mô hình *discriminative* trên cả chuỗi nhãn, có ma trận transition, chuẩn hóa bằng partition function (forward algorithm), giải mã **Viterbi** → loại chuỗi nhãn vô lý. BiLSTM-CRF là baseline kinh điển.
- **Extractive QA**: dự đoán start/end logits trên token context; context dài → sliding window (`max_length=384`, `doc_stride=128`), không có đáp án → trỏ về [CLS]; chọn span hợp lệ (end ≥ start, độ dài giới hạn) có tổng logit lớn nhất.
- **Tóm tắt**: extractive (chọn câu) vs abstractive (sinh mới, rủi ro hallucination). **MT**: enc-dec hoặc LLM; beam search, đánh giá BLEU/chrF/COMET.
- **POS tagging**, **sentiment**, **text classification**: encoder + head [CLS]/mean pooling.

### Metrics NLP (công thức cần nhớ)

- **Perplexity** = `exp(CE trung bình theo nats/token)` (hoặc `2^CE` nếu bits). Word-level PPL = `exp(CE_token × số token/từ)`. Chỉ so sánh PPL khi cùng tokenizer/đơn vị.
- **BLEU** = `BP · exp(Σ w_n log p_n)`, p_n = modified (clipped) n-gram precision, n = 1..4, w = 1/4. `BP = 1` nếu c > r, ngược lại `exp(1 − r/c)`. Corpus BLEU cộng dồn count toàn corpus (≠ trung bình sentence BLEU). Sentence BLEU = 0 nếu không có 4-gram khớp (cần smoothing).
- **ROUGE-N**: n-gram overlap, thiên về recall. **ROUGE-L**: LCS (không cần liên tiếp), `P = LCS/|cand|`, `R = LCS/|ref|`.
- **METEOR**: khớp unigram (exact, stem, synonym), `F_mean` trọng recall (10PR/(R+9P)), phạt phân mảnh theo số chunk.
- **BERTScore**: greedy matching token–token bằng cosine contextual embedding → P/R/F, tùy chọn idf weighting, baseline rescaling.
- **SQuAD EM/F1**: chuẩn hóa (lowercase, bỏ dấu câu, bỏ a/an/the, gộp khoảng trắng); F1 trên **túi token** (không xét thứ tự); lấy max qua các gold answer.
- **NER**: entity-level (seqeval, strict: đúng biên + đúng loại) thấp hơn nhiều so với token-level accuracy (bị O chi phối).

### Sentence embedding & contrastive learning

- **Cosine** = `u·v / (‖u‖·‖v‖)`; với vector đã L2-normalize thì cosine = dot product.
- **Anisotropy**: embedding của LM pre-trained nằm trong một hình nón hẹp → cosine của mọi cặp đều cao. Khắc phục: mean-centering/whitening (BERT-whitening), contrastive fine-tune (SimCSE, SBERT).
- [CLS] / mean pooling BERT chưa fine-tune cho STS **kém hơn** cả average GloVe (SBERT paper).
- **SBERT**: siamese bi-encoder, mean pooling; objective NLI classification `softmax(W·[u; v; |u−v|])`; regression cosine; triplet.
- **Bi-encoder vs cross-encoder**: bi-encoder precompute + ANN (retrieval); cross-encoder chính xác hơn nhưng O(số cặp) forward → reranker.
- **InfoNCE / MultipleNegativesRankingLoss**: `−log(exp(sim(q,p⁺)/τ) / Σ_j exp(sim(q,p_j)/τ))`, in-batch negatives → batch lớn tốt hơn; τ nhỏ → phạt hard negative mạnh hơn.
- **SimCSE unsupervised**: cùng câu qua encoder 2 lần với dropout khác nhau = positive; τ = 0.05. Supervised SimCSE: NLI entailment = positive, contradiction = hard negative.

### Zero-shot / few-shot classification

- **NLI-based zero-shot** (bart-large-mnli): premise = văn bản, hypothesis = `This example is about {label}.` → điểm = P(entailment). Chi phí ∝ số nhãn (1 forward / nhãn).
- **Embedding zero-shot**: cosine giữa embedding văn bản và embedding mô tả nhãn (rẻ, nhanh).
- **LLM prompting**: zero-shot / few-shot in-context learning — **không cập nhật trọng số**. Nhạy với thứ tự ví dụ (recency bias), tỷ lệ nhãn (majority-label bias), từ phổ biến (common-token bias) → contextual calibration, cân bằng demo, chọn demo bằng kNN.
- **SetFit**: contrastive fine-tune sentence-transformer trên vài cặp + logistic head → mạnh với 8–64 ví dụ/nhãn.
- Khi có vài nghìn nhãn chất lượng, fine-tune encoder nhỏ thường rẻ hơn và tốt ngang/hơn LLM zero-shot về accuracy, latency, cost.

### Bẫy & câu phỏng vấn hay gặp

- Nhầm LSTM PyTorch (2 bias) với Keras (1 bias).
- Nghĩ MHA tốn gấp h lần tham số (sai), quên score memory tăng theo h (đúng).
- Nghĩ O(n²) luôn chiếm ưu thế (sai với n nhỏ hơn ~4d).
- Nhân mask 0/1 *sau* softmax (sai — hàng không còn tổng = 1).
- RoPE xoay cả V (sai). Sinusoidal PE nhân với embedding (sai — cộng).
- RoBERTa vẫn giữ NSP (sai). ELECTRA là GAN đối kháng (sai — generator train bằng MLM).
- Loss MLM trên toàn chuỗi (sai — chỉ 15%).
- Corpus BLEU = trung bình sentence BLEU (sai).
- So PPL giữa 2 tokenizer khác nhau (sai).
- Token accuracy cho NER (sai — dùng entity F1).
- Đưa câu tiếng Việt thô vào PhoBERT (thiếu word segmentation), quên NFC.
- Off-by-one: số merge BPE = vocab − base − special.


---

## 🧠 LLM: Kiến trúc, Training & Alignment

### Pretraining objective & dữ liệu

**Causal LM (next-token):** `L = −(1/T)·Σ_t log p_θ(x_{t+1} | x_≤t)` — teacher forcing + causal mask → 1 forward cho T dự đoán song song.

- **Perplexity** = `exp(loss nats)` (= `2^loss` nếu loss tính bằng bit). Chỉ so sánh được khi **cùng tokenizer**; khác tokenizer → dùng **bits-per-byte** = tổng NLL / (ln2 · số byte).
- **Pipeline dữ liệu:** extract text → lọc ngôn ngữ → heuristic/quality classifier (FineWeb-Edu, perplexity filter) → **dedup exact + fuzzy (MinHash-LSH)** → lọc PII/toxic → **decontamination** (n-gram overlap với test set) → **data mixture** (web, code, math, sách, đa ngữ; tăng tỷ trọng dữ liệu chất lượng cao ở cuối — annealing).
- **Lặp dữ liệu:** tới ~4 epoch gần như ngang dữ liệu mới (Muennighoff 2023), nhiều hơn → lợi ích giảm, tăng memorization.
- **Dấu hiệu contamination:** điểm cao bất thường nhưng tụt mạnh trên bản rephrase; model hoàn thành nguyên văn câu hỏi benchmark.
- **Tokenizer:** byte-level BPE (không OOV). Vocab lớn → chuỗi ngắn hơn (tốt cho tiếng Việt/đa ngữ) nhưng embedding + LM head tăng `V·d` tham số. Llama 2: 32K, Llama 3: ~128K.

### Scaling laws & ước lượng compute

| | Kaplan 2020 | Chinchilla 2022 |
|---|---|---|
| N_opt | ∝ C^0.73 | ∝ C^0.5 |
| D_opt | ∝ C^0.27 | ∝ C^0.5 |
| Kết luận | ưu tiên model to | **~20 token / tham số** |

- **Compute training:** `C ≈ 6·N·D` FLOPs (2 forward + 4 backward). Inference ≈ `2·N` FLOPs/token. Activation checkpointing → phần cứng chạy ~`8·N·D`.
- **Compute-optimal:** `C = 6·N·20N = 120·N²` → `N = √(C/120)`. Ví dụ C = 1e23 → N ≈ 28.9B, D ≈ 577B.
- **Thời gian:** `T = 6ND / (#GPU × peak × MFU)`. 70B × 1.4T trên 1024 A100 (312 TFLOPS), MFU 40% → ~53 ngày.
- **MFU** = FLOPs hữu ích (6ND/s) / peak; **HFU** tính cả recompute. Thực tế 35–55% (PaLM 46%, Llama 3 405B 38–43%).
- **Mốc:** Chinchilla 70B/1.4T vs Gopher 280B/300B (cùng compute). GPT-3: 175B/300B ≈ 3.15e23 FLOPs. Llama 3 405B/15.6T ≈ 3.8e25.
- **Overtraining có chủ đích** (Llama 3 8B: 15T token ≈ 1875 token/tham số): Chinchilla không tính chi phí inference; model nhỏ train lâu rẻ hơn khi serve.

### Kiến trúc LLM hiện đại (Llama-style)

Decoder-only + **pre-norm RMSNorm** + **RoPE** + **SwiGLU** + **GQA** + linear không bias.

- **RMSNorm:** `y = x/√(mean(x²)+ε)·γ` — không trừ mean, không β.
- **RoPE:** xoay cặp chiều của Q, K góc `m·θ_i`, `θ_i = base^(−2i/d)` → `q_m·k_n` chỉ phụ thuộc `m−n`. Áp ở mọi layer, KHÔNG lên V. Base 10K (Llama 2), 500K (Llama 3).
- **SwiGLU:** `W_down(SiLU(W_gate x) ⊙ W_up x)` — 3 ma trận nên hidden ≈ `8d/3` (Llama-2 7B: 11008 thay vì 16384).

| Attention | KV head | KV cache | Ghi chú |
|---|---|---|---|
| MHA | = H | 1× | chất lượng chuẩn |
| GQA | G (vd 8) | H/G nhỏ hơn | cân bằng, Llama 2 70B / Llama 3 |
| MQA | 1 | H nhỏ hơn | nhanh nhất, dễ giảm chất lượng |

- GQA giảm **KV cache & băng thông**, không giảm FLOPs QKᵀ (vẫn H query head).
- **Sliding window (Mistral, W=4096):** O(n·W)/layer, rolling buffer KV cố định W, receptive field ≈ n_layer·W ≈ 131K.

### Mixture of Experts

- Router (linear + softmax) chọn **top-k** expert FFN cho mỗi token; output = tổng có trọng số. Attention/embedding **dùng chung**.
- **Mixtral 8x7B:** ~46.7B total, ~12.9B active (top-2) — KHÔNG phải 56B.
- **Compute/token ∝ active params; VRAM ∝ total params** → MoE tiết kiệm FLOPs, không tiết kiệm bộ nhớ.
- **Cách tính:** active = n_layer·(k·expert + attention + router) + embedding + LM head. Expert SwiGLU = `3·d·d_ff`.
- **Load-balancing loss (Switch):** `L_aux = α·N_e·Σ f_i·P_i` — f_i (tỷ lệ token, không khả vi) × P_i (xác suất router trung bình, khả vi). Tránh router collapse.
- Biến thể: expert capacity + token dropping (Switch), shared expert + fine-grained experts, aux-loss-free balancing bằng bias (DeepSeek-V3).
- Khó khăn: all-to-all communication (expert parallel), mất cân bằng tải, fine-tune dễ overfit, serving cần nhiều VRAM.

### Long context & mở rộng context

| Kỹ thuật | Ý chính | Cần fine-tune? |
|---|---|---|
| Ngoại suy thẳng | dùng vị trí > L_train | vỡ (góc quay lạ) |
| **Position Interpolation** | `m' = m·L/L'`, nén đều | ngắn (~1K step) |
| **NTK-aware** | tăng base: `b' = b·s^(d/(d−2))` | có thể không (dynamic NTK) |
| **YaRN** | NTK-by-parts theo dải tần số + scale nhiệt độ logits | rất ít token |

- PI nén cả tần số cao → mất độ phân giải cục bộ; NTK/YaRN giữ tần số cao, nội suy tần số thấp.
- RoPE scaling **không** giảm KV cache; context dài vẫn tốn bộ nhớ tuyến tính, attention O(n²).
- **Lost in the middle:** chữ U — đầu/cuối tốt, giữa kém. RAG: ít chunk tốt (rerank), đặt chunk quan trọng ở đầu/cuối.
- Pass needle-in-a-haystack ≠ suy luận tốt trên context dài (RULER).

### Ổn định training: precision, optimizer, LR

| Format | Exponent | Mantissa | Max | Loss scaling |
|---|---|---|---|---|
| fp32 | 8 | 23 | 3.4e38 | không |
| fp16 | 5 | 10 | 65504 | **cần** |
| bf16 | 8 | 7 | 3.4e38 | thường không |

- **Loss scaling:** nhân loss × S → backward → unscale → skip step nếu inf/NaN.
- bf16 kém chính xác hơn fp16 → vẫn giữ **master weights fp32**.
- **AdamW:** decay tách khỏi adaptive scaling (`w −= lr·λ·w`); Adam+L2 bị chia √v̂. WD ~0.1, không decay bias/norm.
- **LR schedule:** linear warmup (vài trăm–vài nghìn step; Adam's v chưa ổn định lúc đầu) → cosine decay về ~10% max LR (hoặc WSD: warmup-stable-decay).
- **Gradient clipping** global norm 1.0. Loss spike → rollback checkpoint, skip batch, giảm LR; QK-norm, z-loss giúp ổn định logits.
- Adam β₂ = 0.95 cho LLM (phản ứng nhanh hơn với gradient lớn).

### Bộ nhớ training & parallelism

**Model states Adam mixed precision = 16 byte/tham số:** 2 (bf16 W) + 2 (bf16 grad) + 4 (fp32 master) + 4 (m) + 4 (v). 7B → 112 GB (+ activation).

| Chiến lược | Shard | Bộ nhớ/GPU (Ψ tham số, N_d GPU) | Comm |
|---|---|---|---|
| DDP | không | 16Ψ | 2Ψ |
| ZeRO-1 | optimizer | 4Ψ + 12Ψ/N_d | 2Ψ |
| ZeRO-2 | + grad | 2Ψ + 14Ψ/N_d | 2Ψ |
| ZeRO-3 / FSDP | + params | 16Ψ/N_d | 3Ψ (1.5×) |

- Ví dụ 7B, 8 GPU: ZeRO-1 38.5 GB, ZeRO-2 26.25 GB, ZeRO-3 14 GB.
- **Tensor parallel:** chia matmul trong layer, all-reduce mỗi layer → trong node (NVLink). **Pipeline parallel:** chia layer theo stage, p2p → giữa node; bubble ≈ `(p−1)/(m+p−1)`.
- **Gradient accumulation:** global batch = #GPU × micro-batch × accum. Loss phải chuẩn hoá theo **tổng token hợp lệ của cả global batch** (bug HF 10/2024).
- **Activation checkpointing:** tăng ~33% compute, giảm mạnh activation memory; MFU không tăng (HFU tăng).

### SFT, chat template & PEFT

- **SFT:** dữ liệu (instruction, response) theo **chat template** (ChatML, Llama 3...). **Loss masking**: label −100 cho system/user, chỉ tính loss trên assistant **kể cả token EOS/end-of-turn** (thiếu → model không dừng).
- Dùng `apply_chat_template` giống nhau ở train và inference; cẩn thận `pad = eos` rồi mask nhầm EOS.
- **Catastrophic forgetting:** replay dữ liệu chung, LoRA/LR nhỏ/ít epoch, eval regression (MMLU, IFEval).

**LoRA:** `W' = W + (α/r)·B·A`; A random, **B = 0**; params = `r·(d_in + d_out)` mỗi ma trận (chú ý k/v nhỏ hơn ở GQA). Merge được → 0 latency. α/r ≈ hệ số LR; rsLoRA dùng α/√r.

**QLoRA:** base **NF4** (quantile của N(0,1), block 64) frozen + dequantize bf16 khi tính; **double quantization** 0.5 → 0.127 bit/tham số (tiết kiệm 0.373 bit); **paged optimizers** (unified memory). 65B trên 1 GPU 48GB.

| PEFT | Merge được | Chi phí inference |
|---|---|---|
| LoRA | có | 0 |
| Adapter (Houlsby) | không | + latency |
| Prompt tuning | không | + token input |
| Prefix tuning | không | + KV mỗi layer |

### Alignment: RLHF, DPO & biến thể

**RLHF:** SFT → Reward model (Bradley–Terry: `L = −log σ(r(x,y_w) − r(x,y_l))`) → PPO tối đa `r − β·KL(π‖π_ref)`. Cần 4 model: policy, ref, RM, critic.

**DPO:** `L = −log σ(β[log(π/π_ref)(y_w) − log(π/π_ref)(y_l)])`. Implicit reward `β·log(π/π_ref)`. Không RM, không sampling online, vẫn cần ref. Ban đầu loss = ln2 ≈ 0.693. β lớn → bám ref chặt hơn.

| Phương pháp | Dữ liệu | Ref model? | Ý chính |
|---|---|---|---|
| DPO | cặp | có | logistic trên log-ratio |
| IPO | cặp | có | loss bình phương, chống overfit |
| KTO | **nhãn đơn tốt/xấu** | có | prospect theory |
| ORPO | cặp | **không** | SFT + odds-ratio, 1 giai đoạn |
| SimPO | cặp | **không** | reward = log-prob trung bình theo độ dài + margin γ |

- **Reward hacking:** dài dòng, sycophancy; chống bằng KL, ensemble RM, length normalization, verifiable rewards. β → 0 làm tệ hơn.
- **RLAIF / Constitutional AI:** self-critique & revision theo nguyên tắc (SL) → preference do AI gán nhãn (RL).

### GRPO, reasoning models & test-time compute

- **GRPO** (DeepSeekMath): mỗi prompt sample G output, `Â_i = (r_i − mean)/std` trong nhóm; **không critic**; clip kiểu PPO + KL vào loss. Nhóm toàn đúng/toàn sai → advantage 0 (lãng phí; DAPO dùng dynamic sampling).
- **RLVR** (RL with verifiable rewards): đáp án toán, unit test, format — ít bị hack hơn neural RM.
- **DeepSeek-R1-Zero:** RL thuần từ V3-Base, reward theo luật (accuracy + format) → CoT dài tự phát ('aha moment'), nhưng khó đọc. **R1:** cold-start SFT → RL → rejection sampling SFT → RL; distill sang Qwen/Llama bằng SFT.
- **Test-time compute:** CoT, **self-consistency** (sample T>0, majority vote đáp án cuối), best-of-N + verifier/PRM, tree search, long CoT. Greedy N lần = 1 lần.
- ORM (outcome reward) vs PRM (process reward, chấm từng bước).

### Distillation, hallucination, emergent & ICL

**Distillation:**
- Logit KD (Hinton): `softmax(z/T)` cho cả teacher & student, loss mềm × **T²**; cần **cùng tokenizer**. T > 1 → mềm hơn.
- Forward KL(p‖q): mean-seeking/mode-covering; Reverse KL(q‖p): mode-seeking (MiniLLM).
- Sequence-level KD (Kim & Rush): SFT trên output teacher → dùng được với API, khác tokenizer (synthetic data, R1-distill).

**Hallucination:** objective likelihood ≠ factuality; long-tail/knowledge cutoff; SFT trên kiến thức mới làm tăng bịa (Gekhman 2024); eval nhị phân thưởng đoán; snowballing. Temperature 0 và RAG **không** loại bỏ hoàn toàn.

**Emergent abilities:** Wei 2022 (xuất hiện đột ngột) vs Schaeffer 2023 (mirage do metric không liên tục như exact match ≈ p^L).

**ICL:** không cập nhật trọng số; nhạy định dạng/thứ tự; nhãn random chỉ giảm nhẹ ở nhiều task (Min 2022) nhưng model rất lớn có thể theo nhãn lật (Wei 2023); base GPT-3 đã few-shot được.

### Câu hỏi phỏng vấn hay gặp & lỗi thường gặp

**Hay hỏi:**
- Tính nhanh: FLOPs (6ND), thời gian train với MFU, VRAM full FT (16 byte/param), LoRA params, MoE active params, KV cache của GQA.
- Vì sao Llama dùng RMSNorm/RoPE/SwiGLU/GQA? Vì sao pre-norm?
- Chinchilla vs overtraining; khi nào chọn model nhỏ train lâu?
- ZeRO 1/2/3 khác gì FSDP, TP, PP? Chọn layout cho N GPU, M node.
- LoRA vs QLoRA vs full FT; chọn r, α, target modules.
- RLHF vs DPO vs GRPO: cần model nào, dữ liệu gì, ưu/nhược.

**Lỗi thường gặp:**
- So sánh perplexity giữa tokenizer khác nhau.
- Dùng 2ND thay 6ND; quên MFU; nhầm GB/GiB, bit/byte.
- Coi MoE tiết kiệm VRAM; tính Mixtral = 56B.
- Mask mất EOS khi SFT; template train ≠ inference.
- Gradient accumulation chuẩn hoá loss theo micro-batch.
- Khởi tạo cả A, B của LoRA bằng 0; đổi r mà giữ α và LR.
- Giảm β KL tưởng giảm reward hacking.


---

## ⚡ LLM Inference & Model Optimization

### Decoding & sampling

**Pipeline xử lý logits (thứ tự phổ biến):** penalties / logit_bias → chia cho temperature → lọc top-k / top-p / min-p → chuẩn hóa lại → sample.

| Kỹ thuật | Cách làm | Ghi nhớ |
|---|---|---|
| Greedy | argmax mỗi bước | tất định về thuật toán, dễ lặp |
| Beam search | giữ k chuỗi có Σlog p cao nhất | hợp dịch máy/ASR; văn bản mở dễ nhạt, lặp; tốn k× KV cache; KHÔNG đảm bảo tối ưu toàn cục |
| Temperature | `softmax(z/T)` | T<1 nhọn hơn, T>1 phẳng hơn, T→0 ≈ greedy |
| Top-k | giữ k token xác suất cao nhất | k cố định, không thích nghi |
| Top-p (nucleus) | tập NHỎ NHẤT có tổng ≥ p | phải thêm cả token làm tổng vượt p, rồi chuẩn hóa lại |
| Min-p | giữ token có p ≥ min_p × p_max | ngưỡng tự co giãn theo độ tự tin của model |

**Penalties:**
- OpenAI: `mu[j] − c[j]·frequency_penalty − 1[c[j]>0]·presence_penalty` (frequency nhân theo số lần, presence trừ đúng 1 lần).
- HF `repetition_penalty` (CTRL): logit > 0 thì CHIA, logit < 0 thì NHÂN cho penalty — luôn làm token kém khả năng hơn, không phụ thuộc số lần.
- `logit_bias`: cộng vào logit theo **token ID** (không phải chuỗi); −100 ≈ cấm, +100 ≈ ép.

**Không tất định ở temperature=0:** nguyên nhân chính trong serving là kernel không *batch-invariant* (kích thước batch đổi theo tải → thứ tự reduction đổi → logits lệch nhỏ → argmax đổi khi 2 token sát nhau). `seed` của provider chỉ là best-effort (xem `system_fingerprint`).

### Prefill vs decode, metrics, roofline

| | Prefill | Decode |
|---|---|---|
| Xử lý | toàn bộ prompt song song | 1 token/bước/chuỗi |
| Bottleneck | compute-bound (matmul lớn) | memory-bandwidth-bound (đọc lại weights + KV mỗi bước) |
| Metric | **TTFT** | **TPOT / ITL** |

- **E2E latency = TTFT + (n − 1) · TPOT** (token đầu đã nằm trong TTFT).
- **Throughput hệ thống** ≈ batch / TPOT (tokens/s) ở pha decode; **goodput** = throughput thỏa SLO.
- TTFT tăng do: prompt dài, hàng đợi, cache miss. `max_tokens` và độ dài output KHÔNG ảnh hưởng TTFT.

**Roofline:** arithmetic intensity AI = FLOPs / bytes. Ridge point = peak FLOPS / bandwidth. AI < ridge → memory-bound.
- Ví dụ H100 SXM: 989 TFLOPS BF16 dense / 3.35 TB/s ≈ **295 FLOP/byte**; A100 80GB: 312 / 2.04 ≈ 153.
- Decode linear layer BF16: ≈ 2B FLOPs / 2 bytes mỗi param → AI ≈ **B** (batch size). Batch 1 ≈ 1 FLOP/byte → cực kỳ memory-bound.
- Attention trên KV cache: mỗi chuỗi đọc KV riêng → AI gần như không tăng theo batch → context dài giữ decode memory-bound.

### Toán bộ nhớ: weights, GB vs GiB, tokens/s

| dtype | bytes/param | 7B | 13B | 70B |
|---|---|---|---|---|
| FP32 | 4 | 28 GB | 52 GB | 280 GB |
| FP16/BF16 | 2 | 14 GB | 26 GB | 140 GB |
| INT8/FP8 | 1 | 7 GB | 13 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 6.5 GB | 35 GB |

- **GB = 10⁹ byte, GiB = 2³⁰ = 1.0737·10⁹ byte.** 26 GB = 24.21 GiB. nvidia-smi hiển thị MiB.
- INT4 thực tế > 4 bit/weight: group-wise 4-bit, group 128 + scale FP16 + zero FP16 → 4 + 32/128 = **4.25 bpw**; GGUF Q4_K_M ≈ 4.8–4.9 bpw.
- Tổng VRAM = weights + KV cache + activation/workspace + CUDA context (≈ vài GB).
- **Cận trên tokens/s decode (batch 1)** ≈ bandwidth / bytes weights. Vd 8B FP16 (16 GB) trên 2 TB/s ≈ 125 tok/s; 70B INT4 (35 GB) trên 3.35 TB/s ≈ 96 tok/s. Thực tế đạt ~60–80%.
- Quantization weight giúp decode nhanh gần tỉ lệ giảm bytes, nhưng không giảm FLOPs nếu vẫn tính FP16 (W4A16).

### KV cache

**Công thức:** `KV bytes = 2 (K và V) × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes/elem`

| Model | layers | kv_heads | head_dim | KV/token FP16 |
|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 128 | 512 KiB |
| Llama-3-8B (GQA) | 32 | 8 | 128 | 128 KiB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | 320 KiB |

- Llama-3-8B, batch 16 × 8192 token → 16 GiB; Llama-3-70B, 1 request 128K token → 40 GiB = 42.95 GB.
- **GQA** giảm KV theo n_q_heads / n_kv_heads (8B: 32/8 = 4×); **MQA** = 1 KV head (giảm nhiều nhất, chất lượng giảm hơn).
- **KV cache quantization** (FP8/INT8/INT4) giảm 2–4× → nhiều chuỗi đồng thời hơn / context dài hơn.
- vLLM: ngân sách KV = `gpu_memory_utilization × VRAM − weights − activation peak`; số chuỗi tối đa = ngân sách / KV mỗi chuỗi.
- Lỗi hay gặp: dùng n_q_heads thay n_kv_heads, quên hệ số 2, lẫn GB/GiB.

### Kỹ thuật serving: PagedAttention, batching, prefix caching, FlashAttention, TP

- **PagedAttention (vLLM):** KV cache chia block cố định (mặc định 16 token), cấp phát theo nhu cầu, không cần liên tục → lãng phí bộ nhớ < 4% (so với cấp phát trước max_len), chia sẻ block bằng copy-on-write (parallel sampling, beam, prefix). Không xấp xỉ attention.
- **Static batching:** chờ cả batch xong (bị chuỗi dài nhất giữ). **Dynamic batching** (Triton): gom request theo cửa sổ thời gian/batch size tối đa — ở mức request. **Continuous / in-flight batching:** lập lịch mỗi iteration, chuỗi xong rời ngay, chuỗi mới vào ngay → throughput cao hơn nhiều.
- **Chunked prefill:** chia prefill dài thành chunk, trộn với decode → ITL ổn định, không bị khựng; TTFT prompt dài có thể tăng nhẹ. Thay thế khác: disaggregated prefill/decode.
- **Prefix caching (vLLM APC, SGLang RadixAttention, prompt caching của provider):** tái dùng KV của prefix trùng khớp CHÍNH XÁC theo token → giảm TTFT và giá input cached. Đặt phần tĩnh lên đầu, phần động xuống cuối.
- **FlashAttention:** attention CHÍNH XÁC, IO-aware: tiling vào SRAM + online softmax, không ghi ma trận N×N ra HBM → bộ nhớ O(N), ít truy cập HBM; FLOPs vẫn O(N²). FA2 song song tốt hơn, FA3 cho Hopper (FP8, async).
- **Tensor parallel:** chia ma trận trong mỗi layer (Megatron: column rồi row), 2 all-reduce/block → cần NVLink trong node; giảm weights/GPU và latency/token. Pipeline parallel tăng throughput nhưng không giảm latency/token ở batch 1. TP > n_kv_heads → nhân bản KV head.

### Speculative decoding

- Draft (model nhỏ / head) đề xuất γ token → target verify song song trong 1 forward → chấp nhận tiền tố đúng + 1 token từ target.
- **Rejection sampling chuẩn** (Leviathan/Chen 2023): chấp nhận với xác suất min(1, p/q); nếu từ chối, sample từ norm(max(0, p − q)) → **phân phối output y hệt target**.
- Số token kỳ vọng mỗi vòng (α i.i.d.): **(1 − α^(γ+1)) / (1 − α)**. Speedup walltime ≈ **(1 − α^(γ+1)) / ((1 − α)(γc + 1))**, c = chi phí 1 bước draft / 1 bước target.
  - α = 0.8, γ = 4, c = 0.1 → 3.36 token/vòng, speedup ≈ 2.40×.
- **Medusa:** thêm nhiều decoding head + tree attention, không cần draft model; *typical acceptance* KHÔNG đảm bảo giữ đúng phân phối (greedy thì lossless).
- **EAGLE:** draft head nhẹ tự hồi quy ở mức feature (hidden state) của target, lossless.
- Lợi ích giảm khi batch lớn (GPU đã compute-bound) và khi α thấp (task sáng tạo, temperature cao). Dạng chuẩn cần chung tokenizer.

### Quantization: nền tảng & tính tay

- **Asymmetric (uint8 [0,255]):** `scale = (max − min)/255`, `zp = round(−min/scale)`, `q = clamp(round(x/scale) + zp)`, `x̂ = (q − zp)·scale`.
  - Vd [−1, 3]: scale = 0.01569, zp = 64; x = 1.5 → q = 160, x̂ = 1.5059.
- **Symmetric (int8 [−127,127]):** `scale = max|x|/127`, zp = 0. Đơn giản, nhanh; lãng phí range nếu phân phối lệch (vd sau ReLU).
- **Granularity:** per-tensor < per-channel < group-wise (vd 128 phần tử/group) về độ chính xác; tốn thêm metadata. Một outlier làm scale per-tensor phình to → giá trị nhỏ bị làm tròn về 0.
- **PTQ:** lượng tử model đã train, ít/không cần calibration data, nhanh (GPTQ, AWQ, SmoothQuant đều là PTQ). **QAT:** fake-quant + straight-through estimator khi train → tốt hơn ở bit thấp, tốn compute.
- **Weight-only (W4A16, W8A16)** lợi cho decode memory-bound; **W8A8 / FP8** lợi cả compute (Tensor Core INT8/FP8) → tốt cho prefill & batch lớn.
- Activation LLM có **outlier features** (xuất hiện rõ từ ~6.7B params), tập trung ở một số kênh → per-tensor activation INT8 hỏng.

### Các phương pháp lượng tử hóa LLM

| Phương pháp | Ý tưởng | Loại |
|---|---|---|
| LLM.int8() | vector-wise INT8 + tách cột outlier (>6) tính FP16 (mixed-precision decomposition) | W8A8 hỗn hợp, thường chậm hơn FP16 |
| GPTQ | lượng tử từng cột, dùng xấp xỉ Hessian (từ calibration) để cập nhật weight còn lại bù sai số (OBQ) | weight-only 3/4-bit |
| AWQ | tìm ~1% kênh quan trọng theo độ lớn ACTIVATION, bảo vệ bằng per-channel scaling; không backprop | weight-only 4-bit |
| SmoothQuant | chia activation, nhân weights theo kênh: `s_j = max|X_j|^α / max|W_j|^(1−α)` (α≈0.5) → chuyển độ khó sang weights | W8A8 |
| FP8 | E4M3 (max 448, chính xác hơn) cho W/A; E5M2 (max 57344, range rộng) cho gradient | Hopper/Ada+ |
| NF4 (QLoRA) | 16 mức theo quantile N(0,1) + double quantization cho scale | lưu trữ 4-bit, tính BF16 |
| GGUF k-quants | super-block 256 weight, block con có scale/min lượng tử 6-bit; Q4_K_M ≈ 4.8–4.9 bpw | llama.cpp / Ollama, CPU/Apple/GPU |

Hay hỏi: *Vì sao W4A16 không nhanh ở batch lớn?* → compute-bound, vẫn tính FP16, thêm chi phí dequant.

### Pruning, distillation, low-rank

**Pruning**
- Unstructured (từng weight): giữ chất lượng tốt nhưng ít tăng tốc GPU (vẫn dense kernel; sparse kernel cần độ thưa rất cao).
- Structured (head/neuron/layer): tăng tốc thật với dense kernel, mất chất lượng nhiều hơn → cần fine-tune/distill.
- **2:4 semi-structured:** mỗi 4 weight liên tiếp có 2 số 0 (50%), Sparse Tensor Core (Ampere+) tối đa ~2× math.
- **SparseGPT:** one-shot, cập nhật weight còn lại theo xấp xỉ Hessian. **Wanda:** điểm = |W|·‖X‖₂, so sánh trong từng output, không cập nhật weight. Magnitude thuần kém ở LLM.

**Knowledge distillation (Hinton)**
- `L = α·CE(y, p_s) + (1 − α)·T²·KL(p_t^T ‖ p_s^T)`, `p^T = softmax(z/T)`.
- T lớn làm lộ *dark knowledge* (quan hệ giữa các lớp sai). Nhân T² vì gradient soft term ∝ 1/T².
- Biến thể LLM: sequence-level KD (train trên output teacher), on-policy/GKD, reverse KL.

**Low-rank / weight sharing**
- W (d×d) ≈ A(d×r)·B(r×d): tham số 2dr, có lợi khi r < d/2. d = 4096, r = 256 → 8×. Là xấp xỉ, cần fine-tune hồi phục. LoRA áp ý tưởng này cho phần cập nhật.
- Weight sharing (ALBERT dùng chung layer) giảm params nhưng KHÔNG giảm FLOPs.

### Serving engines

| Engine | Điểm mạnh |
|---|---|
| vLLM | PagedAttention, continuous batching, APC, chunked prefill, speculative, nhiều định dạng quant, API OpenAI-compatible |
| SGLang | RadixAttention (radix tree KV, LRU) — mạnh khi nhiều prefix chung: agent, multi-turn, few-shot, structured output |
| TensorRT-LLM | compile engine tối ưu cho NVIDIA, FP8/INT4, in-flight batching; thường chạy sau Triton |
| TGI | server của Hugging Face, continuous batching, streaming |
| Triton Inference Server | server đa backend (TensorRT, ONNX, PyTorch, Python, vLLM, TRT-LLM), dynamic batching, ensemble |
| ONNX Runtime | inference engine đa nền tảng, graph optimization, execution providers (CUDA, TensorRT, CPU, DirectML) |
| Ollama / llama.cpp | chạy local GGUF, CPU/Apple Silicon, quant k-quants |

**Streaming SSE:** `text/event-stream`, một chiều server → client; giảm *perceived latency* (người dùng thấy token đầu sau TTFT) nhưng không giảm E2E latency hay chi phí. Nhớ tắt buffering ở reverse proxy (nginx `X-Accel-Buffering: no`).

### Production: caching, routing, rate limit, chi phí, token optimization

- **Chi phí/request** = uncached_in × giá_in + cached_in × giá_cached + out × giá_out. Output thường đắt 3–5× input.
- **Prompt caching (provider):** prefix trùng chính xác; giảm TTFT + giá phần cached; không trả lại response cũ.
- **Exact cache:** hash(prompt + params) → an toàn, hit thấp. **Semantic cache:** embedding + ngưỡng cosine → hit cao, rủi ro false hit (khác thực thể/số/phủ định), cần phân vùng theo tenant/quyền, TTL/invalidation, tránh câu hỏi thời gian thực.
- **Rate limit:** RPM và TPM, cái nào chạm trước thì giới hạn. Azure OpenAI ước lượng TPM = prompt + `max_tokens` (+ best_of) tại lúc nhận request → đặt `max_tokens` sát thực tế. RPM được đánh giá theo cửa sổ 1–10 s → tránh burst.
- **Retry:** exponential backoff + jitter, tôn trọng `retry-after(-ms)`; request bị 429 vẫn tính vào limit. Circuit breaker cho provider lỗi.
- **Fallback đa provider:** khác tokenizer, context window, tool-calling schema, structured output, safety filter → cần lớp adapter + eval; fallback an toàn trước token đầu tiên khi stream.
- **Token optimization:** cắt context RAG (rerank, top-k nhỏ, nén), system prompt gọn và tĩnh (để cache), output ngắn bằng structured output, `max_tokens` hợp lý, tránh gửi lại lịch sử dài (tóm tắt), chọn model nhỏ cho task dễ (routing theo độ khó).


---

## 🔎 RAG, Retrieval & Vector DB

### Kiến trúc RAG: naive → advanced → modular → agentic

| Thế hệ | Đặc điểm | Thành phần điển hình |
|---|---|---|
| Naive | index → retrieve top-k → stuff vào prompt → generate | chunk cố định, 1 dense retriever |
| Advanced | thêm pre-retrieval & post-retrieval | query rewrite/HyDE, hybrid search, rerank, compression |
| Modular | các khối thay thế được, có routing/fusion | router, nhiều retriever, RRF, memory, module eval |
| Agentic | LLM/agent **quyết định động**: có retrieve không, dùng tool nào, lặp lại không | router, Self-RAG, CRAG, iterative/multi-hop, web search, tool calling |

- Pipeline offline: **load → parse → clean → chunk → enrich metadata → embed → index**. Online: **query transform → retrieve → fuse → rerank → context build → generate → cite**.
- Câu hỏi phỏng vấn hay gặp: *Khi nào KHÔNG cần RAG?* (kiến thức ổn định, nhỏ, vừa context → long-context/prompt caching hoặc fine-tune cho style/format); *RAG vs fine-tune?* (RAG cho kiến thức thay đổi/cần trích dẫn/ACL; fine-tune cho hành vi, format, ngôn ngữ domain).
- Failure modes kinh điển: thiếu nội dung trong corpus, retrieve trượt (top-k không chứa đáp án), có nhưng bị cắt khỏi context (token budget), có trong context nhưng LLM không dùng (lost in the middle/nhiễu), format sai, câu trả lời không đầy đủ, hallucination ngoài context.

### Ingestion & Chunking

**Ingestion**: phát hiện trang scan → OCR; giữ bảng dạng Markdown/HTML (không cắt đôi bảng), bỏ header/footer/số trang lặp, chuẩn hóa unicode/khoảng trắng, dedup (exact: hash; near-dup: **MinHash+LSH / SimHash**), metadata (source, page, heading path, ngày hiệu lực, ACL, tenant).

| Chiến lược | Ý tưởng | Lưu ý |
|---|---|---|
| Fixed-size + overlap | cắt theo token, cửa sổ trượt | đơn giản, có thể cắt giữa câu |
| Recursive (LangChain) | thử separator `["\n\n", "\n", " ", ""]` | `chunk_size` mặc định đo bằng **ký tự** (`len`), dùng `from_tiktoken_encoder` để đo token |
| Semantic | cắt ở điểm embedding câu liên tiếp đổi mạnh | tốn embed, ngưỡng khó chỉnh |
| Structure-aware | theo heading/section/Markdown/HTML | giữ heading path làm metadata |
| Parent-child / small-to-big | search chunk con, trả chunk cha | parent ở docstore, dedup theo parent_id |
| Sentence window | embed câu, trả câu ± w câu lân cận | LlamaIndex `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` |
| Contextual retrieval | LLM viết 50–100 token ngữ cảnh prepend vào chunk trước khi embed **và** BM25 | Anthropic: giảm 35% (emb) / 49% (+BM25) / 67% (+rerank) lỗi retrieval top-20 |

- Số chunk với cửa sổ trượt: stride `s = c − o`; số chunk = `ceil((L − c)/s) + 1` (với L > c). Cẩn thận off-by-one: `ceil(L/s)` có thể dư 1 chunk nằm trọn trong chunk trước.
- Chunk to: recall tài liệu ↑ nhưng embedding bị pha loãng, tốn token, có thể **vượt max_seq_length của model → bị truncate âm thầm**. Chunk nhỏ: chính xác nhưng thiếu ngữ cảnh → dùng small-to-big.

### Embeddings

- **Bi-encoder**: encode query và doc độc lập → doc tính trước được → ANN. Cross-encoder thì không.
- Độ đo: nếu ‖v‖ = 1 thì `dot = cosine` và `‖a − b‖² = 2 − 2·cos` ⇒ **cả 3 cho cùng thứ hạng**. Vector chưa chuẩn hóa: dot thiên vị norm lớn.
- Mặc định cần nhớ: Chroma `hnsw:space = l2`; Qdrant `Cosine` tự chuẩn hóa vector khi upload; FAISS `IndexFlatIP` chỉ là cosine khi bạn tự `normalize_L2`.
- **Asymmetric prefix**: E5 `query: ` / `passage: `; BGE thêm instruction cho query; Nomic `search_query: ` / `search_document: `. Quên prefix → recall tụt.
- **Matryoshka (MRL)**: thông tin dồn vào các chiều ĐẦU; cắt lấy d chiều đầu rồi **chuẩn hóa lại L2** (OpenAI `text-embedding-3-*` có tham số `dimensions`).
- Bộ nhớ: `N × d × bytes` (float32 = 4, float16 = 2, int8 = 1, binary = 1/8). Ví dụ 10M × 1536 × 4 = 61,44 GB (≈ 57,2 GiB).
- **Fine-tune**: MultipleNegativesRankingLoss/InfoNCE (in-batch negatives → batch lớn tốt), hard negatives từ BM25/model cũ nhưng coi chừng **false negatives** (lọc bằng cross-encoder, bỏ vài rank đầu).
- **MTEB**: xem tab Retrieval (nDCG@10; bản tiếng Anh gốc dựa trên các dataset BEIR), đúng ngôn ngữ; luôn đánh giá lại trên dữ liệu domain.
- Đổi model ⇒ **re-index toàn bộ** (không gian vector khác nhau dù cùng dimension). Cache key = `model_id + version + config (prefix, dims, normalize) + hash(text)`.

### Sparse retrieval: TF-IDF, BM25, SPLADE

**BM25** (term t, doc D):

`score = IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))`

- Lucene IDF: `IDF = ln(1 + (N − n + 0.5)/(n + 0.5))` (luôn dương; IDF Robertson cổ điển có thể âm khi n > N/2).
- Mặc định Lucene/Elasticsearch: **k1 = 1.2, b = 0.75**. Lucene 8+ bỏ hệ số (k1+1) ở tử (không đổi thứ hạng).
- **k1**: độ bão hòa tf (k1 = 0 → chỉ còn có/không có term). **b**: chuẩn hóa độ dài (b = 0 tắt, b = 1 đầy đủ). Doc dài hơn avgdl bị phạt.
- tf tăng → điểm tăng nhưng **bão hòa** (tiệm cận IDF·(k1+1)), khác TF-IDF tuyến tính.
- Ví dụ: IDF = 2, tf = 3, dl = 2·avgdl → 2·6,6/5,1 ≈ 2,588; nếu dl = avgdl → 3,143.
- **SPLADE**: learned sparse — dùng MLM head của BERT, trọng số `log(1 + ReLU(logit))` + max pooling trên vocabulary, có **term expansion**, vẫn chạy trên inverted index (Qdrant sparse vectors, ES `sparse_vector`/ELSER tương tự ý tưởng).
- Sparse mạnh với: mã sản phẩm, số điều luật, tên riêng, từ hiếm, OOV với dense model.

### Hybrid search & Fusion

| Cách | Công thức | Ưu/nhược |
|---|---|---|
| RRF | `score(d) = Σ_r 1/(k + rank_r(d))`, rank từ 1 | không cần thang điểm, robust; mất thông tin độ lớn điểm |
| Min-max weighted | `α·norm(dense) + (1−α)·norm(sparse)` | chỉnh được α; nhạy outlier & kích thước tập ứng viên |
| DBSF (Qdrant) | chuẩn hóa theo μ ± 3σ rồi cộng | giữ độ lớn điểm, cần phân phối ổn định |

- Hằng số k: paper gốc (Cormack 2009) & **Elasticsearch `rank_constant` = 60**; `rank_window_size` mặc định = `size`. **Qdrant RRF mặc định k = 2** (tham số hóa từ v1.16, weighted RRF từ v1.17). k nhỏ → thưởng mạnh vị trí top của từng list.
- Ví dụ k = 60: rank 1 & 2 → 1/61 + 1/62 ≈ 0,03252.
- Không bao giờ cộng thẳng BM25 thô (0…∞) với cosine (−1…1).
- Qdrant: Query API với `prefetch` (dense + sparse) + `query: {fusion: "rrf" | "dbsf"}`; prefetch lồng nhau cho multi-stage (vd BQ → rescore → ColBERT).
- Elasticsearch 8.14+: `retriever: { rrf: { retrievers: [standard (BM25), knn] } }`; hoặc `linear` retriever với normalizer.

### Reranking

| Loại | Cách tính | Chi phí | Dùng ở đâu |
|---|---|---|---|
| Bi-encoder | cos(E(q), E(d)) | doc tính trước, ANN | first-stage, hàng triệu doc |
| ColBERT (late interaction) | Σ_q max_d (q_i·d_j) — MaxSim | lưu vector cho MỌI token (ColBERTv2 nén residual) | first-stage/rerank, chính xác hơn bi-encoder |
| Cross-encoder | 1 forward cho cặp (q, d), full attention | N forward/query | rerank top 50–200 |
| LLM rerank | pointwise / pairwise O(N²) / listwise | đắt, chậm nhất | top 20–50, query khó |

- **RankGPT listwise**: sliding window (vd window 20, step 10) đi từ cuối lên đầu để doc tốt nổi lên.
- **Position bias**: LLM thiên vị thứ tự đầu vào → xáo trộn/nhiều lượt/đối xứng hóa khi pairwise.
- Pointwise khó hiệu chuẩn điểm giữa các query; ngưỡng cắt (score threshold) phải tune trên dev set.
- Latency budget: cross-encoder base trên GPU vài chục ms cho 50 cặp; LLM rerank có thể vài giây.

### Query transformation

- **Rewrite cho hội thoại**: condense lịch sử + câu cuối → câu hỏi độc lập (standalone) trước khi retrieve.
- **Multi-query**: sinh n paraphrase → retrieve song song → hợp nhất (RRF/union + dedup). Tăng recall, tăng chi phí/latency.
- **Decomposition**: tách câu multi-hop/so sánh thành sub-question, retrieve từng cái (có thể tuần tự khi câu sau phụ thuộc kết quả câu trước).
- **Step-back** (Zheng et al., 2023): hỏi câu tổng quát hơn (nguyên lý/khái niệm) để lấy nền tảng.
- **HyDE** (Gao et al., 2022): LLM viết tài liệu giả định → embed tài liệu đó → tìm doc thật (doc-doc similarity). Rủi ro: domain hiếm/thuật ngữ nội bộ LLM không biết → doc giả lệch hướng; tốn 1 lần gọi LLM.
- **Self-query / metadata extraction**: LLM trích filter có cấu trúc (năm, phòng ban) từ câu hỏi → filter vector DB.
- Routing: phân loại query → vector / SQL / graph / web / trả lời trực tiếp.

### Vector index & quantization

| Index | Ý tưởng | Tham số | Ghi chú |
|---|---|---|---|
| Flat | brute force, exact | — | ground truth để đo recall ANN |
| IVF | k-means nlist cụm, chỉ quét nprobe cụm | nlist (~4√N–16√N), nprobe | cần train; số phép tính ≈ nlist + N·nprobe/nlist |
| PQ | chia d thành m subvector, mỗi cái nbits | m, nbits | mã = m·nbits/8 byte/vector; ADC: query không nén, tra bảng |
| IVF-PQ | IVF + PQ trên residual | nlist, nprobe, m | + 8 byte id/vector trong FAISS; refine bằng vector gốc |
| HNSW | graph nhiều tầng | M, efConstruction, efSearch | Qdrant mặc định m = 16, ef_construct = 100; ef ≥ k |
| ScaNN | anisotropic quantization + reorder | — | Google |

- HNSW: ↑efSearch → recall ↑, latency ↑ (đổi lúc query); ↑M → RAM ↑, recall ↑ (rebuild); ↑efConstruction → build chậm, graph tốt hơn.
- Quantization: **SQ int8 = 4×**, **binary = 32×** so với float32; dùng **oversampling + rescore** bằng vector gốc.
- PQ ví dụ: 100M × (m = 48, nbits = 4) → 24 byte/vector → 2,4 GB (float32 768-d là 307,2 GB).
- **Filtering**: post-filter → thiếu kết quả khi filter chặt; pre-filter ngây thơ trên HNSW → graph đứt, recall giảm. Qdrant: payload index (tạo TRƯỚC khi ingest) + filterable HNSW + query planner (cardinality thấp → dùng payload index + exact). Elasticsearch: `filter` trong `knn` là pre-filter, `post_filter` là post-filter.

### So sánh Vector DB & GraphRAG

| | FAISS | Qdrant | Chroma | Elasticsearch |
|---|---|---|---|---|
| Bản chất | thư viện in-process | vector DB (Rust) server | DB nhẹ, dev/prototype | search engine (Lucene) |
| Filter metadata | hạn chế (IDSelector) | payload index mạnh | where filter | bool query đầy đủ |
| Hybrid | tự làm | sparse + prefetch + RRF/DBSF | hạn chế | BM25 + kNN + RRF/linear retriever |
| Điểm mạnh | tốc độ, GPU, nhiều loại index | filter, quantization, multitenancy | dễ dùng | full-text, aggregations, ops sẵn |

**GraphRAG (Microsoft)**: LLM trích entity/relation/claim → graph → **Leiden** phân cộng đồng phân cấp → LLM viết **community summary**.
- **Global search**: map-reduce trên community summaries → câu hỏi toàn cục ("chủ đề chính của corpus").
- **Local search**: từ entity liên quan → hàng xóm, relationship, text unit.
- Indexing rất tốn LLM call; graph tốt khi cần multi-hop, quan hệ, tổng hợp toàn cục; vector tốt cho tra cứu fact cục bộ.

**Neo4j Cypher**: `MATCH (p:Person)-[:WORKS_AT]->(c:Company)-[:SUPPLIES*1..3]->(:Company {name:'Acme'}) RETURN DISTINCT p.name`. Chú ý **chiều mũi tên**, `*1..3` cho đường đi độ dài biến thiên, `DISTINCT` vì nhiều path. Neo4j cũng có vector index (`CREATE VECTOR INDEX ... OPTIONS {indexConfig: {...}}` với dimensions + similarity function, truy vấn qua `db.index.vector.queryNodes`) → kết hợp vector search để tìm entity khởi đầu rồi duyệt graph (GraphRAG kiểu local).

### Context optimization

- **Lost in the Middle** (Liu et al., 2023): hiệu năng theo vị trí thông tin có dạng chữ U — tốt ở đầu/cuối, kém ở giữa. → ít chunk hơn, đặt chunk tốt nhất ở đầu và cuối (LangChain `LongContextReorder`).
- **Token budget**: `window − max_output − system − history − question` rồi chia cho (chunk + header/metadata). Luôn **trừ phần output**.
- **Compression**: lọc câu liên quan (LLMChainExtractor/EmbeddingsFilter), LongLLMLingua, tóm tắt; cẩn thận làm mất số liệu cần trích dẫn.
- **Dedup context**: trùng parent, near-dup giữa các nguồn, overlap giữa chunk liền kề → gộp/merge chunk liền kề.
- **Citation**: gắn id chunk (`[doc3#p12]`) trong prompt, yêu cầu trích dẫn; kiểm tra citation tồn tại thật (post-validation).
- Prompt caching: đặt phần tĩnh (system, few-shot) ở đầu, phần động (chunks, question) ở cuối.

### Evaluation: retrieval & RAG

| Metric | Công thức | Ghi chú |
|---|---|---|
| Precision@k | #relevant trong top-k / k | |
| Recall@k | #relevant trong top-k / tổng #relevant | cần biết tổng relevant |
| Hit rate@k | 1 nếu có ≥ 1 relevant trong top-k | trung bình theo query |
| MRR | trung bình 1/rank đầu tiên đúng (không có → 0) | chia cho TẤT CẢ query |
| nDCG@k | DCG = Σ rel_i / log2(i+1); chia IDCG | graded relevance; biến thể (2^rel − 1) |

- Ví dụ MRR: rank 1, 3, –, 2, 5 → (1 + 1/3 + 0 + 1/2 + 1/5)/5 ≈ 0,4067.
- **RAGAS**: Faithfulness = #claim trong answer được context hỗ trợ / #claim (không cần reference); Answer Relevancy = mean cos(câu hỏi sinh ngược từ answer, câu hỏi gốc); Context Precision@K = Σ(Precision@k·v_k) / #relevant trong top K (thưởng khi chunk relevant nằm ở đầu danh sách); Context Recall = #claim của reference được context hỗ trợ / #claim reference (CẦN reference).
- Chẩn đoán: faithfulness cao + context recall thấp → lỗi retrieval; context recall cao + faithfulness thấp → lỗi generation/hallucination; context precision thấp → nhiễu, cần rerank/giảm k.
- Bộ test: golden set (query, doc relevant, answer), synthetic test từ LLM nhưng phải review; theo dõi theo slice (loại query, ngôn ngữ, tenant).

### Agentic RAG, web ingestion, multi-tenant & cache

- **Self-RAG**: LM được train sinh reflection token `Retrieve`, `ISREL`, `ISSUP`, `ISUSE` → tự quyết retrieve và tự phê bình.
- **CRAG**: retrieval evaluator nhẹ (T5) → Correct (refine: decompose-then-recompose), Incorrect (bỏ, web search), Ambiguous (kết hợp).
- **Adaptive/Router**: phân loại độ khó/loại query → no-retrieval / single-shot / iterative. LangGraph: conditional edges, state, max iterations, grade documents → rewrite query → retrieve lại.
- Luôn có điều kiện dừng & budget (số vòng, token, thời gian); log trace để debug.
- **Web ingestion**: robots.txt + rate limit, sitemap `lastmod`, render JS (Playwright) cho SPA, boilerplate removal (trafilatura/readability), canonical URL + bỏ `utm_*`, ETag/Last-Modified + content hash cho incremental, lưu `fetched_at` để ưu tiên độ mới.
- **Multi-tenant & ACL**: filter tenant/ACL ở tầng retrieval từ identity phía server (không tin prompt); Qdrant `is_tenant` + `payload_m`; cập nhật payload ACL ngay khi quyền đổi (không cần re-embed); cache key phải có tenant/ACL.
- **Cache**: embedding cache (key gồm model+version+config+hash), result cache, semantic cache (ngưỡng similarity quá thấp → trả nhầm câu trả lời cho câu hỏi khác nghĩa, vd khác mỗi con số/phủ định).


---

## 🤖 AI Agents, Tool Calling & MCP

### Agent vs Workflow & 5 pattern (Anthropic)

**Workflow**: LLM + tool điều phối qua *đường code định sẵn*. **Agent**: LLM *tự điều hướng* quá trình và việc dùng tool. Nguyên tắc: bắt đầu đơn giản, chỉ thêm độ phức tạp khi đo được lợi ích; giảm bớt tầng abstraction của framework khi lên production.

| Pattern | Khi nào dùng | Ví dụ |
|---|---|---|
| Prompt chaining | Subtask cố định, tuần tự; có thể chèn *gate* bằng code | Dàn ý → kiểm tra → viết |
| Routing | Input chia thành nhóm rõ ràng | Câu dễ → model nhỏ, khó → model lớn |
| Parallelization | *Sectioning* (subtask độc lập) / *Voting* (nhiều phiếu) | Guardrail chạy song song; nhiều reviewer bảo mật |
| Orchestrator-workers | Subtask **không biết trước**, LLM trung tâm quyết định động | Sửa code nhiều file |
| Evaluator-optimizer | Có tiêu chí đánh giá rõ, lặp để cải thiện | Dịch văn học |

**ACI**: đầu tư thiết kế tool như thiết kế UI. Mô tả rõ, có ví dụ, có edge case; dùng *poka-yoke* (vd bắt buộc đường dẫn tuyệt đối).

### ReAct & các biến thể lập kế hoạch

| Kỹ thuật | Cơ chế | Điểm mạnh / yếu |
|---|---|---|
| CoT | Suy luận từng bước, không có tool | Dễ bịa fact |
| Self-consistency | N chuỗi CoT độc lập → đa số phiếu | Chi phí ×N; chỉ hiệu quả khi p > 0,5 |
| **ReAct** | Thought → Action → Observation lặp | Neo suy luận vào dữ liệu thật; context phình theo số vòng |
| Plan-and-Execute | Planner lập kế hoạch, executor làm từng bước, có replan | Ít gọi model lớn; kế hoạch cứng |
| ReWOO | Planner viết plan với biến #E1.. trước mọi Observation → Worker → Solver | Tiết kiệm token; khó thích ứng |
| Reflexion | Tự phản tỉnh bằng lời, lưu vào episodic memory, thử lại | Không cập nhật trọng số; cần tín hiệu đánh giá |
| Tree of Thoughts | Nhiều thought mỗi bước + tự đánh giá + BFS/DFS có backtrack | Mạnh với bài cần tìm kiếm; rất tốn |

**Failure modes của ReAct**: lặp vô hạn (cần max_iterations và phát hiện lời gọi trùng), hallucinated Observation (dùng stop sequence hoặc native tool calling), hallucinated tool/args (validate rồi trả lỗi có hướng dẫn), dừng sớm hoặc trả lời khi chưa đủ bằng chứng (grounding check).

### Function / tool calling

- Model **không thực thi** tool. Model sinh tên hàm + args JSON, ứng dụng thực thi và trả kết quả kèm `tool_call_id`/`call_id`.
- Chat Completions: assistant message có `tool_calls` → đủ message role `tool` cho **mỗi** id (thiếu một id là lỗi 400). Tool lỗi vẫn phải trả message mô tả lỗi.
- `tool_choice`: `auto` (mặc định) · `required` (≥1 tool, tool nào tuỳ model) · `none` · forced function (đúng một hàm) · `allowed_tools` (giới hạn tập tool, giữ danh sách để cache). Gemini: AUTO / ANY (+`allowed_function_names`) / NONE / VALIDATED.
- `parallel_tool_calls=false` → mỗi lượt tối đa một tool call. Các lời gọi trong cùng lượt **không thấy kết quả của nhau**.
- `strict: true`: `additionalProperties:false` ở mọi object, mọi field nằm trong `required`, field tuỳ chọn dùng union với `null`. Chỉ bảo đảm *cấu trúc*, không bảo đảm *ngữ nghĩa*.
- Nhiều tool: OpenAI khuyên dưới ~20 function ở đầu lượt → dùng tool retrieval/dynamic discovery, routing hai tầng, tool search/`defer_loading`. Lưu ý trade-off với prompt cache.
- Tool tốt: tên rõ có namespace, mô tả khi nào dùng/không dùng, tham số khó dùng sai, lỗi có hướng dẫn, output gọn và dễ đọc thay vì JSON khổng lồ.

### Memory cho agent

| Loại | Nội dung | Lưu ở đâu |
|---|---|---|
| Short-term | Lịch sử/state của một thread | State + checkpointer (theo `thread_id`) |
| Semantic | Fact về user/thế giới | Store/profile có namespace, upsert |
| Episodic | Trải nghiệm cụ thể (trajectory) | Vector store, dùng làm few-shot động |
| Procedural | Quy tắc, cách làm | System prompt, có thể tự cập nhật |

Kỹ thuật: cửa sổ K lượt gần nhất + rolling summary; trích fact quan trọng thành state có cấu trúc; khi trim **không tách cặp tool call ↔ tool result**. Vector memory dễ trả fact cũ hoặc mâu thuẫn, nên cần timestamp và cơ chế cập nhật.

### Multi-agent

- **Supervisor**: điều phối trung tâm. **Hierarchical**: supervisor của các supervisor. **Swarm/handoff**: agent chuyển quyền trực tiếp cho nhau, agent nhận tiếp tục nói chuyện với user.
- Số liệu của Anthropic (multi-agent research): agent ≈ **4×** token so với chat, multi-agent ≈ **15×**. Multi-agent hơn single-agent 90,2% trên bộ eval nội bộ. Lượng token giải thích ~80% phương sai hiệu năng (BrowseComp). Song song hoá cắt tới ~90% thời gian nghiên cứu.
- **Nên** dùng: tác vụ giá trị cao, theo chiều rộng, song song hoá được, vượt một context window, nhiều tool phức tạp.
- **Không nên**: các agent cần chung toàn bộ context, phụ thuộc chặt, phần lớn tác vụ coding, ngân sách token hẹp.
- Rủi ro: chi phí phối hợp, mất ngữ cảnh khi giao việc, lỗi lan truyền. Cần mô tả nhiệm vụ rõ cho sub-agent (mục tiêu, định dạng output, ranh giới).

### MCP: kiến trúc, lifecycle, transport

- **Host** (ứng dụng) → nhiều **client** (mỗi client 1:1 với một **server**). Giao thức JSON-RPC 2.0, UTF-8.
- Server cung cấp **Prompts** (user-controlled), **Resources** (application-controlled), **Tools** (model-controlled). Client cung cấp **Sampling**, **Roots**, **Elicitation**.
- **Lifecycle (≤ 2025-11-25)**: `initialize` (protocolVersion, capabilities, clientInfo) → result (version, capabilities, serverInfo, instructions) → notification `notifications/initialized`. Server không hỗ trợ version client yêu cầu thì trả version khác mà nó hỗ trợ; client không hỗ trợ thì ngắt kết nối. Chỉ dùng capability đã đàm phán.
- **stdio**: message phân tách bằng newline, không có newline bên trong; stdout chỉ chứa message MCP, log ra **stderr**. Shutdown: đóng stdin → SIGTERM → SIGKILL.
- **Streamable HTTP** (thay HTTP+SSE của 2024-11-05): một endpoint POST+GET; trả JSON hoặc SSE; `Mcp-Session-Id` (404 thì initialize lại); `MCP-Protocol-Version` ở mọi request; resumability bằng `Last-Event-ID`; validate `Origin`, bind 127.0.0.1.
- **ping** (≤ 2025-11-25): request do bất kỳ bên nào gửi, bên nhận MUST trả `{}`; timeout thì coi là chết và reconnect. Nên đặt timeout cho mọi request và gửi cancellation khi quá hạn.
- **Revision hiện hành 2026-07-28 (stateless)**: bỏ `initialize`/`initialized`, bỏ session `Mcp-Session-Id`, bỏ `ping`; version và capabilities đi trong `_meta` của từng request; `server/discover` bắt buộc; `subscriptions/listen` thay GET stream; Multi Round-Trip Requests thay các request do server khởi tạo; Roots/Sampling/Logging bị deprecated.

### MCP: tools chi tiết

- `tools/list` (có phân trang `cursor`/`nextCursor`) · `tools/call` · `notifications/tools/list_changed` (khi capability `tools.listChanged` = true) → client gọi lại `tools/list`. Đây là nền tảng của **dynamic tool discovery**.
- Tool: `name`, `title`, `description`, `inputSchema`, `outputSchema` (tuỳ chọn), `annotations`.
- Result: `content` (text/image/audio/resource_link/resource) · `structuredContent` (khi có outputSchema: MUST khớp schema, SHOULD kèm JSON dạng text) · `isError`.
- Lỗi: **protocol error** (JSON-RPC: tool không tồn tại, request sai) và **tool execution error** (`isError: true`: lỗi API, nghiệp vụ, input sai) để LLM tự sửa.
- Annotations (đều là *hint*, không tin cậy nếu server không tin cậy): `readOnlyHint` = false · `destructiveHint` = **true** · `idempotentHint` = false · `openWorldHint` = true (giá trị mặc định).
- Bảo mật: server validate input, kiểm soát truy cập, rate limit, sanitize output. Client xác nhận thao tác nhạy cảm, hiển thị input trước khi gọi, validate kết quả, đặt timeout, ghi log audit.

### MCP: bảo mật & authorization

- Authorization **tuỳ chọn**; HTTP dùng OAuth 2.1; stdio thì lấy credential từ env.
- Server = resource server: 401 + `WWW-Authenticate` → Protected Resource Metadata (RFC 9728) → AS metadata (RFC 8414) → (dynamic client registration RFC 7591 / Client ID Metadata Documents) → **PKCE** + tham số `resource` (RFC 8707) → `Authorization: Bearer` (không đặt trong query string).
- Server MUST validate **audience**. **Token passthrough bị cấm**: gọi API upstream phải dùng token riêng.
- **Confused deputy**: proxy dùng static client ID + dynamic registration + consent cookie → MUST có consent theo từng client, kiểm tra redirect_uri khớp chính xác, `state` dùng một lần.
- **Tool poisoning / rug pull**: chỉ dẫn độc hại trong mô tả tool, hoặc mô tả đổi sau khi đã được tin → pin/review, diff khi `list_changed`.
- **Indirect prompt injection** qua output tool: cách ly dữ liệu, least privilege, human-in-the-loop, egress allow-list, validate args bằng code.
- Khác: session hijacking (session ID không dùng để xác thực; gắn `user_id:session_id`), SSRF khi discovery OAuth, server local chạy lệnh tuỳ ý (cần consent và sandbox), DNS rebinding.

### LangGraph · LangChain · LlamaIndex · FastMCP

- **LangGraph**: `StateGraph(State)` → `add_node` / `add_edge` / `add_conditional_edges` → `compile(checkpointer=...)`. Key không có reducer chỉ nhận một giá trị mỗi superstep (ghi song song gây `InvalidUpdateError`). Dùng `Annotated[list, operator.add]` hoặc `add_messages`.
- **Checkpointer** lưu state mỗi superstep theo `thread_id` (cho phép HITL, time travel, chạy tiếp sau lỗi). InMemorySaver chỉ dành cho dev. **Store** dùng cho long-term memory xuyên thread.
- **interrupt()** + `Command(resume=...)`: cần checkpointer; khi resume, **node chạy lại từ đầu** → side effect trước interrupt phải idempotent.
- **Send**: conditional edge trả list `Send(node, state)` → map-reduce động, fan-in bằng reducer.
- **recursion_limit** đếm superstep, khi vượt thì ném `GraphRecursionError`. Default đổi theo phiên bản (≤ 1.0.x: 25; từ 1.1: 10.000, bản 1.2.x: 10.007; override bằng env `LANGGRAPH_DEFAULT_RECURSION_LIMIT`) → đặt tường minh, kèm bộ đếm bước trong state để dừng mềm.
- **LCEL**: `prompt | model | parser` = RunnableSequence; `RunnableParallel`; invoke/batch/stream/async; `.with_retry()`, `.with_fallbacks()`.
- **LlamaIndex**: Document → Node → `VectorStoreIndex`; `as_query_engine()` (retriever + synthesizer, một lượt) vs `as_chat_engine()`; `QueryEngineTool` cho agent.
- **FastMCP**: `@mcp.tool` (tên hàm, docstring, type hints/`Field` → schema; trả dict/Pydantic → structured output) · `@mcp.resource("x://{id}")` (template) · `@mcp.prompt` · `ctx: Context` (inject, ẩn khỏi schema; log, progress, read_resource) · `ToolError`.

### Prompt engineering & structured output

- System prompt: vai trò, mục tiêu, ràng buộc, định dạng output, chính sách khi thiếu thông tin. Dùng **delimiter** (XML tag, ```) để tách chỉ thị với dữ liệu.
- Few-shot: ví dụ đa dạng, đúng định dạng, có cả ca biên; với agent có thể chọn few-shot động từ episodic memory.
- CoT / thinking: tăng độ chính xác cho bài nhiều bước nhưng tốn token và latency. Self-consistency = N mẫu + bỏ phiếu.
- **JSON mode** = JSON hợp lệ. **Structured Outputs / strict** = khớp schema. Cả hai không bảo đảm đúng nghiệp vụ → vẫn validate.
- **Prompt injection**: *direct* (từ user, jailbreak) và *indirect* (từ web/email/tài liệu/tool output). Không có cách chặn tuyệt đối ở tầng prompt → defense-in-depth.
- Azure OpenAI: gọi theo **deployment name** + `api-version`, có content filter riêng. Gemini: `response_schema`, function calling modes.

### Guardrails, anti-hallucination & độ tin cậy

- **Input guardrails**: phát hiện injection, PII, phạm vi chủ đề. **Output guardrails**: schema (Pydantic), quy tắc nghiệp vụ, **grounding check** (mọi claim phải có evidence trong output tool hoặc tài liệu), allow-list (bảng SQL, domain, người nhận).
- Validate → gửi lỗi cụ thể → LLM sửa, **giới hạn retry**, fail thì chuyển human-in-the-loop.
- Ràng buộc quan trọng enforce bằng **code ở tầng tool** (precondition, hạn mức), không phó mặc cho prompt.
- **Idempotency key** gắn với thao tác nghiệp vụ, giữ nguyên qua các lần retry. Timeout ≠ thất bại.
- **Timeout + retry + exponential backoff + jitter**, tôn trọng `Retry-After`, chỉ retry thao tác idempotent, có deadline tổng cho request.
- Giới hạn số bước (max_iterations / recursion_limit), phát hiện vòng lặp, ngân sách token và chi phí cho mỗi request.
- Temperature 0 **không** loại bỏ hallucination và không bảo đảm deterministic.

### Đánh giá agent & công thức tính tay

- **Chỉ số**: task success rate, trajectory eval (khớp chính xác / không theo thứ tự / theo ràng buộc), tool-call accuracy (tool, args, số lần), state-based eval (τ-bench), chi phí, latency, tỷ lệ chuyển người.
- **Golden-label regression**: pin snapshot model, chấm deterministic phần có cấu trúc, LLM-judge đã hiệu chỉnh cho phần tự do, chạy nhiều lần, so baseline có tính nhiễu; mỗi bug production thành một case mới; không sửa nhãn cho CI xanh.

| Công thức | Ý nghĩa |
|---|---|
| `p^n` | Chuỗi n bước độc lập (0,9¹⁰ ≈ 34,9%) |
| `1 − (1−p)^(r+1)` | Một bước có r lần retry |
| `pass@k = 1 − C(n−c,k)/C(n,k)` | ≥1 trong k mẫu đúng (ước lượng không chệch) |
| `pass^k = C(c,k)/C(n,k)` (≈ pᵏ) | Cả k lần đều đúng: đo độ nhất quán |
| `Σ_{i=1..N} (P₀ + (i−1)·Δ)` = `N·P₀ + Δ·N(N−1)/2` | Tổng input tokens của ReAct N vòng (context tăng Δ mỗi vòng) |
| tuần tự = Σtᵢ; song song = max tᵢ; giới hạn concurrency → lập lịch theo slot | Latency |
| backoff: chờ b·2ⁱ (có cap); worst-case = (r+1)·timeout + Σ chờ | Thời gian retry |


---

## 🧩 Agent Engineering: Skills, Subagents, Frameworks

### Context engineering: nguyên tắc & chiến lược

**Context engineering** = chọn và duy trì *tập token nhỏ nhất có tín hiệu cao* cho mỗi bước suy luận của agent (system prompt, tools, examples, lịch sử, dữ liệu ngoài).

- **Attention budget** hữu hạn; **context rot**: càng nhiều token, khả năng nhớ lại chính xác càng giảm, *trước cả* khi chạm giới hạn window.
- Ba chiến lược cho tác vụ dài:

| Kỹ thuật | Cơ chế | Hợp khi |
|---|---|---|
| Compaction | Tóm tắt lịch sử, mở context mới; giữ quyết định kiến trúc, bug chưa xử lý, việc đang dở | Hội thoại/tác vụ dài liên tục |
| Structured note-taking | Ghi `NOTES.md`/memory tool ngoài context, đọc lại khi cần | Tác vụ có mốc, tiến độ rõ |
| Sub-agent | Subagent đọc nhiều trong context riêng, trả tóm tắt ~1.000–2.000 token | Nghiên cứu, khám phá song song |

- **Just-in-time**: giữ định danh nhẹ (path, query, URL), nạp bằng tool khi cần, đổi lại chậm hơn dữ liệu tính sẵn. **Hybrid** (Claude Code): CLAUDE.md nạp sẵn, glob/grep tìm đúng lúc.
- Tinh chỉnh prompt compaction: **tối đa recall trước**, sau đó mới tăng precision.

Nguồn: anthropic.com/engineering/effective-context-engineering-for-ai-agents (kiểm tra 10/2026).

### Token tích luỹ & prompt caching

**Tổng input của vòng agent tăng theo bậc hai**: lần gọi thứ i (i = 1..n) có input = P + (i−1)·(o + r), với P là prefix, o là output, r là tool result. Tổng = n·P + (o + r)·n(n−1)/2.

**Prompt caching (Claude, docs 10/2026)**

| Mục | Giá trị |
|---|---|
| Thứ tự prefix | `tools` → `system` → `messages` |
| Đổi tools | mất cache của tools + system + messages |
| Đổi system | mất cache của system + messages |
| Cache write TTL 5 phút / 1 giờ | 1,25× / 2× giá input |
| Cache read | 0,1× (tuỳ model) |
| Số breakpoint tối đa | 4 |

- Cache khớp **chính xác từng token** của prefix, không theo ngữ nghĩa và không phụ thuộc temperature.
- Mọi thứ thay đổi (timestamp, user id, kết quả retrieval, tool động) đặt **sau** phần ổn định. Timestamp ở đầu system prompt có thể làm chi phí *tăng* (chỉ ghi, không bao giờ đọc).
- Tool RAG đổi tập tool mỗi lượt sẽ phá cache; nên giữ tool ổn định hoặc dùng cơ chế tool search/defer loading của provider.

Nguồn: platform.claude.com/docs/en/build-with-claude/prompt-caching.

### Tool result clearing / context editing

- Dạng compaction "nhẹ" an toàn nhất: xoá **tool result cũ** (đã dùng xong) và thay bằng placeholder.
- Claude API *context editing* (beta `context-management-2025-06-27`, docs 10/2026), strategy `clear_tool_uses_20250919`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `trigger` | 100.000 input token | Ngưỡng kích hoạt |
| `keep` | 3 tool use | Số cặp gần nhất giữ lại |
| `clear_at_least` | không | Xoá tối thiểu bao nhiêu token để đáng mất cache |
| `exclude_tools` | không | Tool không bao giờ bị xoá |
| `clear_tool_inputs` | false | Xoá cả args của tool call |

- Xoá diễn ra **phía server**, client vẫn giữ lịch sử đầy đủ. Mỗi lần xoá **làm mất cache** từ điểm bị sửa.
- Kết hợp **memory tool**: Claude được báo trước khi bị xoá và có thể lưu điều quan trọng.
- Có thêm `clear_thinking_20251015` cho thinking block (phải đứng trước khi dùng cùng `clear_tool_uses`).

### Agent Skills (SKILL.md)

**Skill** = thư mục chứa `SKILL.md` (YAML frontmatter + hướng dẫn) kèm file phụ, script và tài nguyên. Chuẩn mở agentskills.io từ 12/2025.

**Progressive disclosure** (docs Claude, 10/2026):

| Level | Khi nạp | Chi phí |
|---|---|---|
| 1. Metadata (`name`, `description`) | Luôn, lúc khởi động | ~100 token/skill |
| 2. Thân `SKILL.md` | Khi skill được kích hoạt | khuyến nghị < 5k token |
| 3+. File phụ, script | Khi được đọc/chạy | file đọc vào thì tính; script chỉ tính **output** |

**Frontmatter**: `name` ≤ 64 ký tự, chỉ chữ thường/số/gạch ngang, không chứa "anthropic"/"claude", không thẻ XML. `description` không rỗng, ≤ 1.024 ký tự, phải nói *làm gì* + *khi nào dùng*.

**Chọn chỗ đặt kiến thức**

| Loại | Đặt ở |
|---|---|
| Quy ước ngắn, luôn đúng | System prompt / CLAUDE.md (luôn nạp) |
| Quy trình dài, dùng thỉnh thoảng, có script tất định | Skill |
| Dữ liệu sống, hành động trên hệ thống ngoài | MCP server / tool |

Bảo mật: chỉ cài skill từ nguồn tin cậy; skill có thể chạy code và gọi tool. Claude Code: `~/.claude/skills/`, `.claude/skills/`; `disable-model-invocation: true` để chỉ chạy khi user gọi.

### Thiết kế tool cho agent (ACI)

Theo *Writing effective tools for agents* (Anthropic, 2025):

- **Chọn đúng tool**: đừng bọc 1:1 API. `search_contacts` tốt hơn `list_contacts`; gộp thao tác liên quan (`schedule_event`).
- **Namespacing**: `asana_search`, `jira_search`; tiền tố hay hậu tố ảnh hưởng khác nhau tuỳ model, cần đo.
- **Trả ngữ cảnh có ý nghĩa**: tên dễ đọc thay vì UUID; tham số `response_format` (concise ≈ 1/3 token so với detailed trong ví dụ 72 vs 206).
- **Hiệu quả token**: phân trang, filter, range, truncate với mặc định hợp lý, kèm chỉ dẫn bước tiếp theo. Claude Code giới hạn tool response 25.000 token theo mặc định (tại thời điểm bài viết).
- **Lỗi có hướng dẫn**: nói rõ sai gì, đúng phải thế nào, ví dụ, tool thay thế. Không trả stack trace hay chuỗi rỗng.
- Tham số rõ nghĩa (`user_id` thay vì `user`), mô tả như viết cho đồng nghiệp mới; tối ưu bằng **eval**.

Nguồn: anthropic.com/engineering/writing-tools-for-agents.

### Subagents, handoff & multi-agent

- **Subagent** (Claude Code): file Markdown + frontmatter (`name`, `description`, `tools`, `model`, `permissionMode`, `maxTurns`…), ở `.claude/agents/` hoặc `~/.claude/agents/`. Chạy trong **context riêng**, trả tóm tắt. Agent chính chọn subagent dựa vào `description`.
- Số liệu Anthropic (multi-agent research): agent ≈ **4×** token so với chat, multi-agent ≈ **15×**; lead Opus 4 + subagent Sonnet 4 hơn single-agent **90,2%**; 3–5 subagent song song cắt tới **90%** thời gian.
- **Không nên** tách: cần chung toàn bộ context, phụ thuộc chặt, phần lớn tác vụ coding, tác vụ đơn giản, ngân sách chặt.
- Giao việc rõ: objective, output format, tool/nguồn, ranh giới; scale số subagent theo độ phức tạp; lưu output ra filesystem để tránh *game of telephone*.

**OpenAI Agents SDK** (openai-agents 0.23.1): handoff = tool `transfer_to_<agent_name>`; agent đích **tiếp quản**, mặc định thấy toàn bộ lịch sử; `input_filter` cắt lịch sử; `on_handoff`, `input_type`. `Agent.as_tool()` khi muốn giữ quyền điều phối.

### A2A vs MCP

| | MCP | A2A |
|---|---|---|
| Hướng | *Vertical*: agent ↔ tool/dữ liệu | *Horizontal*: agent ↔ agent (opaque) |
| Khởi xướng | Anthropic | Google, nay thuộc Linux Foundation |
| Khám phá | `initialize`/list tools… | **Agent Card** tại `/.well-known/agent-card.json` |
| Đơn vị | tool call, resource, prompt | **Task** (stateful), Message/Part, Artifact |

- Agent Card: name, description, endpoint, capabilities (streaming, push notifications), security schemes, skills (`id, name, description, tags, examples`).
- Trạng thái Task (spec 1.0): submitted, working, input-required, auth-required, completed, failed, canceled, rejected.
- Binding: JSON-RPC 2.0, gRPC, HTTP+JSON/REST; streaming qua SSE/gRPC; push notification qua webhook.
- Ví dụ trong docs: quản lý xưởng ↔ thợ ↔ nhà cung cấp dùng A2A; thợ ↔ máy chẩn đoán dùng MCP.

Nguồn: a2a-protocol.org/latest/specification, /topics/a2a-and-mcp (kiểm tra 10/2026).

### Hooks, permissions, plugins (Claude Code)

**Hooks** là lệnh **tất định** do harness chạy tại lifecycle event: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `PreCompact`, `SessionStart`, `SessionEnd`… Loại hook: command, http, mcp_tool, prompt, agent.

| Exit code | Ý nghĩa |
|---|---|
| 0 | Thành công, đọc JSON từ stdout |
| 2 | Lỗi chặn: PreToolUse chặn tool, stderr đưa cho Claude |
| Khác | Lỗi không chặn |

- PreToolUse: `permissionDecision` = allow / deny / ask / defer. PostToolUse **không chặn, không hoàn tác** được (tool đã chạy); có thể sửa output Claude thấy.
- Matcher MCP: `mcp__<server>__<tool>`.
- **Permission rules**: deny → ask → allow, rule khớp đầu tiên thắng, **độ cụ thể không đổi thứ tự**. Deny chặn ở mọi mode kể cả `bypassPermissions`. Hook allow **không vượt** deny/ask rule; hook exit 2 chặn được cả lệnh có allow rule.
- **Modes**: `default` (Manual), `acceptEdits`, `plan`, `auto` (classifier), `dontAsk` (cái gì cần hỏi thì từ chối, hợp CI), `bypassPermissions` (chỉ dùng trong container/VM).
- **Plugin**: thư mục skills, agents, hooks, MCP servers với manifest `.claude-plugin/plugin.json`; cài từ marketplace; skill có namespace `/plugin:skill`.

Nguồn: code.claude.com/docs/en/hooks, /permissions, /permission-modes, /plugins (kiểm tra 10/2026).

### Sandbox cho code agent (smolagents)

- `CodeAgent` mặc định chạy `LocalPythonExecutor`: interpreter tự viết theo AST; import bị cấm trừ khi nằm trong `additional_authorized_imports` (submodule phải cho phép riêng, vd `numpy.*`); giới hạn số thao tác; chặn truy cập như `random._os`. **Không sandbox local nào an toàn tuyệt đối.**
- Remote executor: `executor_type` ∈ {`e2b`, `docker`, `modal`, `blaxel`} (smolagents 1.26.0). Chỉ code chạy trong sandbox, model vẫn gọi local; **không hỗ trợ managed agents**.
- Muốn multi-agent cách ly: chạy **toàn bộ** hệ agent trong sandbox (phải đưa API key vào).
- Best practice: giới hạn CPU/RAM, timeout, chạy user không đặc quyền, tắt mạng nếu không cần, pin version, luôn cleanup (`with` hoặc `agent.cleanup()`).
- Rủi ro: lỗi của LLM, prompt injection từ web, supply chain, agent public bị lạm dụng.

Nguồn: huggingface.co/docs/smolagents/tutorials/secure_code_execution.

### So sánh framework agent

| Framework | Điểm mạnh | API cần nhớ |
|---|---|---|
| **LangGraph** | Điều khiển tường minh, persistence | `StateGraph`, reducer, `Command(update, goto, graph=Command.PARENT, resume)`, `Send` (map-reduce), subgraph, checkpointer (`thread_id`) vs Store (namespace, xuyên thread), `recursion_limit` |
| **LlamaIndex** | Data/RAG + agent | Workflows (`@step`, Event, StartEvent/StopEvent, Context), `RouterQueryEngine` (Pydantic/LLM Single/Multi selector), `SubQuestionQueryEngine`, `AgentWorkflow` (`root_agent`, `can_handoff_to`), FunctionAgent/ReActAgent |
| **smolagents** | Gọn, code-as-action | `CodeAgent` vs `ToolCallingAgent`, `managed_agents`, `max_steps` (mặc định 20), `final_answer_checks`, `executor_type` |
| **OpenAI Agents SDK** | Ít abstraction, tích hợp sẵn | `Agent`, `Runner.run/run_sync/run_streamed`, handoff, guardrails (input: agent đầu, output: agent cuối, tool guardrail), `max_turns` (mặc định 10, `None` = tắt giới hạn; vượt → `MaxTurnsExceeded`), tracing bật mặc định |

- LangGraph: `Command` chỉ **thêm** edge động, edge tĩnh vẫn chạy; schema khác nhau thì gọi subgraph trong node function.
- LlamaIndex: câu hỏi cần nhiều nguồn → SubQuestion; mỗi câu hỏi thuộc một nguồn → Router.
- Workflow cố định thì viết code thường; chỉ dùng agent khi cần LLM quyết định luồng.

### Agentic RAG nâng cao

- **Retrieval là tool**: agent quyết định có retrieve không, retrieve gì, ở nguồn nào, bao nhiêu lần.
- **Multi-hop**: query của bước sau phụ thuộc kết quả bước trước; tăng top-k không thay được việc lặp retrieve.
- **Router / query planning**: chọn index theo description (nêu rõ phạm vi, thời gian, loại tài liệu); tách câu hỏi so sánh thành câu hỏi con.
- **Corrective / self-reflective** (CRAG, Self-RAG): grade tài liệu → viết lại query hoặc đổi nguồn → generate → kiểm groundedness; **giới hạn vòng** và có đường thoát "không đủ thông tin".
- Latency kỳ vọng với p đạt mỗi vòng, tối đa R vòng: E[số vòng] = Σ_{k=0}^{R−1} (1−p)^k.
- **Khi nào pipeline cố định tốt hơn**: phần lớn câu hỏi single-hop, SLA latency chặt, cần tái lập/audit, ngân sách thấp → pipeline cố định + router đẩy ca khó sang agent.
- Failure mode: lặp query gần trùng, trả lời từ kiến thức có sẵn không retrieve, chọn nhầm index, context phình vì kết quả dài.

### Observability & đánh giá agent

- **Trace** cho mỗi task, **span** lồng nhau: agent step, generation (token, cost), tool (args, output, lỗi), retriever, guardrail, handoff. Gắn session/user id, version prompt/model. Langfuse có observation type `agent`, `tool`, `chain`, `retriever`, `evaluator`, `embedding`, `guardrail`… (Python SDK ≥ 3.3.1).
- OpenAI Agents SDK: tracing bật mặc định; `OPENAI_AGENTS_DISABLE_TRACING=1`; `trace_include_sensitive_data` mặc định True; `add_trace_processor()`; không khả dụng với ZDR.

| Chỉ số | Công thức |
|---|---|
| Success rate | task thành công / tổng task |
| Cost per successful task | tổng chi phí / số task **thành công** |
| Tool-call success rate | 1 − call lỗi / tổng call |
| Tool precision / recall | trên tập tool duy nhất so với tham chiếu |
| pass^k (độ ổn định) | pᵏ |

- **Outcome eval** (state cuối, kiểm bằng code) + **trajectory eval** dạng ràng buộc (tool bắt buộc/cấm, thứ tự, số bước tối đa) thay vì exact match một đường duy nhất. LLM-judge có rubric, hiệu chỉnh với nhãn người.
- **Regression với golden task**: chạy nhiều lần mỗi task; với p = 0,8 và luật 2/3 thì P(fail) = 0,104, tức có fail "giả". So với baseline có tính nhiễu.
- **Online**: max steps/turns + ngân sách cost, phát hiện call trùng, dashboard step count, cost/task, p95 latency, tool error rate theo version.
- Anthropic: bắt đầu eval sớm với ~20 truy vấn thực tế; LLM-judge chấm theo rubric; tập trung vào end-state.


---

## 📊 LLM Evaluation, Observability & LLMOps

### Vì sao eval LLM khó & bản đồ phương pháp

**Khó vì:** output mở (nhiều đáp án đúng), non-deterministic (kể cả `temperature=0`), chất lượng đa chiều (đúng, đủ, an toàn, giọng văn, format), phân phối input production thay đổi, provider đổi model sau alias.

| Nhóm | Ví dụ | Khi dùng | Hạn chế |
|---|---|---|---|
| Reference-based | EM, token F1, BLEU/ROUGE, BERTScore | QA ngắn, extraction, dịch | Phạt paraphrase đúng, không hiểu phủ định |
| Rule/assertion | JSON schema, regex, unit test (pass@k) | Format, code, tool-call | Chỉ kiểm tra được cái định nghĩa được |
| Reference-free | LLM-as-judge rubric, faithfulness với context | Task mở, RAG | Bias của judge, tốn cost |
| Human eval | Chuyên gia chấm, pairwise | Gold standard, kiểm định judge | Đắt, chậm, cần guideline + IAA |
| Online | A/B, feedback, implicit signals | Tác động KPI thật | Chậm, selection bias |

**Nguyên tắc:** metric phải bám KPI sản phẩm. Báo cáo kèm CI và breakdown theo segment. Tách quality / cost / latency / robustness.

### Metric phân loại: P/R/F1, macro/micro, mAP

- Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN).
- **Accuracy paradox:** 5% positive thì model “luôn âm tính” vẫn đạt 95% accuracy. Hãy dùng P/R/F1, PR-AUC cho lớp hiếm.
- **Micro** (gộp TP/FP/FN toàn cục): với bài đơn nhãn thì micro-P = micro-R = micro-F1 = accuracy.
- **Macro:** trung bình đều F1 từng lớp, nhạy với lớp hiếm.
- **Weighted:** trọng số theo support, nghiêng về lớp phổ biến, nên gần accuracy.
- Đọc **confusion matrix**: hàng = thật, cột = dự đoán. Tổng cột → precision, tổng hàng → recall.
- **AP** = diện tích dưới đường PR của một lớp/truy vấn. **mAP** = trung bình AP qua các lớp (detection, theo ngưỡng IoU, vd. mAP@0.5 hoặc @[.5:.95]) hoặc qua các truy vấn (retrieval).
- Chọn ngưỡng theo chi phí lỗi: guardrail an toàn ưu tiên recall, auto-block ưu tiên precision. Dùng Fβ (β > 1 nghiêng về recall).

### LLM-as-judge: thiết kế, G-Eval, bias

**Kiểu:** pointwise (chấm thang, hợp với monitoring), pairwise (A vs B, hợp để chọn phiên bản), reference-guided (có đáp án chuẩn: toán, fact).

**Rubric tốt:** tiêu chí tách riêng, có mô tả từng mức (anchor), yêu cầu lý do trước điểm, output JSON. Thang nhị phân hoặc 3 mức thường ổn định hơn 1–10.

**G-Eval:** tiêu chí → LLM tự sinh evaluation steps (CoT) → form-filling. Điểm = Σ p(sᵢ)·sᵢ theo xác suất token điểm, cho điểm liên tục và ít hoà hơn. Cần logprobs (hoặc sample nhiều lần).

**Bias (Zheng et al. 2023):** position, verbosity, self-enhancement, yếu khi chấm toán/suy luận.
- Giảm bias: chạy 2 thứ tự, chỉ tính thắng khi nhất quán. Rubric “độ dài không phải tiêu chí”. Judge khác họ với model được chấm hoặc panel nhiều judge. Reference/CoT cho toán. Calibrate trên nhãn người.

**Kiểm định judge** (bắt buộc trước khi dùng làm gate): 100–300 mẫu có nhãn người, gồm cả mẫu lỗi.
- Nhị phân: Cohen's κ = (pₒ − pₑ)/(1 − pₑ), kèm precision/recall theo từng lớp (cẩn thận prevalence paradox).
- Ordinal: Spearman/Kendall (thứ hạng), weighted kappa (giá trị tuyệt đối).
- Thang Landis–Koch: < 0.2 slight, 0.21–0.4 fair, 0.41–0.6 moderate, 0.61–0.8 substantial, > 0.8 almost perfect.

### Benchmark & contamination

| Benchmark | Nội dung | Chấm |
|---|---|---|
| MMLU | 57 subject, trắc nghiệm 4 đáp án | Accuracy |
| GSM8K | ~8.5K bài toán grade-school nhiều bước | Đáp số cuối |
| HumanEval | 164 bài viết hàm Python | Unit test, pass@k |
| MT-Bench | 80 câu multi-turn, 8 nhóm | GPT-4 judge thang 1–10 |
| Chatbot Arena | Vote pairwise ẩn danh của người dùng | Bradley–Terry (trước đây Elo online) |

- **pass@k không chệch:** 1 − C(n−c, k)/C(n, k), với n mẫu và c mẫu đúng. Không dùng 1 − (1 − c/n)^k vì bị chệch.
- **Elo:** E_A = 1/(1 + 10^((R_B − R_A)/400)), R'_A = R_A + K(S_A − E_A). Chênh 100 điểm thì E ≈ 0.64 cho bên mạnh hơn.
- Elo online phụ thuộc thứ tự trận. Model tĩnh thì nên fit BT bằng MLE, CI lấy bằng bootstrap.
- **Contamination:** test lọt vào train. Phát hiện: n-gram overlap, perturb/paraphrase làm điểm tụt, so trước/sau cutoff. Giảm: private/held-out, benchmark động, tách dev/test cho eval nội bộ.
- Benchmark công khai ≠ hiệu năng trên domain của bạn. Luôn có eval set riêng.

### Thiết kế eval set & thống kê so sánh

**Golden dataset:** sample từ production (ẩn PII), stratify theo intent và độ khó, có edge case, câu ngoài phạm vi, adversarial/red-team (prompt injection, jailbreak). Version hoá data và guideline gán nhãn. Đo IAA giữa các annotator.

**CI cho accuracy (Wald):** p ± 1.96·√(p(1−p)/n). Ví dụ p = 0.85, n = 400 cho ±3.5 điểm %. Muốn CI hẹp một nửa thì cần n gấp 4. Khi p gần 0/1 hoặc n nhỏ, dùng Wilson.

**Cỡ mẫu:** n = 1.96²·p(1−p)/E². Trường hợp xấu nhất p = 0.5, E = ±2 điểm % thì n = 2401. Áp dụng cho **từng** segment muốn báo cáo riêng.

**So sánh 2 hệ thống trên cùng tập (paired):**
- McNemar: χ² = (|b − c| − 1)²/(b + c), với b, c là số cặp bất đồng. Ngưỡng 3.841 (α = 0.05, df = 1). b + c nhỏ thì dùng exact binomial.
- Paired bootstrap: resample **cùng** chỉ số item cho A và B, tính Δ, lấy percentile 2.5–97.5. CI không chứa 0 thì có ý nghĩa.
- Hai CI riêng chồng lấn ≠ không có ý nghĩa.
- So nhiều biến thể thì cẩn thận multiple comparisons (Bonferroni/Holm).

**Báo cáo thử nghiệm:** giả thuyết, setup (model snapshot, prompt version, dataset version, seed/temperature), metric chính và guardrail, Δ ± CI, p-value, breakdown theo segment, cost/latency, ví dụ lỗi, khuyến nghị.

### Hallucination, faithfulness & RAG metrics

- **Faithfulness/groundedness:** mọi claim có được context hỗ trợ không. Cách đo: claim decomposition + verify (LLM hoặc NLI). RAGAS faithfulness = claim được hỗ trợ / tổng claim.
- **NLI-based:** premise = context, hypothesis = claim, phân loại entail/neutral/contradict.
- **SelfCheckGPT:** không có nguồn thì sample nhiều lần, nội dung không nhất quán giữa các lần là dấu hiệu bịa.
- **Answer relevancy (RAGAS):** sinh câu hỏi ngược từ câu trả lời, tính cosine với câu hỏi gốc. **Không** dùng ground truth, **không** đo tính đúng.
- **Context precision/recall:** đánh giá retriever (context recall cần reference).
- **Factual correctness** (so với reference) khác **faithfulness** (so với context): câu trả lời có thể faithful với một context sai.
- Không dùng perplexity hay BLEU để đo hallucination.

### Regression testing trong CI & online eval

**Trong CI (Pytest + GitLab CI):**
1. Lớp deterministic: JSON schema, field bắt buộc, không lộ system prompt, tool-call đúng tên/tham số. Gate cứng.
2. Lớp chất lượng: chạy golden set, metric tổng hợp ≥ baseline − margin (margin lấy từ noise đo được khi chạy baseline nhiều lần).
3. Case quan trọng: chạy lặp k lần, yêu cầu tỷ lệ pass ≥ ngưỡng.
4. Ghim model snapshot + prompt version + dataset version. Cache response để rẻ. Lưu kết quả làm experiment để xem xu hướng.
- **Flaky:** không skip, chuyển sang assertion ngữ nghĩa hoặc chạy lặp. `temperature=0` ≠ deterministic (float non-associativity, dynamic batching, MoE, provider đổi model). `seed` chỉ best-effort.

**Online:** shadow, canary, A/B (random theo user, tính MDE và cỡ mẫu trước, không peeking, có guardrail: p95 latency, cost/request, error rate, safety). Explicit feedback (thumbs, selection bias) kết hợp implicit signals (rephrase, bỏ dở, escalate, copy, thời gian giải quyết). Đưa lỗi production vào golden set để khép vòng.

### Observability: Langfuse & OpenTelemetry

**Langfuse data model:**
- **Trace:** một request/tương tác. Chứa các **observation** lồng nhau.
- Observation types: `span` (đơn vị công việc có thời lượng), `generation` (gọi LLM: model, input/output, usage, cost), `event` (sự kiện rời rạc), cùng agent, tool, chain, retriever, embedding, evaluator, guardrail.
- **Session:** gom nhiều trace theo `sessionId` (hội thoại nhiều lượt). Còn có user, tags, metadata, environment, release/version.
- **Score:** gắn với đúng một Trace / Observation / Session / DatasetRun. Kiểu numeric / categorical / boolean / text. Source là API / EVAL (LLM-as-judge) / ANNOTATION. Có score config để chuẩn hoá.
- **Datasets:** item gồm input, expected_output, metadata, có thể link tới trace nguồn. **Experiments/dataset runs:** chạy app trên dataset, mỗi item thành một trace, chấm score, so sánh giữa các run.
- **Prompt management:** version tự tăng. Label `production` (mặc định khi `get_prompt` không truyền gì) và `latest`. Rollback bằng cách chuyển label, không cần deploy. SDK cache client-side TTL mặc định 60s (stale-while-revalidate). Nên khai báo `fallback`.

**OpenTelemetry GenAI semconv** (trạng thái Development): `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`/`output_tokens`. Tên span có dạng `chat gpt-4o-mini`. Nội dung message là opt-in (PII).

**Thực hành:** redact PII trước khi export, retention policy, tail-based sampling (giữ 100% trace lỗi/chậm/feedback xấu), cost tracking theo token × giá theo model/route/tenant.

### Latency: percentile, TTFT, throughput

- Dùng **p50/p95/p99**, không dùng mean: phân phối latency lệch phải, mean bị outlier kéo nhưng không mô tả được đuôi.
- **Nearest-rank:** rank = ⌈q·n⌉. Numpy mặc định nội suy tuyến tính nên cho kết quả khác. Ghi rõ phương pháp khi báo cáo.
- **Không trung bình các percentile** (giữa replica hay giữa các khoảng thời gian). Hãy gộp histogram hoặc sketch (t-digest, HDR, DDSketch).
- LLM latency ≈ TTFT (queue + prefill, tăng theo độ dài prompt) + số output token × TPOT (time per output token).
- Tối ưu TTFT: prompt caching, rút gọn prompt, streaming, model nhỏ hơn. Tối ưu throughput: batching, giới hạn max_tokens.
- Throughput (tokens/s, req/s) đánh đổi với latency: batch lớn tăng throughput nhưng tăng TTFT.
- p95 với n nhỏ rất nhiễu. Cần đủ mẫu, và nên báo cáo theo route/model/độ dài input.

### LLMOps: deploy, routing, độ tin cậy, SLO

- **Prompt versioning:** prompt là artifact có version, label theo môi trường, gắn version vào trace. Model registry hoặc config ghim snapshot model, tham số, provider.
- **Deploy:** offline eval → **shadow** (nhân bản request, user không thấy, chi phí ×2, tắt side effect) → **canary** (x% user) → A/B → rollout. Blue-green để rollback nhanh.
- **Multi-provider routing & fallback:** route theo cost, latency, task. Fallback khi lỗi hoặc rate limit. Mỗi provider cần prompt và regression riêng.
- **Retry:** chỉ với 429/5xx/timeout. Exponential backoff + **jitter** (full jitter: Uniform(0, min(cap, base·2^i))). Tôn trọng `Retry-After`. Đặt deadline tổng. Chú ý idempotency.
- **Circuit breaker:** closed → open (fail-fast/fallback) → half-open (thử) → closed.
- **Rate limit:** token bucket theo RPM/TPM, queue, ưu tiên theo tenant.
- **SLO/SLA:** SLI là số đo, SLO là mục tiêu nội bộ, SLA là cam kết hợp đồng. Error budget = (1 − SLO)·thời gian. Ví dụ 99.5% trong 30 ngày cho 216 phút.
- **Drift:** data drift (P(X) đổi), concept drift (P(Y|X) đổi), model/prompt drift (provider cập nhật model sau alias). Giám sát bằng metric online + judge chạy trên sample production + alert.

### Cost optimization & caching

- **Cost** = input_tokens·giá_in + output_tokens·giá_out (+ cache write/read). Theo dõi theo route, model, tenant, feature.
- **Cascade** (rẻ → đắt): chi phí kỳ vọng = c_rẻ + P(escalate)·c_đắt. Đừng quên request escalate đã trả tiền model rẻ. Latency của request escalate là tổng hai model.
- **Router** phân loại trước tránh chi phí kép, nhưng phụ thuộc độ chính xác router.
- **Exact cache:** key = prompt chuẩn hoá. An toàn, hit rate thấp.
- **Semantic cache:** cosine ≥ ngưỡng. Hit rate cao, nhưng có **false hit** (khác số tiền, tài khoản, phủ định). Phân vùng theo user/tenant, không cache nội dung cá nhân hoặc thời gian thực, đo false-hit rate.
- **Provider prompt caching:** cache **tiền tố** giống hệt. Đặt phần tĩnh (system, tools, few-shot) trước. Giảm cost input và TTFT, không đổi output. Có giá ghi/đọc riêng và TTL.
- **Token optimization:** rút gọn system prompt, chọn lọc context RAG (rerank, top-k nhỏ), giới hạn `max_tokens`, structured output ngắn gọn, tóm tắt lịch sử hội thoại.
- Mọi tối ưu cost phải kèm eval chất lượng: chỉ báo cáo “tiết kiệm X% ở mức chất lượng không đổi (Δ ± CI)”.


---

## 📈 Machine Learning & Deep Learning

### Bias–variance & regularization

**Error = Bias² + Variance + Irreducible noise.**

| Triệu chứng | Chẩn đoán | Xử lý |
|---|---|---|
| Train tốt, val kém, gap lớn | High variance (overfit) | thêm data/augmentation, L1/L2, dropout, early stopping, giảm độ phức tạp, ensemble (bagging) |
| Train kém, val kém tương đương | High bias (underfit) | model mạnh hơn, thêm feature, giảm regularization, train lâu hơn, boosting |

- Learning curve: overfit → gap train/val thu hẹp khi thêm data; underfit → hai đường hội tụ sớm ở mức lỗi cao, thêm data **không** giúp.
- **L1 (Lasso)** λΣ|w|: gradient độ lớn hằng số λ → đẩy w về **đúng 0** → sparse, chọn feature. **L2 (Ridge)** λΣw²: gradient 2λw → co nhỏ, hiếm khi = 0, ổn định khi feature tương quan. **ElasticNet** = kết hợp.
- **Weight decay vs L2**: với SGD thuần tương đương (đổi hệ số). Với Adam thì KHÔNG: số hạng λw bị chia √v̂ → **AdamW** tách rời decay: θ ← θ − lr·λ·θ ngoài bước Adam.
- Regularization khác: dropout, early stopping, data augmentation, label smoothing, max-norm, giảm số tham số.
- Lỗi hay gặp: regularize cả bias/LayerNorm weight (thường loại khỏi weight decay); tune λ trên test set.

### Cross-validation & data leakage

| Tình huống | Splitter |
|---|---|
| i.i.d., mất cân bằng | `StratifiedKFold` |
| Nhiều mẫu/1 thực thể (bệnh nhân, user, session) | `GroupKFold` / `StratifiedGroupKFold` |
| Chuỗi thời gian | `TimeSeriesSplit` / walk-forward (expanding hoặc sliding window), có `gap` |
| Chọn hyperparameter + ước lượng không lệch | **Nested CV** (inner chọn, outer đánh giá) |

**Các dạng leakage kinh điển**
- Preprocessing fit trên toàn bộ data trước khi split (scaler, imputer, PCA, feature selection, target encoding) → luôn đặt trong `Pipeline` để fit theo từng fold.
- Oversampling/SMOTE trước khi split → bản sao/nội suy của mẫu val nằm trong train.
- **Target leakage**: feature chỉ biết được SAU thời điểm dự đoán (vd 'số lần gọi hỗ trợ sau khi huỷ').
- Time-series: rolling/lag không `shift(1)`, random shuffle, scale bằng thống kê tương lai.
- Duplicate / near-duplicate giữa train và test; cùng group ở 2 phía.
- Dấu hiệu: val score đẹp bất thường, feature importance dồn vào 1 feature lạ.

### Metrics phân loại

Confusion: TP, FP, FN, TN.

- Precision = TP/(TP+FP) — trong số báo dương, bao nhiêu đúng.
- Recall (TPR, sensitivity) = TP/(TP+FN) — bắt được bao nhiêu dương thật.
- Specificity = TN/(TN+FP); FPR = 1 − specificity.
- F1 = 2PR/(P+R) (trung bình điều hoà, bị kéo về giá trị nhỏ hơn).
- **Fβ = (1+β²)·P·R / (β²·P + R)**: β>1 coi trọng recall (F2), β<1 coi trọng precision (F0.5).
- Macro (trung bình đều các lớp) vs micro (gộp TP/FP/FN) vs weighted.

| | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|
| Trục | TPR vs FPR | Precision vs Recall |
| Baseline ngẫu nhiên | 0.5 | = prevalence (vd 0.002) |
| Mất cân bằng nặng | lạc quan (TN khổng lồ) | phản ánh tốt lớp hiếm |

- ROC-AUC = P(score dương ngẫu nhiên > score âm ngẫu nhiên), tie tính 0.5.
- **Log loss** = −(1/N)Σ[y·ln p + (1−y)·ln(1−p)]; phạt rất nặng dự đoán tự tin mà sai.
- **Calibration**: reliability diagram, Brier score, ECE; sửa bằng Platt (sigmoid) / isotonic trên tập riêng. Biến đổi đơn điệu tăng không đổi AUC.
- Threshold: chọn trên validation/OOF theo mục tiêu business (recall ≥ x, cost matrix, max F1), không chọn trên test. Class weight/SMOTE làm xác suất bị lệch → cần calibrate lại nếu dùng xác suất.

### Metrics hồi quy & class imbalance

| Metric | Công thức | Ghi chú |
|---|---|---|
| MAE | mean(|y−ŷ|) | bền với outlier, tối ưu → trung vị |
| MSE / RMSE | mean((y−ŷ)²), √MSE | phạt lỗi lớn, tối ưu → trung bình; RMSE ≥ MAE |
| MAPE | mean(|y−ŷ|/|y|)·100 | **nổ khi y≈0**, bất đối xứng (phạt over-forecast nặng hơn) |
| sMAPE | mean(2|y−ŷ|/(|y|+|ŷ|)) | bị chặn ở 200% nhưng vẫn bất ổn khi cả y, ŷ ≈ 0 |
| MASE | MAE / MAE(naive in-sample) | <1: tốt hơn naive; không phụ thuộc scale |
| R² | 1 − SS_res/SS_tot | có thể **âm** trên test (tệ hơn dự đoán hằng = mean) |

- **Huber**: bậc 2 khi |e| ≤ δ, tuyến tính khi |e| > δ.

**Class imbalance**
- Metric: PR-AUC, F1/Fβ, recall@precision, không dùng accuracy.
- `class_weight='balanced'`: w_c = n_samples / (n_classes × n_c).
- Focal loss: FL = −α(1−p_t)^γ·log p_t (γ≈2, α≈0.25 trong RetinaNet) – giảm trọng số mẫu dễ.
- Resampling (random over/under, SMOTE, ADASYN) **chỉ trên train fold** – dùng `imblearn.pipeline.Pipeline`.
- Chỉnh threshold thường hiệu quả và rẻ hơn resampling.

### Model cổ điển: linear, logistic, SVM, kNN, tree

- **Linear regression**: MSE, nghiệm đóng (XᵀX)⁻¹Xᵀy hoặc GD. Giả định: tuyến tính, sai số độc lập, phương sai đều.
- **Logistic regression**: p = σ(wᵀx+b), σ(z)=1/(1+e^−z). Loss BCE. **Gradient: ∂L/∂w = (p − y)·x, ∂L/∂b = p − y** (gọn nhờ σ' = σ(1−σ) triệt tiêu).
- **SVM**: tối đa margin 2/||w||. **C lớn** → phạt vi phạm nặng, margin hẹp, dễ overfit; C nhỏ → margin rộng. RBF **gamma lớn** → mỗi điểm ảnh hưởng vùng nhỏ, biên ngoằn ngoèo → overfit. Cần scale feature.
- **kNN**: lazy, k nhỏ → variance cao; cần scale; chậm ở inference; curse of dimensionality.
- **Decision tree**:
  - Gini = 1 − Σp_k² (max 0.5 với 2 lớp); Entropy = −Σp_k·log₂p_k (max 1 với 2 lớp).
  - Information Gain = H(parent) − Σ(n_i/n)·H(child_i).
  - Ví dụ: [6,3,1] → Gini = 1 − (0.36+0.09+0.01) = 0.54.
  - Không cần scale, overfit nếu không giới hạn max_depth/min_samples_leaf; không ngoại suy.

| Cần scale feature | Không cần |
|---|---|
| kNN, SVM, k-means, PCA, linear/logistic có regularization, neural nets | Decision tree, Random Forest, GBDT (XGBoost/LightGBM/CatBoost) |

### Ensemble: bagging, boosting, stacking

- Variance của trung bình M model tương quan ρ: **ρσ² + (1−ρ)σ²/M** → M→∞ còn ρσ²; muốn ensemble hiệu quả cần model **đa dạng**.
- **Bagging / Random Forest**: bootstrap (≈63.2% mẫu unique, **36.8% OOB** vì (1−1/n)ⁿ → e⁻¹) + chọn ngẫu nhiên `max_features` mỗi split → giảm tương quan → giảm variance. Thêm cây không làm overfit nặng hơn (chỉ tốn compute).
- **Boosting**: các cây nối tiếp fit residual/gradient; giảm bias (và cả variance nhờ shrinkage). learning_rate ↓ ⇒ cần n_estimators ↑ (≈ tỉ lệ nghịch) + early stopping.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Mọc cây | depthwise mặc định (`lossguide` tuỳ chọn) | **leaf-wise** (best-first) | cây đối xứng (oblivious) |
| Tham số chính | max_depth, eta, subsample, colsample, lambda | **num_leaves** (< 2^max_depth), min_data_in_leaf | depth, l2_leaf_reg |
| Categorical | native (`enable_categorical`) bản mới | native (`categorical_feature`) | **ordered target statistics** + ordered boosting |
| NaN | native (default direction) | native | native |

- **Stacking**: meta-model học trên **out-of-fold predictions** của base models (không dùng in-sample prediction → leakage). **Blending**: dùng 1 holdout thay K-fold – đơn giản hơn nhưng phí data.
- Meta-model nên đơn giản (logistic/ridge).

### k-means & PCA

**k-means**: minimize inertia (WCSS) = Σ||x − μ_c||². Lloyd: gán cụm ↔ cập nhật tâm, hội tụ về **local optimum**.
- **k-means++**: chọn tâm kế tiếp với xác suất ∝ D(x)² → khởi tạo tốt hơn; vẫn nên chạy nhiều `n_init`.
- Chọn k: **elbow** (inertia luôn giảm khi k tăng → tìm 'khuỷu', không chọn min), **silhouette** s = (b−a)/max(a,b) ∈ [−1, 1], gap statistic.
- Giả định cụm cầu, kích thước tương đương; nhạy outlier & scale. Cụm hình dạng lạ → DBSCAN, GMM, spectral.

**PCA**: chiếu lên các vector riêng của ma trận hiệp phương sai (sau khi **center**, thường **standardize**).
- Explained variance ratio của PC_i = λ_i / Σλ.
- Chọn k nhỏ nhất để cumulative ratio ≥ ngưỡng (90–95%).
- Ví dụ λ = [5.1, 3.0, 1.6, 1.2, 0.7, 0.4] (tổng 12): cum = 0.425, 0.675, 0.808, **0.908** → k = 4 cho ≥ 90%.
- PCA tuyến tính, không dùng nhãn; fit trên train rồi transform val/test (fit trên toàn bộ = leakage).

### Feature engineering

- **Encoding**: one-hot (cardinality thấp; cho linear/NN), ordinal (cho tree khi có thứ tự), **target encoding** (cardinality cao) → bắt buộc **out-of-fold + smoothing**: enc = (n·mean_cat + m·global_mean)/(n + m); frequency/count encoding; embedding (NN); hashing.
- **Scaling**: StandardScaler (z-score), MinMaxScaler, RobustScaler (median/IQR, bền outlier); log/Box-Cox/Yeo-Johnson cho phân phối lệch. Tree-based không cần.
- **Missing values**: impute (median/mode/KNN/iterative) **fit trên train**; thêm cờ `is_missing` khi thiếu có ý nghĩa (MNAR); GBDT xử lý NaN native.
- **Binning**: quantile/uniform bins – giúp linear model học phi tuyến, nhưng mất thông tin với tree.
- **Interaction**: tích/tỷ lệ (giá/m², thu nhập/khoản vay), polynomial features.
- **Datetime**: day-of-week, month, holiday, cyclical encoding sin/cos(2π·t/T).
- **Time-series**: lag_k = y.shift(k); rolling mean/std **phải shift(1) trước khi rolling**; khi nhiều chuỗi dùng `groupby(id)` trước shift; horizon h > 1 thì lag nhỏ nhất ≥ h (direct strategy).
- Kiểm soát chất lượng data: duplicate, outlier, range check, schema/drift check, label noise (confident learning/cleanlab).

### Time-series forecasting

- **Stationarity**: mean/variance/autocovariance không đổi theo thời gian.
  - **ADF**: H0 = có unit root (**không dừng**); p < 0.05 → bác bỏ → dừng.
  - **KPSS**: H0 = **dừng** (ngược ADF) → nên dùng cả hai.
  - Biến đổi: differencing (d), seasonal differencing (D, lag m), log để ổn định variance.
- **ARIMA(p,d,q)**: p = bậc AR, d = số lần sai phân, q = bậc MA. SARIMA(p,d,q)(P,D,Q)_m.

| | ACF | PACF |
|---|---|---|
| AR(p) | tắt dần | **cắt sau lag p** |
| MA(q) | **cắt sau lag q** | tắt dần |
| ARMA | tắt dần | tắt dần |

- **Prophet**: y = trend (piecewise linear/logistic) + seasonality (Fourier) + holidays + noise; dễ dùng, vẫn phải backtest.
- ML approach: lag/rolling/calendar features + GBDT; recursive vs direct multi-step.
- **Đánh giá**: walk-forward backtest (expanding/sliding), không shuffle; so với **naive** (y_{t−1}) và **seasonal naive** (y_{t−m}); MASE, sMAPE, RMSE; tránh MAPE khi có y≈0.
- Leakage: feature tương lai (giá khuyến mãi chưa biết), scale bằng thống kê toàn chuỗi, rolling không shift.

### Hyperparameter optimization

| Phương pháp | Ý tưởng | Khi nào |
|---|---|---|
| Grid search | thử mọi tổ hợp | ít tham số, rời rạc |
| Random search | lấy mẫu ngẫu nhiên | nhiều tham số, chỉ vài cái quan trọng (Bergstra & Bengio 2012) |
| Bayesian (GP, **TPE**) | mô hình hoá p(score|config) từ các lần thử trước → chọn điểm hứa hẹn | train đắt |
| Hyperband / ASHA / pruning | dừng sớm trial tệ | DL, budget lớn |

- **Optuna**: TPE sampler mặc định, `MedianPruner`, define-by-run search space.
- Learning rate, regularization lấy mẫu **log-uniform**.
- **Early stopping**: theo dõi val metric, `patience`; với GBDT dùng `early_stopping_rounds` (eval set không được là test).
- **Nested CV**: số lần fit = outer_k × (n_configs × inner_k + 1 refit). Score của inner CV được chọn là **lạc quan** (optimistic bias) → cần outer loop/holdout để báo cáo.
- Thứ tự ưu tiên tune GBDT: learning_rate + n_estimators (early stop) → num_leaves/max_depth → min_child_samples → subsample/colsample → reg_lambda/alpha.

### Deep learning cốt lõi

- **Backprop** = chain rule: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w. Softmax + CE: ∂L/∂z = p − y.
- **Activation**: sigmoid σ' ≤ 0.25 (saturation → vanishing); tanh; ReLU (dying ReLU); Leaky ReLU; GELU = x·Φ(x) (BERT/GPT); SiLU/Swish.
- **Softmax ổn định**: softmax(x) = softmax(x − max x) – kết quả giữ nguyên, tránh overflow. Dùng log-sum-exp.
- **Init**: Xavier Var = 2/(fan_in+fan_out) (tanh/sigmoid); **He** Var = 2/fan_in (ReLU). Init toàn 0 → đối xứng, các neuron học như nhau.
- **BatchNorm**: train dùng thống kê batch + cập nhật running stats; eval dùng running stats; batch nhỏ → kém → GroupNorm/LayerNorm. **LayerNorm** chuẩn hoá theo feature của từng mẫu, train = eval (Transformer).
- **Dropout (inverted)**: train: giữ với xác suất 1−p và **chia cho (1−p)**; eval: identity. PyTorch `p` = xác suất **drop**.
- **Optimizer**: SGD+momentum v = μv + g; Adam: m = β1m + (1−β1)g, v = β2v + (1−β2)g², m̂ = m/(1−β1ᵗ), v̂ = v/(1−β2ᵗ), θ −= lr·m̂/(√v̂+ε). Bước 1: |Δθ| ≈ lr bất kể |g|. Warmup + cosine/linear decay; OneCycle.
- **Vanishing/exploding**: ReLU, init đúng, normalization, residual connection, gradient clipping, LSTM/GRU.
- **Loss**: CE (multi-class, logits), `BCEWithLogitsLoss` (multi-label/binary, ổn định hơn sigmoid+BCE), MSE/MAE/Huber, label smoothing.
- **Đếm tham số**: Linear = in·out + out; Conv2d = C_in·C_out·k_h·k_w (+C_out nếu bias); BatchNorm = 2C trainable (+running_mean/var là buffer); pooling/ReLU/Flatten = 0; Embedding = V·d; LSTM (PyTorch) = 4·(h·in + h·h + 2h). Ví dụ MLP 784→256→128→10 = 200,960 + 32,896 + 1,290 = **235,146**.

### PyTorch thực chiến & fine-tuning

```python
model.train()
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, y)   # CrossEntropyLoss nhận LOGITS
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)        # trước khi clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    running += loss.item()            # không cộng tensor

model.eval()
with torch.no_grad():                 # hoặc torch.inference_mode()
    preds = model(x_val).argmax(1)
```
- `model.eval()` đổi hành vi Dropout/BatchNorm; `no_grad()` tắt autograd. Cần **cả hai** khi inference.
- `.detach()` cắt tensor khỏi graph (vd target network, logging); `.item()` lấy số Python.
- `DataLoader(num_workers>0, pin_memory=True)` tăng throughput; trên Windows cần `if __name__ == '__main__':`; seed worker để reproducible.
- bf16 không cần GradScaler (dải mũ như fp32); fp16 cần.
- **Transfer learning**: data ít + domain gần → freeze backbone, train head; sau đó unfreeze dần, **LR nhỏ cho layer sớm** (discriminative LR), LR nhỏ hơn train từ đầu; normalize theo mean/std của pretrained.
- **Augmentation** chỉ trên train (trừ TTA); tránh augmentation đổi nghĩa nhãn (flip chữ số/biển rẽ trái). Mixup/CutMix, RandAugment.
- **Label noise**: label smoothing, loss bền (MAE/GCE), co-teaching, làm sạch bằng confident learning; kiểm tra nhãn ở các mẫu loss cao nhất.


---

## 👁️ Computer Vision, Speech, OCR & RecSys

### CNN: công thức phải tính được bằng tay

**Output size** (PyTorch, floor):

`out = floor((W − K + 2P) / S) + 1`

- 'same' với stride 1: `P = (K − 1)/2`. Với K=3, P=1 → `out = ceil(W/S)`.
- Pooling dùng cùng công thức (MaxPool 2×2, S=2 → chia đôi, floor).

**Số tham số conv**: `(K·K·C_in + 1)·C_out` (bỏ +1 nếu bias=False, thường khi có BatchNorm phía sau). KHÔNG phụ thuộc H×W (weight sharing).

**FLOPs (MAC)** ≈ `K²·C_in·C_out·H_out·W_out`.

**Depthwise separable** (MobileNet): depthwise `K²·C_in` + pointwise `C_in·C_out`. Tỷ lệ tiết kiệm = `1 / (1/C_out + 1/K²)` ≈ K² (≈ 8–9 lần với 3×3). Phần pointwise chiếm ~95%+ chi phí.

**Receptive field**: `r_l = r_{l−1} + (K_l − 1)·j_{l−1}`, `j_l = j_{l−1}·S_l` (r₀ = 1, j₀ = 1).
- 2 lớp 3×3 (S=1) → 5×5 với 18C² tham số (< 25C²); 3 lớp → 7×7 (27C² < 49C²) + thêm phi tuyến (ý tưởng VGG).
- Stride/pooling làm RF tăng nhanh vì jump nhân lên.

**1×1 conv**: tuyến tính trên chiều kênh tại từng pixel → đổi số kênh, bottleneck (ResNet-50, Inception), trộn kênh.

**Pooling**: không tham số, giữ số kênh, tạo bất biến tịnh tiến nhỏ. **Global Average Pooling** thay FC lớn ở cuối mạng.

**Lỗi hay gặp**: quên floor, quên C_in hoặc bias khi đếm tham số, nghĩ padding 1 luôn giữ kích thước (chỉ đúng khi S=1).

### Kiến trúc: ResNet, EfficientNet, ViT, CLIP & augmentation

| Model | Ý tưởng cốt lõi | Câu phỏng vấn hay gặp |
|---|---|---|
| ResNet | `y = F(x) + x`; bottleneck 1×1–3×3–1×1 | Giải **degradation** (training error mạng plain sâu tăng), không phải overfitting; gradient có thành phần identity |
| EfficientNet | Compound scaling d=α^φ, w=β^φ, r=γ^φ, `α·β²·γ² ≈ 2` | Vì FLOPs ∝ d·w²·r² → mỗi φ tăng 1, FLOPs ×2; B0: α=1.2, β=1.1, γ=1.15 |
| ViT | Cắt patch P×P → linear projection → + position embedding + [CLS] → Transformer encoder | Số token = (H/P)·(W/P) + 1: 224/16 → 197, 384/16 → 577. Ít inductive bias → cần data lớn (JFT, IN-21k) hoặc DeiT |
| CLIP | Image encoder + text encoder, contrastive InfoNCE đối xứng trên ma trận N×N, ~400M cặp | Zero-shot: prompt 'a photo of a {label}', chọn argmax cosine; không sinh caption |

**ViT**: attention ∝ N² → patch 16 → 8 làm token ×4, attention ×16. Đổi resolution → nội suy position embedding.

**Augmentation**
- Mixup: `x = λx₁ + (1−λ)x₂`, nhãn **mềm** cùng λ, λ ~ Beta(α, α).
- CutMix: dán patch ảnh 2 vào ảnh 1, nhãn theo tỷ lệ diện tích.
- Mosaic (YOLOv4): ghép 4 ảnh; detection phải biến đổi box tương ứng.
- Flip/rotate phải giữ ngữ nghĩa nhãn: KHÔNG flip cho OCR, biển số, ảnh y tế trái/phải.

### Object Detection: kiến trúc, IoU, NMS

| | Two-stage | One-stage | Set prediction |
|---|---|---|---|
| Ví dụ | Faster R-CNN (RPN + RoI head), Mask R-CNN | YOLO, SSD, RetinaNet, FCOS | DETR, Deformable DETR, RT-DETR |
| Ưu | Chính xác, tốt với vật nhỏ | Nhanh, realtime | Không anchor, không NMS |
| Nhược | Chậm hơn | Cần NMS, mất cân bằng lớp (RetinaNet dùng focal loss) | DETR gốc hội tụ chậm (~500 epoch), yếu vật nhỏ |

- **Anchor-based**: dự đoán offset so với anchor box định trước (cần tune tỷ lệ/kích thước). **Anchor-free**: dự đoán từ điểm/center (FCOS, CenterNet, YOLOX/v8). Anchor-free ≠ NMS-free (FCOS vẫn NMS).
- **NMS-free** thực sự nhờ gán nhãn một-một: DETR (Hungarian matching), YOLOv10.

**IoU** = `area(A∩B) / (area(A) + area(B) − area(A∩B))`. Giao: `w = max(0, min(x2) − max(x1))`, tương tự h. Biến thể loss: GIoU, DIoU, CIoU (xử lý khi không giao nhau).

**Greedy NMS** (theo từng class): sort score → giữ box cao nhất → loại các box có IoU > ngưỡng với nó → lặp với box còn lại. Box đã bị loại **không** dùng để loại box khác. Soft-NMS: giảm score thay vì xóa (tốt cho cảnh đông người).

### Detection metrics: Precision-Recall, AP, mAP

**Cách tính AP cho 1 class**
1. Sort mọi detection theo confidence giảm dần.
2. Gán TP nếu IoU ≥ ngưỡng với GT chưa được khớp (mỗi GT chỉ khớp 1 lần; trùng → FP).
3. Tích lũy: `P = TP/(TP+FP)`, `R = TP / #GT` (mẫu số là tổng GT, kể cả object bị bỏ sót).
4. Nội suy: `P_interp(r) = max P(r') với r' ≥ r`; AP = diện tích dưới đường đã nội suy.

| | PASCAL VOC | COCO |
|---|---|---|
| Ngưỡng IoU | 0.5 (mAP@0.5) | 10 ngưỡng 0.50:0.05:0.95 (AP = mAP@[.5:.95]) |
| Nội suy | VOC2007: 11 điểm; VOC2010+: all-point | 101 điểm recall |
| Phụ | — | AP50, AP75, APs/APm/APl (theo kích thước) |

- AP50 cao nhưng AP thấp → phát hiện được nhưng box lệch (localization kém).
- mAP = trung bình AP qua class (COCO: qua class và ngưỡng IoU).
- Production còn cần **latency** (ms/ảnh, p95), **throughput** (ảnh/s, FPS) — đổi bằng batch size, resolution, TensorRT/FP16/INT8.

**Bẫy**: chia cho số TP thay vì số GT; nhầm AP (diện tích) với precision tại 1 ngưỡng confidence.

### Segmentation: loại bài toán, model, Dice vs IoU

| Loại | Output | Ví dụ |
|---|---|---|
| Semantic | Class cho mỗi pixel, không tách instance | FCN, DeepLab, U-Net, SegFormer |
| Instance | Mask riêng cho mỗi object (things) | Mask R-CNN, YOLACT, YOLOv8-seg |
| Panoptic | Mỗi pixel 1 class + 1 instance id, gồm stuff + things | Panoptic FPN, Mask2Former; metric PQ = SQ × RQ |

- **U-Net**: encoder-decoder + skip connection (concat) giữ chi tiết không gian; mạnh với ảnh y tế ít data.
- **Mask R-CNN**: Faster R-CNN + nhánh mask theo RoI; **RoIAlign** (bilinear) thay RoIPool tránh lệch lượng tử hóa.
- **SAM**: promptable (point/box/mask), class-agnostic; ViT image encoder nặng chạy 1 lần + prompt encoder + mask decoder nhẹ; SA-1B ~1.1 tỷ mask / 11M ảnh. Cần thêm detector (vd Grounding DINO) nếu muốn có class.

**Metric**
- `IoU = |A∩B| / |A∪B|`
- `Dice = 2|A∩B| / (|A| + |B|)` = F1 trên pixel
- `Dice = 2·IoU / (1 + IoU)` → Dice ≥ IoU; IoU = 0.5 ↔ Dice ≈ 0.667.
- mIoU = trung bình IoU qua class. Dice loss chống mất cân bằng foreground/background.

### Generative: GAN, Diffusion, Latent Diffusion, VLM

**GAN**: minimax giữa generator và discriminator. Vấn đề: mode collapse (ít đa dạng), train bất ổn. Chữa: WGAN-GP, spectral norm, minibatch discrimination. Sampling 1 bước → rất nhanh.

**Diffusion (DDPM)**
- Forward: thêm Gaussian noise dần; dạng đóng `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`.
- Train: chọn t ngẫu nhiên, model ε_θ(x_t, t) dự đoán ε, loss MSE.
- Reverse: khử nhiễu từng bước (DDPM ~1000 bước, stochastic).
- **DDIM**: cùng model đã train, sampling non-Markovian, η=0 → deterministic, ít bước (20–50). Nhanh hơn nữa: DPM-Solver, LCM, consistency distillation.

**Latent Diffusion / Stable Diffusion**: VAE nén 512×512×3 → 64×64×4 (f=8); U-Net (hoặc DiT) khử nhiễu trong latent; text qua CLIP text encoder → cross-attention.

**Classifier-free guidance**: `ε̂ = ε(x,∅) + w·(ε(x,y) − ε(x,∅))`
- Train: drop condition ~10% để 1 model học cả cond/uncond.
- w = 0: unconditional; w = 1: conditional thuần; w > 1: bám prompt mạnh hơn, giảm đa dạng; SD mặc định 7.5. Chi phí 2 forward/bước.
- Classifier guidance (khác!) cần classifier train trên ảnh nhiễu.

| | GAN | Diffusion |
|---|---|---|
| Chất lượng/đa dạng | Sắc nét, dễ mode collapse | Đa dạng, phủ phân phối tốt |
| Train | Bất ổn | Ổn định (regression) |
| Sampling | 1 bước | Nhiều bước |

**VLM (LLaVA)**: CLIP ViT-L/14 (frozen) → projector (linear/MLP) → LLM. Ảnh 336/14 → 576 visual token. Stage 1: chỉ train projector (alignment); stage 2: projector + LLM (visual instruction tuning).

### Speech: tín hiệu, framing, Mel, MFCC

- **Nyquist**: lấy mẫu fs biểu diễn tối đa fs/2. 16 kHz → 8 kHz (chuẩn ASR); 8 kHz điện thoại → 4 kHz. Upsample không khôi phục dải đã mất. Sai sample rate → model 'nghe' audio nhanh/chậm → WER tăng vọt.
- **Framing**: window 25 ms (400 mẫu @16k), hop 10 ms (160 mẫu) → ~100 frame/s.
  - Không padding: `n = floor((N − win)/hop) + 1` (10 s → 998).
  - center=True (librosa/torch.stft): `n = 1 + floor(N/hop)` (10 s → 1001).
- **Pipeline đặc trưng**: waveform → pre-emphasis (tùy) → frame + Hann window → STFT → |·|² → Mel filterbank (80 bin) → log = **log-Mel** → DCT → **MFCC** (13 hệ số + Δ, ΔΔ).
- Mel: filter hẹp ở tần số thấp, rộng ở cao (tai người phân biệt tốt âm trầm).
- Spectrogram/Mel/MFCC bỏ phase → tái tạo audio cần vocoder.
- Deep ASR dùng log-Mel (Whisper 80 bin; large-v3 128 bin) hoặc raw waveform (wav2vec 2.0); MFCC hợp với GMM-HMM cổ điển.
- Augment: SpecAugment (mask time/frequency), speed perturbation, thêm noise/reverb.

### ASR: CTC, Whisper, WER/CER

**CTC**
- Thêm token **blank**; giải mã: (1) gộp ký hiệu giống nhau liền kề → (2) xóa blank. `a a _ a b _ b b` → `aabb`.
- Blank cho phép ký tự lặp thật ('ll' trong hello) và frame 'không nói gì'.
- Loss = tổng xác suất mọi alignment hợp lệ (forward-backward) → không cần nhãn căn chỉnh frame. Giả định độc lập có điều kiện → hay ghép LM khi beam search.
- Họ khác: attention encoder-decoder (Whisper), RNN-T/Transducer (streaming tốt).

**Whisper**
- Transformer encoder-decoder, 680k giờ weakly-supervised, đa ngôn ngữ.
- Audio 16 kHz, cửa sổ **30 s** → 3000 frame log-Mel (80 bin; large-v3: 128) → 2 conv (stride 2) → 1500 vector.
- Multitask token: `<|startoftranscript|>`, `<|vi|>`, `<|transcribe|>` / `<|translate|>` (chỉ X → English), `<|notimestamps|>`.
- Không streaming native; audio dài xử lý theo cửa sổ 30 s; hallucination ở đoạn im lặng → dùng VAD.

**Metric**
- `WER = (S + D + I) / N`, N = số từ của **reference** → có thể **> 100%** (nhiều insertion).
- CER: cùng công thức ở mức ký tự; dùng cho tiếng Trung/Nhật (không có khoảng trắng), OCR.
- Tiếng Việt: tách theo khoảng trắng = âm tiết; sai dấu = 1 substitution. Chuẩn hóa trước khi tính: lowercase, bỏ dấu câu, NFC, số ↔ chữ.

### Hệ thống speech: streaming, VAD, diarization, TTS, RTF

| Thành phần | Trả lời câu hỏi | Ví dụ |
|---|---|---|
| VAD | Khi nào có tiếng nói? | Silero VAD, WebRTC VAD |
| Diarization | Ai nói khi nào? (SPEAKER_00, 01…) | pyannote: segmentation + speaker embedding (x-vector/ECAPA) + clustering; metric DER |
| Streaming ASR | Ra chữ khi đang nói | RNN-T, CTC chunk, Emformer; lookahead hạn chế → WER cao hơn offline |
| Offline ASR | Chính xác nhất | Whisper, attention 2 chiều trên toàn đoạn |

- Pattern two-pass: partial result streaming → offline pass sửa bản cuối.

**TTS pipeline**: text normalization + G2P → **acoustic model** (Tacotron 2, FastSpeech 2: text/phoneme → mel, duration/pitch/energy) → **vocoder** (WaveNet, HiFi-GAN: mel → waveform). End-to-end: VITS. TTS kiểu LLM: dự đoán token neural codec (EnCodec) rồi decode. Đánh giá: MOS, WER qua ASR, speaker similarity.

**RTF** = thời gian xử lý / thời lượng audio. RTF 0.15 ↔ nhanh 1/0.15 ≈ 6.7× realtime. Streaming cần RTF < 1 **và** latency thấp (partial/first-token latency); throughput = số stream đồng thời / GPU.

### OCR & Document AI

**Pipeline cổ điển**: tiền xử lý → layout analysis → text detection → recognition → post-processing.
- Tiền xử lý: perspective correction, **deskew** (Hough, minAreaRect), **binarization** (Otsu toàn cục vs Sauvola/adaptive cục bộ khi ánh sáng không đều), denoise, CLAHE. Đừng resize quá nhỏ → mất dấu tiếng Việt.
- Detection: DBNet, EAST, CRAFT (polygon, chữ cong/nghiêng).
- Recognition: **CRNN** = CNN → BiLSTM → **CTC** (không cần vị trí từng ký tự); attention/Transformer: TrOCR, PARSeq.
- Layout: phát hiện vùng title/table/figure, reading order (LayoutParser, DocLayNet).

| Model | Đầu vào | Đặc điểm |
|---|---|---|
| TrOCR | Ảnh **dòng chữ** | ViT/BEiT encoder + text decoder; vẫn cần detector |
| LayoutLM v1/v2/v3 | Text + bbox từ OCR (+ ảnh ở v2/v3) | 2D position embedding; mạnh cho KIE, form, classification |
| Donut | Ảnh tài liệu | OCR-free: Swin encoder + BART decoder sinh JSON; có thể hallucinate |
| VLM (Qwen-VL…) | Ảnh | Linh hoạt, phụ thuộc resolution, khó truy vết bbox |

**Đánh giá**: CER/WER (CER cho OCR), field-level accuracy/F1 cho KIE.

**Unicode tiếng Việt**: NFC (dựng sẵn, 'à' = U+00E0) vs NFD ('a' + U+0300). 'Hà Nội': NFC 6 code point, NFD 9. Luôn `unicodedata.normalize('NFC', s)` cho cả ref/hyp, dữ liệu train, query search. Nguồn NFD: macOS filename, copy từ web/Word, một số bộ gõ/OCR.

### RecSys: mô hình & kiến trúc nhiều tầng

| Cách tiếp cận | Dựa trên | Ưu | Nhược |
|---|---|---|---|
| Content-based | Feature item + hồ sơ user | Gợi ý được item mới | Overspecialization, cần feature tốt |
| User-based CF | User tương tự | Đơn giản | User nhiều, đổi nhanh → khó scale |
| Item-based CF | Item tương tự (co-occurrence) | Ổn định, tính trước được (Amazon) | Item cold start |
| Matrix factorization | `r̂ = p_u·q_i + b_u + b_i + μ` | Gọn, mạnh | Cold start, chỉ ID |
| Two-tower | User tower, item tower → dot product | Precompute item emb + ANN | Không có feature chéo |

- **ALS**: cố định một bên → ridge regression nghiệm đóng cho bên kia; song song hóa tốt (Spark).
- **Implicit feedback** (Hu-Koren-Volinsky): preference p ∈ {0,1}, confidence `c = 1 + α·r`; ô trống = chưa thấy, không phải ghét.
- **Cold start**: item mới → content feature, exploration; user mới → popularity theo ngữ cảnh, onboarding, session-based.

**Phễu**: candidate generation (triệu → nghìn, ưu tiên recall, nhiều nguồn) → ranking (nghìn → trăm, model nặng, feature chéo, multi-task pCTR/pCVR) → re-ranking (diversity MMR/DPP, business rule, dedup, freshness, fairness).

**Negative sampling**: in-batch negatives rẻ nhưng phạt item hot quá mức → **logQ correction** `s − log Q(i)` (YouTube 2019); mixed negatives; hard negatives (cẩn thận false negative). Giống bi-encoder vs cross-encoder trong RAG.

### RecSys: metrics, offline vs online, bias

**Ranking metrics @k** (rel nhị phân trừ NDCG)
- `Precision@k = #relevant trong top-k / k`; `Recall@k = #relevant trong top-k / R` (R = tổng relevant của user).
- Hit rate@k: tỷ lệ user có ≥ 1 item relevant trong top-k.
- MRR: trung bình 1/rank của item relevant đầu tiên.
- `AP@k = (1/min(R,k))·Σ P@i·rel(i)`; MAP = trung bình AP qua user. (Mẫu số khác nhau giữa thư viện → nói rõ quy ước.)
- `DCG@k = Σ gain_i / log2(i + 1)`, gain = rel hoặc 2^rel − 1; `NDCG = DCG / IDCG`, IDCG từ thứ tự lý tưởng của **toàn bộ** item relevant (không chỉ item đã trả về).

**Beyond accuracy**: coverage (tỷ lệ catalog từng được gợi ý), diversity (1 − similarity trung bình trong danh sách), novelty, serendipity.

**Offline vs online**
- Offline trên log của policy cũ → exposure bias, popularity bias; offline tăng ≠ online tăng.
- Online A/B: CTR, CVR, watch time, retention; chú ý novelty effect, đủ thời gian/cỡ mẫu, guardrail metrics.
- Giảm bias: IPS (inverse propensity scoring), exploration (bandit), popularity debiasing.

**Feedback loop**: model gợi ý → item được tương tác → retrain trên log đó → gợi ý item đó nhiều hơn → 'rich get richer', coverage giảm, filter bubble.


---

## 🛠️ Python, Backend & MLOps cho AI

### Python gotchas hay bị hỏi (đoán output)

| Hiện tượng | Ví dụ | Kết quả / cách sửa |
|---|---|---|
| Mutable default argument | `def f(x, acc=[])` | Default tạo **1 lần** lúc `def` (nằm trong `f.__defaults__`) → dùng chung giữa các lần gọi. Sửa: `acc=None` rồi `acc = [] if acc is None else acc` |
| Late binding closure | `[lambda: i for i in range(3)]` | Closure tra `i` lúc **gọi** → mọi lambda thấy giá trị cuối. Sửa: `lambda i=i: i` hoặc `functools.partial` |
| Generator chỉ chạy 1 lần | `4 in g` rồi `list(g)` | `in` tiêu thụ generator tới phần tử tìm thấy; lần duyệt thứ 2 chỉ còn phần dư (hoặc rỗng) |
| `t[0] += [x]` với `t` là tuple | | List bị extend **rồi** mới raise `TypeError` (gán lại vào tuple) |
| `[[0]*2]*2` | | 2 phần tử cùng trỏ 1 list con. `deepcopy` **giữ nguyên** quan hệ alias nhờ `memo` |
| `is` vs `==` | | `is` so identity (`id`), `==` so giá trị (`__eq__`). Chỉ dùng `is` cho `None`/sentinel |
| `copy` vs `deepcopy` | | Shallow copy chỉ copy container ngoài, object con dùng chung |

- **Decorator** chạy lúc định nghĩa hàm (lúc import). Luôn dùng `@functools.wraps(f)` để giữ `__name__`, `__doc__`, `__wrapped__` (FastAPI đọc signature qua `inspect.signature` → thiếu `wraps` làm sai tham số endpoint).
- **Context manager**: `__exit__` trả về truthy ⇒ **nuốt exception**. Với `@contextmanager`, bọc `yield` trong `try/finally` để cleanup luôn chạy.
- `yield from sub` chuyển tiếp mọi giá trị và trả về giá trị `return` của sub-generator.

### GIL & chọn mô hình concurrency

**GIL** (CPython build mặc định): tại 1 thời điểm chỉ 1 thread chạy bytecode Python. GIL được **nhả** khi chờ I/O (socket, file, `time.sleep`) và trong nhiều C extension (NumPy, PyTorch op, tokenizers Rust).

| Bài toán | Lựa chọn | Lý do |
|---|---|---|
| I/O-bound, rất nhiều kết nối (gọi LLM API, DB) | `asyncio` | 1 thread, coroutine vài KB, chuyển ngữ cảnh tại `await` |
| I/O-bound với thư viện đồng bộ | `threading` / `to_thread` | GIL nhả khi chờ I/O |
| CPU-bound Python thuần | `multiprocessing` / `ProcessPoolExecutor` | Mỗi process 1 GIL riêng; chi phí pickle + RAM riêng |
| CPU-bound trong NumPy/PyTorch | thread thường đủ | C code nhả GIL |

- Python **3.13**: có bản build *free-threaded* tuỳ chọn (PEP 703, `python3.13t`, experimental). 3.14: được hỗ trợ chính thức nhưng **vẫn là build riêng**, bản mặc định vẫn có GIL; C extension phải tương thích.
- asyncio là **cooperative**: code CPU nặng không có `await` chặn cả event loop.
- Câu hỏi hay gặp: *"Vì sao thêm thread mà xử lý ảnh bằng vòng lặp Python không nhanh hơn?"* → GIL.

### asyncio: coroutine, task, gather, TaskGroup, timeout

- **Coroutine** `coro()` chỉ là object, chưa chạy. `await coro` chạy nó **ngay** trong task hiện tại. `asyncio.create_task(coro)` **lên lịch**; task chỉ chạy khi task hiện tại nhường quyền (`await` thứ gì đó thật sự chờ).
- **Tính thời gian tay**:
  - `for` + `await` tuần tự: Σ thời gian.
  - `gather(...)` không giới hạn: max thời gian.
  - `Semaphore(k)` với n job cùng thời gian d: `ceil(n/k) · d`.
  - Có `time.sleep`/code blocking: phần blocking cộng dồn **tuần tự**, timer `asyncio.sleep` chỉ bắt đầu khi task được chạy tới.

| | `gather` | `TaskGroup` (3.11+) |
|---|---|---|
| Lỗi ở 1 task | Exception đầu tiên ném ra cho caller, **task khác vẫn chạy** | **Huỷ** các task còn lại, ném `ExceptionGroup` (`except*`) |
| `return_exceptions=True` | Exception thành giá trị trong list | Không có |
| Thứ tự kết quả | Theo thứ tự truyền vào | Lấy qua `task.result()` |

- `asyncio.timeout(s)` (3.11+) / `wait_for(coro, s)`: hết giờ thì **cancel** coroutine bên trong rồi raise `TimeoutError` (từ 3.11 `asyncio.TimeoutError is TimeoutError`).
- `CancelledError` kế thừa **`BaseException`** (từ 3.8). Bắt để dọn dẹp thì phải `raise` lại, nếu không timeout/TaskGroup hỏng.
- Blocking call trong `async def` → dùng thư viện async (`httpx.AsyncClient`, `motor`, `redis.asyncio`) hoặc `await asyncio.to_thread(fn)`; CPU-bound → `loop.run_in_executor(ProcessPoolExecutor(), fn)`.
- Async generator (`async def ... yield`) + `async for`: chuẩn để stream token LLM.

### FastAPI cho AI service

| Khai báo | Chạy ở đâu | Lưu ý |
|---|---|---|
| `async def` endpoint | Trực tiếp trên event loop | **Không** gọi hàm blocking (`requests`, `time.sleep`, `model.predict` nặng) |
| `def` endpoint/dependency | Threadpool (AnyIO, mặc định 40 slot) | An toàn với lib đồng bộ, giới hạn bởi số slot |

- **lifespan**: `@asynccontextmanager async def lifespan(app): load model; yield; cleanup` rồi `FastAPI(lifespan=lifespan)`. `@app.on_event` đã deprecated. Lưu model vào `app.state`, lấy qua `Depends` để dễ override khi test.
- **Workers**: `uvicorn --workers N` hoặc `gunicorn -k uvicorn.workers.UvicornWorker -w N` → N process, **mỗi process load model riêng**: RAM ≈ N × (model + overhead). Với GPU: N × VRAM → hay OOM. Model lớn nên tách ra model server (Triton, vLLM, TGI).
- **Depends**: cache kết quả theo request (cùng dependency chỉ gọi 1 lần); dependency `yield` + `try/finally` để đóng session; `app.dependency_overrides` để fake LLM client khi test.
- **BackgroundTasks**: chạy sau khi gửi response, cùng process, không bền vững, không retry → job nặng/quan trọng dùng Celery/RQ/Arq/Kafka consumer.
- **Streaming**: `StreamingResponse(agen(), media_type="text/event-stream")`; mỗi event `data: ...` + dòng trống; tắt buffering của proxy (`X-Accel-Buffering: no`); dừng gọi LLM khi client ngắt. SSE một chiều; cần hai chiều → WebSocket.
- Pydantic v2: lax mode `"5"→5`, `"5.5"` cho `int` → lỗi 422; default **không** được validate (trừ `validate_default=True`); default mutable được copy mỗi instance. `@dataclass` cấm default `list` (phải `field(default_factory=list)`) và không validate kiểu.

### Rate limiting, retry, circuit breaker, throughput

| Thuật toán | Ý tưởng | Ưu / nhược |
|---|---|---|
| Fixed window | `INCR key:<phút>` + `EXPIRE` | Đơn giản; **burst 2× ở ranh giới** (limit 100/phút → 200 trong ~1 giây) |
| Sliding window log | Redis ZSET: `ZREMRANGEBYSCORE` bỏ cũ, `ZCARD`, `ZADD` (Lua/MULTI cho atomic) | Chính xác; tốn RAM theo số request |
| Sliding window counter | Nội suy 2 cửa sổ | Xấp xỉ, rẻ |
| Token bucket | Capacity C, refill r/s | Cho burst C, tốc độ dài hạn r. Max trong T giây = C + r·T |
| Leaky bucket | Hàng đợi chảy đều | Làm mượt output |

- **Exponential backoff**: `wait_k = min(cap, base·2^(k−1))`. Full jitter: `random(0, wait_k)` (kỳ vọng = wait_k/2) → tránh *thundering herd*. Tôn trọng header `Retry-After`.
- Retry chỉ lỗi tạm thời (timeout, 429, 5xx), **không** retry 400/401/422. Luôn đặt timeout + giới hạn số lần.
- **Idempotency key**: client gửi key, server lưu kết quả theo key (Redis `SET NX` + TTL) → retry không tạo trùng.
- **Circuit breaker**: CLOSED → (lỗi vượt ngưỡng) → OPEN (fail-fast/fallback provider khác) → hết cooldown → HALF-OPEN (thử vài request) → CLOSED/OPEN.
- **Little's Law**: L = λ·W → throughput λ = concurrency / latency. Ví dụ 96 slot, 4 s/call → 24 req/s = 1440 req/phút.
- Queue/worker (Celery, RQ, Kafka) cho batch: at-least-once (`acks_late`) → task phải **idempotent**.

### Redis cho AI backend

| Kiểu | Dùng cho |
|---|---|
| String | cache response, counter rate limit (`INCR`), lock |
| Hash | session, metadata object |
| List | queue đơn giản (`LPUSH`/`BRPOP`) |
| Set / Sorted set (ZSET) | dedup; leaderboard, sliding window log, delay queue |
| Stream | log sự kiện bền, consumer group, ack |
| Pub/Sub | broadcast fire-and-forget, **không lưu**, subscriber offline là mất |

- **Cache-aside**: đọc cache → miss thì đọc nguồn → ghi cache với TTL. Latency kỳ vọng = h·t_hit + (1−h)·(t_check + t_source + t_set).
- **Cache stampede**: key hot hết hạn → hàng trăm request cùng gọi LLM. Chống: lock `SET k token NX PX`, TTL jitter, early/probabilistic refresh, stale-while-revalidate.
- **Lock đúng cách**: `SET lock token NX PX 30000` (1 lệnh atomic, không dùng `SETNX`+`EXPIRE` rời); release bằng Lua: `if GET == token then DEL`. Nhiều node: Redlock (còn tranh cãi), cần fencing token khi yêu cầu an toàn tuyệt đối.
- **Eviction** (`maxmemory-policy`): mặc định `noeviction` (ghi bị lỗi OOM). `allkeys-lru/lfu` cho cache thuần; `volatile-*` chỉ evict key có TTL → không còn key TTL thì hành xử như noeviction. LRU/LFU của Redis là xấp xỉ (sampling). LFU từ Redis 4.0.
- Redis single-thread thực thi lệnh → tránh `KEYS *`, dùng `SCAN`.

### MongoDB & Elasticsearch

**MongoDB** (document store, BSON):
- Mô hình document: embed khi dữ liệu luôn đọc cùng nhau và giới hạn kích thước (document ≤ 16 MB); reference khi quan hệ lớn/không giới hạn (vd hàng triệu message).
- Compound index `{a:1, b:-1}` hỗ trợ query theo **prefix**: `a`, `a+b`; không seek được nếu chỉ lọc `b`. Sort có thể duyệt index **ngược** chiều. Quy tắc **ESR**: Equality → Sort → Range khi thiết kế thứ tự field.
- Aggregation pipeline: `$match` (đặt sớm để dùng index) → `$group` → `$sort` → `$project`, `$lookup` (join), `$unwind`.
- `explain()` xem `IXSCAN` vs `COLLSCAN`.

**Elasticsearch**:
- **Inverted index**: term → danh sách document. Chấm điểm mặc định BM25.
- `text`: qua **analyzer** (tokenizer + filter: lowercase, stop, stemming) → dùng `match` cho full-text. `keyword`: lưu nguyên văn, phân biệt hoa thường → dùng `term`, filter, aggregation, sort. Dynamic mapping cho string tạo `text` + sub-field `.keyword`.
- Lỗi kinh điển: `term` lên field `text` với chuỗi gốc có hoa/khoảng trắng → không khớp.
- **Shard/replica**: tổng bản shard = primary × (1 + replica). Số primary cố định khi tạo index (đổi phải reindex/split/shrink); replica đổi được bất kỳ lúc nào; replica không nằm cùng node với primary của nó.
- Hybrid search: BM25 + kNN `dense_vector`, gộp bằng RRF.

### Docker, GitLab CI, Git

**Docker**
- Layer cache: layer bị invalidate thì mọi layer sau build lại → `COPY requirements.txt` + `pip install` **trước**, `COPY . .` **sau**. BuildKit: `RUN --mount=type=cache,target=/root/.cache/pip`.
- Multi-stage: builder (gcc, build deps) → runtime (`python:3.12-slim`, chỉ copy venv/wheel). `.dockerignore`: `.git`, `.venv`, data, `.env`.
- Secrets: không dùng `ARG`/`ENV`/`COPY .env` (lộ trong layer/history); dùng `--mount=type=secret` hoặc inject lúc runtime.
- `ENTRYPOINT` = chương trình cố định; `CMD` = tham số mặc định (bị thay bởi tham số `docker run`). Exec form `["uvicorn", ...]` để process là PID 1 nhận SIGTERM; shell form chạy qua `/bin/sh -c` → không chuyển tiếp signal.
- GPU: host cài NVIDIA driver + **NVIDIA Container Toolkit**, chạy `--gpus all`; image chỉ chứa CUDA user-space; driver host phải hỗ trợ version CUDA của image. K8s: device plugin, request `nvidia.com/gpu`.
- Container **stateless**, chạy user non-root, có `HEALTHCHECK`/readiness probe.

**GitLab CI**: `stages` chạy tuần tự, job cùng stage chạy song song; `needs:` tạo DAG. `artifacts` = output của job truyền cho job sau (đảm bảo); `cache` = tăng tốc dependency (best-effort, có thể không có). Biến bí mật: CI/CD variables masked/protected.

**Git**: `merge` giữ lịch sử + merge commit; `rebase` viết lại commit (hash mới) → không rebase branch đã chia sẻ, nếu cần dùng `push --force-with-lease`. `cherry-pick` copy thay đổi của 1 commit thành commit **mới** (hay dùng cho hotfix sang release branch).

### Pytest cho AI service

- Fixture scope: `function` (mặc định) < `class` < `module` < `package` < `session`. Scope rộng = nhanh nhưng dễ rò state giữa test.
- `yield` fixture: code sau `yield` là teardown.
- `@pytest.mark.parametrize` xếp chồng → **tích Descartes** (3 × 4 = 12 case).
- **Mock LLM call**: `patch` ở **nơi được dùng** (`app.service.call_llm` nếu service đã `from app.llm import call_llm`). Hàm async → `AsyncMock` (từ 3.8 `patch` tự tạo AsyncMock cho async function). FastAPI: `app.dependency_overrides[get_llm] = lambda: FakeLLM()`.
- `monkeypatch.setenv/setattr` tự hoàn tác sau mỗi test.
- pytest-asyncio: `@pytest.mark.asyncio` hoặc `asyncio_mode = auto`; test API bằng `httpx.AsyncClient` + `ASGITransport` hoặc `TestClient`.
- Test output LLM: không assert chuỗi chính xác; assert schema (Pydantic), thuộc tính, hoặc dùng bộ eval riêng. Đặt `temperature=0` không đảm bảo deterministic tuyệt đối.
- Phân loại: unit (mock hết I/O) → integration (Redis/Mongo thật trong service container của CI) → e2e/eval.

### MLOps: registry, tracking, triển khai, giám sát

| Thành phần | Công cụ | Vai trò |
|---|---|---|
| Experiment tracking | MLflow, W&B | log params, metrics, artifacts mỗi run |
| Model registry | MLflow Model Registry | version, alias/stage, lineage, rollback |
| Data versioning | DVC, lakeFS | version dataset theo commit Git (DVC lưu pointer `.dvc`, dữ liệu ở remote storage) |
| Feature store | Feast | cùng định nghĩa feature cho offline train & online serve |
| Serving | Triton, vLLM, TorchServe, BentoML | batching, multi-model, GPU |

- **Training-serving skew**: tiền xử lý/feature khác giữa train và serve → dùng chung code/feature store, log input serving để so sánh.
- **Data drift** P(X) đổi; **concept drift** P(Y|X) đổi. Đo: PSI, KS test, khoảng cách embedding; drift ≠ chắc chắn giảm accuracy → cần nhãn/feedback.
- Triển khai: **shadow** (chạy song song, không trả kết quả cho user) → **canary** (1% → 10% → 100%, theo dõi metric) / **blue-green** (2 môi trường, switch toàn bộ, rollback tức thì) / A/B test (so sánh metric kinh doanh).
- **ONNX**: `torch.onnx.export` nhớ `dynamic_axes` cho batch/seq; chạy bằng ONNX Runtime (CPU/GPU/…), có thể convert tiếp TensorRT. **Triton**: backends TensorRT/ONNX/PyTorch/Python, **dynamic batching**, concurrent model instances.
- **Kubernetes HPA**: `desired = ceil(current × currentMetric / targetMetric)` (tolerance mặc định 10%, scale-down có stabilization window). Với LLM/GPU nên scale theo queue length / in-flight requests (custom metrics, KEDA) thay vì CPU.

### Bảo mật API AI & n8n

**Bảo mật**
- Secrets: secret manager (Vault, AWS/GCP Secret Manager, K8s Secret + encryption), CI masked variables; không commit, không bake vào image, xoay vòng key.
- **Prompt injection** (trực tiếp và gián tiếp qua tài liệu RAG, web, email): không chặn triệt để bằng system prompt. Phòng thủ nhiều lớp: least privilege cho tool, **authz ở backend** (lấy `user_id` từ token, không từ output LLM), allow-list tool/tham số, human-in-the-loop cho hành động nguy hiểm, validate output (schema), tách dữ liệu không tin cậy.
- PII: che/mask trước khi log hoặc gửi provider, chính sách retention, mã hoá at-rest/in-transit.
- Rate limit + quota theo API key/user chống lạm dụng chi phí; giới hạn `max_tokens`, kích thước input.

**n8n**
- Nền tảng workflow automation (trigger, webhook, cron, hàng trăm node tích hợp, node AI Agent/LLM, node Code JS/Python), self-host được, giấy phép fair-code (Sustainable Use License).
- Dùng khi: luồng tích hợp nhiều SaaS, lưu lượng thấp–vừa, người không chuyên code cần sửa, prototype nhanh.
- Viết code khi: throughput cao, logic phức tạp, cần unit test/CI/code review/versioning chặt, latency thấp. Kết hợp: n8n gọi API FastAPI qua node HTTP Request.

### C/C++ nền & Big-O, cấu trúc dữ liệu trong hệ AI

- **Stack**: biến cục bộ, cấp phát/giải phóng tự động khi vào/ra hàm, rất nhanh, kích thước giới hạn (thường vài MB). **Heap**: `new`/`malloc`, sống tới khi giải phóng; dùng RAII (`std::vector`, `std::unique_ptr`) để tránh leak.
- Trả về con trỏ/tham chiếu tới biến cục bộ → **dangling pointer**, undefined behavior (có thể vẫn "chạy đúng" ngẫu nhiên).
- Con trỏ: `p[i] == *(p+i)`; `nullptr`; truyền tham chiếu `const T&` để tránh copy lớn.

| Cấu trúc | Độ phức tạp | Dùng trong hệ AI |
|---|---|---|
| Hash map | O(1) trung bình tra cứu | vocab token→id, cache, dedup, inverted index |
| Min-heap kích thước k | O(n log k) cho top-k | rerank top-k, ANN search, `heapq.nlargest` |
| Sort toàn bộ | O(n log n) | khi cần cả thứ tự đầy đủ |
| Binary search | O(log n) | tìm trong mảng đã sort (log2(10^6) ≈ 20) |
| Quickselect / `np.argpartition` | O(n) trung bình | top-k không cần thứ tự |
| Trie | O(độ dài chuỗi) | tokenizer, autocomplete |

- Tỷ số sort vs heap top-k: log(n)/log(k); n = 10^6, k = 100 → 6/2 = 3.


---

## 🏗️ ML System Design, Ranking & Experimentation

### Khung trả lời ML system design (8 bước)

1. **Làm rõ yêu cầu**: mục tiêu kinh doanh, người dùng, quy mô (user, item, peak QPS), latency/cost budget, ràng buộc pháp lý/dữ liệu.
2. **Metric**: offline (recall@K, nDCG, PR-AUC, calibration) ↔ online (CTR, conversion, GMV, retention) + **guardrail** (latency p99, error rate, complaint).
3. **Dữ liệu & nhãn**: nguồn, label explicit/implicit, label delay, selection bias, chia train/test **theo thời gian**.
4. **Feature**: user/item/context/cross, batch vs real-time, point-in-time correctness.
5. **Model**: heuristic/baseline → GBDT/two-tower → deep/LLM; giải thích trade-off.
6. **Serving**: batch/online/streaming, cascade, caching, fallback, latency budget, capacity.
7. **Monitoring**: data/feature drift, prediction drift, KPI, freshness, tỉ lệ fallback.
8. **Iteration**: shadow → canary → A/B, retrain, data flywheel (sản phẩm tốt hơn → nhiều tương tác → nhiều nhãn → model tốt hơn).

**Lỗi hay gặp:** nhảy vào model ngay; không nói metric; quên baseline; quên cold start và feedback loop; không nêu fallback.

### Batch vs online vs streaming

| Kiểu | Khi nào | Ưu | Nhược |
|---|---|---|---|
| Batch (offline) | Biết trước tập cần chấm, không cần tức thì (email hằng ngày, gợi ý tính sẵn) | Rẻ, batch lớn, dễ kiểm tra trước | Kết quả cũ, tốn công cho user không hoạt động |
| Online (đồng bộ) | Cần quyết định trong request (fraud lúc thanh toán, search) | Dùng ngữ cảnh mới nhất | Ràng buộc latency, cần autoscale, fallback |
| Streaming | Feature/score cập nhật theo event (velocity, session) | Feature tươi, độ trễ giây | Phức tạp (Kafka/Flink), exactly-once, late events |

Kiến trúc thường gặp là lai: embedding item tính batch, feature velocity tính streaming, ranking online.

### Feature store & training-serving skew

- **Online store**: KV latency thấp (Redis/DynamoDB), giữ giá trị mới nhất → serving.
- **Offline store**: lịch sử có timestamp → dựng tập train bằng **point-in-time join** (chỉ lấy giá trị có `ts ≤ event_ts`).
- **Nguồn skew**: hai codebase tính feature khác nhau; join snapshot hiện tại (leakage tương lai); feature online bị trễ/thiếu; xử lý null khác nhau; phân phối thay đổi (drift).
- **Phòng tránh**: định nghĩa feature một lần; log feature lúc serving để train lại ("log-and-wait"); so sánh phân phối train vs serve (PSI, KS); unit test cho transform.
- Dấu hiệu leakage: AUC offline cao bất thường (0.97+) rồi rơi mạnh online.

### Latency budget, capacity, cost

- **Latency budget**: bước song song lấy max, tuần tự thì cộng; cộng các p99 là ước lượng bảo thủ.
- **Little's law**: `L = λ × W` → số request đồng thời = QPS × latency (giây).
- **Số replica** = ⌈QPS × latency / (concurrency_mỗi_replica × utilization_mục_tiêu)⌉, cộng thêm N+1 dự phòng.
- **Batch tuần tự**: số ứng viên tối đa = ⌊budget / t_batch⌋ × batch_size (batch không đầy vẫn tốn trọn thời gian).
- **Cost per request (API)** = in_tok × giá_in + out_tok × giá_out.
- **Hoà vốn self-host**: N* = chi phí cố định/tháng (GPU 24/7 + **người vận hành**) / cost_per_request_API. Kiểm tra utilization thật và chất lượng model.
- Ví dụ: 800 QPS × 0.15 s = 120 đồng thời; 16 slot × 70% = 11.2 → 11 replica.

### Cascade, cold start, feedback loop, deploy an toàn

- **Cascade**: retrieval (recall@K, ANN, two-tower) → ranking (nDCG, GBDT/DNN, cross features) → re-ranking (đa dạng, business rule, fairness). Recall của tầng đầu là trần chất lượng; recall nối tiếp = tích recall các tầng.
- **Cold start**: user mới → popular theo segment, onboarding, session features; item mới → content embedding, exploration slot.
- **Feedback loop**: model chỉ thấy nhãn trên cái nó hiển thị/cho qua → cần exploration, holdout ngẫu nhiên, log propensity.
- **Label delay**: chỉ train trên dữ liệu đã qua label maturity window, hoặc dùng delayed-feedback model.
- **Deploy**: shadow (sức khoẻ, không đo được hành vi user) → canary (lỗi, guardrail) → A/B (KPI) → ramp-up; luôn có rollback và fallback (popular, cache, model nhẹ, circuit breaker).

### Case study nhanh

| Bài toán | Điểm then chốt |
|---|---|
| Fraud | < 100 ms, velocity feature streaming, rule + ML, split theo thời gian, label delay (chargeback), ngưỡng theo chi phí, selective labels |
| Document AI/OCR | detect → recognize → layout/KIE → validate rule → human-in-the-loop theo confidence; ngưỡng theo tổng chi phí review + lỗi |
| Chatbot CSKH RAG | tri thức tĩnh → RAG; dữ liệu cá nhân/động → tool/API có xác thực; guardrail, handoff sang người, eval faithfulness |
| Search | query understanding → hybrid retrieval → LTR → rerank; metric nDCG/MRR, zero-result rate, CTR@k |
| Moderation | classifier rẻ recall cao → LLM/model nặng → người; recall nhân dồn, sample ngẫu nhiên bài đã đăng để đo bỏ sót |
| RecSys | two-tower + ANN → ranker đa mục tiêu → re-rank đa dạng; cold start, position bias |

### Learning-to-rank & metric

| Họ | Loss trên | Ví dụ |
|---|---|---|
| Pointwise | từng doc | regression/logloss |
| Pairwise | cặp (d_i ≻ d_j) | RankNet `P = σ(s_i − s_j)` |
| Listwise | cả danh sách | ListNet, ListMLE, LambdaRank (lai pairwise–listwise) |

- **LambdaRank**: λ_ij = gradient RankNet × |ΔNDCG_ij| (nDCG không khả vi → định nghĩa thẳng gradient). **LambdaMART** = lambda + GBDT (LightGBM `lambdarank`).
- `DCG@k = Σ (2^rel − 1)/log2(i + 1)`; nDCG = DCG/IDCG, với IDCG từ **mọi** doc relevant đã biết.
- MRR = mean(1/rank của kết quả đúng đầu tiên); MAP = mean AP (nhị phân).
- Precision@k tăng không kéo theo nDCG tăng.

### Position bias, click model, IPS, interleaving, exploration

- **Examination hypothesis**: P(click) = P(xem | vị trí) × P(hấp dẫn). Hấp dẫn ≈ CTR / propensity.
- **Click model**: position-based (PBM), cascade (người dùng duyệt từ trên xuống, dừng khi click).
- **Ước lượng propensity**: randomization (swap/RandPair), EM trên log; KHÔNG dùng CTR theo vị trí thô.
- **IPS**: trọng số 1/p_k → không chệch nếu p đúng, variance cao → clip / self-normalized. Không cứu được item chưa từng hiển thị.
- **Interleaving** (team-draft): nhạy hơn A/B nhiều lần cho so sánh ranker, nhưng chỉ cho preference, không cho lift KPI.
- **Exploration**: ε-greedy, UCB, Thompson; log propensity để off-policy eval.
- **Offline↑ online↔**: nhãn sinh từ policy cũ, metric lệch KPI, novelty, hiệu ứng toàn trang.

### Anomaly detection cheat-sheet

| Phương pháp | Ý tưởng | Lưu ý |
|---|---|---|
| z-score | \|x − μ\|/σ > 3 | Bị **masking**; với n nhỏ \|z\| ≤ √(n−1); dùng modified z (median, MAD, > 3.5) |
| IQR (Tukey) | ngoài [Q1 − 1.5·IQR, Q3 + 1.5·IQR] | Robust; quy ước quantile khác nhau |
| Isolation Forest | path ngắn = dễ cô lập = bất thường; s = 2^(−E[h]/c(n)) | Không cần nhãn, nhanh, sub-sample 256 |
| LOF | mật độ cục bộ so với láng giềng; ≈1 bình thường, ≫1 bất thường | Hợp cụm mật độ khác nhau; O(n²) |
| One-Class SVM | biên bao dữ liệu bình thường; ν = cận trên tỉ lệ outlier | Nhạy scale & kernel |
| Autoencoder | reconstruction error cao = bất thường | Train trên data sạch; dễ học luôn anomaly |

- Loại: **point**, **contextual** (theo giờ/mùa), **collective** (chuỗi).
- Đánh giá ít nhãn: precision@k (k = năng lực review), PR-AUC, recall trên case đã biết, backtest sự cố.
- Ngưỡng theo chi phí: tối thiểu `#alert × c_review + #miss × c_miss`.

### A/B testing: thiết kế & thống kê

- **Đơn vị randomize** = đơn vị phân tích (user); hash ổn định; pre-register primary/guardrail/thời lượng; chạy trọn chu kỳ tuần.
- **Sample size**: `n ≈ 16σ²/δ²` mỗi nhóm (α = 0.05 hai phía, power 80%); tỉ lệ: σ² = p(1 − p). Ví dụ p = 5%, δ = 0.5 điểm → 30 400/nhóm.
- **MDE**: `δ = √(16σ²/n)` với n **mỗi nhóm**; n ∝ 1/δ².
- **Multiple testing**: FWER = 1 − (1 − α)^m khi m kiểm định độc lập và mọi H0 đúng (m = 20 → 64%); Bonferroni α/m; Holm; BH cho FDR.
- **Peeking**: nhìn 10 lần ≈ 19% false positive → fixed horizon hoặc sequential (O'Brien–Fleming, mSPRT).
- **SRM**: χ² = Σ(O − E)²/E, df = 1; 50 600 vs 49 400 → χ² = 14.4, p ≈ 0.00015 → dừng, tìm bug.
- **CUPED**: Y − θ(X − X̄), θ = cov/var, variance × (1 − ρ²) → ρ = 0.6 giảm 36% cỡ mẫu.
- **A/A**: tỉ lệ false positive kỳ vọng ≈ α; p-value phân phối đều.
- **Novelty** (lift giảm dần) vs **primacy** (ban đầu tệ, tốt dần): xem effect theo thời gian, tách user mới/cũ.

### Interference, bandit & khi nào không A/B

- **SUTVA bị vi phạm**: marketplace (chung nguồn cung), mạng xã hội, đấu giá quảng cáo chung ngân sách → chệch, tăng n không sửa được.
- Thiết kế: **switchback** (khu vực × khung giờ), **cluster/geo randomization** (variance ↑), ego-network clusters.
- **Bandit** vs A/B: bandit giảm regret, hợp chiến dịch ngắn/nhiều arm; A/B cho ước lượng hiệu ứng không chệch và CI chuẩn.
- ε-greedy: arm tốt nhất nhận 1 − ε + ε/K (ε = 0.1, K = 4 → 92.5%). Thompson: sample posterior (Beta) rồi argmax.
- Offline eval → online A/B mismatch: luôn xác nhận bằng thí nghiệm online trước khi ship thay đổi lớn.

### Causal inference khi không thí nghiệm được

| Phương pháp | Giả định then chốt | Ghi nhớ |
|---|---|---|
| Propensity (matching/IPW) | Không có confounder ẩn, có overlap | e(X) = P(T=1\|X); kiểm tra balance; không đưa biến sau treatment |
| DiD | Parallel trends | (A_sau − A_trước) − (B_sau − B_trước); kiểm tra event study |
| IV | Relevance + exclusion | Ước lượng LATE cho compliers; encouragement design |
| RDD | Không thao túng quanh ngưỡng | Hiệu ứng cục bộ tại ngưỡng |
| Uplift modeling | Có cả treatment & control | Nhắm persuadables; Qini curve; đừng nhắm "sure things" |

- **Correlation ≠ causation**: self-selection, confounder (mức độ gắn bó).
- **Simpson's paradox**: thắng từng tầng nhưng thua tổng do phân phối tầng khác nhau → so theo tầng/chuẩn hoá.


---

## ∑ Nền tảng: Toán & Thống kê cho AI

### Vector, norm & similarity

- **Dot product** a·b = Σ aᵢbᵢ = ‖a‖‖b‖·cos θ. Dương: cùng phía, 0: **trực giao**, âm: ngược phía.
- **Norm**: ‖v‖₁ = Σ|vᵢ| · ‖v‖₂ = √Σvᵢ² · ‖v‖∞ = max|vᵢ|. Ví dụ (3, −4, 0, 12): 19 / 13 / 12.
- **Cosine** = a·b / (‖a‖₂‖b‖₂) ∈ [−1, 1], chỉ đo hướng. Embedding đã L2-normalize → cosine = dot product (vector DB dùng inner product).
- Khoảng cách Euclid với vector đơn vị: ‖a − b‖² = 2 − 2cos θ → xếp hạng theo L2 và cosine **trùng nhau**.
- Pearson correlation = cosine của hai vector đã trừ mean.
- Liên hệ: L1 → Lasso/sparse; L2 → weight decay, gradient clipping; L∞ → adversarial ε-ball.

### Ma trận: shape, chi phí, tính chất

| Phép | Shape | Ghi chú |
|---|---|---|
| (m×k)·(k×n) | m×n | m·k·n phép nhân, ≈ 2mkn FLOP |
| outer u·vᵀ (u∈ℝᵐ, v∈ℝⁿ) | m×n | rank 1 |
| Q·Kᵀ (T×d)(d×T) | T×T | O(T²d) mỗi head |

- Nhân n×n thông thường O(n³). LLM forward ≈ 2·N FLOP/token.
- **Không giao hoán**: AB ≠ BA. (AB)ᵀ = BᵀAᵀ, (AB)⁻¹ = B⁻¹A⁻¹.
- 2×2: det[[a, b], [c, d]] = ad − bc; nghịch đảo = (1/det)·[[d, −b], [−c, a]]. det = 0 ⇔ rank < n ⇔ không khả nghịch ⇔ Ax = b vô nghiệm hoặc vô số nghiệm.
- **Rank** = số hàng/cột độc lập tuyến tính.
- **Broadcasting**: so từ chiều cuối, bằng nhau hoặc một bên = 1; thiếu chiều thì thêm 1 ở đầu. Bẫy kinh điển: (32, 1) − (32,) → (32, 32).

### Eigen, SVD, PCA, low-rank (LoRA)

- **Eigen**: Av = λv. 2×2: λ² − (trace)·λ + det = 0; Σλ = trace, Πλ = det. Ví dụ [[4, 1], [2, 3]] → λ = 5 (v = (1, 1)), λ = 2 (v = (1, −2)).
- Ma trận đối xứng: trị riêng thực, vector riêng trực giao.
- **SVD**: A = UΣVᵀ cho mọi ma trận; σᵢ = √eig(AᵀA) ≥ 0. σᵢ = |λᵢ| chỉ đúng với ma trận đối xứng (tổng quát: normal), không đúng cho ma trận bất kỳ.
- **Eckart–Young**: giữ k singular value lớn nhất = xấp xỉ rank-k tốt nhất.
- **PCA** = SVD của X đã **center**; trục chính = cột V, phương sai giải thích = σᵢ²/(n − 1).
- **LoRA**: ΔW = B·A, B ∈ ℝ^(d×r), A ∈ ℝ^(r×k) → r(d + k) tham số, rank ≤ r. d = k = 4096, r = 16 → 131,072 (≈ 0.78% của W).

### Đạo hàm & backprop

| Hàm | Đạo hàm | Ghi chú |
|---|---|---|
| σ(z) | σ(1 − σ) | max 0.25 tại 0 → vanishing |
| tanh z | 1 − tanh² | max 1, output (−1, 1) |
| ReLU | 0 (z<0), 1 (z>0) | dead ReLU |
| ln x | 1/x | |
| eˣ | eˣ | |

- **Chain rule**: ∂L/∂w = ∂L/∂h · ∂h/∂z · ∂z/∂w. Backprop = chain rule ngược, tái sử dụng gradient upstream.
- **Softmax + cross-entropy**: ∂L/∂z = p − y (tổng = 0). Logistic + BCE: ∂L/∂w = (p − y)·x. Truyền **logits** vào `CrossEntropyLoss`.
- **Jacobian** f: ℝⁿ→ℝᵐ: m×n. **Hessian**: n×n đối xứng; PD → min, ND → max, trị riêng trái dấu → saddle.
- **Taylor bậc 1**: f(x + Δ) ≈ f(x) + f'(x)Δ → cơ sở của GD: L(θ − ηg) ≈ L(θ) − η‖g‖².

### Tối ưu: GD, learning rate, momentum, Lagrange

- GD: θ ← θ − η∇L. f = (w − 3)², w₀ = 0, η = 0.1 → 0.6 → 1.08.
- **Ổn định**: với hàm bậc hai có độ cong lớn nhất L, cần η < 2/L. f = w²: η < 1; η = 1 dao động, η > 1 phân kỳ. Đổi dấu ≠ phân kỳ.
- **Lồi**: f'' ≥ 0 / Hessian PSD → local min = global min (linear/logistic regression). Neural net: không lồi, saddle phổ biến hơn local min tồi.
- **Batch GD** chính xác nhưng đắt; **SGD/mini-batch** nhiễu không chệch, rẻ, giúp thoát saddle.
- **Momentum** v ← βv + g: làm mượt, giữ quán tính; gradient không đổi → v → g/(1 − β) (β = 0.9 → 10g). Adam = momentum + chia √v (thích nghi theo từng tham số).
- **Lagrange**: tối ưu có ràng buộc g = 0 ⇒ ∇f = λ∇g; λ = độ nhạy của giá trị tối ưu theo ràng buộc. Max entropy + ràng buộc → softmax/Gibbs.

### Xác suất cơ bản & Bayes

- P(A | B) = P(A ∩ B)/P(B). **Độc lập**: P(A ∩ B) = P(A)P(B). **Xung khắc** (P > 0) ⇒ KHÔNG độc lập.
- **Bayes**: P(H | E) = P(E | H)P(H)/P(E); posterior ∝ likelihood × prior.
- Bài kinh điển: prevalence 1%, sensitivity 99%, specificity 95% → P(bệnh | +) = 0.0099/0.0594 ≈ **16.7%** (không phải 99%: base rate neglect).
- Mẹo: đếm trên 10,000 người thay vì dùng công thức.
- Liên hệ: precision trên lớp hiếm thấp dù recall/specificity cao; Naive Bayes; diễn giải p-value sai cũng là nhầm P(E | H) với P(H | E).

### Phân phối, kỳ vọng, phương sai, CLT

| Phân phối | E | Var | Dùng ở đâu |
|---|---|---|---|
| Bernoulli(p) | p | p(1−p) | BCE |
| Binomial(n,p) | np | np(1−p) | pass@k, đếm thành công |
| Poisson(λ) | λ | λ | số request/phút |
| Uniform[a,b] | (a+b)/2 | (b−a)²/12 | random init |
| N(μ,σ²) | μ | σ² | init, nhiễu diffusion, MSE |
| Categorical(p) | – | – | softmax, next-token |

- Var(aX + c) = a²Var(X); Var(X ± Y) = VarX + VarY ± 2Cov. Độc lập ⇒ Cov = 0 (ngược lại sai: Y = X²).
- ρ = Cov/(σxσy) ∈ [−1, 1], chỉ đo tuyến tính, bất biến khi đổi đơn vị.
- 68-95-99.7 cho ±1σ/±2σ/±3σ. Một đuôi > μ + 2σ ≈ 2.5%.
- **LLN**: x̄ → μ. **CLT**: x̄ ≈ N(μ, σ²/n) với mọi phân phối có phương sai hữu hạn; dữ liệu gốc KHÔNG trở thành chuẩn.
- q·k của vector d chiều (thành phần độc lập, mean 0, var 1) có Var = d → chia √d trong attention.

### MLE & MAP ↔ loss & regularization

- **MLE**: θ̂ = argmax Σ log p(xᵢ | θ) = argmin NLL.
  - Bernoulli: p̂ = k/n; NLL = BCE. Categorical: NLL = cross-entropy (train LLM).
  - Gaussian: μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ − x̄)² (chệch); với σ cố định, MLE ⇔ MSE.
- **MAP**: argmax [log p(D | θ) + log p(θ)].
  - Prior Gaussian N(0, τ²) → **L2/Ridge/weight decay**, λ = σ²/τ².
  - Prior Laplace → **L1/Lasso** (sparse).
  - Prior đều → MAP = MLE.
- Phương sai mẫu không chệch chia n − 1 (Bessel). NumPy `std` mặc định ddof=0, pandas/torch mặc định chia n − 1.

### Lý thuyết thông tin cho LLM

- **Entropy** H(P) = −Σp log p (bit nếu log₂, nats nếu ln). (1/2, 1/4, 1/8, 1/8) → 1.75 bit; đều K lớp → log₂K.
- **Cross-entropy** H(P, Q) = −Σp log q = H(P) + KL(P ‖ Q) → train bằng CE = minimize forward KL.
- **KL** = Σp log(p/q) ≥ 0, = 0 ⇔ P = Q, **bất đối xứng**, = ∞ nếu q = 0 mà p > 0. Ví dụ P = (0.5, 0.5), Q = (0.9, 0.1): 0.511 vs đảo chiều 0.368 nats.
- Forward KL(P‖Q): mode-covering (MLE). Reverse KL(Q‖P): mode-seeking (VI, phạt KL trong RLHF).
- **Mutual information** I(X; Y) = H(X) − H(X|Y) = KL(p(x,y) ‖ p(x)p(y)); = 0 ⇔ độc lập; InfoNCE tối đa cận dưới của MI.
- **Perplexity** = exp(CE trung bình/token) cùng cơ số: loss 2.0 nats → e² ≈ 7.39.
- **Temperature**: softmax(z/T); T < 1 nhọn, T > 1 phẳng, argmax không đổi.
- **Log-sum-exp**: LSE(z) = m + log Σe^(zᵢ − m), m = max z; softmax(z) = softmax(z − c).

### Thống kê mô tả & suy diễn

- Mean nhạy outlier; **median** bền (breakdown 50%); mode = giá trị hay gặp nhất. MSE ↔ mean, MAE ↔ median.
- Std mẫu s = √(Σ(x − x̄)²/(n − 1)). **z** = (x − μ)/σ; Φ(1.5) ≈ 0.933, Φ(1.96) = 0.975.
- **SE** = s/√n; CI 95% ≈ x̄ ± 1.96·SE. CI nói về **tham số** (mean), không phải từng quan sát. Giảm một nửa CI cần gấp 4 lần mẫu.
- **p-value** = P(dữ liệu cực đoan như vậy | H₀), KHÔNG phải P(H₀ đúng). p lớn ≠ chứng minh H₀.
- **Loại I** (α): bác bỏ H₀ đúng – false positive. **Loại II** (β): bỏ sót hiệu ứng thật. Power = 1 − β.
- **A/B** hai tỷ lệ: z = (p₂ − p₁)/√(p̂(1 − p̂)(1/n₁ + 1/n₂)); |z| > 1.96 → có ý nghĩa ở 5%. Tránh peeking và multiple testing.
- **Correlation ≠ causation** (confounder). **Simpson's paradox**: xu hướng trong từng nhóm đảo ngược khi gộp → so trên cùng eval set/phân tầng.
- **Sampling/selection bias**: mẫu tự nguyện, survivorship, eval contamination.

### Số thực trên máy tính

| Kiểu | Sign/Exp/Mantissa | Max | eps (quanh 1) | Ghi chú |
|---|---|---|---|---|
| fp32 | 1/8/23 | ~3.4e38 | 2⁻²³ ≈ 1.2e-7 | master weights |
| fp16 | 1/5/10 | 65504 | 2⁻¹⁰ ≈ 9.8e-4 | cần loss scaling |
| bf16 | 1/8/7 | ~3.4e38 | 2⁻⁷ ≈ 7.8e-3 | range như fp32 |
| fp64 | 1/11/52 | ~1.8e308 | 2⁻⁵² | khoa học |

- Exponent → **range**, mantissa → **precision**.
- Overflow: exp(z) tràn fp32 khi z > ~88.7, fp16 khi z > ~11.1 → trừ max trong softmax.
- Underflow: gradient < ~6e-8 thành 0 trong fp16 → loss scaling.
- bf16: 256 + 1 = 256 (khoảng cách 2 trong [256, 512)) → accumulate ở fp32.
- Bộ nhớ: 7B tham số × 2 byte = 14 GB (bf16).


---

## 🧱 Nền tảng: ML & Deep Learning cốt lõi

### Bức tranh chung & các kiểu học

**AI ⊃ ML ⊃ DL.** AI: máy làm tác vụ cần trí thông minh (gồm cả luật viết tay). ML: học hàm từ dữ liệu thay vì lập trình luật. DL: ML bằng neural network nhiều lớp, tự học biểu diễn.

| Kiểu học | Dữ liệu | Ví dụ |
|---|---|---|
| Supervised | (x, y) có nhãn | phân loại spam, dự đoán giá nhà |
| Unsupervised | chỉ x | clustering, PCA, phát hiện bất thường |
| Self-supervised | nhãn tự sinh từ dữ liệu | next-token (GPT), masked LM (BERT), contrastive |
| Semi-supervised | ít nhãn + nhiều không nhãn | pseudo-labeling |
| Reinforcement | reward từ môi trường | game, robot, RLHF |

**Loại task** (hỏi 'output là gì?'): classification (nhãn rời rạc), regression (số liên tục), clustering (nhóm, không nhãn), ranking (thứ tự – NDCG/MAP), generation (sinh dữ liệu).

### Các cặp khái niệm dễ nhầm

| Cặp | Phân biệt | Ví dụ |
|---|---|---|
| Generative vs discriminative | học P(x, y) = P(x\|y)P(y) vs học thẳng P(y\|x) | Naive Bayes, GMM, HMM vs logistic regression, SVM, NN classifier |
| Parametric vs non-parametric | số tham số cố định vs độ phức tạp tăng theo dữ liệu | linear/logistic, NN vs kNN, cây không giới hạn, kernel SVM |
| Online vs batch learning | cập nhật dần theo từng mẫu/mini-batch đến liên tục vs train lại trên toàn bộ dữ liệu | SGD trên stream, recommender cập nhật realtime vs train định kỳ hằng tuần |
| Lazy vs eager | không train, tính lúc dự đoán vs học model trước | kNN vs hầu hết model khác |
| Parameter vs hyperparameter | học từ data vs chọn bằng validation | trọng số w vs learning rate, k, max_depth |

- 'Non-parametric' **không** có nghĩa là không có hyperparameter.
- 'Generative' trong ML cổ điển ≠ 'GenAI': Naive Bayes là generative.

### Quy trình: chia dữ liệu, baseline, giả định

- **Train**: fit trọng số. **Validation**: chọn hyperparameter/model/threshold/epoch. **Test**: dùng **một lần** cuối để báo cáo. Dữ liệu nhỏ → k-fold CV, vẫn giữ test riêng.
- **Baseline** trước tiên: majority class (`DummyClassifier`), mean/median (`DummyRegressor`), rule đơn giản, model tuyến tính.
- **No Free Lunch**: trung bình trên mọi bài toán, mọi thuật toán như nhau → không có model tốt nhất tuyệt đối; phải thử & đánh giá.
- **Inductive bias**: giả định giúp tổng quát hoá (CNN: locality + weight sharing; kNN: hàng xóm giống nhau; linear: quan hệ tuyến tính; Naive Bayes: độc lập có điều kiện).
- **i.i.d.**: các mẫu độc lập, train và test cùng phân phối. Vi phạm: time-series, nhiều mẫu cùng 1 user/bệnh nhân, dữ liệu deploy khác train.

| Distribution shift | Cái gì đổi | Ví dụ |
|---|---|---|
| Covariate | P(x), P(y\|x) giữ nguyên | ảnh scan → ảnh điện thoại |
| Label/prior | P(y), P(x\|y) giữ nguyên | mùa dịch tỷ lệ bệnh tăng |
| Concept | P(y\|x) | chiêu gian lận mới, định nghĩa spam đổi |

### Confusion matrix & metric phân loại

| | Dự đoán + | Dự đoán − |
|---|---|---|
| **Thực tế +** | TP | FN (bỏ sót, lỗi loại II) |
| **Thực tế −** | FP (báo động giả, lỗi loại I) | TN |

Mẹo: chữ thứ hai = model đoán gì; chữ đầu = đoán đúng hay sai.

- Accuracy = (TP + TN)/N – vô nghĩa khi mất cân bằng.
- **Precision** = TP/(TP + FP) – 'trong các ca báo dương, bao nhiêu đúng'.
- **Recall / sensitivity / TPR** = TP/(TP + FN) – 'bắt được bao nhiêu ca dương'.
- **Specificity / TNR** = TN/(TN + FP); FPR = 1 − specificity.
- **F1** = 2PR/(P + R) (trung bình điều hoà, ≤ trung bình cộng). F-beta: β > 1 nghiêng về recall.
- **ROC**: TPR theo FPR khi quét threshold. **AUC** = P(score mẫu dương ngẫu nhiên > score mẫu âm ngẫu nhiên); 0.5 = ngẫu nhiên. Không phụ thuộc threshold. Lớp dương rất hiếm → xem thêm PR-AUC.
- Tăng threshold → thường precision ↑, recall ↓.

### Metric hồi quy & chọn metric

- **MAE** = mean|y − ŷ| – dễ hiểu, bền outlier.
- **MSE** = mean(y − ŷ)² – khả vi, hay dùng làm loss, phạt lỗi lớn.
- **RMSE** = √MSE – cùng đơn vị y; luôn RMSE ≥ MAE, chênh nhiều ⇒ có vài lỗi rất lớn.
- Ví dụ e = [1, 0, −2, 3]: MAE 1.5, MSE 3.5, RMSE 1.871.

| Tình huống | Ưu tiên |
|---|---|
| Bỏ sót đắt (sàng lọc bệnh, fraud) | Recall (kèm ràng buộc precision) |
| Báo động giả đắt (lọc spam) | Precision |
| Mất cân bằng nặng | PR-AUC, F1, recall@precision |
| Cần xếp hạng tốt, không phụ thuộc threshold | ROC-AUC |
| Lỗi lớn rất tệ | RMSE |
| Nhiều outlier | MAE / Huber |

### Overfitting, bias–variance & regularization

**Expected error = Bias² + Variance + Irreducible noise.**

| Triệu chứng | Chẩn đoán | Xử lý |
|---|---|---|
| Train tốt, val kém, gap lớn | Overfit / high variance | thêm data, augmentation, L1/L2, dropout, early stopping, model nhỏ hơn, bagging |
| Train kém ≈ val kém | Underfit / high bias | model mạnh hơn, thêm feature, giảm regularization, train lâu hơn, boosting |

- **Learning curve** (error theo lượng data): overfit → gap thu hẹp khi thêm data (thêm data có ích); underfit → hai đường hội tụ sớm ở mức cao (thêm data vô ích).
- **L2** (λΣw²): co trọng số nhỏ. **L1** (λΣ|w|): đẩy về đúng 0 → sparse. **Dropout**: tắt ngẫu nhiên neuron khi train (inverted dropout chia 1 − p). **Early stopping**: theo val loss + patience, khôi phục best checkpoint. **Data augmentation**: biến thể hợp lệ của dữ liệu train.
- kNN: k nhỏ → variance cao; k lớn → bias cao.

### Thuật toán ML kinh điển

| Thuật toán | Ý tưởng | Cần scale? | Ghi nhớ |
|---|---|---|---|
| Linear regression | min Σ(y − ŷ)²; normal equation w = (XᵀX)⁻¹Xᵀy | có (nếu regularize/GD) | giả định: tuyến tính, phần dư độc lập, phương sai đều, không đa cộng tuyến hoàn hảo |
| Logistic regression | p = σ(w·x + b), loss BCE | có | boundary **tuyến tính** w·x + b = 0; z là log-odds |
| kNN | vote k láng giềng gần nhất | **bắt buộc** | lazy, non-parametric |
| Naive Bayes | P(y\|x) ∝ P(y)ΠP(xᵢ\|y) | không | độc lập có điều kiện; Laplace smoothing; generative |
| Decision tree | split greedy theo Gini/entropy | không | sâu → overfit; dễ diễn giải |
| Random forest | bagging cây sâu + random feature | không | giảm variance, song song |
| Gradient boosting | cây nông tuần tự khớp residual | không | giảm bias, cần early stopping |
| SVM | max margin, support vector, kernel trick | có | C lớn → ít vi phạm, dễ overfit |
| k-means | gán cụm gần nhất ↔ cập nhật centroid = mean | có | chọn k trước, k-means++ |
| PCA | chiếu lên hướng phương sai lớn nhất | có | unsupervised, component = tổ hợp tuyến tính |

### Neural network cơ bản

- **Neuron**: z = Σwᵢxᵢ + b → a = activation(z). Perceptron dùng hàm bước, chỉ tách tuyến tính (không giải XOR).
- **Non-linearity**: thiếu activation, nhiều layer tuyến tính = 1 layer tuyến tính.
- **Activation**: ReLU max(0, z) mặc định cho layer ẩn; sigmoid cho xác suất nhị phân; softmax cho đa lớp; GELU trong Transformer.
- **Loss**: regression → MSE/MAE/Huber; multi-class → softmax + CE (= −log p_đúng); multi-label/binary → sigmoid + BCE. PyTorch `CrossEntropyLoss` nhận logits.
- **Backprop** = chain rule từ output ngược về input để tính ∂L/∂w; optimizer mới là thứ cập nhật trọng số.
- **Epoch** = 1 lượt toàn bộ data; **iteration** = 1 batch. Số iteration/epoch = ⌈N / batch⌉ (drop_last=False).
- **Optimizer**: SGD θ ← θ − ηg; Momentum v ← βv + g; RMSProp chia √E[g²]; Adam = momentum + RMSProp + bias correction (lr mặc định 1e-3). AdamW tách weight decay.
- **Learning rate**: lớn → dao động/NaN; nhỏ → chậm. Dùng warmup + scheduler.
- **Init**: không init hằng số (đối xứng). Xavier (tanh/sigmoid), He (ReLU).
- **Vanishing**: σ' ≤ 0.25 nhân dồn → dùng ReLU, He init, Norm, residual. **Exploding** → gradient clipping.

### Kiến trúc nền: CNN, RNN, attention, Transformer, autoencoder

- **CNN**: kernel trượt, weight sharing → params = k·k·C_in·C_out + C_out, không phụ thuộc H×W. Output: **⌊(n + 2p − k)/s⌋ + 1**. Padding 'same' (stride 1): p = (k − 1)/2. Pooling: giảm kích thước, không tham số.
- **RNN**: h_t = tanh(W h_{t−1} + U x_t) – tuần tự, vanishing qua thời gian. **LSTM**: cell state cập nhật cộng + forget/input/output gate → nhớ xa; ~4× tham số RNN. GRU: 2 gate.
- **Attention**: Attention(Q, K, V) = softmax(QKᵀ/√d_k)·V – tra cứu mềm; chia √d_k tránh softmax bão hoà.
- **Transformer**: self-attention (mọi token nhìn nhau, train song song), multi-head, FFN, residual + LayerNorm, positional encoding (attention không biết thứ tự), chi phí O(n²) theo độ dài; decoder dùng causal mask.
- **Autoencoder**: encoder → bottleneck → decoder, reconstruction loss; dùng giảm chiều, denoising, anomaly detection. VAE: sinh dữ liệu.
- **Embedding**: bảng tra V × d học được, vector dày, ngữ nghĩa gần → vector gần.
- **Transfer learning**: dùng pretrained; ít data → freeze backbone, train head; nhiều/khác domain → fine-tune sâu hơn với lr nhỏ.

### Thực hành dữ liệu

- **Missing**: drop (ít, ngẫu nhiên), impute mean/median/mode (median nếu lệch/outlier), thêm cột `is_missing`; tree-based (XGBoost/LightGBM) xử lý được NaN.
- **Outlier**: điều tra trước (lỗi hay tín hiệu?); clip/winsorize, log transform, model/loss bền.
- **Min-max**: (x − min)/(max − min) → [0, 1], nhạy outlier. **Standardization**: (x − μ)/σ → mean 0, std 1 (sklearn dùng population std). Cây không cần scale.
- **Encoding**: nominal → one-hot; ordinal → label/ordinal encoding; cardinality cao → target encoding (cẩn thận leakage), hashing, embedding.
- **Imbalanced**: class_weight, over/undersampling, SMOTE (chỉ trên train), chỉnh threshold, metric PR-AUC/F1.
- **Augmentation**: chỉ khi train; val/test giữ nguyên.
- **Feature engineering**: tạo feature có ý nghĩa từ domain – tách ngày/giờ/thứ trong tuần, tỷ lệ (nợ/thu nhập), tổng hợp theo nhóm, log của biến lệch.
- **Leakage cơ bản**: fit scaler/imputer/encoder trên toàn bộ data, resample trước khi split, feature chỉ có sau thời điểm dự đoán → dùng `Pipeline`, fit chỉ trên train.

### Công cụ: tensor, GPU, sklearn, PyTorch

- **Shape**: ảnh PyTorch (N, C, H, W). Broadcasting so từ phải sang trái, chiều bằng nhau hoặc = 1. Matmul: (…, n, k) @ (k, m) → (…, n, m). `reshape(N, -1)` để flatten.
- **GPU**: hàng nghìn core song song + Tensor Core + băng thông HBM cao → nhân ma trận nhanh. Model và data phải cùng device.
- **Bộ nhớ train** = weights + gradients + optimizer state (cố định) + activations (∝ batch size). OOM → giảm batch + gradient accumulation, mixed precision, gradient checkpointing.
- **sklearn**: `fit` học tham số, `transform` áp dụng, `fit_transform` cho train, `predict`/`predict_proba` cho model. Chỉ fit trên train; gói trong `Pipeline`.
- **PyTorch**: `model.train()` bật dropout & BatchNorm dùng batch stats; `model.eval()` tắt dropout, BatchNorm dùng running stats; `torch.no_grad()` tắt tính gradient khi inference. Vòng lặp: `optimizer.zero_grad()` → forward → `loss.backward()` → `optimizer.step()`.


---

## 📘 Nền tảng: NLP, LLM, RAG & Agent

### Trả lời phỏng vấn 30 giây (câu kinh điển)

- **RAG là gì?** — Retrieval-Augmented Generation: trước khi LLM trả lời, hệ thống tìm các đoạn tài liệu liên quan (thường bằng embedding + vector DB, có thể kèm BM25 và reranker) rồi chèn vào prompt để mô hình trả lời *dựa trên* tài liệu. Lợi ích: kiến thức mới/riêng tư không cần train lại, giảm hallucination, có trích dẫn, phân quyền được. Đánh giá tách 2 tầng: retrieval (recall/hit rate) và generation (faithfulness, relevancy).
- **Transformer khác RNN thế nào?** — RNN đọc tuần tự, state truyền qua từng bước → khó song song, quên phụ thuộc xa. Transformer dùng self-attention: mọi token nhìn trực tiếp mọi token khác, cả chuỗi tính song song → train nhanh, scale lớn, phụ thuộc xa tốt. Đổi lại attention O(n²) theo độ dài và cần positional encoding.
- **Agent là gì?** — LLM + tools + vòng lặp + memory, trong đó LLM tự quyết định bước tiếp theo (gọi tool nào, khi nào dừng) dựa trên kết quả quan sát. Khác workflow (các bước do code định sẵn) và chatbot (chỉ hội thoại). Rủi ro: tốn token, khó đoán, prompt injection → cần guardrails, giới hạn vòng lặp, human-in-the-loop.
- **Vì sao LLM hallucinate?** — Được huấn luyện để sinh token tiếp theo *nghe hợp lý*, không có cơ chế kiểm chứng sự thật; thiếu/cũ kiến thức vẫn trả lời trôi chảy. Giảm bằng RAG + trích dẫn, cho phép nói 'không biết', tool, kiểm chứng đầu ra.
- **RAG hay fine-tune?** — RAG cho *kiến thức* (thay đổi, lớn, cần nguồn, cần phân quyền); fine-tune cho *hành vi* (format, giọng văn, kỹ năng hẹp, model nhỏ rẻ). Có thể kết hợp.

### Token, tokenization & tiền xử lý

- **Token**: đơn vị mô hình xử lý (từ, mảnh từ, ký tự hoặc byte). **Vocabulary**: tập token cố định; mỗi token có id → tra bảng embedding V × d.

| Cách tách | Ưu | Nhược |
|---|---|---|
| Word-level | token có nghĩa | vocab khổng lồ, OOV → `<UNK>` |
| Character-level | không OOV, vocab nhỏ | chuỗi rất dài, mỗi token ít nghĩa |
| **Subword** (BPE, WordPiece, Unigram/SentencePiece) | vocab vừa, ghép được mọi từ lạ | token không trùng ranh giới từ |

- Tiền xử lý cổ điển (cho BoW/TF-IDF): lowercase, bỏ stopword, **stemming** (cắt hậu tố theo luật, nhanh, có thể ra 'từ' không có thật: `studies → studi`), **lemmatization** (về dạng từ điển theo từ loại: `better → good`, chậm hơn, chính xác hơn).
- LLM hiện đại gần như **không** dùng các bước này: `không`, `not` mang nghĩa quyết định; tokenizer subword đã xử lý biến thể.
- Chi phí và context đều tính theo **token**, không theo từ; tiếng Việt thường tốn nhiều token/từ hơn tiếng Anh với tokenizer thiên về tiếng Anh.

### Biểu diễn văn bản: BoW → TF-IDF → embedding

| Biểu diễn | Ý tưởng | Hạn chế |
|---|---|---|
| Bag-of-words | đếm từ, bỏ thứ tự | thưa, nhiều chiều, không ngữ nghĩa, `chó cắn mèo` = `mèo cắn chó` |
| TF-IDF | tf × idf: nhiều trong văn bản, hiếm trong corpus | vẫn không hiểu đồng nghĩa |
| Static embedding (word2vec, GloVe, fastText) | vector dày, từ cùng ngữ cảnh gần nhau | 1 vector/từ, không xử lý đa nghĩa |
| Contextual embedding (BERT, LLM) | vector phụ thuộc câu | phải chạy mô hình |

- TF-IDF: `tf = count/len`, `idf = log(N/df)`; từ có ở mọi văn bản → idf = 0. (scikit-learn: smooth idf `ln((1+N)/(1+df)) + 1`.)
- Analogy: `king − man + woman ≈ queen` — quan hệ trở thành hướng gần cố định trong không gian vector.
- Cosine: `cos = a·b / (‖a‖‖b‖)`; vector đã chuẩn hoá thì cosine = dot product.

### Language model & n-gram

- **Language model** gán xác suất cho chuỗi: `P(w1..wn) = Π P(wi | w1..wi−1)` (chain rule) ⇔ dự đoán token kế tiếp.
- **n-gram**: giả định Markov — chỉ phụ thuộc n−1 từ trước. Bigram: `P(wi | wi−1) = count(wi−1 wi) / count(wi−1)`.
- Vấn đề: n-gram chưa gặp → xác suất 0 → cần **smoothing** (Laplace, Kneser-Ney); ngữ cảnh ngắn; dữ liệu thưa khi n lớn. Tích xác suất nhỏ → dùng tổng log-prob.
- Neural LM → RNN LM → Transformer LM (GPT): cùng mục tiêu next-token, nhưng ngữ cảnh dài và biểu diễn học được.
- **Perplexity** = exp(trung bình NLL/token): 'số lựa chọn tương đương' mô hình phân vân; chỉ so sánh được khi cùng tokenizer và cùng tập dữ liệu.

| Tác vụ | Đầu ra | Mô hình hợp |
|---|---|---|
| Classification | 1 nhãn/văn bản | encoder + head, hoặc prompt LLM |
| NER | 1 nhãn/token (BIO) | token classification |
| Extractive QA | span start/end trong đoạn văn | encoder (BERT) |
| Summarization, translation | chuỗi mới | encoder-decoder hoặc decoder-only |

### Transformer: self-attention, positional encoding

- `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`, với Q = X·W_Q, K = X·W_K, V = X·W_V.
- Q = 'token này tìm gì', K = 'token kia chứa gì', V = 'nội dung được lấy về'. Chia √d_k để softmax không bão hoà.
- **Multi-head**: nhiều bộ Q/K/V song song, mỗi head học một kiểu quan hệ, rồi nối lại và chiếu bằng W_O.
- **Positional encoding** cần vì attention không tự biết thứ tự. **Causal mask** (khác PE!) cấm nhìn token tương lai trong decoder.
- Chi phí attention **O(n²)**: chuỗi dài gấp 4 → ma trận điểm lớn gấp 16.

| | RNN/LSTM | Transformer |
|---|---|---|
| Xử lý | tuần tự | song song |
| Phụ thuộc xa | qua n bước, dễ quên | trực tiếp 1 bước |
| Chi phí theo n | tuyến tính | bậc hai (attention) |
| Thứ tự | sẵn trong cách tính | cần positional encoding |

### Ba họ kiến trúc & vòng đời LLM

| Họ | Ví dụ | Attention | Dùng cho |
|---|---|---|---|
| Encoder-only | BERT, RoBERTa | hai chiều | phân loại, NER, embedding, rerank |
| Decoder-only | GPT, Llama, Qwen | causal | sinh văn bản, chat, agent |
| Encoder-decoder | T5, BART | 2 chiều + cross-attention | dịch, tóm tắt |

Vòng đời: **Pretraining** (next-token trên dữ liệu khổng lồ → kiến thức, base model chỉ 'viết tiếp') → **SFT/instruction tuning** (cặp chỉ dẫn–trả lời → biết làm theo yêu cầu, chat template) → **Alignment** (RLHF: reward model từ xếp hạng của người + RL; hoặc DPO) → hữu ích, an toàn.

- Kiến thức chủ yếu từ pretraining; SFT/RLHF dạy *hành vi*.
- **Autoregressive**: sinh từng token, nối vào, lặp tới EOS/max_tokens → streaming, độ trễ tỷ lệ số output token.
- **Embedding model** (vector, không sinh chữ) ≠ **generative model** (sinh chữ).

### Sinh văn bản: context window, sampling, chi phí

- **Context window** = giới hạn tổng input + output. Ngân sách lịch sử = context − system − câu hỏi − phần dành cho output; số lượt giữ được = **floor**(ngân sách / token mỗi lượt).
- **Temperature**: chia logits cho T; T thấp → nhọn, gần greedy (ổn định, *không* đảm bảo đúng); T cao → đa dạng.
- **Top-p**: giữ tập token nhỏ nhất có tổng xác suất ≥ p (số token thay đổi theo bước). **Top-k sampling**: giữ k token xác suất cao nhất (khác top-k retrieval!).
- **Chi phí** = (input × giá_in + output × giá_out) / 10⁶; giá output thường gấp 3–5 lần input. Ví dụ 20K req/ngày × (1,500 in, 400 out) với $0.40/$1.60 → $744/tháng.
- Giảm chi phí: prompt caching, rút gọn output, model nhỏ cho tác vụ dễ, RAG thay vì nhồi toàn bộ tài liệu.
- **Hallucination**: nội dung nghe hợp lý nhưng sai/không có căn cứ. Gốc: mục tiêu next-token, không kiểm chứng sự thật.

### Prompting & in-context learning vs fine-tuning

- **Zero-shot**: chỉ dẫn, không ví dụ. **Few-shot**: kèm vài ví dụ input → output. **Chain-of-thought**: suy luận từng bước trước khi kết luận. **System prompt**: vai trò, quy tắc, format cho cả phiên (ưu tiên cao nhưng không phải rào chắn tuyệt đối).
- **In-context learning** không đổi trọng số: hiệu quả chỉ trong request đó.

| | In-context learning (prompt) | Fine-tuning |
|---|---|---|
| Dữ liệu cần | 0–vài chục ví dụ | hàng trăm–nghìn+ mẫu |
| Tốc độ thử | vài phút–giờ | ngày, cần pipeline train |
| Đổi yêu cầu | sửa prompt | train lại |
| Chi phí chạy | prompt dài → tốn token | prompt ngắn, có thể dùng model nhỏ |
| Hợp với | prototype, yêu cầu hay đổi | tác vụ ổn định, khối lượng lớn, format/giọng văn |

Thứ tự nên thử: prompt tốt → few-shot → RAG (nếu thiếu kiến thức) → fine-tune (nếu thiếu hành vi/kinh tế).

### RAG: pipeline & các khái niệm gốc

```
INDEXING (offline):  load → parse/clean → chunk → embed → store (vector + text + metadata)
QUERY (online):      embed query → retrieve top-k → (rerank) → augment prompt → generate (+ trích dẫn)
```

- Query và tài liệu phải embed bằng **cùng một** embedding model; đổi model → index lại toàn bộ.
- **Chunk** cửa sổ trượt: stride s = c − o; số chunk = ceil((L − c)/s) + 1. Ví dụ L = 2,000, c = 500, o = 100 → 5 chunk. Overlap giữ ý bị cắt ở ranh giới.
- **Vector DB**: lưu vector + payload, tìm láng giềng gần bằng ANN (HNSW, IVF), lọc metadata.
- **Top-k**: số chunk lấy về; k lớn → recall ↑ nhưng nhiễu, token, độ trễ ↑.
- **Keyword (BM25)**: mạnh với mã, tên riêng, từ hiếm. **Semantic (dense)**: mạnh với paraphrase. **Hybrid**: chạy cả hai rồi fuse (RRF).
- **Reranker** (cross-encoder): chấm cặp (query, chunk) chung → chính xác nhưng chậm → chỉ rerank top 20–100 → giữ 3–5.

### RAG vs fine-tuning vs long context; đánh giá RAG

| Nhu cầu | Chọn |
|---|---|
| Kiến thức hay thay đổi, cần trích dẫn, phân quyền | **RAG** |
| Format/giọng văn/kỹ năng hẹp, nhiều dữ liệu gán nhãn | **Fine-tune** |
| Kho nhỏ, ít đổi, có prompt caching | **Long context** (nhồi thẳng) có thể đủ |

- RAG **giảm** chứ không **loại bỏ** hallucination; không thay đổi trọng số.
- Ví dụ chi phí: sổ tay 120K token nhồi mỗi câu vs RAG 2,400 token → 1,000 câu/ngày ở $2.5/1M tiết kiệm $295/ngày.

**Đánh giá 2 tầng:**

| Tầng | Câu hỏi | Metric | Sửa ở đâu |
|---|---|---|---|
| Retrieval | chunk đúng có trong top-k? | hit rate@k, recall@k, MRR, context recall/precision | chunking, embedding, hybrid, rerank, query rewrite |
| Generation | trả lời bám context, đúng câu hỏi? | faithfulness/groundedness, answer relevancy, correctness | prompt, trích dẫn, giảm nhiễu context, model |

Lỗi kinh điển: tài liệu không có trong kho → retrieve trượt → có nhưng bị cắt khỏi prompt → có trong prompt nhưng LLM không dùng → trả lời sai format.

### Agent: định nghĩa, tool calling, ReAct, memory

| | Chatbot | Workflow | Agent |
|---|---|---|---|
| Gọi tool | không | có thể | có |
| Ai quyết định bước tiếp | — | code định sẵn | **LLM** |
| Độ dự đoán / chi phí | cao / thấp | cao / thấp | thấp / cao |

- **Tool calling**: app khai báo tool (name, description, JSON Schema) → model trả về *lời gọi* (tên + đối số) → **app thực thi** → gửi kết quả lại → model trả lời hoặc gọi tiếp. Luôn validate đối số.
- **ReAct**: Thought → Action → Observation (do môi trường trả về) → … → Final Answer. Lỗi kinh điển: model tự bịa Observation; luôn có max iterations.
- **Planning**: plan-and-execute (lập kế hoạch trước, re-plan khi cần) vs ReAct (từng bước).
- **Memory**: ngắn hạn = context window hiện tại (cắt, tóm tắt); dài hạn = lưu ngoài (DB/vector store) rồi truy xuất. Không thay đổi trọng số.
- **Multi-agent**: orchestrator–worker, chuyên môn hoá, song song; tốn token hơn, khó debug. Bắt đầu với single-agent.
- **Token agent**: mỗi lần gọi gửi lại toàn bộ lịch sử → tổng input tăng gần bậc hai theo số bước (vd 4 lần gọi: 1,600 + 2,100 + 2,600 + 3,100 = 9,400).

### MCP, guardrails & an toàn

- **MCP (Model Context Protocol)**: chuẩn mở (Anthropic, 11/2024) kết nối ứng dụng LLM ↔ tool/dữ liệu; biến tích hợp M×N thành M+N (6 app × 10 hệ thống: 60 → 16).
- Kiến trúc: **Host** (ứng dụng: IDE, chat app, chứa LLM) → nhiều **Client** (mỗi client 1–1 với một server) → **Server** (expose một hệ thống).
- Primitives phía server: **Tools** (hành động, model quyết định gọi), **Resources** (dữ liệu đọc làm context, ứng dụng quyết định), **Prompts** (mẫu prompt/quy trình, người dùng chọn).
- **Guardrails**: input (lọc, phát hiện injection), output (PII, độc hại, format), hành động (quyền tool, ngưỡng tiền, max iterations). Ràng buộc quan trọng phải nằm trong **code**, không chỉ trong prompt.
- **Human-in-the-loop**: duyệt hành động rủi ro cao, khó đảo ngược (hoàn tiền, gửi email ra ngoài, xoá dữ liệu); không duyệt hành động chỉ đọc.
- **Prompt injection**: direct (user gõ) vs **indirect** (chỉ dẫn ẩn trong web/email/tài liệu/kết quả tool). Không có cách chặn 100% → least privilege, coi output tool là dữ liệu không tin cậy, xác nhận hành động nhạy cảm, allowlist, giám sát.


---

## 🎓 Nền tảng: Lý thuyết AI tổng quát

### Mốc lịch sử AI cần nhớ

| Năm | Mốc | Ý nghĩa |
|---|---|---|
| 1950 | Turing, 'Computing Machinery and Intelligence' | Imitation game / Turing test – tiêu chí **hành vi** |
| 1956 | Hội thảo Dartmouth | Khai sinh lĩnh vực, tên 'artificial intelligence' (McCarthy) |
| 1958 | Perceptron (Rosenblatt) | Mạng nơ-ron một lớp học trọng số từ dữ liệu (perceptron learning rule) |
| 1970s | AI winter #1 | Báo cáo Lighthill (1973), giới hạn perceptron một lớp, bùng nổ tổ hợp |
| 1970–80s | Expert systems (MYCIN, XCON) | Symbolic AI thương mại hoá |
| 1986 | Backprop (Rumelhart, Hinton, Williams) | Train được mạng nhiều lớp |
| cuối 1980s | AI winter #2 | Expert system khó bảo trì, thị trường Lisp machine sụp đổ (1987) |
| 1997 | Deep Blue thắng Kasparov | Tìm kiếm alpha-beta + hàm đánh giá viết tay, **không** phải deep learning |
| 2012 | AlexNet thắng ImageNet | Mở đầu kỷ nguyên deep learning (CNN + GPU) |
| 2014 | GAN | Mô hình sinh đối kháng |
| 2016 | AlphaGo thắng Lee Sedol | Policy/value network (SL từ ván người + RL self-play) + MCTS |
| 2017 | Transformer ('Attention Is All You Need') | Nền tảng của LLM |
| 2020 | GPT-3 (175B) | Few-shot / in-context learning |
| 2022 | ChatGPT (30/11) | GPT-3.5 + RLHF, AI phổ cập đại chúng |

- **Narrow AI**: giỏi một nhiệm vụ hẹp (AlphaGo, model phân loại ảnh). **AGI**: năng lực tổng quát ngang người trên nhiều nhiệm vụ – chưa có định nghĩa/tiêu chí thống nhất.
- **Chinese Room** (Searle 1980): vượt Turing test về hành vi ≠ hiểu thật.

### Trường phái AI & intelligent agent

| Trường phái | Tri thức nằm ở đâu | Mạnh | Yếu | Ví dụ |
|---|---|---|---|---|
| Symbolic (GOFAI) | ký hiệu, luật, logic do người viết | minh bạch, suy luận chặt | khó mở rộng, giòn, nút thắt thu nhận tri thức | expert system, Prolog |
| Connectionism | trọng số mạng nơ-ron | học từ dữ liệu thô, chịu nhiễu | khó giải thích, cần nhiều dữ liệu | MLP, CNN, Transformer |
| Statistical learning | mô hình xác suất, tối ưu loss | nền tảng lý thuyết (VC, PAC) | cần feature engineering | SVM, Naive Bayes, boosting |

**Agent** = nhận percept qua sensors → chọn action qua actuators để tối đa performance measure. **PEAS**: Performance, Environment, Actuators, Sensors.

Tính chất môi trường:
- fully vs **partially observable** (poker, xe tự lái)
- deterministic vs **stochastic** (đối thủ chiến lược ≠ ngẫu nhiên: cờ vua vẫn deterministic)
- episodic vs sequential; static vs dynamic; discrete vs continuous; single vs multi-agent.

### Tìm kiếm: uninformed & informed

| Thuật toán | Mở node theo | Complete | Optimal | Bộ nhớ |
|---|---|---|---|---|
| BFS | tầng nông nhất | có (b hữu hạn) | chỉ khi chi phí bước bằng nhau | O(b^d) |
| DFS | sâu nhất | không (vô hạn/chu trình) | không | O(b·m) |
| Iterative deepening | DFS giới hạn độ sâu tăng dần | có | như BFS | O(b·d) |
| UCS | g(n) nhỏ nhất | có (chi phí ≥ ε > 0) | có | O(b^(1+C*/ε)) |
| Greedy best-first | h(n) nhỏ nhất | không | không | – |
| A* | f = g + h nhỏ nhất | có | có nếu h admissible (tree) / consistent (graph) | lớn |

- **Admissible**: h(n) ≤ chi phí thật tới goal (lạc quan). **Consistent**: h(n) ≤ c(n, n') + h(n').
- h = 0 → A* thành UCS; h càng sát h* (vẫn admissible) → mở càng ít node.
- A* chỉ dừng khi **lấy** goal ra khỏi hàng đợi, không phải khi vừa sinh ra goal.
- Số node cây đầy đủ tới độ sâu d: (b^(d+1) − 1)/(b − 1), tầng cuối chiếm phần lớn.

### Adversarial search, CSP & biểu diễn tri thức

- **Minimax**: MAX chọn max, MIN chọn min, truyền từ lá lên. **Alpha-beta**: cắt nhánh khi α ≥ β → kết quả **y hệt** minimax; thứ tự tốt nhất O(b^(d/2)) (tìm sâu gấp đôi), ngẫu nhiên ≈ O(b^(3d/4)).
- **MCTS**: selection – expansion – simulation – backpropagation (UCT); AlphaGo = MCTS + policy/value network.
- **CSP**: biến + miền giá trị + ràng buộc (Sudoku, tô màu bản đồ, xếp lịch). Giải bằng backtracking + MRV + forward checking + arc consistency (AC-3).
- **Logic mệnh đề**: mệnh đề đúng/sai + ∧ ∨ ¬ →. **Logic vị từ bậc nhất (FOL)**: thêm đối tượng, vị từ, lượng từ ∀ ∃ → biểu diễn tổng quát gọn hơn.
- **Knowledge graph**: triple (subject, relation, object); **ontology**: lược đồ lớp, thuộc tính, quan hệ, ràng buộc (OWL, schema.org). Dùng cho tìm kiếm, QA, GraphRAG.
- **Bayesian network**: DAG + bảng CPT, P(X₁…Xₙ) = Π P(Xᵢ | cha(Xᵢ)); mã hoá độc lập có điều kiện, suy luận xác suất 'nguyên nhân → triệu chứng'.

### Reinforcement Learning: khái niệm & công thức

**Vòng lặp**: state s → agent chọn action a theo policy π → môi trường trả reward r và state s'. Mục tiêu: tối đa **return** G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + …

- **MDP** (S, A, P, R, γ) + **Markov property**: tương lai chỉ phụ thuộc (s, a) hiện tại. Markov ≠ deterministic.
- **V^π(s)** = E[G | s]; **Q^π(s, a)** = E[G | s, a]; V^π(s) = Σ_a π(a|s)·Q^π(s, a).
- **Bellman**: V(s) = E[r + γ·V(s')]; tối ưu: Q*(s, a) = E[r + γ·max_a' Q*(s', a')].
- γ → 0: thiển cận; γ → 1: nhìn xa, khó học hơn. G_t = r_{t+1} + γ·G_{t+1} (tính ngược từ cuối).
- **Q-learning**: Q ← Q + α·[r + γ·max_a' Q(s', a') − Q] (off-policy). **SARSA**: dùng Q(s', a') của action thực tế (on-policy).
- **ε-greedy** (ngẫu nhiên trên mọi action): P(greedy) = 1 − ε + ε/|A|.
- **Exploration vs exploitation**: ε-greedy, softmax, UCB, Thompson sampling; thường giảm ε dần.
- Khác supervised: không có nhãn hành động đúng, reward trễ/thưa (credit assignment), dữ liệu phụ thuộc policy.

### RL: các họ thuật toán, reward & liên hệ LLM

| Họ | Học gì | Ví dụ | Ghi chú |
|---|---|---|---|
| Value-based | Q(s, a) → policy = argmax | Q-learning, DQN | action rời rạc; DQN: replay buffer + target network |
| Policy-based | π_θ trực tiếp | REINFORCE | action liên tục được, variance cao |
| Actor-critic | π (actor) + V/advantage (critic) | A2C, PPO, SAC | critic giảm variance |
| Model-based | mô hình P(s'∣s,a), R để lập kế hoạch | Dyna, AlphaZero, MuZero | hiệu quả mẫu cao, sai model thì lệch |

- **On-policy** (SARSA, REINFORCE, PPO): học từ dữ liệu của chính policy hiện tại. **Off-policy** (Q-learning, DQN, SAC): học từ dữ liệu của policy khác → tái sử dụng replay, offline RL.
- Cliff walking: SARSA chọn đường an toàn, Q-learning chọn đường sát vực (tối ưu cho greedy) nhưng reward online thấp hơn khi còn khám phá.
- **Reward hacking / specification gaming**: tối ưu proxy lệch mục tiêu (thuyền CoastRunners chạy vòng). **Potential-based shaping** F = γΦ(s') − Φ(s) không đổi policy tối ưu → giúp học nhanh nhưng **không** sửa được reward gốc đã lệch; chữa hacking phải sửa đặc tả reward.
- **RLHF ↔ RL**: policy = LLM; state = prompt + token đã sinh; action = token kế tiếp; reward = reward model cuối câu − β·KL so với model tham chiếu; PPO (actor-critic, value head).

### Mô hình sinh: 4 họ + autoregressive

**Generative** học p(x) hoặc p(x, y) (sinh được mẫu); **discriminative** học p(y|x) / ranh giới. Naive Bayes ↔ logistic regression là cặp kinh điển.

| Họ | Ý tưởng | Sinh | Likelihood | Ưu | Nhược |
|---|---|---|---|---|---|
| Autoregressive | p(x) = Π p(x_t ∣ x_<t) | tuần tự | chính xác | đơn giản, mạnh (GPT) | chậm theo độ dài |
| VAE | encoder q(z∣x), decoder p(x∣z), tối đa ELBO | 1 bước | cận dưới | ổn định, latent có cấu trúc | mẫu mờ |
| GAN | G vs D chơi minimax | 1 bước | không có | sắc nét, nhanh | train bất ổn, mode collapse |
| Flow | biến đổi khả nghịch, đổi biến | 1 bước | chính xác | likelihood chính xác | ràng buộc kiến trúc, giữ số chiều |
| Diffusion | thêm nhiễu dần, học khử nhiễu (dự đoán ε) | nhiều bước | cận dưới | chất lượng + đa dạng cao, ổn định | chậm (giảm bằng DDIM, distillation, latent) |
| EBM | p(x) ∝ exp(−E(x)) | MCMC | Z khó tính | linh hoạt | train/sample khó |

- **ELBO** = E_q[log p(x|z)] − KL(q(z|x) ‖ p(z)) ≤ log p(x). **Reparameterization**: z = μ + σ·ε để backprop.
- GAN: D quá mạnh → gradient G biến mất → loss **non-saturating** (max log D(G(z))); WGAN, spectral norm.
- Đánh giá: **FID** (thấp = tốt, so ảnh thật vs sinh), **IS** (cao = tốt, chỉ ảnh sinh), CLIPScore cho text–image, đánh giá của người.

### Lý thuyết học máy

- **True risk** R(h) = E[L] trên phân phối thật; **empirical risk** R̂(h) = trung bình trên train. **ERM**: chọn h tối thiểu R̂. **Generalization gap** = R − R̂.
- **VC dimension**: số điểm lớn nhất lớp H shatter được. Siêu phẳng có bias trong R^d: **d + 1**. VC ≠ số tham số (sign(sin(ωx)) có VC vô hạn).
- **PAC**: với xác suất ≥ 1 − δ, lỗi ≤ ε, số mẫu đa thức theo 1/ε, 1/δ. H hữu hạn (realizable): m ≥ (1/ε)(ln|H| + ln(1/δ)).
- **Curse of dimensionality**: hypercube con chứa tỷ lệ p dữ liệu có cạnh p^(1/d) (d = 10, p = 10% → 0.79); khoảng cách mất ý nghĩa, cần dữ liệu tăng theo hàm mũ. Cứu cánh: manifold hypothesis, giảm chiều, inductive bias.
- **Occam's razor**: ưu tiên giải thích đơn giản khi khớp như nhau (ưu tiên, không phải bảo đảm).
- **No Free Lunch**: trung bình trên mọi bài toán, không thuật toán nào tốt hơn → hiệu quả đến từ giả định khớp bài toán.
- **Double descent**: test error giảm → đỉnh tại ngưỡng nội suy (#tham số ≈ #mẫu) → giảm lại khi over-parameterized; có cả theo epoch và theo lượng dữ liệu.
- **Lottery ticket** (Frankle & Carbin 2019): có mạng con thưa train lại từ khởi tạo gốc đạt độ chính xác tương đương; tìm bằng iterative magnitude pruning (tốn kém).

### Inductive bias, representation & self-supervised

| Kiến trúc | Inductive bias | Hệ quả |
|---|---|---|
| MLP | gần như không có | cần nhiều dữ liệu, không khai thác cấu trúc |
| CNN | locality + weight sharing → translation **equivariance**; pooling → bất biến xấp xỉ | hiệu quả dữ liệu cao cho ảnh |
| RNN | tuần tự, hidden state tóm tắt quá khứ | hợp chuỗi, khó song song, khó phụ thuộc xa |
| Transformer | attention toàn cục, permutation-equivariant nếu không có PE | bias yếu → cần nhiều dữ liệu, scale rất tốt |
| GNN | bất biến hoán vị node | hợp dữ liệu đồ thị |

- **Representation learning**: học feature tự động thay vì viết tay; biểu diễn tốt → linear probe/fine-tune ít nhãn.
- **Self-supervised**: nhãn tự sinh từ dữ liệu. **Contrastive** (SimCLR, MoCo; CLIP cho ảnh–text): kéo gần positive, đẩy xa negative (InfoNCE). **Masked/generative** (BERT, MAE, GPT next-token).
- **Scaling hypothesis / The Bitter Lesson** (Sutton 2019): phương pháp tổng quát + nhiều compute/data thắng tri thức viết tay về lâu dài.
- **Emergent abilities**: năng lực 'xuất hiện đột ngột' ở model lớn – đang tranh luận, một phần có thể do chọn metric không liên tục.

### Ứng dụng: bài toán → input/output → metric

| Miền | Bài toán | Output | Metric chính |
|---|---|---|---|
| CV | classification | 1 nhãn/ảnh | accuracy, top-k |
| CV | detection | box + class | mAP (theo ngưỡng IoU) |
| CV | semantic / instance / panoptic segmentation | lớp mỗi pixel / mask từng vật | mIoU, Dice / mask AP / PQ |
| CV | generation | ảnh | FID, IS, CLIPScore |
| Speech | ASR / TTS | text / audio | WER, CER / MOS |
| Speech | speaker verification (1:1) / identification (1:N) / diarization | chấp nhận/từ chối / ai / ai nói lúc nào | EER / accuracy / DER |
| OCR | detection + recognition (CRNN + CTC, TrOCR) | chuỗi ký tự | CER, WER |
| RecSys | retrieval + ranking | danh sách xếp hạng | NDCG@k, MAP@k, Recall@k; CTR online |
| Time-series | forecasting / anomaly detection | giá trị tương lai / cờ bất thường | MAE, RMSE, MAPE, MASE / precision–recall |
| Multimodal | CLIP (embedding chung) / VLM (vision encoder + LLM) | similarity / văn bản | retrieval recall, zero-shot acc / VQA acc |

- Ít nhãn bất thường → anomaly detection không giám sát (autoencoder, forecasting residual, Isolation Forest).
- CLIP không tự sinh văn bản; VLM sinh văn bản.

### Responsible AI: bias, fairness, XAI

**Các loại bias**: historical (dữ liệu phản ánh bất công quá khứ), representation/sampling (thiếu nhóm), selection, measurement (proxy đo lệch), label, aggregation, automation bias (người tin máy quá mức). Đừng nhầm với *bias thống kê* trong bias–variance.

| Tiêu chí | Điều kiện bằng nhau giữa các nhóm |
|---|---|
| Demographic parity | P(ŷ = 1) |
| Equal opportunity | TPR |
| Equalized odds | TPR **và** FPR |
| Calibration theo nhóm | P(y = 1 ∣ score = s) |

- Disparate impact = tỷ lệ dương nhóm thấp / nhóm cao; quy tắc 4/5 (≥ 0.8).
- **Impossibility** (Kleinberg 2016, Chouldechova 2017): base rate khác nhau → không thể vừa calibration vừa cân bằng FPR/FNR.
- Bỏ thuộc tính nhạy cảm không đủ: proxy (mã bưu chính, tên…).
- **XAI**: SHAP (Shapley, additive, cục bộ + toàn cục), LIME (surrogate tuyến tính cục bộ), permutation importance (lệch khi feature tương quan), Grad-CAM cho CNN. **Attention ≠ explanation**; importance ≠ nhân quả.

### Privacy, robustness, safety & quy định + trả lời 30 giây

- **Differential privacy**: output gần như không đổi khi thêm/bớt 1 bản ghi; ε nhỏ → riêng tư mạnh; DP-SGD = clip gradient từng mẫu + nhiễu Gauss.
- **Federated learning**: dữ liệu ở lại máy, gửi cập nhật; gradient vẫn rò rỉ → secure aggregation + DP. Thách thức non-IID.
- **PII**: phát hiện/che (masking), tối thiểu hoá dữ liệu, kiểm soát truy cập, không đưa PII vào prompt/log.
- **Adversarial example**: nhiễu nhỏ được tối ưu (FGSM x + ε·sign(∇ₓL)); khác data poisoning (tấn công lúc train) và distribution shift. Phòng thủ: adversarial training.
- **Alignment**: hành vi khớp ý định/giá trị con người (helpful, honest, harmless); RLHF, Constitutional AI, red teaming, guardrail.
- **Hallucination** mitigation: RAG + trích dẫn, cho phép 'không biết', verify, eval liên tục; temperature 0 không chữa được.
- **Model card** (Mitchell 2019), **datasheet/data card** (Gebru): mục đích, giới hạn, dữ liệu, hiệu năng theo nhóm. **Human-in-the-loop** cho quyết định rủi ro cao.
- **EU AI Act** (hiệu lực 8/2024): cấm (social scoring) / rủi ro cao (tuyển dụng, tín dụng, y tế, giáo dục) / minh bạch (chatbot, deepfake) / tối thiểu; áp dụng cả nhà cung cấp ngoài EU. Lộ trình: điều cấm từ 2/2/2025; nghĩa vụ GPAI từ 2/8/2025; minh bạch (Điều 50) từ 2/8/2026; nghĩa vụ high-risk vốn từ 2/8/2026 đã được **Digital Omnibus on AI** (Regulation (EU) 2026/1744, hiệu lực 27/7/2026) lùi sang 2/12/2027 (Annex III) và 2/8/2028 (Annex I).

**Trả lời 30 giây**
- *RL khác supervised?* Không có nhãn đúng, chỉ có reward trễ; hành động ảnh hưởng dữ liệu; phải cân bằng explore/exploit.
- *On- vs off-policy?* Học giá trị của policy đang hành động (SARSA) vs của policy khác/greedy (Q-learning) → off-policy dùng được replay.
- *GAN vs diffusion?* GAN nhanh, sắc nét nhưng train bất ổn/mode collapse; diffusion ổn định, đa dạng, chất lượng cao nhưng sinh chậm.
- *Vì sao ViT cần nhiều data?* Ít inductive bias hơn CNN (không có locality/equivariance sẵn).

