# Tổng hợp kiến thức ôn phỏng vấn AI Engineer

9 chủ đề · 412 câu hỏi trong web quiz. File này được sinh tự động từ `data/src/*.json`.

## Mục lục

- 📝 NLP & Transformer
- 🧠 LLM: Kiến trúc, Training & Alignment
- ⚡ LLM Inference & Model Optimization
- 🔎 RAG, Retrieval & Vector DB
- 🤖 AI Agents, Tool Calling & MCP
- 📊 LLM Evaluation, Observability & LLMOps
- 📈 Machine Learning & Deep Learning
- 👁️ Computer Vision, Speech, OCR & RecSys
- 🛠️ Python, Backend & MLOps cho AI

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

