# Hướng dẫn soạn ngân hàng câu hỏi (dành cho người/agent đóng góp)

Mỗi chủ đề là 1 file `data/src/<id>.json`, UTF-8, đúng schema:

```json
{
  "id": "rag",
  "name": "RAG & Retrieval",
  "icon": "🔎",
  "order": 4,
  "notes": [
    { "title": "Chunking", "md": "Markdown...\n\n- ý 1\n- ý 2" }
  ],
  "questions": [
    {
      "id": "rag-001",
      "sub": "Hybrid search",
      "level": 2,
      "type": "single",
      "trap": true,
      "q": "Nội dung câu hỏi (Markdown, có thể có `code`, ```block```, bảng)",
      "options": ["...", "...", "...", "..."],
      "answer": 2,
      "explain": "Giải thích VÌ SAO đáp án đúng và VÌ SAO từng phương án sai/bẫy sai."
    },
    { "type": "multi",   "answer": [0, 2], "...": "chọn TẤT CẢ đáp án đúng" },
    { "type": "numeric", "answer": 12.5, "tolerance": 0.1, "unit": "GB", "...": "người dùng nhập số" }
  ]
}
```

## Quy tắc
- `level`: 1 = nền tảng (nhưng không ngây ngô), 2 = trung cấp/phỏng vấn thường gặp, 3 = khó/sâu/edge case. Tỷ lệ mục tiêu ~ 20% / 45% / 35%.
- `trap: true` cho câu có bẫy (khoảng 30%): hiểu lầm phổ biến, phương án "nghe hợp lý" nhưng sai, phủ định kép, đơn vị (bit vs byte, GB vs GiB), off-by-one, nhầm khái niệm gần nhau (precision vs recall, bi-encoder vs cross-encoder, top-k vs top-p...), câu "chọn phát biểu SAI".
- Loại câu: ~50% single, ~25% multi (2–4 đáp án đúng, KHÔNG phải luôn cùng số lượng), ~25% numeric (tính toán) khi chủ đề cho phép.
- `options`: 4 phương án (3–6 cho phép), KHÔNG tự đánh A/B/C (UI tự thêm). Các phương án dài tương đương nhau, tránh để đáp án đúng luôn dài nhất. Phân bố vị trí đáp án đúng đều. Không dùng "Tất cả các đáp án trên".
- `numeric`: ghi rõ đơn vị mong muốn và cách làm tròn ngay trong đề (vd: "làm tròn 2 chữ số thập phân", "đơn vị GB, 1 GB = 10^9 byte"). `tolerance` là sai số tuyệt đối chấp nhận. `answer` phải là số đã tính chính xác — hãy TỰ KIỂM tính lại bằng Python trước khi ghi.
- `explain`: bắt buộc chi tiết (3–8 câu), có lời giải từng bước với câu tính toán, nêu cái bẫy. Đây là phần HỌC nên phải chất lượng.
- Ngôn ngữ: tiếng Việt, giữ nguyên thuật ngữ tiếng Anh (attention, retriever, KV cache...). Không dùng LaTeX; dùng ký hiệu unicode (√, Σ, ×, ≈, ², ·, ∈) hoặc `code`.
- `notes`: phần tổng hợp kiến thức của chủ đề (cheat-sheet ôn tập), 6–12 mục, mỗi mục Markdown súc tích nhưng đầy đủ: định nghĩa, công thức, so sánh dạng bảng, con số cần nhớ, câu hỏi phỏng vấn hay gặp, lỗi thường gặp.
- Tính đúng đắn là ưu tiên số 1. Với kiến thức thay đổi nhanh (MCP spec, model mới, API), chỉ hỏi điều đã được xác minh; nếu không chắc, kiểm tra bằng web search hoặc bỏ câu đó.
- Validate: `python tools/validate.py data/src/<id>.json` phải ra `[OK]`.

## Quy tắc bổ sung (rút ra từ vòng kiểm định)
- UI **xáo trộn thứ tự options**: trong `q`/`explain` không nhắc phương án bằng chữ cái/vị trí ("A sai", "phương án 2"); nhắc bằng nội dung. Options không tham chiếu lẫn nhau. Tránh viết `(a)`, `b)`, `c)` trong văn bản (audit hiểu nhầm là nhãn). Bắt buộc giữ thứ tự thì thêm `"noShuffle": true`.
- **Chống đoán theo độ dài**: đáp án đúng KHÔNG được thường xuyên là phương án dài nhất (mục tiêu ≤ 35% câu single) và cũng không thường xuyên ngắn nhất; độ dài TB đáp án đúng ≈ sai (chênh ≤ 15%), áp dụng cả câu multi. Distractor phải cụ thể, "nghe hợp lý".
- Không dùng ký tự `U+FFFD` (�) thô trong nội dung.
- Kiểm tra: `python tools/validate.py <file>` và `python tools/audit.py <file>`.
