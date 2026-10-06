# AI Engineer Drill — web trắc nghiệm ôn phỏng vấn AI

Web tĩnh, không cần server/backend. Trọng tâm NLP · LLM · Inference · RAG · Agent/MCP · Evaluation, kèm ML/DL, CV/Speech/OCR/RecSys và Python/Backend/MLOps.

## Chạy

```bash
python tools/build.py        # validate + sinh data/bank.js, dist/ai-quiz.html, KIEN_THUC.md
```

Sau đó mở `index.html` trực tiếp bằng trình duyệt (double-click), hoặc:

```bash
python -m http.server 8000   # rồi vào http://localhost:8000
```

`dist/ai-quiz.html` là bản 1 file duy nhất (đã nhúng CSS/JS/dữ liệu) để gửi cho người khác.

## Tính năng

- **Làm bài**: chọn chủ đề, số câu, độ khó, dạng câu (một đáp án / nhiều đáp án / tính toán nhập số), nguồn câu (chưa làm, còn yếu, sai lần trước, đã đánh dấu), chỉ câu bẫy.
- **2 chế độ**: *Luyện tập* (kiểm tra từng câu, xem giải thích ngay) và *Thi thử* (đếm ngược, chỉ chấm khi nộp).
- **Chấm điểm**: Cơ bản 1đ · Trung cấp 2đ · Nâng cao 3đ.
  - Nhiều đáp án: chấm từng phần `max(0, (chọn đúng − chọn sai) / số đáp án đúng)`, hoặc bật "đúng hết mới có điểm".
  - Tính toán: so với đáp án ± tolerance; chấp nhận `12,5`, `1e9`, `1/3`.
  - Tuỳ chọn trừ 25% điểm câu khi trả lời sai.
- **Kết quả**: điểm %, xếp loại, tỉ lệ né bẫy, phân tích theo chủ đề / độ khó / dạng câu, mảng cần ôn, xem lại từng câu kèm giải thích, làm lại câu sai.
- **Kiến thức**: cheat-sheet từng chủ đề, tìm kiếm toàn văn (cũng có trong `KIEN_THUC.md`).
- **Ngân hàng câu**: duyệt/tìm/lọc mọi câu, xem đáp án, luyện nhanh danh sách đang lọc.
- **Thống kê**: biểu đồ điểm, độ thành thạo từng chủ đề, mảng yếu nhất, lịch sử. Lưu ở `localStorage` của trình duyệt.
- Phím tắt khi làm bài: `1–6`/`A–F` chọn, `Enter` kiểm tra/tiếp, `←/→` chuyển câu, `S` đánh dấu.

## Cấu trúc

```
index.html            giao diện
assets/app.css        style (sáng/tối)
assets/app.js         engine: render, chấm điểm, thống kê, markdown renderer
data/src/*.json       ngân hàng câu hỏi + ghi chú kiến thức theo chủ đề (nguồn)
data/bank.js          sinh ra từ data/src (đừng sửa tay)
tools/validate.py     kiểm tra schema câu hỏi
tools/build.py        build
tools/AUTHORING.md    hướng dẫn thêm câu hỏi
```

Thêm câu: sửa/ thêm file trong `data/src/` theo `tools/AUTHORING.md`, rồi chạy lại `python tools/build.py`.
