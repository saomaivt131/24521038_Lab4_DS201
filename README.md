# DS201 Lab 4 – Neural Machine Translation

**Sinh viên:** Võ Thụy Sao Mai  
**MSSV:** 24521038  
**Môn học:** DS201

## Giới thiệu

Repository lưu bài thực hành Lab 4 về bài toán dịch máy từ tiếng Anh sang tiếng Việt trên bộ dữ liệu **small-PhoMT**. Thí nghiệm xây dựng và so sánh ba kiến trúc Encoder–Decoder sử dụng LSTM:

1. Encoder–Decoder LSTM không sử dụng attention.
2. Encoder–Decoder LSTM với Bahdanau attention.
3. Encoder–Decoder LSTM với Luong global-general attention và input-feeding.

Ba mô hình sử dụng cùng dữ liệu, vocabulary, seed và các siêu tham số chính để việc so sánh được công bằng. Mỗi encoder và decoder gồm 3 lớp LSTM, với embedding size và hidden size đều bằng 256.

## Kết quả thực nghiệm

| Mô hình | Dev ROUGE-L F1 | Test Precision | Test Recall | Test ROUGE-L F1 | Số tham số | Giây/epoch |
|---|---:|---:|---:|---:|---:|---:|
| LSTM không attention | 0.2282 | 0.2644 | 0.2146 | 0.2352 | 7,500,963 | 122.01 |
| Bahdanau attention | **0.3021** | 0.3404 | 0.2958 | 0.3142 | 8,025,763 | 191.59 |
| Luong attention | 0.3010 | **0.3445** | 0.2957 | **0.3158** | 7,959,971 | 183.65 |

Hai mô hình attention cải thiện rõ rệt so với baseline. Bahdanau đạt kết quả tốt nhất trên tập validation, trong khi Luong đạt ROUGE-L F1 cao nhất và NLL thấp nhất trên tập test. Chênh lệch giữa hai mô hình attention khá nhỏ, vì vậy chưa thể khẳng định mô hình nào vượt trội hoàn toàn nếu chưa lặp lại thí nghiệm với nhiều seed.

## Cấu trúc repository

```text
24521038_Lab4_DS201/
├── 24521038_Lab4.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── train_eval.py
│   └── visualization.py
├── checkpoints/
│   ├── bai1_baseline.pt
│   ├── bai2_bahdanau.pt
│   └── bai3_luong.pt
└── reports/
    └── Bao_cao_Lab4_DS201_24521038.pdf
```

## Nội dung chính

- [`24521038_Lab4.ipynb`](24521038_Lab4.ipynb): notebook chính, gồm quá trình chuẩn bị dữ liệu, huấn luyện, đánh giá và nhận xét cho cả ba bài.
- [`src/`](src): mã nguồn xử lý dữ liệu, định nghĩa mô hình, huấn luyện, đánh giá và trực quan hóa.
- [`checkpoints/`](checkpoints): checkpoint tốt nhất của ba mô hình sau khi huấn luyện.
- [`reports/Bao_cao_Lab4_DS201_24521038.pdf`](reports/Bao_cao_Lab4_DS201_24521038.pdf): báo cáo tổng hợp thiết lập thí nghiệm, kết quả và nhận xét.

## Môi trường sử dụng

- Python 3.12
- PyTorch
- NumPy, pandas
- Matplotlib, seaborn
- rouge-score
- Jupyter Notebook

Có thể cài các thư viện chính bằng lệnh:

```bash
pip install torch numpy pandas matplotlib seaborn rouge-score jupyter
```

## Xem kết quả

Clone repository:

```bash
git clone https://github.com/saomaivt131/24521038_Lab4_DS201.git
cd 24521038_Lab4_DS201
```

Mở notebook:

```bash
jupyter notebook 24521038_Lab4.ipynb
```

Notebook đã lưu các output của lần chạy hoàn chỉnh. Để huấn luyện lại từ đầu, cần chuẩn bị bộ dữ liệu small-PhoMT theo cấu trúc đường dẫn được mô tả trong notebook. Dữ liệu không được đưa vào repository này.

## Ghi chú

- Các mô hình được huấn luyện tối đa 10 epoch với seed 42.
- Checkpoint được chọn dựa trên dev ROUGE-L F1; nếu hòa thì ưu tiên dev NLL thấp hơn.
- Quá trình đánh giá sử dụng greedy decoding và ROUGE-L theo từng câu.
- Các file checkpoint có tổng dung lượng khoảng 90 MB.

