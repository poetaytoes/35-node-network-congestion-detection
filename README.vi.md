# Phát hiện tắc nghẽn mạng 35 nút

[English](README.md)

Mã nguồn tái lập thí nghiệm dự báo tắc nghẽn cho mô hình tập trung và phân tán trên cấu trúc mạng
Việt Nam 35 nút. Quy trình tạo lưu lượng ghép cặp, tổng hợp số liệu theo cửa sổ 5 giây, huấn luyện
các mô hình, đánh giá trên tập kiểm thử được giữ riêng và xuất báo cáo PNG/SVG.

## Cài đặt nhanh trên Windows (không cần tạo hoặc kích hoạt môi trường ảo)

Yêu cầu: Git và Python 3.11 trở lên.

```powershell
git clone https://github.com/poetaytoes/35-node-network-congestion-detection.git
cd 35-node-network-congestion-detection
py -3.11 -m pip install --user -e .
py -3.11 -m congestion_lab.cli config validate --config configs/experiments/core_35node.yaml
```

Chạy một giá trị khởi tạo ngẫu nhiên:

```powershell
.\scripts\run_vietnam35.ps1 -RunId vietnam35_v1
```

Chạy thí nghiệm 5 giá trị khởi tạo dùng cho báo cáo:

```powershell
.\scripts\run_vietnam35_multiseed.ps1 -RunPrefix vietnam35_multiseed_v1
```

Mỗi lần chạy phải dùng `RunId` hoặc `RunPrefix` mới. Kết quả nằm trong `runs/` và `results/`; hai
thư mục này không được đưa lên Git.

## Thành phần chính

- `src/congestion_lab/`: mô phỏng, tạo đặc trưng, mô hình, chính sách, đánh giá và xuất báo cáo.
- `configs/`: cấu trúc mạng 35 nút và cấu hình YAML của thí nghiệm.
- `scripts/`: lệnh chạy một lần, chạy nhiều lần, kiểm tra và tạo lại báo cáo.
- `tests/`: kiểm thử đơn vị, hợp đồng, thuộc tính, tích hợp và hồi quy.

## Kiểm tra mã nguồn (không bắt buộc)

```powershell
py -3.11 -m pip install --user -e ".[dev]"
py -3.11 -m pytest
```

Giấy phép: MIT.
