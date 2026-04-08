diff --git a/README.md b/README.md
index cd7335f4fbf13b099c7389ed6e0ca6238cb4db15..67eadf14ceee0dce6582b974ced04897bb1ed191 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,41 @@
-# -YOLOv8-
-课程作业，还存在很多问题。
+# 抬头检测系统（YOLOv8 + 深度学习分类）
+
+## 1) 数据地址（需补充下载）
+
+- SCUT-HEAD 官方页面（检测标注来源）：https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
+- Ultralytics YOLO 预训练权重：`yolov8n.pt` / `yolov8s.pt`（可通过 `ultralytics` 自动下载）
+- TorchVision 预训练权重：
+  - `resnet18-f37072fd.pth`
+  - `mobilenet_v2-7ebf99e0.pth`
+  - `vit_b_16-c867db91.pth`
+
+> 建议将原始压缩包放在 *(`data/downloads`)* 目录中，再运行转换脚本。
+
+## 2) 关键数据文件与目录（必须存在）
+
+- 原始检测图片：*(`data/raw/images/train`, `data/raw/images/val`, `data/raw/images/test`)*
+- 原始检测标签：*(`data/raw/labels/train`, `data/raw/labels/val`, `data/raw/labels/test`)*
+- YOLO 配置文件：*(`data/raw/data.yaml`)*
+- 分类样本目录：*(`data/processed/up`, `data/processed/down`)*
+- 划分后分类目录：*(`data/split/train`, `data/split/val`, `data/split/test`)*
+- 时序样本目录（RNN/LSTM）：*(`data/sequences/train`, `data/sequences/val`, `data/sequences/test`)*
+
+## 3) 当前已修复的关键问题
+
+- 统一了配置变量命名（修复训练、转换、YOLO 训练脚本的导入报错）。
+- 修复了多处 `IMG_SIZE` 的错误传参（避免 `((224,224),(224,224))` 这类尺寸异常）。
+- 修复 `main.py` 调用模型工厂函数参数不匹配问题。
+- 为模型工厂函数增加可选 `num_classes`，兼容后续扩展。
+- 修复数据检查脚本目录结构与 `data.yaml` 检查路径错误。
+
+## 4) 建议清理的冗余文件
+
+- Python 缓存文件：*(`models/__pycache__/*`, `utils/__pycache__/*`)*
+- 运行产物（可按需保留）：*(`results/demo/*.mp4`, `results/figures/*`, `results/reports/evaluation_results.json`)*
+
+## 5) 还需要补充/确认的文件
+
+- YOLO 检测权重：*(`pretrained/weights/yolov8n.pt` 或 `pretrained/weights/yolov8s.pt`)*
+- ResNet18 权重：*(`pretrained/weights/resnet18-f37072fd.pth`)*
+- ViT 权重（使用 transformer 模型时）：*(`pretrained/weights/vit_b_16-c867db91.pth`)*
+- 训练好的分类权重（推理前需要）：*(`results/weights/best_cnn.pth` 等)*
