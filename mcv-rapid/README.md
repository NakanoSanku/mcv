# mcv-rapid

RapidOCR 后端，为 MCV 框架提供基于 ONNX Runtime 的快速文字识别能力。

## 特性

- 基于 RapidOCR，使用 ONNX Runtime 进行推理
- 比 PaddleOCR 更轻量，启动更快
- 支持精确匹配、子串匹配、正则表达式匹配
- 支持 ROI 区域限定
- 支持 CUDA GPU 加速
- 线程安全的客户端缓存

## 安装

```bash
# 基础安装
pip install mcv-rapid

# 安装 RapidOCR 依赖
pip install mcv-rapid[rapidocr]

# GPU 支持
pip install mcv-rapid[rapidocr] onnxruntime-gpu
```

## 使用示例

### 查找特定文字

```python
import cv2
from mcv.rapid import RapidOCRTemplate

# 读取图片
image = cv2.imread("screenshot.png")

# 创建模板，查找"确认"按钮
template = RapidOCRTemplate(pattern="确认", threshold=0.8)
result = template.find(image)

if result:
    print(f"找到文字 '{result.text}' 于 {result.center}")
    print(f"置信度: {result.confidence:.2f}")
```

### 使用正则表达式匹配

```python
from mcv.rapid import RapidOCRTemplate

# 匹配数字
template = RapidOCRTemplate(pattern=r"\d+", regex=True, threshold=0.5)
results = template.find_all(image)

for r in results:
    print(f"找到数字: {r.text}")
```

### 提取指定区域的所有文字

```python
from mcv.rapid import RapidOCRTemplate

# pattern=None 返回所有识别到的文字
template = RapidOCRTemplate(pattern=None, threshold=0.5)
results = template.find_all(image, roi=[100, 100, 300, 200])

for r in results:
    print(f"{r.text}: {r.confidence:.2f}")
```

### GPU 加速

```python
from mcv.rapid import RapidOCRTemplate

# 启用 CUDA 加速（需要安装 onnxruntime-gpu）
template = RapidOCRTemplate(
    pattern="目标文字",
    det_use_cuda=True,
    cls_use_cuda=True,
    rec_use_cuda=True,
)
result = template.find(image)
```

## API 参考

### RapidOCRTemplate

```python
RapidOCRTemplate(
    pattern=None,           # 目标文字/正则表达式，None 表示匹配所有
    regex=False,            # 是否启用正则匹配
    roi=None,               # 默认搜索区域 [x, y, width, height]
    threshold=0.5,          # 置信度阈值 [0, 1]
    max_count=10,           # 最大返回数量
    det_use_cuda=False,     # 检测模型使用 CUDA
    cls_use_cuda=False,     # 分类模型使用 CUDA
    rec_use_cuda=False,     # 识别模型使用 CUDA
)
```

### RapidOCRMatchResult

OCR 识别结果，包含以下属性：

- `text`: 识别的文字
- `confidence`: 置信度 [0, 1]
- `quad`: 四边形坐标（四个角点）
- `top_left`: 边界框左上角
- `bottom_right`: 边界框右下角
- `center`: 边界框中心点
- `width`: 边界框宽度
- `height`: 边界框高度

方法：
- `to_match_result()`: 转换为核心 MatchResult 类型

## 与 mcv-paddle 的比较

| 特性 | mcv-rapid | mcv-paddle |
|------|-----------|------------|
| 推理引擎 | ONNX Runtime | PaddlePaddle |
| 启动速度 | 快 | 较慢 |
| 内存占用 | 较小 | 较大 |
| 语言支持 | 中英文 | 多语言 |
| GPU 支持 | CUDA | CUDA |

## 许可证

MIT License
