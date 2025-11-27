# mcv-image

基于 OpenCV 的图像模版匹配实现，提供灰度化、多种匹配方法和 NMS 去重能力。

## 安装

```bash
# 使用 uv
uv add "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-image"

# 使用 pip
pip install "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-image"
```

**依赖**: Python >= 3.9, opencv-python >= 4.11.0, mcv-base

## 快速开始

```python
import cv2

from mcv.image import ImageTemplate

# 加载图像 (BGR 格式)
screen = cv2.imread("screenshot.png")
button = cv2.imread("button.png")

# 创建模版
template = ImageTemplate(button, threshold=0.9)

# 查找单个目标
result = template.find(screen)
if result:
    print(f"找到目标: {result.center}, 置信度: {result.score:.2f}")

# 查找多个目标，指定搜索区域
results = template.find_all(
    screen,
    roi=[0, 0, 800, 600],  # 限制搜索范围
    threshold=0.8,
    max_count=5,
)
for r in results:
    print(f"位置: {r.center}, 分数: {r.score:.2f}")
```

## API 参考

### ImageTemplate

```python
ImageTemplate(
    template_image: np.ndarray,        # 模版图像 (BGR 或灰度)
    *,
    roi: Optional[ROILike] = None,     # 默认搜索区域
    threshold: float = 0.8,            # 默认匹配阈值 [0, 1]
)
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `template_image` | 模版图像，numpy.ndarray，支持 BGR (HxWx3) 或灰度 (HxW) |
| `threshold` | 默认匹配阈值，范围 [0, 1] |

### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `find(image, roi, threshold)` | `MatchResult \| None` | 返回置信度最高的单个匹配 |
| `find_all(image, roi, threshold, max_count)` | `List[MatchResult]` | 返回所有匹配，按置信度降序，经过 NMS 去重 |

### 匹配方法

| 方法 | 说明 |
|------|------|
| `cv2.TM_CCOEFF_NORMED` | 归一化相关系数 (默认)，抗光照变化 |
| `cv2.TM_SQDIFF_NORMED` | 归一化平方差，自动转换为相似度分数 |

> 平方差方法 (`TM_SQDIFF*`) 会自动反转得分，确保分数范围统一为 [0, 1]，分数越高匹配越好。

## 行为说明

- **ROI 处理**: 支持 `ROI` 对象或 `[x, y, width, height]` 列表；默认为全图搜索；越界自动裁剪，完全越界返回空列表
- **尺寸检查**: 模版尺寸必须小于搜索区域，否则返回空结果
- **类型验证**: `template_image` 和 `image` 必须是 numpy.ndarray (HxW 或 HxWx3)，否则抛出 `TypeError`/`ValueError`
- **参数验证**: `threshold` 必须在 [0, 1] 范围内，`max_count`（如指定）必须为正整数，否则抛出 `ValueError`
- **NMS 去重**: 基于 IoU 去除重叠匹配，避免重复检测
- **分数归一化**: 所有分数统一到 [0, 1] 范围，便于跨方法比较

### 图像格式要求

| 格式 | shape | 说明 |
|------|-------|------|
| BGR | (H, W, 3) | OpenCV 默认格式 |
| 灰度 | (H, W) | 单通道图像 |

## 性能建议

| 场景 | 建议 |
|------|------|
| 大图搜索 | 使用 `roi` 限制搜索范围 |
| 多目标 | 依赖内置 NMS 去重，必要时可通过实例属性微调 |
