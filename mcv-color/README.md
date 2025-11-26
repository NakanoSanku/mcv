# mcv-color

多点找色模板匹配实现，使用首点颜色 + 偏移点颜色的经典算法进行 UI 元素定位。

## 安装

```bash
# 使用 uv
uv add "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-color"

# 使用 pip
pip install "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-color"
```

**依赖**: Python >= 3.9, numpy, mcv-base

## 快速开始

```python
import cv2

from mcv.color import MultiColorTemplate

# 加载图像 (BGR 格式)
screen = cv2.imread("screenshot.png")

# 创建多点找色模板
# first_color: 首点颜色 (十六进制或 RGB 元组)
# offsets: 偏移点列表 [(dx, dy, color), ...]
template = MultiColorTemplate(
    first_color="1E841E",
    offsets=[
        (14, -1, "2E942E"),
        (41, 17, "82C693"),
        (47, 14, "C6D7D7"),
    ],
    threshold=0.95,  # 允许 5% 颜色差异
)

# 查找单个目标
result = template.find(screen)
if result:
    print(f"找到目标: {result.center}")

# 查找多个目标
results = template.find_all(screen, max_count=10)
for r in results:
    print(f"位置: {r.center}")
```

## 算法原理

灵感来自按键精灵的经典多点找色算法：

1. 在图像中搜索所有匹配首点颜色的像素
2. 对每个候选点，检查所有偏移位置的颜色是否匹配
3. 全部匹配则返回首点坐标，否则继续搜索下一个候选点

**优势**: 比图像模板匹配更轻量，对 UI 缩放和轻微变化更具鲁棒性。

## API 参考

### MultiColorTemplate

```python
MultiColorTemplate(
    first_color: ColorLike,            # 首点颜色
    offsets: Sequence[OffsetSpec],     # 偏移点列表
    *,
    roi: Optional[ROILike] = None,     # 默认搜索区域
    threshold: float = 1.0,            # 颜色相似度阈值 [0, 1]
    max_count: int = 1,                # 默认最大返回数量
)
```

### 颜色格式

| 格式 | 示例 | 说明 |
|------|------|------|
| 十六进制 | `"FF5500"` | 6 位 RGB 十六进制字符串 |
| 带 # 前缀 | `"#FF5500"` | 同上，自动去除 # |
| RGB 元组 | `(255, 85, 0)` | RGB 整数元组，每通道 [0, 255] |
| RGB 列表 | `[255, 85, 0]` | 同上 |

### 偏移点格式

```python
offsets = [
    (dx, dy, color),  # dx: x 方向偏移, dy: y 方向偏移, color: 目标颜色
    (14, -1, "2E942E"),
    (41, 17, (130, 198, 147)),
]
```

### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `find(image, roi, threshold)` | `MatchResult \| None` | 返回第一个匹配的点 |
| `find_all(image, roi, threshold, max_count)` | `List[MatchResult]` | 返回所有匹配点，按扫描顺序 |

### 阈值说明

| threshold | 说明 |
|-----------|------|
| `1.0` | 精确匹配，颜色必须完全一致 |
| `0.95` | 允许约 5% 差异 (每通道 ~13) |
| `0.9` | 允许约 10% 差异 (每通道 ~25) |

阈值计算公式：`tolerance = (1 - threshold) * 255`

## 行为说明

- **图像格式**: 输入图像为 BGR 格式 (OpenCV 默认)，内部自动转换为 RGB 进行匹配
- **ROI 处理**: 支持 `ROI` 对象或 `[x, y, width, height]` 列表；默认全图搜索
- **返回类型**: 返回 `MatchResult` 点结果 (`is_point=True`)，`score` 固定为 1.0
- **扫描顺序**: 结果按从上到下、从左到右的顺序返回
- **越界检查**: 偏移点超出图像边界时该候选点不匹配

## 验证规则

- `first_color` 和 `offsets` 中的颜色必须是有效格式
- RGB 值必须在 [0, 255] 范围内
- `threshold` 必须在 [0, 1] 范围内
- `max_count` 必须为正整数
- 图像必须是 3 通道 numpy.ndarray

## 性能建议

| 场景 | 建议 |
|------|------|
| 大图搜索 | 使用 `roi` 限制搜索范围 |
| 多候选点 | 增加偏移点数量提高精确度 |
| 允许颜色波动 | 降低 `threshold` 值 |

## 与图像模板匹配对比

| 特性 | 多点找色 | 图像模板 |
|------|----------|----------|
| 匹配方式 | 颜色点位 | 像素区域 |
| 抗缩放 | 需调整偏移 | 需重新截图 |
| 抗颜色变化 | threshold 容差 | 灰度化 |
| 计算量 | 较低 | 较高 |
| 适用场景 | UI 按钮定位 | 复杂图案匹配 |
