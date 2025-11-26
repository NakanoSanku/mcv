# mcv-base

MCV 核心抽象模块，提供统一的数据结构和模版基类，供各后端实现复用。

## 安装

```bash
# 使用 uv
uv add "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-base"

# 使用 pip
pip install "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-base"
```

**依赖**: Python >= 3.9，无外部依赖

## 快速开始

```python
from typing import List, Optional

import numpy as np

from mcv.base import ROI, MatchResult, Template


# 继承 Template 实现自定义匹配逻辑
class MyTemplate(Template):
    def find(
        self,
        image: np.ndarray,
        roi: Optional[ROI] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        effective_roi = self._resolve_roi(roi, image.shape[1], image.shape[0])
        if effective_roi is None:
            return None
        # 实现匹配逻辑...
        return MatchResult.from_box(
            x=effective_roi.x, y=effective_roi.y, width=10, height=10, score=1.0
        )

    def find_all(
        self,
        image: np.ndarray,
        roi: Optional[ROI] = None,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[MatchResult]:
        result = self.find(image, roi=roi, threshold=threshold)
        return [result] if result else []


# 使用 ROI
roi = ROI(100, 200, 50, 40)
print(roi.to_tuple())  # (100, 200, 50, 40)

# ROI 边界裁剪
clamped = roi.clamp(image_width=1920, image_height=1080)
```

## API 参考

### ROI

感兴趣区域，定义矩形搜索范围。

```python
ROI(x: int, y: int, width: int, height: int)
```

| 方法/属性 | 说明 |
|-----------|------|
| `to_tuple()` | 转换为元组 `(x, y, width, height)` |
| `clamp(image_width, image_height)` | 裁剪到图像边界，越界返回 `None` |
| `from_sequence(seq)` | 从 `[x, y, w, h]` 序列创建 |

**验证规则**:
- `width` 和 `height` 必须为正数
- 字符串和字节类型不能作为 `from_sequence` 的输入

### MatchResult

匹配结果，统一适用于所有模版类型。

```python
MatchResult(
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    center: Tuple[int, int],
    score: float
)
```

| 属性 | 说明 |
|------|------|
| `width` / `height` | 匹配区域尺寸 |
| `is_point` | 是否为点结果 (多点找色) |
| `point` | 等同于 `center` |

| 工厂方法 | 说明 |
|----------|------|
| `from_box(x, y, width, height, score)` | 从边界框创建 |
| `from_point(x, y, score=1.0)` | 创建点结果 |

### Template

模版抽象基类，所有后端实现需继承此类。

```python
Template(
    roi: Optional[ROILike] = None,      # 默认搜索区域
    threshold: float = 0.8,              # 默认匹配阈值 [0, 1]
    max_count: int = 1,                  # 默认最大返回数量
)
```

| 抽象方法 | 说明 |
|----------|------|
| `find(image, roi, threshold)` | 查找单个目标，返回 `MatchResult` 或 `None` |
| `find_all(image, roi, threshold, max_count)` | 查找所有目标，按置信度降序返回 |

**验证规则**:
- `threshold` 必须在 [0, 1] 范围内，否则抛出 `ValueError`
- `max_count` 必须为正整数，否则抛出 `ValueError`
- `roi` 可以是 `ROI` 对象或长度为 4 的序列 `[x, y, w, h]`
- 默认 ROI 为 `None`，表示全图搜索
- ROI 越界时自动裁剪；完全越界则 `find` 返回 `None`，`find_all` 返回空列表

**坐标系**: 左上角为原点，x 向右增长，y 向下增长

## 适用场景

- 作为图像匹配、多点找色、OCR 等后端的公共契约
- 需要统一匹配结果数据结构的场景
- 自定义匹配算法的基类
