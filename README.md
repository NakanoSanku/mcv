# MCV (Multi-Computer Vision)

轻量级视觉识别引擎，专为自动化测试、脚本编写和群控系统设计。可独立使用或作为 AirTest 等框架的替代视觉引擎。

> **项目状态: Pre-Alpha**
>
> 本项目处于早期开发阶段，API 可能发生变化。目前仅 mcv-base 和 mcv-image 模块可用，其他功能正在开发中。

## 特性

- **统一接口**: 所有识别能力通过统一的 `Template` 抽象接口调用，屏蔽底层实现差异
- **可插拔后端**: 设计支持多种后端（OpenCV、PaddleOCR、RapidOCR 等），按需安装
- **轻量依赖**: 核心包仅依赖 NumPy，后端按需引入
- **类型安全**: 完整的类型提示 (Type Hints)

## 能力矩阵

| 能力       | 描述                                  | 后端模块             | 状态   |
| ---------- | ------------------------------------- | -------------------- | ------ |
| 模版匹配   | 传统模版匹配 + 灰度化 + NMS去重       | mcv-image            | 可用   |
| 多点找色   | 基于首点颜色 + 偏移坐标的快速定位算法 | mcv-color            | 可用   |
| OCR        | 文字识别，支持中英文                  | mcv-paddle/mcv-rapid | 可用   |
| 特征点匹配 | ORB/SIFT 等特征点检测与匹配           | mcv-features         | 计划中 |

## 模块结构

```
mcv/
├── mcv-base      # 核心抽象、数据结构 (可用)
├── mcv-image     # OpenCV 模版匹配 (可用)
├── mcv-color     # 多点找色 (可用)
├── mcv-paddle    # PaddleOCR 后端 (可用)
├── mcv-rapid     # RapidOCR 后端 (可用)
└── mcv-features  # OpenCV 特征点匹配 (计划中)
```

## 安装

推荐使用 `uv` 进行安装。

```bash
# 克隆仓库
git clone https://github.com/NakanoSanku/mcv
cd mcv

# 安装依赖 (开发模式)
uv sync
```

### 从 GitHub 安装

```bash
# 全量安装
uv add "git+https://github.com/NakanoSanku/mcv"

# 仅安装 image 模块
uv add "git+https://github.com/NakanoSanku/mcv#subdirectory=mcv-image"
```

### 依赖说明

- Python >= 3.9
- mcv-base: 无外部依赖
- mcv-image: opencv-python >= 4.11.0
- mcv-paddle: paddleocr >= 2.7.0 (可选)
- mcv-rapid: rapidocr >= 1.4.0 (可选)

## API 设计

### 核心数据结构

```python
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass(frozen=True)
class ROI:
    """感兴趣区域 (x, y, width, height)"""
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True)
class MatchResult:
    """匹配结果 - 统一适用于所有模版类型"""
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    center: Tuple[int, int]
    score: float  # 置信度 [0, 1]

    @property
    def is_point(self) -> bool:
        """是否为点结果（多点找色返回点而非区域）"""
        return self.top_left == self.bottom_right

    @property
    def point(self) -> Tuple[int, int]:
        """作为点使用，等同于center"""
        return self.center
```

**设计说明**: 统一使用 `MatchResult` 处理所有模版类型的返回值。

- **区域结果**（图像匹配、OCR、特征点）: `top_left` 和 `bottom_right` 定义边界框
- **点结果**（多点找色）: `top_left = bottom_right = center = point`，可通过 `is_point` 判断

这样设计的好处是用户代码无需区分结果类型：

```python
# 无论是图像匹配还是多点找色，用户代码一致
result = template.find(screen)
if result:
    click(result.center)  # 统一使用 center 进行点击
```

### 模版抽象基类

统一使用 `Template` 作为基类，实例自带默认参数，方法参数可覆盖。

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class Template(ABC):
    """模版抽象基类"""

    def __init__(
        self,
        roi: Optional[ROI] = None,      # 默认搜索区域
        threshold: float = 0.8,          # 默认匹配阈值
        max_count: int = 1,             # 默认最大返回数量
    ) -> None:
        self.default_roi = roi
        self.default_threshold = threshold
        self.default_max_count = max_count

    @abstractmethod
    def find(
        self,
        image,
        roi: Optional[ROI] = None,       # 覆盖默认值
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """查找单个目标"""
        pass

    @abstractmethod
    def find_all(
        self,
        image,
        roi: Optional[ROI] = None,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[MatchResult]:
        """查找所有目标"""
        pass
```

### 模版命名约定

具体模版基类由各子包定义，建议命名：

| 子包         | 模版类                 | 用途                |
| ------------ | ---------------------- | ------------------- |
| mcv-image    | `ImageTemplate`      | 图像模版匹配        |
| mcv-color    | `MultiColorTemplate` | 多点找色匹配        |
| mcv-paddle   | `PaddleOCRTemplate`  | PaddleOCR文字识别   |
| mcv-rapid    | `RapidOCRTemplate`   | RapidOCR文字识别    |
| mcv-features | `FeatureTemplate`    | 特征点匹配 (计划中) |

### 输入规范

- **图像格式**: numpy.ndarray (BGR 格式，与 OpenCV 一致)
- **ROI 格式**: `ROI(x, y, width, height)` 或 `[x, y, width, height]`
- **坐标系**: 左上角为原点，x 向右增长，y 向下增长
- **颜色格式**: 十六进制字符串 (如 "FF5500") 或 RGB 元组

### 示例用法 (计划中)

#### 图像模版匹配

```python
import cv2
from mcv.image import ImageTemplate

# 加载图像
screen = cv2.imread("screenshot.png")
template_img = cv2.imread("button.png")

# 创建模版，设置默认阈值
template = ImageTemplate(template_img, threshold=0.9)

# 使用默认参数查找
result = template.find(screen)

# 或覆盖默认参数
result = template.find(screen, roi=[0, 0, 500, 500], threshold=0.7)
if result:
    print(f"找到目标: {result.center}, 置信度: {result.score:.2f}")

# 查找所有匹配
results = template.find_all(screen, max_count=5)
for r in results:
    print(f"位置: {r.center}, 分数: {r.score:.2f}")
```

#### 颜色模式匹配

```python
# 创建模版，设置默认阈值
template = MultiColorTemplate(first_color="1E841E",
    offsets=[
        (14, -1, "2E942E"),
        (41, 17, "82C693"),
        (47, 14, "C6D7D7"),
    ], threshold=1.0)
result = template.find(screen)
if result:
    print(f"找到颜色: {result.center}")
```

#### OCR 文字匹配

```python
# 使用 PaddleOCR 后端
from mcv.paddle import PaddleOCRTemplate

# 查找特定文字
template = PaddleOCRTemplate(pattern="确认", lang="ch", threshold=0.8)
result = template.find(screen)
if result:
    print(f"找到文字 '{result.text}' 在 {result.center}, 置信度: {result.confidence:.2f}")

# 使用正则表达式匹配数字
template = PaddleOCRTemplate(pattern=r"\d+", regex=True)
results = template.find_all(screen)
for r in results:
    print(f"数字: {r.text}")

# 提取所有文字 (pattern=None)
template = PaddleOCRTemplate(pattern=None, threshold=0.5)
results = template.find_all(screen, roi=[0, 0, 500, 300])
```

```python
# 使用 RapidOCR 后端 (更轻量，启动更快)
from mcv.rapid import RapidOCRTemplate

template = RapidOCRTemplate(pattern="确认", threshold=0.8)
result = template.find(screen)

# GPU 加速 (需要 onnxruntime-gpu)
template = RapidOCRTemplate(
    pattern="目标",
    det_use_cuda=True,
    rec_use_cuda=True,
)
```

## 性能优化 (计划中)

默认配置优先保证**易用性**，高级优化参数可选开启：

| 优化项     | 默认 | 说明                                |
| ---------- | ---- | ----------------------------------- |
| 图像金字塔 | 开启 | 单目标匹配自动使用金字塔加速      |
| ROI 区域   | 全图 | 指定 `roi` 缩小搜索范围           |
| 灰度化     | 开启 | 内部自动灰度化，减少计算量        |
| NMS 去重   | 开启 | 内部固定阈值，多目标去重         |

## 与屏幕采集的集成

MCV 专注于视觉识别，屏幕采集请使用其他工具：

| 场景                    | 推荐工具                                    |
| ----------------------- | ------------------------------------------- |
| Android 模拟器 (MuMu等) | [msc-mumu](https://github.com/NakanoSanku/msc) |
| Android 真机/模拟器     | ADB screencap, Minicap, DroidCast           |
| Windows 桌面            | pyautogui, mss, PIL.ImageGrab               |

### 图像格式说明

MCV 统一使用 **BGR 格式** (与 OpenCV 一致)，如果使用其他采集工具：

- **PIL/Pillow**: RGB 格式，需转换 `cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)`
- **ADB screencap**: RGBA 或 RGB，需相应转换
- **pyautogui**: RGB 格式，需转换

示例：配合 ADB 使用

```python
import subprocess
import numpy as np
import cv2

def adb_screenshot():
    """通过 ADB 获取屏幕截图"""
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        capture_output=True
    )
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # 返回 BGR 格式

screen = adb_screenshot()
# 使用 MCV 进行识别...
```

## 技术背景

### 模版匹配

采用 OpenCV 的 `cv2.matchTemplate()` 实现，支持多种匹配算法：

- **TM_CCOEFF_NORMED** (默认): 归一化相关系数匹配，抗光照变化
- **TM_SQDIFF_NORMED**: 归一化平方差匹配

对比业界方案 ([参考](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623105/))：

- SIFT/SURF: 精度最高但计算开销大，适合复杂场景
- ORB/BRISK: 效率与精度平衡，适合实时应用
- SuperPoint: 深度学习方案，极端场景表现更好

对于自动化测试场景，传统模版匹配 + 图像金字塔优化是**最实用**的选择。

### 多点找色

灵感来自按键精灵的经典算法 ([参考](https://www.cnblogs.com/Evan-fanfan/p/11097850.html))：

1. 在图像中搜索首点颜色
2. 找到后检查所有偏移点的颜色是否匹配
3. 全部匹配则返回首点坐标，否则继续搜索

优势：对于 UI 元素定位，比模版匹配更轻量、更抗干扰。

### OCR

支持多种后端 ([性能对比](https://medium.com/@shah.vansh132/comparison-of-text-detection-techniques-easyocr-vs-kerasocr-vs-paddleocr-vs-pytesseract-vs-opencv-44c2bc22b133))：

- **PaddleOCR/RapidOCR**: 内存效率最高，中英文识别优秀
- **EasyOCR**: 使用简单，支持 70+ 语言

## 对比 AirTest

| 特性      | MCV          | AirTest         |
| --------- | ------------ | --------------- |
| 定位      | 轻量视觉引擎 | 完整自动化框架  |
| 设备控制  | 不包含       | ADB/iOS/Windows |
| UI 树访问 | 不包含       | Poco SDK        |
| 测试报告  | 不包含       | HTML 报告       |
| 视觉识别  | 可插拔多后端 | 内置固定实现    |
| 集成方式  | 作为库嵌入   | 独立框架        |

MCV 适合：已有自动化框架，只需替换/增强视觉识别能力。

## 路线图

### 近期

- [X] mcv-base 核心抽象定义
- [X] mcv-image 模版匹配实现
- [X] mcv-color 多点找色实现
- [X] 坐标缩放映射工具

### 中期

- [X] mcv-paddle PaddleOCR 集成
- [X] mcv-rapid RapidOCR 集成
- [ ] ORB/SIFT 特征点匹配

### 远期

- [ ] SuperPoint/LoFTR 深度学习匹配后端
- [ ] VLM OCR 适配 (Claude/GPT-4 Vision)

## 开发

```bash
# 克隆仓库
git clone https://github.com/NakanoSanku/mcv
cd mcv

# 安装依赖
uv sync

# 运行测试
uv run pytest
```

## 贡献

欢迎贡献代码！新增后端的步骤：

1. 创建新的子包 `mcv-xxx`
2. 继承 `Template` 基类，定义该子包的模版类型
3. 实现 `find`/`find_all` 抽象方法
4. 添加测试用例
5. 更新能力矩阵文档

## 参考

- [AirTest/Poco](https://github.com/AirtestProject/Airtest) - 网易自动化测试框架
- [OpenCV Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度 OCR 引擎
- [RapidOCR](https://github.com/RapidAI/RapidOCR) - 轻量级 OCR

## 许可证

MIT License
