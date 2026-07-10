# Rizline Color Selector

从图片中提取主题色，自动生成 [Rizline](https://store.steampowered.com/app/2272590/Rizline/) 谱面配色方案。

## 功能

- **主题色提取**：使用 K-Means 聚类从图片中提取主导色、强调色和候选色
- **智能配色分配**：根据色相/饱和度/亮度自动分配背景色、音符色和 UI/特效色
- **双方案生成**：每张图生成「常规段落」和「Riztime 段落」两套方案
- **多种方案对比**：标准/高对比/柔和三种风格可选
- **像素排序预览**：将原图所有像素按颜色排序重组成渐变预览图
- **线颜色衍生**：从主色调自动生成多组线颜色

## 安装

```bash
pip install pillow numpy scikit-learn matplotlib
```

## 使用

```bash
python rizline_color_selector.py
```

1. 点击「浏览图片」选择图片
2. 拖动滑块调整主题色数量（1-10）和线颜色数（3-10）
3. 选择配色模式：`single`（单方案）/ `multiple`（三方案对比）
4. 点击「生成配色方案」
5. 结果图片自动保存到 `output/` 文件夹

## 输出说明

结果图包含：
- 原图缩略图 + 像素排序重组图
- 提取到的所有主题色（RGB 标注）
- 每个主题的配色方案（常规 / Riztime 并排）
- 音符颜色、背景色、UI/特效颜色、线颜色
- 所有色块标注 RGB 和十六进制色号

## 配色约束

为保障谱面可读性，算法自动校验：
- 音符颜色不过于接近白色（避免 Tap/Drag 混淆）
- 背景色不过于接近黑色（保证音符描边可见）
- UI 颜色不过于接近白色（保证计量条可见）

## 文件结构

```
Rizline_Color_Selector/
├── rizline_color_selector.py   # 主程序
├── fonts/                      # 字体文件夹
│ └── SourceHanSansCN/          # 思源黑体（简体中文）字体文件
├── output/                     # 输出文件存放目录（运行后自动生成）
├── README.md
└── LICENSE.txt
```

## 依赖

- Python 3.8+
- Pillow
- NumPy
- scikit-learn
- Matplotlib
- tkinter (内置)

## 注意

- 以上内容均为AI生成，仅供参考
- 程序由AI生成，输出结果仅供参考

## 许可证

本项目采用双许可证：

- **代码**（`rizline_color_selector.py` 等文件）：采用 [CC0 1.0 通用公共领域奉献协议](https://creativecommons.org/publicdomain/zero/1.0/)。
- **字体**（`fonts/SourceHanSansCN/` 目录下的所有文件）：为 **思源黑体 (Source Han Sans)**，依据 **SIL 开放字体许可证 1.1 版** 授权。完整版权声明和许可证文本请参见 `fonts/SourceHanSansCN/LICENSE.txt`，官方仓库：[adobe-fonts/source-han-sans](https://github.com/adobe-fonts/source-han-sans)。

### 致谢

感谢 **Adobe** 与 **Google** 为开源社区贡献了高品质的思源黑体字体。