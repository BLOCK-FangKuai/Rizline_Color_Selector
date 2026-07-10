#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rizline Color Selector
从图片中提取主题色，生成 Rizline 谱面配色方案
包含：音符颜色、背景色、UI/打击特效颜色、线的颜色
分别生成「常规段落」和「Riztime段落」两套方案
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import colorsys
import os
import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure
from matplotlib import font_manager

# ── 尝试导入 sklearn ──────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ══════════════════════════════════════════════════════════════════
#  字体加载 — 优先使用工作区中的思源黑体
# ══════════════════════════════════════════════════════════════════

_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'fonts', 'SourceHanSansCN', 'SubsetOTF', 'CN')
_FONT_REGULAR = os.path.join(_FONT_DIR, 'SourceHanSansCN-Regular.otf')
_FONT_BOLD = os.path.join(_FONT_DIR, 'SourceHanSansCN-Bold.otf')
_FONT_MEDIUM = os.path.join(_FONT_DIR, 'SourceHanSansCN-Medium.otf')
_FONT_LIGHT = os.path.join(_FONT_DIR, 'SourceHanSansCN-Light.otf')

if os.path.exists(_FONT_REGULAR):
    for fp in [_FONT_REGULAR, _FONT_BOLD, _FONT_MEDIUM, _FONT_LIGHT]:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    matplotlib.rcParams['font.family'] = 'Source Han Sans CN'
    matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans CN']
    matplotlib.rcParams['axes.unicode_minus'] = False
    FONT_AVAILABLE = True
    _FONT_NAME = 'Source Han Sans CN'
else:
    FONT_AVAILABLE = False
    _FONT_NAME = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False

# ── 字号表（结果图用） ──
FONT_SIZES = {
    'title': 14,
    'subtitle': 9,
    'theme_title': 8,
    'scheme_name': 7,
    'section_label': 6.5,
    'item_name': 6,
    'rgb': 5.5,
    'line_rgb': 5,
    'legend': 6,
}

# ══════════════════════════════════════════════════════════════════
#  颜色工具函数
# ══════════════════════════════════════════════════════════════════

def rgb_to_hsv(r, g, b):
    """RGB (0-255) → HSV (h: 0-360, s: 0-1, v: 0-1)"""
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return h * 360, s, v


def hsv_to_rgb(h, s, v):
    """HSV (h: 0-360, s: 0-1, v: 0-1) → RGB (0-255)"""
    h_norm = h / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def clamp(value, min_v=0, max_v=255):
    return max(min_v, min(int(value), max_v))


def rgb_to_hex(r, g, b):
    return f'#{r:02X}{g:02X}{b:02X}'


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def color_distance(c1, c2):
    """计算两个RGB颜色的感知距离（加权欧几里得）"""
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    # 加权，人眼对绿色更敏感
    dr, dg, db = r1 - r2, g1 - g2, b1 - b2
    return (dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114) ** 0.5


def is_too_similar(c1, c2, threshold=50):
    """判断两个颜色是否过于相似"""
    return color_distance(c1, c2) < threshold


def adjust_brightness(color, factor):
    """调整亮度: factor > 1 变亮, < 1 变暗"""
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    v = clamp(v * factor, 0, 255) / 255.0
    return hsv_to_rgb(h, s, v)


def adjust_saturation(color, factor):
    """调整饱和度: factor > 1 增加饱和度"""
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    s = clamp(s * factor, 0, 100) / 100.0
    return hsv_to_rgb(h, s, v)


def shift_hue(color, degrees):
    """色相偏移"""
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    h = (h + degrees) % 360
    return hsv_to_rgb(h, s, v)


def get_complementary(color):
    """获取互补色"""
    return shift_hue(color, 180)


def get_analogous_colors(color, count=3, step=30):
    """获取类似色（相邻色相）"""
    r, g, b = color
    h, s, v = rgb_to_hsv(r, g, b)
    colors_list = []
    for i in range(count):
        offset = (i - count // 2) * step
        new_h = (h + offset) % 360
        colors_list.append(hsv_to_rgb(new_h, s, v))
    return colors_list


# ══════════════════════════════════════════════════════════════════
#  颜色约束与校验
# ══════════════════════════════════════════════════════════════════

def relative_luminance(r, g, b):
    """计算相对亮度 (WCAG 标准)"""
    def linearize(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def contrast_ratio(c1, c2):
    """计算两个颜色的对比度 (WCAG)"""
    l1 = relative_luminance(*c1)
    l2 = relative_luminance(*c2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def is_near_white(color, threshold=0.9):
    """判断颜色是否接近白色 (value 通道)"""
    r, g, b = color
    _, _, v = rgb_to_hsv(r, g, b)
    return v > threshold


def is_near_black(color, threshold=0.08):
    """判断颜色是否接近黑色"""
    r, g, b = color
    _, _, v = rgb_to_hsv(r, g, b)
    return v < threshold


def enforce_color_constraints(regular, riztime, dominant_color):
    """
    对配色方案执行易读性约束调整：
    - 背景与音符/UI 对比度足够
    - 音符不过于接近白色
    - 背景不过于接近黑色（保证音符黑色描边可见）
    - UI 不过于接近白色
    """
    dh, ds, dv = rgb_to_hsv(*dominant_color)

    for part in [regular, riztime]:
        note = list(part['note'])
        bg = list(part['background'])
        ui = list(part['ui_effect'])

        nh, ns, nv = rgb_to_hsv(*note)

        # 1. 音符不过于接近白色：仅当颜色确实发白（低饱和+高亮）时才压暗
        if ns < 0.20 and nv > 0.85:
            nv = min(0.90, nv * 0.92)
            note = list(hsv_to_rgb(nh, max(ns, 0.08), nv))
        # 音符也不应过暗
        if nv < 0.12:
            nv = 0.20
            note = list(hsv_to_rgb(nh, min(ns, 1.0), nv))

        # 2. 背景不过于接近黑色
        bh, bs, bv = rgb_to_hsv(*bg)
        if bv < 0.10:
            bv = 0.14
            bg = list(hsv_to_rgb(bh, min(bs, 0.40), bv))

        # 3. UI 不过于接近白色（同样仅限低饱和时）
        uh, us, uv = rgb_to_hsv(*ui)
        if us < 0.20 and uv > 0.85:
            uv = min(0.90, uv * 0.92)
            ui = list(hsv_to_rgb(uh, max(us, 0.10), uv))

        # 4. Riztime 背景对比度：仅在极低时微调
        if 'Riztime' in part.get('label', ''):
            cr_note_bg = contrast_ratio(tuple(note), tuple(bg))
            if cr_note_bg < 1.6 and bv > 0.50:
                bv2 = max(0.40, bv * 0.80)
                bg = list(hsv_to_rgb(bh, min(bs, 0.50), bv2))

        # 5. 背景与 UI 对比度
        cr_ui_bg = contrast_ratio(tuple(ui), tuple(bg))
        if cr_ui_bg < 2.0:
            uh2, us2, uv2 = rgb_to_hsv(*ui)
            uv2 = min(0.92, uv2 * 1.25)
            ui = list(hsv_to_rgb(uh2, min(us2 * 1.15, 1.0), uv2))

        # 修正线颜色
        bg_vl = rgb_to_hsv(*bg)[2]
        fixed_lines = []
        for lc in part['line_colors']:
            lr, lg, lb = lc
            lh, ls, lv = rgb_to_hsv(lr, lg, lb)
            lv = max(lv, bg_vl + 0.10)
            if lv > 0.92:
                lv = 0.88
            fixed_lines.append(hsv_to_rgb(lh, min(ls, 1.0), lv))
        part['line_colors'] = fixed_lines

        part['note'] = tuple(note)
        part['background'] = tuple(bg)
        part['ui_effect'] = tuple(ui)

    return regular, riztime


# ══════════════════════════════════════════════════════════════════
#  主题色提取（K-Means 聚类）
# ══════════════════════════════════════════════════════════════════

def extract_dominant_colors(image_path, n_colors=5, sample_size=10000, n_accent=6):
    """
    从图片中提取颜色
    返回: (dominant_colors, accent_colors, minor_colors)
      - dominant:  频率最高的主题色
      - accent:    高饱和高亮的强调色（用于 UI/特效）
      - minor:     非主色但存在于图中的少量颜色（可用于背景等候选）
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((200, int(200 * img.height / img.width)), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3)

    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]

    n_total = max(n_colors + n_accent + 6, 16)
    if HAS_SKLEARN:
        kmeans = KMeans(n_clusters=n_total, random_state=42, n_init='auto')
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
    else:
        colors = simple_quantize(pixels, n_total)
        counts = np.ones(len(colors))

    # 按出现频率排序
    order = np.argsort(-counts)
    colors_sorted = colors[order]

    # 过滤过于接近的颜色，保留唯一色
    filtered = [tuple(colors_sorted[0])]
    for c in colors_sorted[1:]:
        c_tuple = tuple(c)
        if all(not is_too_similar(c_tuple, f, threshold=35) for f in filtered):
            filtered.append(c_tuple)

    # ── 分离：主题色 / accent / minor ──
    if len(filtered) >= n_colors:
        dominants = filtered[:n_colors]
        rest = filtered[n_colors:]
    else:
        dominants = filtered[:]
        rest = []

    # accent: 从剩余中选饱和度高+亮度高的
    def colorful_score(c):
        _, s, v = rgb_to_hsv(*c)
        return s * 0.6 + v * 0.4

    # 也从高饱和像素中采样
    colorful_pixels = pixels[
        (pixels[:, 0] > 60) & (pixels[:, 1] > 60) & (pixels[:, 2] > 60) &
        (np.max(pixels, axis=1) - np.min(pixels, axis=1) > 40)
    ]
    accent_from_pixels = []
    if len(colorful_pixels) > 200:
        if HAS_SKLEARN:
            k2 = KMeans(n_clusters=min(n_accent + 2, len(colorful_pixels)), random_state=42, n_init='auto')
            k2.fit(colorful_pixels)
            accent_from_pixels = [tuple(c.astype(int)) for c in k2.cluster_centers_]
        else:
            idx2 = np.random.choice(len(colorful_pixels), min(n_accent + 2, len(colorful_pixels)), replace=False)
            accent_from_pixels = [tuple(colorful_pixels[i]) for i in idx2]

    all_accent_candidates = list(set(rest + accent_from_pixels))
    all_accent_candidates.sort(key=colorful_score, reverse=True)

    final_accent = []
    final_minor = []
    for ac in all_accent_candidates:
        if len(final_accent) >= n_accent and len(final_minor) >= n_accent:
            break
        if all(color_distance(ac, d) > 45 for d in dominants):
            h, s, v = rgb_to_hsv(*ac)
            if v > 0.45 and s > 0.35 and len(final_accent) < n_accent:
                final_accent.append(ac)  # 高饱和高亮 → accent
            elif len(final_minor) < n_accent:
                final_minor.append(ac)    # 低饱和/低亮 → minor 候选

    if not final_accent:
        for dc in dominants:
            h, s, v = rgb_to_hsv(*dc)
            comp = hsv_to_rgb((h + 150) % 360, min(s * 1.3, 1.0), min(v * 1.3, 0.95))
            final_accent.append(comp)

    if not final_minor:
        # fallback minor: 从 dominants 中衍生低饱和版本
        for dc in dominants[:4]:
            h, s, v = rgb_to_hsv(*dc)
            low_s = max(0.08, s * 0.3)
            low_v = max(0.15, v * 0.4)
            final_minor.append(hsv_to_rgb(h, low_s, low_v))

    return dominants, final_accent[:n_accent], final_minor[:n_accent]


def simple_quantize(pixels, n_colors):
    """不使用 sklearn 的简易颜色量化"""
    # 将颜色空间划分为网格，取每个网格的平均值
    bins = int(np.ceil(n_colors ** (1/3)))
    h, w, _ = (8, 8, 8)  # 将 256^3 划分为 8^3 = 512 个格子
    quantized = (pixels // 32).astype(int)
    unique_bins = np.unique(quantized, axis=0)

    if len(unique_bins) >= n_colors:
        # 采样 n_colors 个格子
        sampled = unique_bins[np.random.choice(len(unique_bins), n_colors, replace=False)]
        return sampled * 32 + 16
    else:
        # 从原始像素中随机采样
        idx = np.random.choice(len(pixels), n_colors, replace=False)
        return pixels[idx]


# ══════════════════════════════════════════════════════════════════
#  像素按颜色排序重组图
# ══════════════════════════════════════════════════════════════════

def create_color_sorted_preview(image_path, target_height=300):
    """
    将原图所有像素按颜色排序重组为渐变预览图
    - 不增减像素、不修改颜色
    - 按色相→饱和度→亮度排序后 reshape
    返回: numpy array (H, W, 3) uint8
    """
    img = Image.open(image_path).convert('RGB')
    img.thumbnail((400, 400), Image.LANCZOS)
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    n = len(pixels)
    aspect = img.width / img.height
    out_h = target_height
    out_w = int(out_h * aspect)
    out_w = max(out_w, 1)

    # 转 HSV 排序: 色相 → 饱和度 → 亮度
    hsv_pixels = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        r, g, b = pixels[i] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        hsv_pixels[i] = [h, s, v]

    order = np.lexsort((-hsv_pixels[:, 1], hsv_pixels[:, 2] + (1 - hsv_pixels[:, 1]) * 0.5))
    sorted_pixels = pixels[order].astype(np.uint8)

    target_n = out_h * out_w
    if target_n > n:
        pad = np.tile(sorted_pixels[-1:], (target_n - n, 1))
        sorted_pixels = np.vstack([sorted_pixels, pad])
    elif target_n < n:
        sorted_pixels = sorted_pixels[:target_n]

    return sorted_pixels.reshape(out_h, out_w, 3)


# ══════════════════════════════════════════════════════════════════
#  Rizline 配色方案生成
# ══════════════════════════════════════════════════════════════════

def generate_color_scheme(dominant_color, accent_colors=None, minor_colors=None,
                          full_pool=None, num_line_colors=5):
    """
    从主色调 + 全部提取色生成 Rizline 配色方案
    full_pool: 所有提取到颜色的合并池（若提供则优先使用）
    """
    r, g, b = dominant_color
    h, s, v = rgb_to_hsv(r, g, b)

    # ── 组装全部候选色池 ──
    if full_pool is not None and len(full_pool) >= 3:
        all_candidates = full_pool
    else:
        all_candidates = _build_color_pool(dominant_color, accent_colors, minor_colors)

    # ── 按亮度排序分配：亮→BG, 中亮高饱→Note, 高饱对比→UI ──
    reg_bg, reg_note, reg_ui = _assign_roles(all_candidates, h, theme='regular')
    riz_bg, riz_note, riz_ui = _assign_roles(all_candidates, h, theme='riztime')

    line_colors = _generate_line_colors(h, s, v, num_line_colors, mode='regular')
    rt_line_colors = _generate_line_colors(h, s, v, num_line_colors, mode='riztime')

    regular = {
        'note': reg_note, 'background': reg_bg, 'ui_effect': reg_ui,
        'line_colors': line_colors, 'label': '常规段落'
    }
    riztime = {
        'note': riz_note, 'background': riz_bg, 'ui_effect': riz_ui,
        'line_colors': rt_line_colors, 'label': 'Riztime段落'
    }

    regular, riztime = enforce_color_constraints(regular, riztime, dominant_color)

    # 确保颜色为普通 int（非 numpy 类型）
    for part in [regular, riztime]:
        for key in ('note', 'background', 'ui_effect'):
            part[key] = tuple(int(v) for v in part[key])
        part['line_colors'] = [tuple(int(v) for v in lc) for lc in part['line_colors']]

    return regular, riztime


def _build_color_pool(dominant_color, accent_colors, minor_colors):
    """组装单个主题的所有候选色"""
    pool = [tuple(dominant_color)]
    if accent_colors:
        pool.extend(accent_colors)
    if minor_colors:
        pool.extend(minor_colors)
    seen = set()
    unique = []
    for c in pool:
        key = (c[0] // 8, c[1] // 8, c[2] // 8)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _build_full_pool(dominant_colors, accent_colors, minor_colors):
    """合并所有主题色 + 强调色 + 候选色为一个全量色池"""
    pool = list(dominant_colors)
    if accent_colors:
        pool.extend(accent_colors)
    if minor_colors:
        pool.extend(minor_colors)
    seen = set()
    unique = []
    for c in pool:
        key = (c[0] // 8, c[1] // 8, c[2] // 8)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _assign_roles(all_colors, dominant_hue, theme='regular'):
    """
    通用角色分配（无特调）：
    - Regular: BG=最亮低饱和 | Note=最鲜艳色 | UI=与Note同色系
    - Riztime: BG=同色系暗版 | Note=同色系超亮版 | UI=补色方向高饱和
    """
    # ── 过滤：只取与主色相接近的暖/冷色 ──
    def hue_dist(c):
        ch = rgb_to_hsv(*c)[0]
        return min(abs(ch - dominant_hue), 360 - abs(ch - dominant_hue))

    related = [c for c in all_colors if hue_dist(c) < 90]
    if len(related) < 3:
        related = all_colors[:]

    by_brightness = sorted(related, key=lambda c: rgb_to_hsv(*c)[2], reverse=True)

    def vibrant_score(c):
        _, cs, cv = rgb_to_hsv(*c)
        return cs * 0.7 + cv * 0.3

    by_vibrant = sorted(related, key=vibrant_score, reverse=True)

    if theme == 'regular':
        # BG: 最亮中低饱和色
        bg_candidates = [c for c in by_brightness if rgb_to_hsv(*c)[1] <= 0.45]
        bg = bg_candidates[0] if bg_candidates else by_brightness[0]

        # Note: 最鲜艳色（排除已做 BG 的）
        note_candidates = [c for c in by_vibrant if c != bg]
        note = note_candidates[0] if note_candidates else by_brightness[1]

        # UI: 同色系内与Note接近但不同的颜色，和谐为主
        note_h = rgb_to_hsv(*note)[0]
        ui_options = [c for c in by_vibrant if c not in (bg, note)]
        nearby = [c for c in ui_options if hue_dist(c) < 25]
        if nearby:
            ui = nearby[0]
        elif ui_options:
            # 微调Note色作为UI
            nh, ns, nv = rgb_to_hsv(*note)
            ui = hsv_to_rgb(nh, min(ns * 0.95, 1.0), min(nv * 0.95, 0.98))
        else:
            ui = note

    else:  # riztime
        # Note: 同色系中取最高 vibrant（比Regular Note更亮更饱和）
        same_family = [c for c in by_vibrant if hue_dist(c) < 30]
        note = same_family[0] if same_family else by_vibrant[0]

        # BG: 同色系中偏亮、饱和度偏高的
        mid_family = [c for c in related if hue_dist(c) < 45
                      and 0.50 <= rgb_to_hsv(*c)[2] <= 0.90]
        if mid_family:
            # 亮度 + 饱和度综合排序（亮且有色彩）
            bg = sorted(mid_family,
                        key=lambda c: -(rgb_to_hsv(*c)[2] * 0.6 + rgb_to_hsv(*c)[1] * 0.4))[0]
        else:
            fallback = sorted([c for c in related if hue_dist(c) < 45],
                              key=lambda c: abs(rgb_to_hsv(*c)[2] - 0.60))
            bg = fallback[0] if fallback else related[0]

        # UI: 补色方向高饱和（与Note色相差距 ~120-180°）
        note_h2 = rgb_to_hsv(*note)[0]
        comp_ui = [c for c in by_vibrant if c not in (note, bg)
                   and 90 <= hue_dist(c) <= 180]
        if comp_ui:
            ui = comp_ui[0]
        else:
            remaining = [c for c in by_vibrant if c not in (note, bg)]
            ui = remaining[0] if remaining else by_vibrant[-1]

    return bg, note, ui


def _pick_low_saturation_bg(minor_colors, dominant_color, dh, ds, dv, darker=False):
    """
    从 minor_colors 中选低饱和背景色（回退方案）
    """
    low_sat_candidates = []
    all_sources = list(minor_colors) if minor_colors else []
    for mc in all_sources:
        mh, ms, mv = rgb_to_hsv(*mc)
        if ms <= 0.45 and 0.08 <= mv <= 0.65:
            if darker:
                mv = max(0.08, mv * 0.65)
            low_sat_candidates.append(hsv_to_rgb(mh, min(ms, 0.40), mv))

    if low_sat_candidates:
        return max(low_sat_candidates, key=lambda c: color_distance(dominant_color, c))

    bg_s = min(max(0.08, ds * 0.25), 0.30)
    bg_v = max(0.18, dv * 0.30) if not darker else max(0.12, dv * 0.20)
    return hsv_to_rgb(dh, bg_s, bg_v)


def _generate_line_colors(h, s, v, count, mode='regular'):
    """
    生成一组线的颜色（近似色相、不同明度/饱和度）
    可以有多组，这里生成一组渐变
    """
    colors = []
    if mode == 'regular':
        # 常规：色相轻微偏移，亮度渐变
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            # 色相在 ±15° 范围内变化
            hue_offset = (t - 0.5) * 30
            line_h = (h + hue_offset) % 360
            # 饱和度从 s*0.7 到 s
            line_s = s * (0.7 + t * 0.3)
            # 亮度从 v*0.6 到 v*1.1
            line_v = v * (0.6 + t * 0.5)
            line_v = min(line_v, 1.0)
            colors.append(hsv_to_rgb(line_h, min(line_s, 1.0), line_v))
    else:
        # Riztime：色相变化更大，更亮更饱和
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0.5
            hue_offset = (t - 0.5) * 45
            line_h = (h + hue_offset) % 360
            line_s = min(s * (0.8 + t * 0.4), 1.0)
            line_v = min(v * (0.7 + t * 0.6), 1.0)
            colors.append(hsv_to_rgb(line_h, line_s, line_v))

    return colors


def build_line_color_groups_from_pool(full_pool, n_line=5, similarity_threshold=40):
    """
    从颜色池中选线颜色并分组
    - 按鲜艳度+独特性选 n_line 个主要线颜色
    - 每个线颜色在池中查找相近色（色距 < 阈值）
    - 返回: [{main: (r,g,b), similars: [(r,g,b), ...]}, ...]
    """
    if not full_pool or len(full_pool) < 2:
        return []

    # 按鲜艳度排序选主要线颜色
    sorted_pool = sorted(full_pool, key=lambda c: rgb_to_hsv(*c)[1] + rgb_to_hsv(*c)[2] * 0.3, reverse=True)
    # 过滤掉过于接近的颜色，确保线颜色之间有区分度
    selected = [sorted_pool[0]]
    for c in sorted_pool[1:]:
        if all(color_distance(c, s) > 60 for s in selected):
            selected.append(c)
        if len(selected) >= n_line:
            break

    # 为每个主要线颜色查找相近色
    groups = []
    for mc in selected:
        similars = []
        for pc in full_pool:
            if pc != mc and color_distance(mc, pc) < similarity_threshold:
                similars.append(pc)
        # 相近色也去重
        deduped = [similars[0]] if similars else []
        for s in similars[1:]:
            if all(color_distance(s, d) > 15 for d in deduped):
                deduped.append(s)
        groups.append({'main': mc, 'similars': deduped[:5]})

    return groups


# ══════════════════════════════════════════════════════════════════
#  备选配色方案（多种风格呈现给用户选择）
# ══════════════════════════════════════════════════════════════════

def generate_multiple_schemes(dominant_color, full_pool=None, num_line_colors=5):
    """
    生成多种配色方案供用户选择
    """
    schemes = []

    # ── 方案1: 标准方案 ──
    reg1, riz1 = generate_color_scheme(dominant_color, full_pool=full_pool, num_line_colors=num_line_colors)
    schemes.append({'name': '标准方案', 'regular': reg1, 'riztime': riz1})

    r, g, b = dominant_color
    h, s, v = rgb_to_hsv(r, g, b)

    # ── 方案2: 高对比方案（从全色池分配） ──
    reg2_bg, reg2_note, reg2_ui = _assign_roles(full_pool, h, theme='regular')
    reg2_note = hsv_to_rgb(h, min(s * 1.2, 1.0), min(v * 1.1, 0.92))
    reg2_lines = _generate_line_colors(h, s * 1.1, v * 1.1, num_line_colors, 'regular')
    reg2_lines = [(clamp(c[0]), clamp(c[1]), clamp(c[2])) for c in reg2_lines]

    reg2 = {
        'note': reg2_note, 'background': reg2_bg,
        'ui_effect': reg2_ui, 'line_colors': reg2_lines,
        'label': '常规段落'
    }

    riz2_bg, riz2_note, riz2_ui = _assign_roles(full_pool, h, theme='riztime')
    riz2_note = hsv_to_rgb(h, 1.0, 0.95)
    riz2_ui = hsv_to_rgb((h + 30) % 360, min(s * 1.2, 1.0), 0.95)
    riz2_lines = _generate_line_colors(h, 1.0, 0.9, num_line_colors, 'riztime')
    riz2_lines = [(clamp(c[0]), clamp(c[1]), clamp(c[2])) for c in riz2_lines]

    riz2 = {
        'note': riz2_note, 'background': riz2_bg,
        'ui_effect': riz2_ui, 'line_colors': riz2_lines,
        'label': 'Riztime段落'
    }
    reg2, riz2 = enforce_color_constraints(reg2, riz2, dominant_color)
    schemes.append({'name': '高对比方案', 'regular': reg2, 'riztime': riz2})

    # ── 方案3: 柔和方案（从全色池分配） ──
    soft_s = max(s * 0.6, 0.2)
    soft_v = max(v * 0.8, 0.3)

    reg3_bg, reg3_note, reg3_ui = _assign_roles(full_pool, h, theme='regular')
    reg3_note = hsv_to_rgb(h, soft_s, soft_v)
    reg3_ui = hsv_to_rgb((h + 60) % 360, soft_s * 0.8, min(soft_v * 1.1, 0.9))
    reg3_lines = _generate_line_colors(h, soft_s, soft_v, num_line_colors, 'regular')
    reg3_lines = [(clamp(c[0]), clamp(c[1]), clamp(c[2])) for c in reg3_lines]

    reg3 = {
        'note': reg3_note, 'background': reg3_bg,
        'ui_effect': reg3_ui, 'line_colors': reg3_lines,
        'label': '常规段落'
    }

    riz3_bg, riz3_note, riz3_ui = _assign_roles(full_pool, h, theme='riztime')
    riz3_note = hsv_to_rgb(h, min(soft_s * 1.3, 1.0), min(soft_v * 1.2, 0.92))
    riz3_lines = _generate_line_colors(h, soft_s * 1.1, soft_v * 1.1, num_line_colors, 'riztime')
    riz3_lines = [(clamp(c[0]), clamp(c[1]), clamp(c[2])) for c in riz3_lines]

    riz3 = {
        'note': riz3_note, 'background': riz3_bg,
        'ui_effect': riz3_ui, 'line_colors': riz3_lines,
        'label': 'Riztime段落'
    }
    reg3, riz3 = enforce_color_constraints(reg3, riz3, dominant_color)
    schemes.append({'name': '柔和方案', 'regular': reg3, 'riztime': riz3})

    return schemes


# ══════════════════════════════════════════════════════════════════
#  结果可视化
# ══════════════════════════════════════════════════════════════════

def create_result_figure(image_path, dominant_colors, all_schemes, all_line_groups,
                         num_line_colors, dpi=150):
    """
    创建结果展示图（v3: 色号在色块内、点击复制HEX、可滚动）
    返回: (fig, color_axes) — color_axes 是 {axes: hex_str} 映射
    """
    fs = FONT_SIZES
    num_themes = len(dominant_colors)
    num_schemes = len(all_schemes[0]) if all_schemes else 1

    scheme_height = 1.25
    line_height = 0.55
    row_height = scheme_height + line_height + 0.6 + 0.3
    header_height = 0.5
    dominant_height = 0.55
    bottom_margin = 0.3

    total_height = header_height + dominant_height + num_themes * row_height + bottom_margin + 0.2
    fig_width = 6
    fig_height = max(12, total_height)

    fig = Figure(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')

    y = 1.0
    margin_left = 0.06
    margin_right = 0.02
    content_width = 1.0 - margin_left - margin_right

    # ── 标题 ──
    ax_title = fig.add_axes([margin_left, y - 0.05, content_width, 0.06])
    ax_title.axis('off')
    img_name = os.path.basename(image_path)
    ax_title.text(0.5, 0.5, f'Rizline 配色 — {img_name}',
                  fontsize=fs['title'], fontweight='bold', color='white',
                  ha='center', va='center', transform=ax_title.transAxes,
                  fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                  if FONT_AVAILABLE else {})
    y -= 0.08

    # ── 原图缩略图 ──
    try:
        img = Image.open(image_path).convert('RGB')
        img_thumb = img.copy()
        img_thumb.thumbnail((80, 80), Image.LANCZOS)
        ax_img = fig.add_axes([margin_left, y - 0.13, 0.10, 0.13])
        ax_img.imshow(np.array(img_thumb))
        ax_img.axis('off')
    except Exception:
        pass

    # ── 像素排序重组预览 ──
    try:
        sorted_preview = create_color_sorted_preview(image_path, target_height=100)
        ax_sort = fig.add_axes([margin_left + 0.12, y - 0.14, 0.28, 0.14])
        ax_sort.imshow(sorted_preview)
        ax_sort.axis('off')
        ax_sort.text(0.5, -0.12, u'像素排序重组',
                     fontsize=5, color='#666', ha='center', va='top',
                     transform=ax_sort.transAxes)
    except Exception:
        pass

    y_preview_bottom = y - 0.14
    y = y_preview_bottom - 0.02

    # ── 提取的主题色（横向排列） ──
    dominant_start = margin_left + 0.17
    dominant_width = content_width - 0.17
    box_w = dominant_width / max(len(dominant_colors), 1)
    for i, dc in enumerate(dominant_colors):
        ax_dc = fig.add_axes([dominant_start + i * box_w, y - 0.10, box_w * 0.85, 0.10])
        r, g, b = dc
        ax_dc.add_patch(patches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.04",
            facecolor=(r/255, g/255, b/255),
            edgecolor='white', linewidth=1.2))
        ax_dc.text(0.5, 1.25, f'主题{i+1}', fontsize=fs['item_name'], color='#aaa',
                   ha='center', va='bottom', transform=ax_dc.transAxes,
                   fontproperties=font_manager.FontProperties(family=_FONT_NAME)
                   if FONT_AVAILABLE else {})
        ax_dc.set_xlim(0, 1)
        ax_dc.set_ylim(0, 1)
        ax_dc.axis('off')
    y -= 0.12

    # ── 每个主题的配色方案 ──
    for theme_idx, dc in enumerate(dominant_colors):
        schemes = all_schemes[theme_idx]

        r, g, b = dc
        ax_theme_title = fig.add_axes([margin_left, y - 0.03, content_width, 0.03])
        ax_theme_title.axis('off')
        ax_theme_title.text(0, 0.5, f'● 主题 {theme_idx + 1} — '
                            f'RGB({r},{g},{b})  {rgb_to_hex(r, g, b)}',
                            fontsize=fs['theme_title'], fontweight='bold', color='white',
                            ha='left', va='center',
                            fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                            if FONT_AVAILABLE else {})
        y -= 0.05

        for scheme_idx, scheme in enumerate(schemes):
            scheme_name = scheme['name']
            regular = scheme['regular']
            riztime = scheme['riztime']

            ax_name = fig.add_axes([margin_left, y - 0.025, content_width, 0.025])
            ax_name.axis('off')
            ax_name.text(0, 0.5, f'  > {scheme_name}',
                         fontsize=fs['scheme_name'], color='#8cf', ha='left', va='center',
                         fontproperties=font_manager.FontProperties(family=_FONT_NAME)
                         if FONT_AVAILABLE else {})
            y -= 0.035

            for part_idx, part in enumerate([regular, riztime]):
                part_x = margin_left + part_idx * (content_width / 2 + 0.015)
                part_w = content_width / 2 - 0.025

                label = part['label']
                note_c = part['note']
                bg_c = part['background']
                ui_c = part['ui_effect']
                line_cs = part['line_colors']

                ax_label = fig.add_axes([part_x, y - 0.025, part_w, 0.025])
                ax_label.axis('off')
                color_tag = '#ff6' if 'Riztime' in label else '#6cf'
                ax_label.text(0.02, 0.5, f'【{label}】', fontsize=fs['section_label'],
                              fontweight='bold', color=color_tag, ha='left', va='center',
                              fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                              if FONT_AVAILABLE else {})
                y -= 0.03

                items = [
                    ('音符', note_c),
                    ('背景', bg_c),
                    ('UI/特效', ui_c),
                ]
                block_w = part_w / 3.5
                block_h = 0.11

                for item_idx, (item_name, item_color) in enumerate(items):
                    ix = part_x + item_idx * (block_w + 0.018)
                    ir, ig, ib = item_color
                    hex_str = rgb_to_hex(ir, ig, ib).lstrip('#')

                    ax_block = fig.add_axes([ix, y - block_h, block_w, block_h])
                    ax_block.add_patch(patches.FancyBboxPatch(
                        (0, 0), 1, 1, boxstyle="round,pad=0.1",
                        facecolor=(ir/255, ig/255, ib/255),
                        edgecolor='white', linewidth=1.5))

                    # 根据亮度选文字颜色
                    text_color = '#000' if relative_luminance(ir, ig, ib) > 0.35 else '#FFF'
                    ax_block.text(0.5, 0.78, item_name, fontsize=fs['item_name'] + 1,
                                  color=text_color if text_color == '#FFF' else '#111',
                                  ha='center', va='center', weight='bold',
                                  transform=ax_block.transAxes,
                                  fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                                  if FONT_AVAILABLE else {})
                    ax_block.text(0.5, 0.45, f'RGB({ir},{ig},{ib})', fontsize=fs['rgb'] + 0.5,
                                  color=text_color if text_color == '#FFF' else '#333',
                                  ha='center', va='center',
                                  transform=ax_block.transAxes,
                                  fontproperties=font_manager.FontProperties(family=_FONT_NAME)
                                  if FONT_AVAILABLE else {})
                    ax_block.text(0.5, 0.18, f'#{hex_str}', fontsize=fs['rgb'] + 1,
                                  color=text_color if text_color == '#FFF' else '#555',
                                  ha='center', va='center', weight='bold',
                                  transform=ax_block.transAxes,
                                  fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                                  if FONT_AVAILABLE else {})

                    ax_block.set_xlim(0, 1)
                    ax_block.set_ylim(0, 1)
                    ax_block.axis('off')

                # 线颜色
                line_y = y - block_h - 0.03
                ax_line_label = fig.add_axes([part_x, line_y - 0.02, part_w, 0.02])
                ax_line_label.axis('off')
                ax_line_label.text(0, 0.5, '线颜色:', fontsize=fs['item_name'], color='#aaa',
                                   ha='left', va='center',
                                   fontproperties=font_manager.FontProperties(family=_FONT_NAME)
                                   if FONT_AVAILABLE else {})
                line_y -= 0.022

                line_block_w = part_w / (len(line_cs) + 1)
                for li, lc in enumerate(line_cs):
                    lx = part_x + li * (line_block_w + 0.01)
                    lr, lg, lb = lc
                    lhex = rgb_to_hex(lr, lg, lb).lstrip('#')
                    ax_l = fig.add_axes([lx, line_y - 0.09, line_block_w * 0.9, 0.09])
                    ax_l.add_patch(patches.FancyBboxPatch(
                        (0, 0), 1, 1, boxstyle="round,pad=0.06",
                        facecolor=(lr/255, lg/255, lb/255),
                        edgecolor='#555', linewidth=1))
                    lt_color = '#000' if relative_luminance(lr, lg, lb) > 0.3 else '#FFF'
                    ax_l.text(0.5, 0.6, f'#{lhex}', fontsize=fs['line_rgb'] + 1,
                              color=lt_color if lt_color == '#FFF' else '#444',
                              ha='center', va='center', weight='bold',
                              transform=ax_l.transAxes,
                              fontproperties=font_manager.FontProperties(family=_FONT_NAME, weight='bold')
                              if FONT_AVAILABLE else {})
                    ax_l.set_xlim(0, 1)
                    ax_l.set_ylim(0, 1)
                    ax_l.axis('off')

            y = min(y, line_y - 0.09) - 0.04

        # ── 线颜色组 (从色池选取, 含相近色) ──
        if theme_idx < len(all_line_groups) and all_line_groups[theme_idx]:
            groups = all_line_groups[theme_idx]
            ax_lg_label = fig.add_axes([margin_left, y - 0.03, content_width, 0.03])
            ax_lg_label.axis('off')
            ax_lg_label.text(0, 0.5, u'线颜色 (从色池选取, 含相近色分组):',
                             fontsize=7, color='#888', ha='left', va='center')
            y -= 0.04

            for gi, grp in enumerate(groups):
                mc = grp['main']
                sims = grp['similars']
                mr, mg, mb = mc
                mhex = f'{mr:02X}{mg:02X}{mb:02X}'

                # 主颜色块 + HEX
                grp_x = margin_left
                main_w = 0.06
                ax_mc = fig.add_axes([grp_x, y - 0.12, main_w, 0.12])
                ax_mc.add_patch(patches.FancyBboxPatch(
                    (0, 0), 1, 1, boxstyle="round,pad=0.06",
                    facecolor=(mr/255, mg/255, mb/255),
                    edgecolor='white', linewidth=1.2))
                mc_text_color = '#000' if relative_luminance(mr, mg, mb) > 0.3 else '#FFF'
                ax_mc.text(0.5, 0.45, f'#{mhex}', fontsize=5.5, color=mc_text_color,
                           ha='center', va='center', weight='bold', transform=ax_mc.transAxes)
                ax_mc.set_xlim(0, 1); ax_mc.set_ylim(0, 1); ax_mc.axis('off')

                # 相近色块
                sim_x = grp_x + main_w + 0.01
                sim_w_total = content_width - main_w - 0.03
                num_sims = len(sims) if sims else 0
                if num_sims > 0:
                    sim_bw = sim_w_total / max(num_sims, 1)
                    for si, sc in enumerate(sims):
                        sx = sim_x + si * sim_bw
                        sr, sg, sb = sc
                        shex = f'{sr:02X}{sg:02X}{sb:02X}'
                        ax_sc = fig.add_axes([sx, y - 0.10, sim_bw * 0.9, 0.10])
                        ax_sc.add_patch(patches.FancyBboxPatch(
                            (0, 0), 1, 1, boxstyle="round,pad=0.04",
                            facecolor=(sr/255, sg/255, sb/255),
                            edgecolor='#555', linewidth=0.7))
                        sc_color = '#000' if relative_luminance(sr, sg, sb) > 0.3 else '#FFF'
                        ax_sc.text(0.5, 0.4, f'#{shex}', fontsize=5, color=sc_color,
                                   ha='center', va='center', weight='bold', transform=ax_sc.transAxes)
                        ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1); ax_sc.axis('off')

                y -= 0.14
            y -= 0.04

        y -= 0.08

    # ── 右上角时间戳 ──
    ax_ts = fig.add_axes([0.65, 0.95, 0.34, 0.035])
    ax_ts.axis('off')
    ax_ts.text(1.0, 0.5, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               fontsize=5, color='#555', ha='right', va='center',
               transform=ax_ts.transAxes)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)
    return fig


def save_and_show_result(fig, output_path=None):
    """保存结果图到文件并显示（高DPI）"""
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'rizline_scheme_{timestamp}.png')

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'✅ 结果已保存: {output_path}')
    return output_path


# ══════════════════════════════════════════════════════════════════
#  主程序入口（带 GUI）
# ══════════════════════════════════════════════════════════════════

class RizlineColorSelectorApp:
    def __init__(self, master):
        self.master = master
        master.title('Rizline 配色方案生成器')
        master.geometry('820x700')
        master.configure(bg='#1a1a2e')
        master.minsize(750, 600)

        # 变量
        self.image_path = tk.StringVar()
        self.num_dominant = tk.IntVar(value=4)
        self.num_line_colors = tk.IntVar(value=5)
        self.similarity_threshold = tk.IntVar(value=40)
        self.output_path = tk.StringVar()
        self.scheme_mode = tk.StringVar(value='single')

        self._build_ui()

    def _build_ui(self):
        # 标题
        title = tk.Label(self.master, text='🎨 Rizline 配色方案生成器',
                         font=('Microsoft YaHei UI', 22, 'bold'),
                         bg='#1a1a2e', fg='white')
        title.pack(pady=(20, 3))

        desc = tk.Label(self.master,
                        text='从图片提取主题色 → 生成谱面配色方案（常规 + Riztime）',
                        font=('Microsoft YaHei UI', 12),
                        bg='#1a1a2e', fg='#aaa')
        desc.pack(pady=(0, 15))

        main_frame = tk.Frame(self.master, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=5)

        # ── 图片选择 ──
        img_frame = tk.LabelFrame(main_frame, text='📷 选择图片',
                                  font=('Microsoft YaHei UI', 12, 'bold'),
                                  bg='#16213e', fg='white',
                                  relief=tk.GROOVE, bd=2)
        img_frame.pack(fill=tk.X, pady=(0, 12))

        btn_frame_inner = tk.Frame(img_frame, bg='#16213e')
        btn_frame_inner.pack(padx=12, pady=10, fill=tk.X)

        btn_select = tk.Button(btn_frame_inner, text='📂 浏览图片...',
                               command=self._select_image,
                               font=('Microsoft YaHei UI', 12),
                               bg='#0f3460', fg='white',
                               activebackground='#1a5276',
                               activeforeground='white',
                               relief=tk.FLAT, padx=22, pady=8,
                               cursor='hand2')
        btn_select.pack(side=tk.LEFT)

        self.lbl_image = tk.Label(btn_frame_inner, text='未选择图片',
                                  bg='#16213e', fg='#888',
                                  font=('Microsoft YaHei UI', 11))
        self.lbl_image.pack(side=tk.LEFT, padx=(15, 0))

        # ── 参数设置（Slider 风格） ──
        param_frame = tk.LabelFrame(main_frame, text='⚙ 参数设置',
                                    font=('Microsoft YaHei UI', 12, 'bold'),
                                    bg='#16213e', fg='white',
                                    relief=tk.GROOVE, bd=2)
        param_frame.pack(fill=tk.X, pady=(0, 12))

        # 主题色数量 Slider
        row1 = tk.Frame(param_frame, bg='#16213e')
        row1.pack(fill=tk.X, padx=15, pady=(12, 5))
        tk.Label(row1, text='主题色数量:', bg='#16213e', fg='#ccc',
                 font=('Microsoft YaHei UI', 11)).pack(side=tk.LEFT)
        self.lbl_dominant_val = tk.Label(row1, text='4', bg='#16213e', fg='#8cf',
                                          font=('Microsoft YaHei UI', 11, 'bold'))
        self.lbl_dominant_val.pack(side=tk.LEFT, padx=(10, 0))
        slider_dominant = tk.Scale(row1, from_=1, to=10, orient=tk.HORIZONTAL,
                                    variable=self.num_dominant,
                                    bg='#16213e', fg='white', troughcolor='#0f3460',
                                    activebackground='#e94560',
                                    highlightbackground='#16213e',
                                    length=280, width=18, sliderlength=26,
                                    font=('Microsoft YaHei UI', 9))
        slider_dominant.pack(side=tk.LEFT, padx=(18, 0), fill=tk.X, expand=True)
        slider_dominant.bind('<Motion>', lambda e: self.lbl_dominant_val.configure(
            text=str(self.num_dominant.get())))

        # 线颜色数量 Slider
        row2 = tk.Frame(param_frame, bg='#16213e')
        row2.pack(fill=tk.X, padx=15, pady=(8, 5))
        tk.Label(row2, text='每组线颜色数:', bg='#16213e', fg='#ccc',
                 font=('Microsoft YaHei UI', 11)).pack(side=tk.LEFT)
        self.lbl_line_val = tk.Label(row2, text='5', bg='#16213e', fg='#8cf',
                                      font=('Microsoft YaHei UI', 11, 'bold'))
        self.lbl_line_val.pack(side=tk.LEFT, padx=(10, 0))
        slider_line = tk.Scale(row2, from_=3, to=10, orient=tk.HORIZONTAL,
                                variable=self.num_line_colors,
                                bg='#16213e', fg='white', troughcolor='#0f3460',
                                activebackground='#e94560',
                                highlightbackground='#16213e',
                                length=280, width=18, sliderlength=26,
                                font=('Microsoft YaHei UI', 9))
        slider_line.pack(side=tk.LEFT, padx=(18, 0), fill=tk.X, expand=True)
        slider_line.bind('<Motion>', lambda e: self.lbl_line_val.configure(
            text=str(self.num_line_colors.get())))

        # 相似色阈值 Slider
        row_sim = tk.Frame(param_frame, bg='#16213e')
        row_sim.pack(fill=tk.X, padx=15, pady=(8, 5))
        tk.Label(row_sim, text='相近色阈值:', bg='#16213e', fg='#ccc',
                 font=('Microsoft YaHei UI', 11)).pack(side=tk.LEFT)
        self.lbl_sim_val = tk.Label(row_sim, text='40', bg='#16213e', fg='#8cf',
                                     font=('Microsoft YaHei UI', 11, 'bold'))
        self.lbl_sim_val.pack(side=tk.LEFT, padx=(10, 0))
        slider_sim = tk.Scale(row_sim, from_=20, to=70, orient=tk.HORIZONTAL,
                               variable=self.similarity_threshold,
                               bg='#16213e', fg='white', troughcolor='#0f3460',
                               activebackground='#e94560',
                               highlightbackground='#16213e',
                               length=280, width=18, sliderlength=26,
                               font=('Microsoft YaHei UI', 9))
        slider_sim.pack(side=tk.LEFT, padx=(18, 0), fill=tk.X, expand=True)
        slider_sim.bind('<Motion>', lambda e: self.lbl_sim_val.configure(
            text=str(self.similarity_threshold.get())))

        # 配色方案模式
        row3 = tk.Frame(param_frame, bg='#16213e')
        row3.pack(fill=tk.X, padx=15, pady=(8, 12))
        tk.Label(row3, text='配色模式:', bg='#16213e', fg='#ccc',
                 font=('Microsoft YaHei UI', 11)).pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(row3, textvariable=self.scheme_mode,
                                        values=['single', 'multiple'],
                                        state='readonly', width=14,
                                        font=('Microsoft YaHei UI', 11))
        self.mode_combo.pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(row3, text='single=单方案  multiple=三方案对比',
                 bg='#16213e', fg='#666',
                 font=('Microsoft YaHei UI', 10)).pack(side=tk.LEFT, padx=(14, 0))

        # ── 输出设置 ──
        out_frame = tk.LabelFrame(main_frame, text='💾 输出设置',
                                  font=('Microsoft YaHei UI', 12, 'bold'),
                                  bg='#16213e', fg='white',
                                  relief=tk.GROOVE, bd=2)
        out_frame.pack(fill=tk.X, pady=(0, 12))

        out_row = tk.Frame(out_frame, bg='#16213e')
        out_row.pack(fill=tk.X, padx=15, pady=10)
        tk.Label(out_row, text='输出文件夹:', bg='#16213e', fg='#ccc',
                 font=('Microsoft YaHei UI', 11)).pack(side=tk.LEFT)

        self.output_path.set(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'))
        lbl_out = tk.Label(out_row, textvariable=self.output_path,
                           bg='#16213e', fg='#8cf',
                           font=('Microsoft YaHei UI', 10))
        lbl_out.pack(side=tk.LEFT, padx=(14, 0))

        btn_out = tk.Button(out_row, text='更改...', command=self._select_output_dir,
                            font=('Microsoft YaHei UI', 10),
                            bg='#0f3460', fg='white',
                            activebackground='#1a5276', relief=tk.FLAT,
                            cursor='hand2', padx=14, pady=4)
        btn_out.pack(side=tk.RIGHT, padx=10)

        # ── 操作按钮 ──
        btn_frame = tk.Frame(main_frame, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, pady=8)

        self.btn_generate = tk.Button(btn_frame, text='🚀 生成配色方案',
                                      command=self._generate,
                                      font=('Microsoft YaHei UI', 15, 'bold'),
                                      bg='#e94560', fg='white',
                                      activebackground='#c73a52',
                                      activeforeground='white',
                                      relief=tk.FLAT, padx=38, pady=13,
                                      cursor='hand2', state=tk.DISABLED)
        self.btn_generate.pack(side=tk.LEFT, padx=(0, 20))

        self.progress = ttk.Progressbar(btn_frame, mode='indeterminate', length=240)
        self.progress.pack(side=tk.LEFT, padx=10)

        # ── 状态信息 ──
        self.status_text = tk.Text(main_frame, height=6, bg='#0d1117', fg='#8b949e',
                                    font=('Consolas', 11), relief=tk.FLAT,
                                    insertbackground='white')
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.status_text.insert(tk.END, '就绪 — 选择一张图片后点击"生成配色方案"\n')
        self.status_text.configure(state=tk.DISABLED)

    def _log(self, msg):
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, msg + '\n')
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)
        self.master.update()

    def _select_image(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('图片文件', '*.png *.jpg *.jpeg *.bmp *.tiff *.webp'),
                       ('所有文件', '*.*')]
        )
        if path:
            self.image_path.set(path)
            self.lbl_image.configure(text=os.path.basename(path), fg='#8cf')
            self.btn_generate.configure(state=tk.NORMAL)

    def _select_output_dir(self):
        path = filedialog.askdirectory(title='选择输出文件夹')
        if path:
            self.output_path.set(path)

    def _generate(self):
        img_path = self.image_path.get()
        if not img_path or not os.path.isfile(img_path):
            messagebox.showerror('错误', '请先选择一张有效的图片')
            return

        self.btn_generate.configure(state=tk.DISABLED)
        self.progress.start()
        self._log('🔄 正在处理...')
        self.master.update()

        try:
            n_colors = self.num_dominant.get()
            n_line = self.num_line_colors.get()
            mode = self.scheme_mode.get()

            # 1. 提取主题色 + accent 色 + minor 色
            self._log(f'📷 提取 {n_colors} 个主题色、强调色与候选色...')
            dominant_colors, accent_colors, minor_colors = extract_dominant_colors(img_path, n_colors)
            self._log(f'✅ 主题色 {len(dominant_colors)} 个, 强调色 {len(accent_colors)} 个, 候选色 {len(minor_colors)} 个')

            for i, dc in enumerate(dominant_colors):
                self._log(f'   主题 {i+1}: RGB{dc} {rgb_to_hex(*dc)}')
            if accent_colors:
                self._log(f'   强调色(UI用): {", ".join(f"RGB{ac}" for ac in accent_colors[:3])}...')
            if minor_colors:
                self._log(f'   候选色(背景用): {", ".join(f"RGB{mc}" for mc in minor_colors[:3])}...')

            # 2. 构建全色池（主题色互相可见）并生成方案
            self._log(f'🎨 生成配色方案（已应用易读性约束）...')
            full_pool = _build_full_pool(dominant_colors, accent_colors, minor_colors)
            all_schemes = []
            all_line_groups = []

            for dc in dominant_colors:
                if mode == 'multiple':
                    schemes = generate_multiple_schemes(dc, full_pool, n_line)
                else:
                    reg, riz = generate_color_scheme(dc, full_pool=full_pool, num_line_colors=n_line)
                    schemes = [{'name': '标准方案', 'regular': reg, 'riztime': riz}]

                all_schemes.append(schemes)

            # 从色池构建线颜色分组（所有主题共用）
            threshold = self.similarity_threshold.get()
            line_groups = build_line_color_groups_from_pool(full_pool, n_line, threshold)
            for _ in dominant_colors:
                all_line_groups.append(line_groups)

            # 3. 创建结果图
            self._log(f'📊 生成结果图表...')
            fig = create_result_figure(img_path, dominant_colors, all_schemes,
                                        all_line_groups, n_line, dpi=150)

            # 4. 保存（高DPI 300）
            output_dir = self.output_path.get()
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(output_dir, f'rizline_scheme_{timestamp}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor=fig.get_facecolor())
            self._log(f'✅ 结果已保存: {save_path}')

            # 5. 关闭 matplotlib figure 释放内存
            plt.close(fig)

            self._log('🎉 完成！')
            self._log(f'📂 文件位置: {save_path}')
            messagebox.showinfo('完成', f'配色方案已生成！\n\n📂 {os.path.basename(save_path)}\n\n保存在 output 文件夹中。')

        except Exception as e:
            self._log(f'❌ 错误: {str(e)}')
            import traceback
            self._log(traceback.format_exc())
            messagebox.showerror('错误', f'处理过程中出错:\n{str(e)}')

        finally:
            self.progress.stop()
            self.btn_generate.configure(state=tk.NORMAL)



# ══════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════

def main():
    if not HAS_SKLEARN:
        print('⚠️  提示: 建议安装 scikit-learn 以获得更好的颜色提取效果')
        print('   运行: pip install scikit-learn\n')

    if FONT_AVAILABLE:
        print(f'✅ 字体已加载: {_FONT_NAME}')
    else:
        print('⚠️  未找到思源黑体字体文件，使用系统默认字体')
        print(f'   期望路径: {_FONT_REGULAR}')

    print(f'🚀 Rizline Color Selector 启动中...\n')

    root = tk.Tk()
    app = RizlineColorSelectorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
