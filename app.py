import re
import math
import cv2
import pandas as pd
import numpy as np
import streamlit as st
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from streamlit_image_coordinates import streamlit_image_coordinates
import colorsys

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Color Detector", layout="wide")

# ---------------------------
# CONFIG / DATA FILE
# ---------------------------
COLOR_FILE = "colors.csv" 

# ---------------------------
# Robust loader for XKCD-style dataset
# ---------------------------
@st.cache_data
def load_colors(filepath=COLOR_FILE):
    try:
        df_try = pd.read_csv(filepath, sep='\t', header=0, engine='python', dtype=str)
        cols = [c.strip().lower() for c in df_try.columns]
        df_try.columns = cols
        if 'name' in df_try.columns and 'hex' in df_try.columns:
            df = df_try[['name', 'hex']].copy()
            df['name'] = df['name'].astype(str).str.strip()
            df['hex'] = df['hex'].astype(str).str.strip().str.lower()
            df['hex'] = df['hex'].apply(lambda x: x if x.startswith('#') else f'#{x}')
            df = df[df['hex'].str.match(r'^#[0-9a-f]{6}$', na=False)]
            if not df.empty:
                return df.reset_index(drop=True)
    except Exception:
        pass

    rows = []
    hex_re = re.compile(r'#([0-9a-fA-F]{6})')
    with open(filepath, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = hex_re.search(line)
            if not m:
                continue
            hex_code = '#' + m.group(1).lower()
            name = line[:m.start()].strip()
            if not name:
                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    name = ' '.join(parts[:-1])
                else:
                    continue
            rows.append((name, hex_code))
    df = pd.DataFrame(rows, columns=['name', 'hex'])
    df['name'] = df['name'].astype(str).str.strip()
    df['hex'] = df['hex'].astype(str).str.strip().str.lower()
    df = df[df['hex'].str.match(r'^#[0-9a-f]{6}$', na=False)]
    return df.reset_index(drop=True)

# ---------------------------
# Convert hex -> RGB and precompute Lab tuples (L,a,b)
# ---------------------------
def hex_to_rgb(hex_code: str):
    h = hex_code.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

@st.cache_data
def precompute_lab(df: pd.DataFrame):
    df2 = df.copy()
    df2[['R', 'G', 'B']] = df2['hex'].apply(lambda x: pd.Series(hex_to_rgb(x)))
    def rgb_to_lab_tuple(row):
        r, g, b = int(row['R']), int(row['G']), int(row['B'])
        srgb = sRGBColor(r, g, b, is_upscaled=True)
        lab = convert_color(srgb, LabColor)
        return (float(lab.lab_l), float(lab.lab_a), float(lab.lab_b))
    df2['lab'] = df2.apply(rgb_to_lab_tuple, axis=1)
    return df2

# ---------------------------
# CIEDE2000 delta E
# ---------------------------
def delta_e_ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    mean_C = (C1 + C2) / 2.0
    mean_C7 = mean_C**7
    G = 0.5 * (1 - math.sqrt(mean_C7 / (mean_C7 + 25**7))) if mean_C != 0 else 0.0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    mean_Cp = (C1p + C2p) / 2.0
    def _hp(ap, b):
        if ap == 0.0 and b == 0.0:
            return 0.0
        angle = math.degrees(math.atan2(b, ap))
        if angle < 0:
            angle += 360.0
        return angle
    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        diff = h2p - h1p
        if diff > 180.0:
            diff -= 360.0
        elif diff < -180.0:
            diff += 360.0
        dhp = diff
    dHp = 2.0 * math.sqrt(max(0.0, C1p * C2p)) * math.sin(math.radians(dhp / 2.0))
    if C1p * C2p == 0:
        mean_hp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180.0:
            mean_hp = (h1p + h2p) / 2.0
        else:
            if (h1p + h2p) < 360.0:
                mean_hp = (h1p + h2p + 360.0) / 2.0
            else:
                mean_hp = (h1p + h2p - 360.0) / 2.0
    T = (1 - 0.17 * math.cos(math.radians(mean_hp - 30.0))
         + 0.24 * math.cos(math.radians(2.0 * mean_hp))
         + 0.32 * math.cos(math.radians(3.0 * mean_hp + 6.0))
         - 0.20 * math.cos(math.radians(4.0 * mean_hp - 63.0)))
    delta_theta = 30.0 * math.exp(-(((mean_hp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * math.sqrt((mean_Cp**7) / (mean_Cp**7 + 25.0**7)) if mean_Cp != 0 else 0.0
    mean_L = (L1 + L2) / 2.0
    Sl = 1 + ((0.015 * ((mean_L - 50.0) ** 2)) / math.sqrt(20.0 + ((mean_L - 50.0) ** 2)))
    Sc = 1 + 0.045 * mean_Cp
    Sh = 1 + 0.015 * mean_Cp * T
    Rt = -math.sin(math.radians(2.0 * delta_theta)) * Rc
    term_L = (dLp / Sl) ** 2
    term_C = (dCp / Sc) ** 2
    term_H = (dHp / Sh) ** 2
    term_RC = Rt * (dCp / Sc) * (dHp / Sh)
    deltaE = math.sqrt(max(0.0, term_L + term_C + term_H + term_RC))
    return float(deltaE)

# ---------------------------
# Matching helpers
# ---------------------------
@st.cache_data
def prepare_dataset():
    raw = load_colors(COLOR_FILE)
    df = precompute_lab(raw)
    return df

colors_df = prepare_dataset()

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def get_top_matches_from_lab(pixel_lab_tuple, n=5):
    results = []
    for _, row in colors_df.iterrows():
        d = delta_e_ciede2000(pixel_lab_tuple, row['lab'])
        results.append({
            'name': row['name'],
            'hex': row['hex'],
            'rgb': (int(row['R']), int(row['G']), int(row['B'])),
            'delta_e': float(d)
        })
    results.sort(key=lambda x: x['delta_e'])
    return results[:n]

def get_closest_color_from_rgb(r, g, b, threshold=20.0):
    srgb = sRGBColor(int(r), int(g), int(b), is_upscaled=True)
    lab = convert_color(srgb, LabColor)
    pixel_lab = (float(lab.lab_l), float(lab.lab_a), float(lab.lab_b))
    top = get_top_matches_from_lab(pixel_lab, n=1)
    if not top:
        return "No match", (int(r), int(g), int(b)), None
    best = top[0]
    if best['delta_e'] > float(threshold):
        return "No close match", (int(r), int(g), int(b)), best['delta_e']
    return best['name'], best['rgb'], best['delta_e']

# ---------------------------
# Improved Color Temperature & Mood Analysis (Hue-based)
# ---------------------------
def color_temperature_and_mood(rgb):
    r, g, b = [x/255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = h * 360

    if s < 0.2 and l > 0.8:
        return "White → purity & simplicity"
    elif s < 0.2 and l < 0.2:
        return "Black → power & mystery"
    elif s < 0.2:
        return "Gray → balance & neutrality"

    if hue < 15 or hue >= 345:
        return "Red → energy & passion"
    elif 15 <= hue < 45:
        return "Orange → enthusiasm & creativity"
    elif 45 <= hue < 75:
        return "Yellow → optimism & cheerfulness"
    elif 75 <= hue < 165:
        return "Green → growth & harmony"
    elif 165 <= hue < 195:
        return "Cyan → clarity & refreshment"
    elif 195 <= hue < 255:
        return "Blue → calm & trust"
    elif 255 <= hue < 285:
        return "Indigo → intuition & sophistication"
    elif 285 <= hue < 345:
        return "Purple/Pink → imagination, love & compassion"
    else:
        return "Neutral / balanced color"

# ---------------------------
# Color-Blindness Simulation
# ---------------------------
def simulate_cvd(img_rgb, cvd_type='protanopia'):
    matrices = {
        'protanopia': np.array([[0.56667, 0.43333, 0],
                                [0.55833, 0.44167, 0],
                                [0, 0.24167, 0.75833]]),
        'deuteranopia': np.array([[0.625, 0.375, 0],
                                  [0.7, 0.3, 0],
                                  [0, 0.3, 0.7]]),
        'tritanopia': np.array([[0.95, 0.05, 0],
                                [0, 0.43333, 0.56667],
                                [0, 0.475, 0.525]])
    }
    matrix = matrices.get(cvd_type, np.eye(3))
    simulated = cv2.transform(img_rgb.astype(np.float32)/255.0, matrix)
    simulated = np.clip(simulated, 0, 1) * 255
    return simulated.astype(np.uint8)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎨 Color Detector")

st.sidebar.header("Options")
avg_kernel = st.sidebar.checkbox("Average 3×3 area (reduce noise)", value=True)
delta_thresh = st.sidebar.slider("ΔE threshold (max accepted)", min_value=1.0, max_value=50.0, value=20.0, step=0.5)
show_topn = st.sidebar.slider("Show Top-N matches", min_value=1, max_value=10, value=5, step=1)

st.write("Upload an image and click on it to detect the color. Also simulates color blindness and shows color temperature & mood.")

uploaded_file = st.file_uploader("Upload an image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Unable to decode image.")
        st.stop()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded image", use_column_width=True)
    st.write("Click on the image (below) to pick a pixel:")
    coords = streamlit_image_coordinates(img_rgb, key="img_coords")

    if coords:
        x, y = int(coords["x"]), int(coords["y"])
        h, w, _ = img_rgb.shape
        if not (0 <= x < w and 0 <= y < h):
            st.error("Clicked outside image bounds.")
        else:
            if avg_kernel:
                x1, y1 = max(0, x - 1), max(0, y - 1)
                x2, y2 = min(w - 1, x + 1), min(h - 1, y + 1)
                region = img_rgb[y1:y2+1, x1:x2+1].reshape(-1, 3)
                r, g, b = np.round(region.mean(axis=0)).astype(int)
            else:
                r, g, b = img_rgb[y, x].astype(int)

            color_name, nearest_rgb, dist = get_closest_color_from_rgb(r, g, b, threshold=delta_thresh)

            st.subheader("Detected Color")
            st.write(f"**Coordinates:** ({x}, {y})")
            st.write(f"**Pixel RGB:** ({r}, {g}, {b}) → **HEX:** {rgb_to_hex(r, g, b)}")
            st.write(f"**Color Temperature & Mood:** {color_temperature_and_mood((r, g, b))}")
            if dist is not None:
                st.write(f"**Closest Named Color:** {color_name}  (ΔE = {dist:.2f})")
            else:
                st.write(f"**Closest Named Color:** {color_name}")

            swatch = np.zeros((100, 200, 3), dtype=np.uint8)
            swatch[:] = [r, g, b]
            st.image(swatch, caption="Pixel color swatch")

            swatch2 = np.zeros((100, 200, 3), dtype=np.uint8)
            swatch2[:] = list(nearest_rgb)
            st.image(swatch2, caption=f"Closest dataset swatch: {color_name}")

            # Top-N matches
            st.markdown(f"**Top {show_topn} closest matches (by ΔE)**")
            srgb = sRGBColor(int(r), int(g), int(b), is_upscaled=True)
            lab = convert_color(srgb, LabColor)
            pixel_lab_tuple = (float(lab.lab_l), float(lab.lab_a), float(lab.lab_b))
            top_matches = get_top_matches_from_lab(pixel_lab_tuple, n=show_topn)
            top_df = pd.DataFrame([{
                "name": t['name'],
                "hex": t['hex'],
                "rgb": t['rgb'],
                "delta_e": round(t['delta_e'], 4)
            } for t in top_matches])
            st.dataframe(top_df, use_container_width=True)

            # ---------------- Color-Blindness Simulation ----------------
            st.subheader("Color-Blindness Simulation")
            col1, col2, col3 = st.columns(3)
            with col1:
                prot_img = simulate_cvd(img_rgb, 'protanopia')
                st.image(prot_img, caption="Protanopia")
            with col2:
                deut_img = simulate_cvd(img_rgb, 'deuteranopia')
                st.image(deut_img, caption="Deuteranopia")
            with col3:
                trit_img = simulate_cvd(img_rgb, 'tritanopia')
                st.image(trit_img, caption="Tritanopia")

else:
    st.info("Upload an image to start detecting colors.")
