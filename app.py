import re
import cv2
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
from rapidfuzz import fuzz, process
from stdnum import ean as std_ean
from paddleocr import PaddleOCR  # działa na Python 3.10 (runtime.txt)

# =========================
# KONFIG
# =========================
st.set_page_config(page_title="FMCG Shelf MVP", layout="wide")

# PaddleOCR nie ma "pol" – użyj "latin" (alfabet łaciński)
OCR_LANG = "latin"
ocr = PaddleOCR(lang=OCR_LANG, use_angle_cls=True, show_log=False)

PRICE_RE   = re.compile(r'(\d{1,3}(?:[.,]\d{2})?)\s*(?:zł|PLN)', re.I)
PERCENT_RE = re.compile(r'(\d{1,3})\s*%')
EAN_RE     = re.compile(r'(?<!\d)(\d{13})(?!\d)')  # 13 cyfr (EAN-13)

# =========================
# FUNKCJE
# =========================
def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess(img: np.ndarray) -> np.ndarray:
    """Odszumianie + wyrównanie kontrastu (CLAHE) dla trudnego światła."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def find_price_tag_candidates(gray: np.ndarray):
    """
    Heurystyka: szukamy jasnych prostokątów (etykiet) głównie w dolnej połowie.
    Zwraca listę bboxów (x,y,w,h).
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, wc, hc = cv2.boundingRect(c)
        area = wc * hc
        if area < 800 or area > (w * h * 0.25):  # odrzuć śmieci i giganty
            continue
        ar = wc / max(hc, 1)
        if 1.5 < ar < 15:
            if y > h * 0.35:  # preferuj dolną część
                boxes.append((x, y, wc, hc))

    # prosty NMS
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for bx in boxes:
        x, y, wc, hc = bx
        ok = True
        for kx, ky, kw, kh in keep:
            rx = max(0, min(x + wc, kx + kw) - max(x, kx))
            ry = max(0, min(y + hc, ky + kh) - max(y, ky))
            if rx * ry > 0.5 * min(wc * hc, kw * kh):
                ok = False
                break
        if ok:
            keep.append(bx)
    return keep[:60]

def safe_paddle_ocr(img_rgb):
    """
    Bezpieczny wrapper na PaddleOCR: zwraca listę [(text, conf), ...]
    niezależnie od struktury/wyjątków w zwrotce.
    """
    try:
        res = ocr.ocr(img_rgb, cls=True)
    except Exception:
        return []

    lines = []
    if not res:
        return lines

    for page in res:
        if not page:
            continue
        for det in page:
            if not isinstance(det, (list, tuple)) or len(det) < 2:
                continue
            txt_pack = det[1]
            if not isinstance(txt_pack, (list, tuple)) or len(txt_pack) < 2:
                continue
            text = str(txt_pack[0]).strip()
            try:
                conf = float(txt_pack[1])
            except Exception:
                conf = 0.0
            if text:
                lines.append((text, conf))
    return lines

def ocr_region(img: np.ndarray, box):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return safe_paddle_ocr(crop_rgb)

def full_image_ocr(img: np.ndarray):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lines = safe_paddle_ocr(rgb)
    return "\n".join([t for t, _ in lines])

def expand_box_up(box, img_shape, up_factor=2.2, side_factor=0.5):
    """
    Powiększ wycinek do góry (łapiemy front opakowania).
    """
    x, y, w, h = box
    H, W = img_shape[:2]
    new_h_up = int(h * up_factor)
    new_y = max(0, y - new_h_up)
    new_x = max(0, x - int(w * side_factor))
    new_w = min(W - new_x, w + int(2 * w * side_factor))
    new_h = min(H - new_y, h + new_h_up)
    return (new_x, new_y, new_w, new_h)

def get_top_strip(box, img_shape, height_factor=1.4, side_factor=0.45):
    """
    Wąski pasek NAD etykietą – tam bywa logo/brand (np. 'Lindt').
    """
    x, y, w, h = box
    H, W = img_shape[:2]
    strip_h = int(h * height_factor)
    new_y = max(0, y - strip_h)
    new_x = max(0, x - int(w * side_factor))
    new_w = min(W - new_x, w + int(2 * w * side_factor))
    new_h = min(H - new_y, strip_h)
    return (new_x, new_y, new_w, new_h)

def ocr_logo_boost(img_bgr):
    """
    „Logo OCR”: powiększenie 2x, CLAHE, unsharp, Otsu.
    Pomaga na złotych/kaligraficznych napisach.
    """
    up = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    up = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)  # unsharp
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return safe_paddle_ocr(rgb)

def parse_prices(texts):
    full = " ".join(t for t, _ in texts)
    prices = [float(p.replace(",", ".")) for p in PRICE_RE.findall(full)]
    promo_flag = bool(
        PERCENT_RE.search(full)
        or "promo" in full.lower()
        or "klub" in full.lower()
        or "teraz" in full.lower()
    )
    price_regular, price_promo = None, None
    if prices:
        price_regular = max(prices)
        if len(prices) >= 2 or promo_flag:
            price_promo = min(prices)
            if price_promo == price_regular:
                price_promo = None
    return price_regular, price_promo, promo_flag

def extract_eans(texts, full_image_text=""):
    found = set()
    for t, _ in texts:
        for m in EAN_RE.findall(t.replace(" ", "")):
            if std_ean.is_valid(m):
                found.add(m)
    for m in EAN_RE.findall(full_image_text.replace(" ", "")):
        if std_ean.is_valid(m):
            found.add(m)
    return list(found)

def guess_name(texts):
    """Wybierz 'nazwę' – linię bogatą w litery, bez cen/procentów."""
    best, score = None, -1
    for t, conf in texts:
        if PRICE_RE.search(t) or PERCENT_RE.search(t):
            continue
        s = sum(ch.isalpha() for ch in t)
        if s > score and len(t) >= 3:
            best, score = t, s
    return best

# =========================
# UI
# =========================
st.title("📸 FMCG Shelf MVP — OCR cen, EAN i udział w półce")

st.markdown("""
Wgraj zdjęcie półki lub etykiet cenowych. Narzędzie:
- wykryje kandydatów etykiet,
- odczyta **nazwę**, **EAN**, **ceny** (standardowa/promocyjna),
- policzy **udział w półce** (na bazie liczby wykrytych etykiet/produktów).
""")

uploaded = st.file_uploader("Wybierz zdjęcie (JPG/PNG)", type=["jpg","jpeg","png"])

brand_hint = st.text_input(
    "(Opcjonalnie) podaj listę marek do rozpoznawania (po przecinku)",
    value="Lindt, Ritter Sport, Raffaello, Wedel, Milka, Merci"
)
brand_list = [b.strip() for b in brand_hint.split(",") if b.strip()]

if uploaded:
    with st.spinner("Przetwarzam obraz..."):
        img = load_image(uploaded)
        gray = preprocess(img)

        whole_text = full_image_ocr(img)

        boxes = find_price_tag_candidates(gray)
        st.subheader(f"Znalezione etykiety: {len(boxes)}")

        vis = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detekcje etykiet (heurystyka)")

        det_results = []
        for i, box in enumerate(boxes, start=1):
            # 1) OCR samej etykiety (ceny)
            texts_tag = ocr_region(img, box)
            price_regular, price_promo, promo_flag = parse_prices(texts_tag)
            eans = extract_eans(texts_tag, whole_text)

            # 2) OCR szerokiego kontekstu nad etykietą (front opakowania)
            big_box = expand_box_up(box, img.shape, up_factor=2.2, side_factor=0.5)
            texts_ctx = ocr_region(img, big_box)

            # 2b) Pasek tuż nad etykietą – tryb „logo”
            sx, sy, sw, sh = get_top_strip(box, img.shape, height_factor=1.4, side_factor=0.45)
            strip_crop = img[sy:sy+sh, sx:sx+sw]
            texts_logo = ocr_logo_boost(strip_crop)

            # Kandydaci nazw: logo > kontekst > etykieta
            name = (guess_name(texts_logo)
                    or guess_name(texts_ctx)
                    or guess_name(texts_tag))

            # Dopasowanie marki: substring w całym tekście (logo+ctx+tag), potem fuzzy
            brand = None
            joined = " ".join([t for t, _ in (texts_logo + texts_ctx + texts_tag)]).lower()
            if brand_list:
                direct = [b for b in brand_list if b.lower() in joined]
                if direct:
                    brand = sorted(direct, key=len)[-1]  # najdłuższa trafiona
                if not brand and name:
                    match = process.extractOne(name, brand_list, scorer=fuzz.token_set_ratio)
                    if match:
                        cand, score, _ = match
                        if score >= 75:
                            brand = cand

            det = {
                "id": i,
                "bbox": box,
                "name": name,
                "brand": brand,
                "ean": eans[0] if eans else None,
                "price_regular": price_regular,
                "price_promo": price_promo,
                "promo_flag": promo_flag
            }
            det_results.append(det)

        # ======= Tabela wyników =======
        st.subheader("📄 Wyniki z etykiet")
        if det_results:
            st.dataframe([
                {
                    "ID": d["id"],
                    "Nazwa": d["name"],
                    "Marka": d["brand"],
                    "EAN": d["ean"],
                    "Cena standardowa": d["price_regular"],
                    "Cena promocyjna": d["price_promo"],
                    "Promocja?": "TAK" if d["promo_flag"] else "NIE"
                }
                for d in det_results
            ])
        else:
            st.info("Nie znaleziono etykiet. Spróbuj ostrzejszego zdjęcia lub bliższego kadru.")

        # ======= Udział w półce (liczymy etykiety per marka) =======
        brand_counts = {}
        for d in det_results:
            key = d["brand"] or "Inne"
            brand_counts[key] = brand_counts.get(key, 0) + 1
        total = sum(brand_counts.values()) or 1

        st.subheader("📊 Udział w półce (wg liczby etykiet)")
        for b, c in sorted(brand_counts.items(), key=lambda kv: -kv[1]):
            st.write(f"- **{b}**: {c}/{total} = {c/total:.1%}")

        # ======= Eksport JSON =======
        result = {
            "photo_id": uploaded.name,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "detections": det_results,
            "share_of_shelf": {
                "level": "brand",
                "counts": brand_counts,
                "total": total
            }
        }
        st.download_button(
            "⬇️ Pobierz JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json"
        )
else:
    st.info("Wgraj zdjęcie, aby rozpocząć.")
