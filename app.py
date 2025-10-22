import io
import re
import cv2
import json
import time
import numpy as np
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR
from rapidfuzz import fuzz, process
from stdnum import ean as std_ean

# -------------------------
# USTAWIENIA
# -------------------------
st.set_page_config(page_title="FMCG Shelf MVP", layout="wide")
OCR_LANG = "pol"  # polski OCR
ocr = PaddleOCR(lang=OCR_LANG, use_angle_cls=True, show_log=False)

PRICE_RE = re.compile(r'(\d{1,3}(?:[.,]\d{2})?)\s*(?:zł|PLN)', re.I)
PERCENT_RE = re.compile(r'(\d{1,3})\s*%')
EAN_RE = re.compile(r'(?<!\d)(\d{13})(?!\d)')  # ciąg 13 cyfr

# -------------------------
# FUNKCJE POMOCNICZE
# -------------------------
def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess(img: np.ndarray) -> np.ndarray:
    """Lekkie odszumianie + wzmocnienie kontrastu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.equalizeHist(gray)
    return gray

def find_price_tag_candidates(gray: np.ndarray):
    """
    Heurystyka: szukamy jasnych prostokątów (etykiet) blisko dołu zdjęcia.
    Zwracamy listę bboxów (x,y,w,h).
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < 800 or area > (w*h*0.2):  # odrzuć śmieci i giganty
            continue
        ar = wc / max(hc,1)
        if 1.5 < ar < 15:  # etykiety zwykle są poziome
            # preferuj dolną połowę zdjęcia
            if y > h*0.35:
                boxes.append((x,y,wc,hc))
    # prosty non-max suppression po nakładających się
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    for bx in boxes:
        x,y,wc,hc = bx
        ok = True
        for kx,ky,kw,kh in keep:
            rx = max(0, min(x+wc, kx+kw) - max(x, kx))
            ry = max(0, min(y+hc, ky+kh) - max(y, ky))
            if rx*ry > 0.5*min(wc*hc, kw*kh):
                ok = False
                break
        if ok:
            keep.append(bx)
    return keep[:50]  # limit

def ocr_region(img: np.ndarray, box):
    x,y,w,h = box
    crop = img[y:y+h, x:x+w]
    # OCR działa lepiej na RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = ocr.ocr(crop_rgb, cls=True)
    lines = []
    for r in res:
        for box, (text, conf) in r:
            lines.append((text.strip(), float(conf)))
    return lines

def parse_prices(texts):
    full = " ".join(t for t,_ in texts)
    prices = [float(p.replace(",", ".")) for p in PRICE_RE.findall(full)]
    promo_flag = bool(PERCENT_RE.search(full) or "promo" in full.lower() or "klub" in full.lower() or "teraz" in full.lower())
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
    # 1) z regionu etykiety
    for t,_ in texts:
        for m in EAN_RE.findall(t.replace(" ", "")):
            if std_ean.is_valid(m):
                found.add(m)
    # 2) z całego OCR zdjęcia jako fallback
    for m in EAN_RE.findall(full_image_text.replace(" ", "")):
        if std_ean.is_valid(m):
            found.add(m)
    return list(found)

def guess_name(texts):
    """
    Zwróć potencjalną nazwę produktu (linia z największą ilością liter i nie-cenowa).
    """
    best, score = None, -1
    for t, conf in texts:
        if PRICE_RE.search(t) or PERCENT_RE.search(t):
            continue
        s = sum(ch.isalpha() for ch in t)
        if s > score and len(t) >= 3:
            best, score = t, s
    return best

def full_image_ocr(img: np.ndarray):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = ocr.ocr(rgb, cls=True)
    all_text = []
    for r in res:
        for box, (text, conf) in r:
            all_text.append(text.strip())
    return "\n".join(all_text)

# -------------------------
# UI
# -------------------------
st.title("📸 FMCG Shelf MVP — OCR cen, EAN i udział w półce")

st.markdown("""
Wgraj zdjęcie półki lub etykiet cenowych. Narzędzie:
- wykryje kandydatów etykiet,
- odczyta **nazwę**, **EAN**, **ceny** (standardowa/promocyjna),
- policzy **udział w półce** (na bazie liczby wykrytych etykiet/produktów).
""")

uploaded = st.file_uploader("Wybierz zdjęcie (JPG/PNG)", type=["jpg","jpeg","png"])

brand_hint = st.text_input("(Opcjonalnie) podaj listę marek do rozpoznawania (po przecinku)", value="Coca-Cola, Pepsi, Sprite, Fanta, Żywiec Zdrój")
brand_list = [b.strip() for b in brand_hint.split(",") if b.strip()]

if uploaded:
    with st.spinner("Przetwarzam obraz..."):
        img = load_image(uploaded)
        gray = preprocess(img)

        # pełny OCR zdjęcia (pomaga w EAN fallback)
        whole_text = full_image_ocr(img)

        boxes = find_price_tag_candidates(gray)
        st.subheader(f"Znalezione etykiety: {len(boxes)}")
        det_results = []

        vis = img.copy()
        for i, box in enumerate(boxes, start=1):
            x,y,w,h = box
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)

        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detekcje etykiet (heurystyka)")

        for i, box in enumerate(boxes, start=1):
            texts = ocr_region(img, box)
            price_regular, price_promo, promo_flag = parse_prices(texts)
            eans = extract_eans(texts, whole_text)
            name = guess_name(texts)

            # prosta próba dopasowania marki po nazwie
            brand = None
            if name and brand_list:
                match, score, _ = process.extractOne(name, brand_list, scorer=fuzz.WRatio)
                if score >= 70:
                    brand = match

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

        # Tabela wyników
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

        # Udział w półce — policz facingi (tu: liczba etykiet na markę)
        brand_counts = {}
        for d in det_results:
            key = d["brand"] or "Inne"
            brand_counts[key] = brand_counts.get(key, 0) + 1
        total = sum(brand_counts.values()) or 1

        st.subheader("📊 Udział w półce (wg liczby etykiet)")
        for b, c in sorted(brand_counts.items(), key=lambda kv: -kv[1]):
            st.write(f"- **{b}**: {c}/{total} = {c/total:.1%}")

        # JSON do pobrania
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
        st.download_button("⬇️ Pobierz JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="result.json", mime="application/json")
else:
    st.info("Wgraj zdjęcie, aby rozpocząć.")
