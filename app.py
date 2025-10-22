import os
import re
import io
import cv2
import json
import time
import math
import numpy as np
import streamlit as st
import requests
from PIL import Image
from rapidfuzz import fuzz, process
from stdnum import ean as std_ean
from paddleocr import PaddleOCR  # działa na Python 3.10 (runtime.txt)

# =========================
# KONFIG
# =========================
st.set_page_config(page_title="FMCG Shelf MVP — produkty & ceny", layout="wide")

# OCR (Paddle nie ma 'pol' – użyj 'latin')
OCR_LANG = "latin"
ocr = PaddleOCR(lang=OCR_LANG, use_angle_cls=True, show_log=False)

# Roboflow (opcjonalnie, zalecane)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "").strip()  # np. "user/pack-front/1"
USE_ROBOFLOW = bool(ROBOFLOW_API_KEY and ROBOFLOW_MODEL)

PRICE_RE   = re.compile(r'(\d{1,3}(?:[.,]\d{2})?)\s*(?:zł|PLN)', re.I)
PERCENT_RE = re.compile(r'(\d{1,3})\s*%')
EAN_RE     = re.compile(r'(?<!\d)(\d{13})(?!\d)')  # 13 cyfr (EAN-13)

# =========================
# FUNKCJE OGÓLNE
# =========================
def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def safe_paddle_ocr(img_rgb):
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

def ocr_text(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return safe_paddle_ocr(rgb)

def parse_prices(texts):
    full = " ".join(t for t,_ in texts)
    prices = [float(p.replace(",", ".")) for p in PRICE_RE.findall(full)]
    promo_flag = bool(PERCENT_RE.search(full) or "promo" in full.lower() or "klub" in full.lower() or "teraz" in full.lower())
    pr, pp = None, None
    if prices:
        pr = max(prices)
        if len(prices) >= 2 or promo_flag:
            pp = min(prices)
            if pp == pr:
                pp = None
    return pr, pp, promo_flag

def extract_eans(texts, full_image_text=""):
    found = set()
    for t,_ in texts:
        for m in EAN_RE.findall(t.replace(" ", "")):
            if std_ean.is_valid(m):
                found.add(m)
    for m in EAN_RE.findall(full_image_text.replace(" ", "")):
        if std_ean.is_valid(m):
            found.add(m)
    return list(found)

def guess_name(texts):
    best, score = None, -1
    for t,_ in texts:
        if PRICE_RE.search(t) or PERCENT_RE.search(t): 
            continue
        s = sum(ch.isalpha() for ch in t)
        if s > score and len(t) >= 3:
            best, score = t, s
    return best

def nms(boxes, scores, iou_thr=0.4):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)  # x,y,w,h
    scores = np.array(scores, dtype=float)
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]; y2 = boxes[:,1] + boxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return [boxes[i].astype(int).tolist() for i in keep]

# =========================
# DETEKCJA PRODUKTÓW
# =========================
def detect_products_roboflow(img_bgr):
    """Detekcja przez chmurowy model Roboflow. Zwraca listę bboxów (x,y,w,h) i etykiety."""
    height, width = img_bgr.shape[:2]
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    resp = requests.post(
        f"https://infer.roboflow.com/{ROBOFLOW_MODEL}",
        params={"api_key": ROBOFLOW_API_KEY},
        files={"file": ("image.jpg", buf.tobytes(), "image/jpeg")},
        timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    boxes, labels, scores = [], [], []
    for p in data.get("predictions", []):
        # Roboflow zwraca x_center, y_center, width, height (w px)
        cx, cy, w, h = p["x"], p["y"], p["width"], p["height"]
        x = int(cx - w/2); y = int(cy - h/2)
        boxes.append([max(0,x), max(0,y), int(w), int(h)])
        labels.append(p.get("class", "product"))
        scores.append(float(p.get("confidence", 0.0)))
    # lekkie NMS
    boxes_nms = nms(boxes, scores, iou_thr=0.4)
    # dopasuj etykiety do NMS-owych boxów (proste nearest)
    final = []
    for bx in boxes_nms:
        x,y,w,h = bx
        best = None; best_iou = -1
        for j,b in enumerate(boxes):
            xx,yy,ww,hh = b
            iou_num = max(0,min(x+w,xx+ww)-max(x,xx)) * max(0,min(y+h,yy+hh)-max(y,yy))
            iou_den = w*h + ww*hh - iou_num + 1e-6
            iou = iou_num / iou_den
            if iou > best_iou:
                best_iou = iou; best = (labels[j], scores[j])
        final.append({"bbox": bx, "label": best[0] if best else "product", "score": best[1] if best else 0.0})
    return final

def detect_products_opencv(img_bgr):
    """
    Heurystyka: wycina prostokątne 'fronty' opakowań:
    - Canny + morfologia
    - kontury ~prostokąt / odpowiedni aspekt
    - grupowanie w rzędy (tolerance po y)
    """
    H, W = img_bgr.shape[:2]
    scale = 1200 / max(H, W)
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr.copy()

    gray = preprocess_gray(small)
    edges = cv2.Canny(gray, 60, 180)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, k, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 0.0005*small.size or area > 0.25*small.size:
            continue
        ar = w / max(h,1)
        # wiele frontów batonów/czekolad: ar ~0.5..0.8 (portrait); pudełka ~0.9..1.6
        if 0.4 <= ar <= 1.8 and h > 25 and w > 25:
            # prostokątność
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) >= 4:
                cand.append([x,y,w,h, area])

    # grupowanie po wierszach (podobne y-środki)
    rows = {}
    for x,y,w,h,_ in cand:
        cy = y + h/2
        key = int(cy/20)  # szeroki kosz
        rows.setdefault(key, []).append([x,y,w,h])

    boxes = []
    for rkey, items in rows.items():
        items = sorted(items, key=lambda b: b[0])
        for x,y,w,h in items:
            boxes.append([x,y,w,h])

    # przeskaluj back do oryginału
    if scale < 1.0:
        inv = 1.0/scale
        boxes = [[int(x*inv), int(y*inv), int(w*inv), int(h*inv)] for x,y,w,h in boxes]

    # NMS
    scores = [b[2]*b[3] for b in boxes]
    boxes = nms(boxes, scores, iou_thr=0.3)
    return [{"bbox": b, "label": "product", "score": 0.5} for b in boxes]

# =========================
# POMOC: marka & nazwa z boxu produktu
# =========================
def crop_from_box(img, box):
    x,y,w,h = box
    x = max(0,x); y = max(0,y)
    return img[y:y+h, x:x+w]

def logo_ocr_boost(img_bgr):
    up = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    up = cv2.cvtColor(cv2.merge([L2,A,B]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(up, (0,0), 1.0)
    sharp = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    rgb = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return safe_paddle_ocr(rgb)

def infer_brand_and_name(img_bgr, brand_list):
    # spróbuj najpierw „logo-ocr” na całym froncie
    t_logo = logo_ocr_boost(img_bgr)
    t_norm = ocr_text(img_bgr)
    joined = " ".join([t for t,_ in (t_logo+t_norm)]).lower()
    # nazwa – bogata w litery linia
    name = guess_name(t_logo) or guess_name(t_norm)

    brand = None
    if brand_list:
        direct = [b for b in brand_list if b.lower() in joined]
        if direct:
            brand = sorted(direct, key=len)[-1]
        elif name:
            match = process.extractOne(name, brand_list, scorer=fuzz.token_set_ratio)
            if match and match[1] >= 75:
                brand = match[0]
    return brand, name

# =========================
# UI
# =========================
st.title("🧃 FMCG Shelf — detekcja produktów, cen i udziałów")

colL, colR = st.columns([1,1])
with colL:
    uploaded = st.file_uploader("Wybierz zdjęcie półki (JPG/PNG)", type=["jpg","jpeg","png"])
with colR:
    brand_hint = st.text_input(
        "Lista marek do rozpoznawania (po przecinku)",
        value="Lindt, Ritter Sport, Raffaello, Wedel, Milka, Merci, Ferrero, Heidi"
    )
brand_list = [b.strip() for b in brand_hint.split(",") if b.strip()]

st.caption(("Tryb detekcji: "
            + ("Roboflow (model: "+ROBOFLOW_MODEL+")" if USE_ROBOFLOW else "OpenCV (fallback – przybliżony)")))

if uploaded:
    with st.spinner("Analizuję zdjęcie..."):
        img = load_image(uploaded)
        H, W = img.shape[:2]

        # 1) DETEKCJA PRODUKTÓW
        if USE_ROBOFLOW:
            detections = detect_products_roboflow(img)
        else:
            detections = detect_products_opencv(img)

        # Wizualizacja boxów produktów
        vis = img.copy()
        for det in detections:
            x,y,w,h = det["bbox"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Znalezione produkty: {len(detections)}")

        # 2) OCR & metadane per produkt
        rows = []
        brand_counts = {}
        for i, det in enumerate(detections, start=1):
            x,y,w,h = det["bbox"]
            front = crop_from_box(img, det["bbox"])
            brand, name = infer_brand_and_name(front, brand_list)

            # OCR cenówki pod produktem (pas pod boxem)
            pad = int(0.25*h)
            y2 = min(H, y+h+pad)
            pricetag_zone = img[y+h:min(H, y+h+pad), x:x+w]
            texts_price = ocr_text(pricetag_zone) if pricetag_zone.size else []
            pr, pp, promo = parse_prices(texts_price)

            # EAN (czasem na cenówkach)
            whole_text = "\n".join([t for t,_ in texts_price])
            eans = extract_eans(texts_price, whole_text)

            rows.append({
                "ID": i,
                "BBox": det["bbox"],
                "Marka": brand,
                "Nazwa": name,
                "EAN": eans[0] if eans else None,
                "Cena standardowa": pr,
                "Cena promocyjna": pp,
                "Promocja?": "TAK" if promo else "NIE"
            })

            key = brand or "Inne"
            brand_counts[key] = brand_counts.get(key, 0) + 1

        # 3) Tabela & udział w półce
        st.subheader("📄 Produkty (1 wiersz = 1 wykryty front)")
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Brak wykrytych produktów. Spróbuj bliższego ujęcia lub włącz tryb Roboflow (zalecany).")

        st.subheader("📊 Udział w półce (liczba frontów)")
        total = sum(brand_counts.values()) or 1
        for b,c in sorted(brand_counts.items(), key=lambda kv: -kv[1]):
            st.write(f"- **{b}**: {c}/{total} = {c/total:.1%}")

        # 4) Eksport JSON
        result = {
            "photo_id": uploaded.name,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "products": rows,
            "share_of_shelf": {"level":"brand","counts":brand_counts,"total":total}
        }
        st.download_button(
            "⬇️ Pobierz JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="result_products.json",
            mime="application/json"
        )
else:
    st.info("Wgraj zdjęcie, aby rozpocząć.")
