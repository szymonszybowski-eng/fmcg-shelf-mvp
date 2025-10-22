import io
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

# UWAGA: PaddleOCR nie ma "pol" – użyj "latin" (alfabet łaciński)
OCR_LANG = "latin"
ocr = PaddleOCR(lang=OCR_LANG, use_angle_cls=True, show_log=False)

PRICE_RE   = re.compile(r'(\d{1,3}(?:[.,]\d{2})?)\s*(?:zł|PLN)', re.I)
PERCENT_RE = re.compile(r'(\d{1,3})\s*%')
EAN_RE     = re.compile(r'(?<!\d)(\d{13})(?!\d)')  # 13 cyfr (EAN-13)

# =========================
# FUNKCJE POMOCNICZE
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
        if area < 800 or area > (w * h * 0.25):  # odrzuć śmieci i gigant
