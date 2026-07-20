import io
import json
import re
import sys

import requests

try:
    import html2text
except ImportError:
    print("html2text kurulu değil: pip install html2text", file=sys.stderr)
    raise

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf kurulu değil: pip install pypdf", file=sys.stderr)
    raise

KB_FILE = "pegasus_kb.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# category: "menu" | "faq"
SOURCES = [
    {
        "url": "https://cdnp.flypgs.com/files/Pegasus_Cafe_PDF_Menu/PegasusCafeMenu__cHatlar.pdf",
        "type": "pdf",
        "category": "menu",
        "prefix": "menu",
    },
    {
        "url": "https://www.flypgs.com/pegasus-bagaj-kurallari",
        "type": "faq_style",
        "category": "faq",
        "prefix": "bagaj",
    },
    {
        "url": "https://www.flypgs.com/seyahat-hizmetlerimiz/ucus-ile-ilgili-hizmetlerimiz/ucus-paketleri",
        "type": "faq_style",
        "category": "faq",
        "prefix": "paket",
    },
    {
        "url": "https://www.flypgs.com/seyahat-hizmetlerimiz/ucus-ile-ilgili-hizmetlerimiz/koltuk-secimi",
        "type": "faq_style",
        "category": "faq",
        "prefix": "koltuk",
    },
    {
        "url": "https://www.flypgs.com/faydali-bilgiler/diger-bilgiler/sikca-sorulan-sorular",
        "type": "faq_style",
        "category": "faq",
        "prefix": "sss",
    },
    {
        "url": "https://www.flypgs.com/yardim-merkezi",
        "type": "helpcenter_style",
        "category": "faq",
        "prefix": "yardim",
    },
]

FAQ_PATTERN_STRICT = re.compile(
    r"###\s+(?P<q>.+?)\s+arrow down arrow down yellow\s*\n+(?P<a>.+?)(?=\n+###|\nPaylaş|\Z)",
    re.DOTALL,
)
FAQ_PATTERN_LOOSE = re.compile(
    r"###\s+(?P<q>.+?)\s*\n+(?P<a>.+?)(?=\n+###|\Z)",
    re.DOTALL,
)
HELPCENTER_PATTERN = re.compile(
    r"!\[arrow icon\]\([^)]*\)\s+\*\*(?P<title>[^\*]{3,80})\*\*\s*\n+(?P<content>.+?)\n+!\[Chatbot\]",
    re.DOTALL,
)


def clean_text(raw):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)  # [text](url) -> text
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)        # **bold** -> bold
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)       # ![alt](url) -> kaldır
    text = re.sub(r"[#*_`>-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def slugify(text, max_len=40):
    text = text.lower()
    tr_map = str.maketrans("çğıöşü", "cgiosu")
    text = text.translate(tr_map)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len]


def fetch_markdown(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = False
    converter.ignore_links = False
    return converter.handle(resp.text)


def scrape_faq_style(url):
    md = fetch_markdown(url)
    matches = list(FAQ_PATTERN_STRICT.finditer(md))
    if not matches:
        matches = list(FAQ_PATTERN_LOOSE.finditer(md))

    seen, items = set(), []
    for m in matches:
        q = clean_text(m.group("q"))
        a = clean_text(m.group("a"))
        if not q or not a or q in seen:
            continue
        seen.add(q)
        items.append({"question": q, "text": f"{q} {a}"})
    return items


def scrape_helpcenter_style(url):
    md = fetch_markdown(url)
    seen, items = set(), []
    for m in HELPCENTER_PATTERN.finditer(md):
        title = clean_text(m.group("title"))
        content = clean_text(m.group("content"))
        if not title or not content or title in seen:
            continue
        seen.add(title)
        items.append({"question": title, "text": f"{title}: {content}"})
    return items


def scrape_menu_pdf(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))

    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    # PDF metnini satır satır işle, boş/çok kısa satırları at
    lines = [ln.strip() for ln in full_text.splitlines() if len(ln.strip()) > 2]
    # Menü PDF'i genelde ürün adı + fiyat satırlarından oluşur; bunları
    # makul boyutlu bloklar (her biri ~8 satır) halinde birleştirip
    # ayrı doküman parçaları olarak sakla.
    chunk_size = 8
    items = []
    for i in range(0, len(lines), chunk_size):
        chunk = " | ".join(lines[i:i + chunk_size])
        if len(chunk) > 20:
            items.append({"question": f"menu_bolum_{i // chunk_size}", "text": chunk})
    return items


def main():
    try:
        with open(KB_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {"destinations": {}, "knowledge_base": []}

    managed_prefixes = tuple(f"{src['prefix']}_" for src in SOURCES)
    preserved = [
        item for item in existing.get("knowledge_base", [])
        if not item.get("id", "").startswith(managed_prefixes)
    ]
    print(f"Korunan (elle eklenmiş) kayıt sayısı: {len(preserved)}")

    new_kb = []
    for src in SOURCES:
        print(f"Çekiliyor: {src['url']}")
        try:
            if src["type"] == "pdf":
                items = scrape_menu_pdf(src["url"])
            elif src["type"] == "faq_style":
                items = scrape_faq_style(src["url"])
            elif src["type"] == "helpcenter_style":
                items = scrape_helpcenter_style(src["url"])
            else:
                items = []
        except Exception as e:
            print(f"  UYARI: {src['url']} çekilemedi ({e}), atlanıyor.", file=sys.stderr)
            continue

        print(f"  {len(items)} kayıt bulundu.")
        for item in items:
            doc_id = f"{src['prefix']}_{slugify(item['question'])}"
            new_kb.append({
                "id": doc_id,
                "category": src["category"],
                "text": item["text"],
            })

    if not new_kb:
        print("Hiçbir kaynaktan veri çekilemedi, pegasus_kb.json değiştirilmeyecek.", file=sys.stderr)
        sys.exit(1)

    dedup = {item["id"]: item for item in preserved + new_kb}
    final_kb = list(dedup.values())

    output = {
        "destinations": existing.get("destinations", {}),
        "knowledge_base": final_kb,
    }

    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nToplam {len(final_kb)} bilgi tabanı kaydı yazıldı -> {KB_FILE}")


if __name__ == "__main__":
    main()
