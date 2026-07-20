!pip install -q scikit-learn groq redis requests gradio

import os
import json
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import redis

try:
    from google.colab import userdata
    def get_secret(name, default=""):
        try:
            val = userdata.get(name)
            return val if val else default
        except Exception:
            return os.environ.get(name, default)
except ImportError:
    def get_secret(name, default=""):
        return os.environ.get(name, default)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
OPENTRIPMAP_API_KEY = get_secret("OPENTRIPMAP_API_KEY")
REDIS_URL = get_secret("REDIS_URL")
KNOWLEDGE_BASE_URL = get_secret(
    "KNOWLEDGE_BASE_URL",
    "https://raw.githubusercontent.com/rmeysaaltay/pegasus_kb.json/main/pegasus_kb.json"
)

MODEL_NAME = "llama-3.3-70b-versatile"

def load_knowledge_base(url, local_fallback="pegasus_kb.json"):
    """Önce uzak URL'den çekmeyi dener; başarısız olursa yerel yedeğe düşer."""
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        print(f"✅ Bilgi tabanı uzak kaynaktan yüklendi: {url}")
    except Exception as e:
        print(f"⚠️ Uzak kaynağa ulaşılamadı ({e}). Yerel yedek dosya deneniyor...")
        with open(local_fallback, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ Bilgi tabanı yerel yedekten yüklendi.")
    return data["destinations"], data["knowledge_base"]


FLOWN_COUNTRIES, KNOWLEDGE_BASE = load_knowledge_base(KNOWLEDGE_BASE_URL)

ALL_KNOWN_CITIES = set()
for cities in FLOWN_COUNTRIES.values():
    ALL_KNOWN_CITIES.update(cities)

_KNOWN_CITIES_TEXT = ", ".join(sorted(c.capitalize() for c in ALL_KNOWN_CITIES))


_corpus = [doc["text"] for doc in KNOWLEDGE_BASE]
_vectorizer = TfidfVectorizer()
_doc_vectors = _vectorizer.fit_transform(_corpus)


def retrieve(query, top_k=3):
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _doc_vectors)[0]
    ranked = sorted(zip(sims, KNOWLEDGE_BASE), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:top_k] if score > 0]



class RedisCache:
    def __init__(self, url, ttl_seconds=3600):
        self.ttl = ttl_seconds
        self.enabled = bool(url)
        self._local_store = {}
        if self.enabled:
            try:
                self.client = redis.from_url(url, decode_responses=True)
                self.client.ping()
                print("✅ Redis'e bağlanıldı.")
            except Exception as e:
                print(f"⚠️ Redis'e bağlanılamadı ({e}). Bellek-içi mock cache kullanılacak.")
                self.enabled = False

    def get(self, key):
        if self.enabled:
            try:
                return self.client.get(key)
            except Exception:
                return None
        return self._local_store.get(key)

    def set(self, key, value):
        if self.enabled:
            try:
                self.client.set(key, value, ex=self.ttl)
            except Exception:
                pass
        else:
            self._local_store[key] = value


cache = RedisCache(REDIS_URL, ttl_seconds=3600)


def get_cultural_info(city):
    """OpenTripMap API'sinden şehir için öne çıkan kültürel/tarihi yerleri çeker."""
    if not OPENTRIPMAP_API_KEY:
        return "Kültürel bilgi servisi şu anda kullanılamıyor (API anahtarı tanımlı değil)."

    cache_key = f"kultur:{city}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        geo = requests.get(
            "https://api.opentripmap.com/0.1/en/places/geoname",
            params={"name": city, "apikey": OPENTRIPMAP_API_KEY},
            timeout=6,
        ).json()

        if "lat" not in geo or "lon" not in geo:
            return f"{city.capitalize()} için konum bilgisi bulunamadı."

        places = requests.get(
            "https://api.opentripmap.com/0.1/en/places/radius",
            params={
                "radius": 6000,
                "lon": geo["lon"],
                "lat": geo["lat"],
                "kinds": "cultural,historic,museums,architecture,interesting_places",
                "rate": 2,
                "limit": 6,
                "apikey": OPENTRIPMAP_API_KEY,
            },
            timeout=6,
        ).json()

        names = [
            f["properties"]["name"]
            for f in places.get("features", [])
            if f.get("properties", {}).get("name")
        ]

        if not names:
            result = f"{city.capitalize()} için öne çıkan bir kültürel nokta bulunamadı."
        else:
            result = f"{city.capitalize()}'de öne çıkan yerler: " + ", ".join(names[:6]) + "."

        cache.set(cache_key, result)
        return result

    except Exception as e:
        return f"Kültürel bilgi alınırken bir sorun oluştu: {e}"


def extract_city(text):
    t = text.lower()
    for city in sorted(ALL_KNOWN_CITIES, key=len, reverse=True):
        if city in t:
            return city
    return "none"


_PROPER_NOUN_RE = re.compile(r"^[A-ZÇĞİÖŞÜ][a-zçğıöşü'’]{2,}$")


def mentions_possible_place(text, known_city):
    """
    Ucuz bir sezgisel: cümlede (başlangıç kelimesi hariç) büyük harfle
    başlayan bir kelime varsa, bunun bilinmeyen bir şehir/ülke adı olma
    ihtimaline karşı LLM'e şehir listesini gönderiyoruz. known_city zaten
    bulunduysa (Python guardrail'i zaten hallediyor) tekrar eklemeye gerek yok.
    """
    if known_city != "none":
        return False
    words = text.replace("'", " ").split()
    for i, w in enumerate(words):
        clean = w.strip(".,!?;:\"'’")
        if i == 0:
            continue
        if _PROPER_NOUN_RE.match(clean):
            return True
    return False


def extract_intent(text):
    t = text.lower()
    intents = []
    if any(k in t for k in ["yemek", "menü", "menu", "ikram", "sandviç", "atıştırmalık", "içecek"]):
        intents.append("YEMEK")
    if any(k in t for k in ["bagaj", "valiz", "çanta"]):
        intents.append("BAGAJ")
    if any(k in t for k in ["check-in", "checkin", "biniş"]):
        intents.append("CHECKIN")
    if any(k in t for k in ["koltuk"]):
        intents.append("KOLTUK")
    if any(k in t for k in [
        "rota", "direkt uçuş", "uçuş var mı", "sefer", "uçtuğu", "uçuyor",
        "hangi ülke", "hangi şehir", "nereler", "nerelere", "destinasyon",
        "uçuş noktaları", "uçuş ağı",
    ]):
        intents.append("ROTA")
    if any(k in t for k in ["kültür", "gezilecek", "görülecek", "tarihi", "müze", "ne yapabilirim", "aktivite", "gezi"]):
        intents.append("KULTUR")
    if not intents:
        intents.append("GENEL")
    return ",".join(intents)



def guardrail_check(user_query, intent, city):
    """
    Uçuş ağı dışındaki şehirler için bilgi/öneri verilmesini engeller.
    Buraya ileride ek etik kontroller (küfür, nefret söylemi vb. filtreleri
    veya bir moderation API çağrısı) eklenebilir.
    """
    if city != "none" and city not in ALL_KNOWN_CITIES:
        reason = (
            f"✈️ Pegasus olarak şu anda {city.capitalize()} bölgesine direkt uçuşumuz bulunmamaktadır. "
            "Kurallarımız gereği uçuş ağımız dışındaki yerler için bilgi veya öneri sunamıyorum. "
            "Size aktif uçuş ağımızdaki başka bir rota için yardımcı olmamı ister misiniz? 😊"
        )
        return False, reason
    return True, None



MAX_DOC_CHARS = 240
MAX_CONTEXT_CHARS = 1400


def _truncate(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def build_context(user_query, intent, city):
    parts = []
    country_context = ""

    if "YEMEK" in intent:
        menu_docs = [d["text"] for d in KNOWLEDGE_BASE if d["category"] == "menu"]
        parts.extend(menu_docs[:4])  # token tasarrufu için üst sınır

    if "KULTUR" in intent and city != "none":
        parts.append(get_cultural_info(city))

    if "ROTA" in intent and city == "none":
        country_list = ", ".join(sorted(c.title() for c in FLOWN_COUNTRIES.keys()))
        country_context = _truncate(f"Pegasus'un uçtuğu ülkeler: {country_list}.", 700)

    if any(i in intent for i in ["BAGAJ", "CHECKIN", "KOLTUK", "ROTA", "GENEL"]):
        parts.extend(d["text"] for d in retrieve(user_query, top_k=2))

    parts = [_truncate(p, MAX_DOC_CHARS) for p in parts]
    if country_context:
        parts.insert(0, country_context)

    context_text = "\n".join(f"- {p}" for p in parts) if parts else ""
    return _truncate(context_text, MAX_CONTEXT_CHARS)



client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BASE_RULES = (
    "Sen Pegasus Havayolları'nın resmi yapay zeka müşteri destek asistanısın. "
    "Dost canlısı, profesyonel ve saygılısın.\n"
    "DİL KURALI (kesinlikle bağlayıcı): SADECE ve SADECE düzgün, akıcı Türkçe yaz. "
    "Tek bir yabancı kelime, karışık dil ifadesi, anlamsız karakter veya bozuk kelime "
    "ÜRETME. Ne söyleyeceğinden emin değilsen kısa ve net bir Türkçe cümleyle bunu belirt, "
    "asla tutarsız veya karışık dilde cümle kurma.\n"
    "Pegasus'un uçuş ağı dışındaki şehirler/ülkeler hakkında kesinlikle hiçbir öneri, "
    "bilgi veya yorum verme."
)


CITY_GUARDRAIL_RULES = (
    "\nGÜVENLİK KONTROLÜ (ikinci katman): Kullanıcının sorusunda belirli bir şehir veya "
    "ülke ismi geçiyorsa, önce bunun aşağıdaki Pegasus uçuş ağı listesinde olup olmadığını "
    "kontrol et:\n"
    f"{_KNOWN_CITIES_TEXT}\n"
    "Eğer bahsedilen şehir/ülke bu listede YOKSA (örn. Tokyo, New York, Şam, Pekin vb.), "
    "başka HİÇBİR ŞEY söylemeden sadece şu cevabı ver (şehir adını kendi cümlende kullan): "
    "'✈️ Pegasus olarak şu anda [ŞEHİR] bölgesine direkt uçuşumuz bulunmamaktadır. Kurallarımız "
    "gereği uçuş ağımız dışındaki yerler için bilgi veya öneri sunamıyorum. Size aktif uçuş "
    "ağımızdaki başka bir rota için yardımcı olmamı ister misiniz? 😊'"
)

FACTUAL_RULES = (
    "\n\nBu soru Pegasus'un resmi kural/politika/menü bilgisiyle ilgili. "
    "SADECE aşağıdaki Context içindeki bilgiyi kullanarak cevap ver. Context'te olmayan "
    "hiçbir rakam, kural veya detayı uydurma. Eğer Context soruyu yanıtlamak için yeterli "
    "değilse, bunu nazikçe belirt ve güncel bilgi için flypgs.com veya çağrı merkezini "
    "önererek yanıtı sonlandır."
)

CULTURAL_RULES = (
    "\n\nBu soru, Pegasus'un uçtuğu bir şehir hakkında kültürel/gezi tavsiyesiyle ilgili "
    "(şehrin uçuş ağımızda olduğu zaten doğrulandı). Aşağıda dış bir kaynaktan (Context) "
    "veri varsa önce onu kullan. Ancak Context boşsa veya yetersizse, KESİNLİKLE 'bağlam "
    "yok' gibi bir cevap VERME; bunun yerine bu şehir hakkındaki kendi genel kültürel "
    "bilgini (tarihi yerler, mutfak, gelenekler, gezilecek noktalar) kullanarak sıcak, "
    "akıcı ve bilgilendirici bir Türkçe rehberlik yap."
)


def call_llm(user_query, context_text, intent="GENEL", city="none"):
    if "KULTUR" in intent:
        mode_rules = CULTURAL_RULES
    else:
        mode_rules = FACTUAL_RULES

    needs_city_rules = any(i in intent for i in ["ROTA", "KULTUR"]) or mentions_possible_place(user_query, city)
    city_rules = CITY_GUARDRAIL_RULES if needs_city_rules else ""

    context_block = f"\n\nContext:\n{context_text}" if context_text else "\n\nContext: (boş)"
    system_prompt = BASE_RULES + city_rules + mode_rules + context_block


    MAX_SYSTEM_PROMPT_CHARS = 4500
    if len(system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
        system_prompt = system_prompt[:MAX_SYSTEM_PROMPT_CHARS].rsplit(" ", 1)[0] + "…"

    if client is None:
        return f"[MOCK CEVAP - GROQ_API_KEY tanımlı değil]\n{context_text or 'İlgili bilgi bulunamadı.'}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.4,
        top_p=0.9,
        max_tokens=400,
    )
    return response.choices[0].message.content


def pegasus_agent(user_query):
    cache_key = f"soru:{user_query.strip().lower()}"

    cached = cache.get(cache_key)
    if cached:
        return cached + "\n\n(önbellekten)"

    intent = extract_intent(user_query)
    city = extract_city(user_query)

    ok, reason = guardrail_check(user_query, intent, city)
    if not ok:
        cache.set(cache_key, reason)
        return reason

    context = build_context(user_query, intent, city)
    answer = call_llm(user_query, context, intent, city)

    cache.set(cache_key, answer)
    return answer



if __name__ == "__main__":

    RUN_QUICK_TESTS = True

    if RUN_QUICK_TESTS:
        test_scenarios = [
            "Bagaj hakkım ne kadar?",
            "Uçakta yemek menüsünde neler var?",
            "İstanbul'dan Berlin'e direkt uçuş var mı?",
            "Amsterdam'da gezilecek kültürel yerler nereler?",
            "Tokyo'ya gitmek için önerin var mı?",   # -> guardrail
            "Koltuk değişikliği yapabilir miyim?",
            "Bagaj hakkım ne kadar?",                # -> cache
        ]

        for q in test_scenarios:
            print("SORU:", q)
            print("CEVAP:", pegasus_agent(q))
            print("-" * 60)


    import gradio as gr

    PGS_ORANGE = "#F3A000"
    PGS_ORANGE_2 = "#F29F05"
    PGS_ORANGE_DARK = "#c97c00"
    PGS_RED = "#E30613"
    PGS_GRAY = "#EEF2F6"
    TEXT_DARK = "#16375B"

    pegasus_theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
    ).set(
        body_background_fill="linear-gradient(180deg, #3f7fd6 0%, #6fa8e8 30%, #bfe0fb 65%, #eaf4fc 100%)",
        body_background_fill_dark="linear-gradient(180deg, #3f7fd6 0%, #6fa8e8 30%, #bfe0fb 65%, #eaf4fc 100%)",
        body_text_color=TEXT_DARK,
        body_text_color_dark=TEXT_DARK,

        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_width="0px",
        block_radius="24px",
        block_shadow="0 12px 40px rgba(20, 60, 110, 0.18)",
        block_label_text_color=TEXT_DARK,
        block_label_text_color_dark=TEXT_DARK,
        block_title_text_color=TEXT_DARK,
        block_title_text_color_dark=TEXT_DARK,

        body_text_color_subdued="#3a5878",
        color_accent_soft="#fff3e0",

        button_primary_background_fill=PGS_ORANGE,
        button_primary_background_fill_hover=PGS_ORANGE_DARK,
        button_primary_text_color=TEXT_DARK,
        button_secondary_background_fill=PGS_GRAY,
        button_secondary_text_color=TEXT_DARK,

        input_background_fill="#ffffff",
        input_border_color="#e6dccf",
    )

    CUSTOM_CSS = f"""
    :root {{
        --pgs-orange: {PGS_ORANGE};
        --pgs-orange-2: {PGS_ORANGE_2};
        --pgs-orange-dark: {PGS_ORANGE_DARK};
        --pgs-red: {PGS_RED};
        --pgs-gray: {PGS_GRAY};
        --pgs-navy: {TEXT_DARK};
    }}

    * {{ transition: background-color 0.25s ease, box-shadow 0.25s ease, transform 0.15s ease; }}


    .gradio-container {{
        background-attachment: fixed !important;
        position: relative;
        font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
        overflow-x: hidden;
    }}
    .gradio-container::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(ellipse 280px 110px at 12% 10%, rgba(255,255,255,0.95), transparent 70%),
            radial-gradient(ellipse 220px 90px  at 22% 15%, rgba(255,255,255,0.85), transparent 70%),
            radial-gradient(ellipse 320px 130px at 78% 8%,  rgba(255,255,255,0.9), transparent 70%),
            radial-gradient(ellipse 240px 100px at 90% 20%, rgba(255,255,255,0.75), transparent 70%),
            radial-gradient(ellipse 300px 120px at 8% 55%,  rgba(255,255,255,0.7), transparent 70%),
            radial-gradient(ellipse 260px 110px at 95% 60%, rgba(255,255,255,0.65), transparent 70%),
            radial-gradient(ellipse 340px 130px at 50% 78%, rgba(255,255,255,0.55), transparent 70%);
        pointer-events: none;
        z-index: 0;
    }}

    .pgs-header {{
        text-align: center;
        padding: 6px 0 18px;
        position: relative;
        z-index: 1;
    }}
    .pgs-header h1 {{
        font-size: 34px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }}
    .pgs-logo-word {{
        color: var(--pgs-red) !important;
        font-style: italic;
        font-weight: 800;
        letter-spacing: 0.5px;
        -webkit-text-stroke: 0.5px var(--pgs-red);
        text-shadow: 1px 1px 0 rgba(0,0,0,0.15), 0 2px 8px rgba(0,0,0,0.15);
    }}
    .pgs-header p {{
        color: #eef6ff;
        font-size: 16px;
        font-weight: 500;
        margin: 8px 0 0;
        text-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }}

    div[data-testid="chatbot"], .chatbot {{
        background: #ffffff !important;
        border-radius: 24px !important;
        z-index: 1;
        position: relative;
    }}

    /* Sarmalayıcı katmanları sıfırla (iç içe kutu görünmesin) */
    .message-wrap, .message-row, .bubble-wrap {{
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}
    .message, .message-bubble-border, .message-content, .bubble-wrap .message {{
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    .message-bubble-border {{
        border-radius: 20px !important;
        padding: 14px 20px !important;
        max-width: 78%;
        font-size: 16px !important;
        line-height: 1.7 !important;
        font-weight: 500 !important;
    }}

    .bot-row .message-bubble-border, .message.bot .message-bubble-border,
    [data-testid="bot"] .message-bubble-border, .message-row.bot-row .message-bubble-border,
    .message.bot:not(:has(.message-bubble-border)) {{
        background: #fff6e6 !important;
        color: var(--pgs-navy) !important;
        box-shadow: 0 4px 14px rgba(243, 160, 0, 0.15) !important;
        border: 1px solid #ffe4ad !important;
        margin-right: auto !important;
        border-radius: 20px !important;
        padding: 14px 20px !important;
        max-width: 78%;
    }}

    .user-row .message-bubble-border, .message.user .message-bubble-border,
    [data-testid="user"] .message-bubble-border, .message-row.user-row .message-bubble-border,
    .message.user:not(:has(.message-bubble-border)) {{
        background: linear-gradient(135deg, var(--pgs-orange) 0%, var(--pgs-orange-2) 100%) !important;
        color: var(--pgs-navy) !important;
        box-shadow: 0 4px 14px rgba(243, 160, 0, 0.35) !important;
        margin-left: auto !important;
        border-radius: 20px !important;
        padding: 14px 20px !important;
        max-width: 78%;
        font-weight: 600 !important;
    }}

    .bot-row .message-bubble-border *, .message.bot .message-bubble-border *,
    [data-testid="bot"] .message-bubble-border *, .message.bot:not(:has(.message-bubble-border)) * {{
        color: var(--pgs-navy) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        font-weight: 500 !important;
    }}
    .user-row .message-bubble-border *, .message.user .message-bubble-border *,
    [data-testid="user"] .message-bubble-border *, .message.user:not(:has(.message-bubble-border)) * {{
        color: var(--pgs-navy) !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
        font-weight: 600 !important;
    }}

    textarea, input[type="text"] {{
        background: #ffffff !important;
        color: var(--pgs-navy) !important;
        border-radius: 999px !important;
        border: 1.5px solid #ffe4ad !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        padding: 14px 22px !important;
        box-shadow: 0 2px 10px rgba(243, 160, 0, 0.08) !important;
    }}
    textarea:focus, input[type="text"]:focus {{
        border-color: var(--pgs-orange) !important;
        box-shadow: 0 0 0 3px rgba(243, 160, 0, 0.20) !important;
        outline: none !important;
    }}
    textarea::placeholder, input[type="text"]::placeholder {{ color: #9aa7b8 !important; }}

    /* Gönder butonu: Pegasus kırmızısı (logo ile uyumlu vurgu) */
    button[aria-label="Submit"], .submit-button, button.primary {{
        background: var(--pgs-red) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(227, 6, 19, 0.30) !important;
    }}
    button[aria-label="Submit"]:hover, .submit-button:hover, button.primary:hover {{
        background: #c00510 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(227, 6, 19, 0.40) !important;
    }}

    .example, button.example {{
        background: #ffffff !important;
        color: var(--pgs-navy) !important;
        border: 1px solid #ffe4ad !important;
        border-radius: 16px !important;
        font-weight: 500 !important;
        font-size: 15px !important;
        padding: 12px 18px !important;
        box-shadow: 0 3px 10px rgba(243, 160, 0, 0.10) !important;
    }}
    .example:hover, button.example:hover {{
        border-color: var(--pgs-orange) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(243, 160, 0, 0.20) !important;
    }}
    """

    def chat_fn(message, history):
        return pegasus_agent(message)

    with gr.Blocks(theme=pegasus_theme, css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="pgs-header">
                <h1><span class="pgs-logo-word">PEGASUS</span> AI Destek Agent'ı ✈️</h1>
                <p>Bagaj, check-in, rota, menü ve kültürel bilgi sorularınızı sorun.</p>
            </div>
            """
        )
        gr.ChatInterface(
            fn=chat_fn,
            examples=[
                "Bagaj hakkım ne kadar?",
                "İstanbul'dan Berlin'e direkt uçuş var mı?",
                "Amsterdam'da gezilecek kültürel yerler nereler?",
            ],
        )

    demo.launch(share=True, debug=False)
