# ✈️ Pegasus AI Destek Agent'ı (RAG & Multi-Agent Architecture)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-orange?style=for-the-badge)
![Redis](https://img.shields.io/badge/Cache-Upstash%20Redis-red?style=for-the-badge&logo=redis)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue?style=for-the-badge&logo=githubactions)

**Pegasus AI Destek Agent'ı**, Pegasus Hava Yolları müşterilerinin bagaj hakları, online check-in, koltuk seçimi, Pegasus Cafe menüsü ve kurumsal bilgiler hakkındaki sorularını anlık yanıtlayan; aynı zamanda aktif uçuş ağındaki şehirler için canlı gezi/kültür rehberliği sunan **üretken yapay zeka (GenAI) destekli müşteri hizmetleri asistanıdır**.

---

## 🌟 Öne Çıkan Özellikler

- **Dinamik RAG (Retrieval-Augmented Generation):** SSS, kural ve menü verilerini statik kod yerine GitHub ve Web Scraper altyapısıyla dinamik taranan JSON veritabanından (`pegasus_kb.json`) TF-IDF benzerlik araması ile çeker.
- **Otomatik Veri Güncelleme (CI/CD & Scraping):** `scrape_pegasus_kb.py` botu ve **GitHub Actions** iş akışları sayesinde web sitesindeki menü/fiyat değişiklikleri otomatik taranarak veritabanı taze tutulur.
- **Canlı Kültür Rehberliği (OpenTripMap REST API):** Pegasus'un uçtuğu şehirler için **OpenTripMap API** ile enlem/boylam koordinatlarından anlık turistik ve tarihi lokasyon verisi çekilir.
- **Yüksek Hızlı LLM Altyapısı (Groq LPU):** Groq çipleri üzerinde çalışan `Llama-3.3-70B-Versatile` modeli ile saniyede 300+ token işleme kapasitesi ve milisaniyeler seviyesinde (Ultra Low-Latency) Türkçe yanıtlar üretilir.
- **Çok Katmanlı Guardrail (Güvenlik & Etik Filtresi):** Pegasus uçuş ağı dışındaki şehirler (ör. Tokyo, New York) tespit edilerek kapsam dışı bilgi üretimi (hallucination) kesin olarak engellenir.
- **Yüksek Performanslı Cache (Upstash Redis & Mock Fallback):** Sık sorulan sorular **AWS eu-central-1** lokasyonlu Redis'te 1 saatlik (3600 sn) TTL ile önbelleklenir. Ağ kesintilerine karşı **Fault-Tolerant In-Memory Mock Cache** mimarisine sahiptir.

---

## 📐 Sistem Mimarisi ve Akış Şeması

```text
[Kullanıcı Sorusu]
       │
       ▼
[Redis Önbellek Sorgusu] ──(Soru var mı?)──► [EVET] ──► (Saniyeler İçinde Önbellekten Döner)
       │ [HAYIR]
       ▼
[Güvenlik & Etik Filtresi (Guardrail)] ──(Uçuş Ağında mı?)──► [HAYIR] ──► (Nazikçe Reddet)
       │ [EVET]
       ▼
[Niyet Analizi (Intent Extraction)]
       │
       ├─► YEMEK / SSS ──► [TF-IDF Retrieval] ──► [GitHub Dynamik KB]
       ├─► KÜLTÜR     ──► [OpenTripMap REST API] ──► [Geocoding + Radius Search]
       │
       ▼
[Bağlam Oluşturma (Context Building)]
       │
       ▼
[Groq LLM Engine (Llama-3.3-70B)]
       │
       ▼
[Gradio Arayüzü & Redis Cache Kaydı]

📂 Proje Yapısı
├── .github/
│   └── workflows/          # Otomatik veri kazıma ve JSON güncelleme CI/CD pipeline'ı
├── pegasus_kb.json          # Dinamik bilgi tabanı (Destinasyonlar, Menü, SSS, Kurumsal)
├── scrape_pegasus_kb.py     # Pegasus web sitesini tarayan Web Scraper scripti
├── pegasus_agent.py         # Ana agent motoru, RAG, Redis, API Entegrasyonları ve Gradio UI
└── README.md                # Proje dokümantasyonu
