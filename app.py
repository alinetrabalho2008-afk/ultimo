
"""
PowerNutri AI — Identificador de Calorias
Aplicação Streamlit para análise nutricional de refeições por imagem ou descrição.

Execute:
    pip install -r requirements.txt
    streamlit run app.py
"""

import base64
import io
import json
import os
from datetime import datetime

import streamlit as st
from openai import OpenAI
from PIL import Image

APP_TITLE = "PowerNutri AI"
MODEL = "gpt-5-mini"

def get_openai_api_key():
    """Lê a chave com segurança no Streamlit Cloud ou por variável de ambiente."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, RuntimeError):
        key = ""
    return key or os.getenv("OPENAI_API_KEY", "")

INSTRUCTIONS = """
Você é a PowerNutri AI, um assistente de estimativa nutricional.
Analise a refeição fornecida por imagem e/ou descrição textual.

REGRAS:
1. Identifique os alimentos visíveis ou descritos.
2. Estime a porção e os valores nutricionais da refeição completa.
3. Avalie se há características claras da culinária mineira.
4. Não invente ingredientes que não possam ser razoavelmente identificados.
5. Deixe claro que os valores são estimativas e não substituem avaliação profissional.
6. Se a entrada não for sobre comida/refeição, retorne o campo "erro".

CULINÁRIA MINEIRA:
Considere elementos claramente característicos como tutu de feijão, feijão tropeiro,
couve refogada, torresmo, linguiça artesanal, frango com quiabo, angu, pão de queijo,
doce de leite, queijo minas e outros preparos reconhecidamente mineiros.
Um prato brasileiro genérico não deve ser classificado como mineiro apenas por ser brasileiro.

Responda SOMENTE em JSON:
{
  "descricao": "descrição curta da refeição",
  "culinaria_mineira": true ou false,
  "prato_tipico": "nome ou null",
  "justificativa_classificacao": "frase curta",
  "calorias": number,
  "carboidratos_g": number,
  "proteinas_g": number,
  "gorduras_g": number,
  "porcao_estimada": "ex.: 1 prato / aproximadamente 450 g",
  "alimentos": ["item 1", "item 2"],
  "observacao": "observação curta sobre a estimativa",
  "fibras_g": number,
  "alimentos": [{"nome":"string","quantidade_estimada":"string","calorias":number,"carboidratos_g":number,"proteinas_g":number,"gorduras_g":number,"fibras_g":number}],
  "fontes_nutrientes": {
    "carboidratos": [{"alimento":"string","contribuicao_g":number,"explicacao":"string"}],
    "proteinas": [{"alimento":"string","contribuicao_g":number,"explicacao":"string"}],
    "gorduras": [{"alimento":"string","contribuicao_g":number,"explicacao":"string"}],
    "fibras": [{"alimento":"string","contribuicao_g":number,"explicacao":"string"}]
  },
  "insights": ["string"]
}
"""


# ----------------------------
# Estilo visual
# ----------------------------
def apply_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

        :root {
            --red: #c62828;
            --red-dark: #8e1717;
            --cream: #fffaf5;
            --ink: #241f1d;
            --muted: #756b66;
            --line: #eadfd8;
            --card: #ffffff;
        }

        * { font-family: 'DM Sans', sans-serif; }

        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(198,40,40,.08), transparent 25%),
                linear-gradient(180deg, #fffaf7 0%, #ffffff 42%, #fffaf7 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #211d1b;
            border-right: 1px solid #332c29;
        }

        [data-testid="stSidebar"] * { color: #fff !important; }

        .brand {
            padding: 8px 0 24px;
        }

        .brand-mark {
            display: inline-flex;
            width: 48px;
            height: 48px;
            align-items: center;
            justify-content: center;
            border-radius: 15px;
            background: var(--red);
            font-size: 24px;
            box-shadow: 0 10px 25px rgba(198,40,40,.25);
        }

        .brand-name {
            font-size: 23px;
            font-weight: 700;
            margin-top: 12px;
            letter-spacing: -.5px;
        }

        .brand-name span { color: #ff6b6b; }

        .brand-sub {
            color: #bdb5b0 !important;
            font-size: 12px;
            margin-top: 2px;
        }

        .hero {
            padding: 28px 0 12px;
        }

        .eyebrow {
            color: var(--red);
            font-weight: 700;
            font-size: 13px;
            letter-spacing: 1.3px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: clamp(34px, 5vw, 54px);
            line-height: 1.04;
            margin: 0;
            letter-spacing: -1.5px;
            color: #211d1b;
        }

        .hero h1 span { color: var(--red); }

        .hero p {
            color: var(--muted);
            max-width: 720px;
            font-size: 16px;
            line-height: 1.65;
            margin-top: 14px;
        }

        .feature-card, .result-card, .upload-card, .history-card {
            background: rgba(255,255,255,.92);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 12px 35px rgba(63,42,31,.06);
        }

        .feature-card {
            min-height: 130px;
        }

        .feature-icon {
            font-size: 25px;
            margin-bottom: 10px;
        }

        .feature-title {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 5px;
        }

        .feature-text {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 27px;
            margin: 25px 0 5px;
        }

        .section-sub {
            color: var(--muted);
            margin-bottom: 18px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 8px 12px;
            border-radius: 999px;
            background: #fff0ee;
            color: var(--red);
            border: 1px solid #ffd5d0;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .badge.neutral {
            background: #f5f2ef;
            color: #625953;
            border-color: #e8dfda;
        }

        .food-description {
            font-size: 19px;
            font-weight: 700;
            line-height: 1.45;
            margin: 5px 0 15px;
        }

        .portion {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 18px;
        }

        .metric-card {
            background: #fff;
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 16px;
            min-height: 105px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
        }

        .metric-value {
            font-size: 25px;
            font-weight: 700;
            margin-top: 6px;
            color: #211d1b;
        }

        .metric-unit {
            font-size: 12px;
            color: var(--muted);
        }

        .food-list {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin: 12px 0 0;
        }

        .food-chip {
            padding: 7px 10px;
            border-radius: 999px;
            background: #f7f3f0;
            border: 1px solid #ebe1db;
            color: #514944;
            font-size: 12px;
        }

        .disclaimer {
            background: #faf7f4;
            border-left: 3px solid var(--red);
            padding: 11px 14px;
            border-radius: 8px;
            color: #6e625c;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 16px;
        }

        .empty {
            text-align: center;
            padding: 55px 20px;
            border: 1px dashed #d9ccc4;
            border-radius: 22px;
            background: rgba(255,255,255,.65);
        }

        .empty-icon { font-size: 42px; }
        .empty-title { font-size: 18px; font-weight: 700; margin-top: 8px; }
        .empty-text { color: var(--muted); font-size: 13px; margin-top: 5px; }

        div.stButton > button {
            border-radius: 12px;
            min-height: 44px;
            font-weight: 700;
            border: 1px solid var(--red);
        }

        div.stButton > button[kind="primary"] {
            background: var(--red);
            color: white;
        }

        div.stButton > button[kind="primary"]:hover {
            background: var(--red-dark);
            border-color: var(--red-dark);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 7px;
            background: #f6f0ec;
            padding: 5px;
            border-radius: 13px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 8px 15px;
        }

        .stTabs [aria-selected="true"] {
            background: #fff;
        }

        .footer {
            color: #8b807a;
            font-size: 11px;
            text-align: center;
            padding: 30px 0 15px;
        }

        @media (max-width: 700px) {
            .hero h1 { font-size: 37px; }
            .feature-card { margin-bottom: 10px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Utilidades
# ----------------------------
def image_to_base64_jpeg(image: Image.Image, max_size=(1200, 1200)) -> str:
    image = image.convert("RGB")
    image.thumbnail(max_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_result(data: dict) -> dict:
    numeric_fields = ["calorias", "carboidratos_g", "proteinas_g", "gorduras_g"]
    for field in numeric_fields:
        try:
            data[field] = float(data.get(field, 0))
        except (TypeError, ValueError):
            data[field] = 0.0

    data.setdefault("alimentos", [])
    data.setdefault("porcao_estimada", "Não informado")
    data.setdefault("observacao", "Estimativa realizada por inteligência artificial.")
    return data


def analyze_dish(client: OpenAI, image=None, description="") -> dict:
    content = [{"type": "text", "text": INSTRUCTIONS}]

    if description.strip():
        content.append({
            "type": "text",
            "text": f"Descrição fornecida pelo usuário:\n{description.strip()}"
        })

    if image is not None:
        encoded = image_to_base64_jpeg(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
        })

   response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": content}],
    response_format={"type": "json_object"},
    max_completion_tokens=4000,
)

   choice = response.choices[0]
raw = (choice.message.content or "").strip()

if not raw:
    raise ValueError(
        f"Resposta vazia. Motivo: {choice.finish_reason}. "
        f"Tokens utilizados: {response.usage.completion_tokens if response.usage else 'não informado'}"
    )

raw = raw.replace("```json", "").replace("```", "").strip()

return normalize_result(json.loads(raw))


# ----------------------------
# Componentes
# ----------------------------
def render_metric(label, value, unit):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value} <span class="metric-unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(dish: dict):
    if dish.get("erro"):
        st.error(dish["erro"])
        return

    mineira = bool(dish.get("culinaria_mineira"))
    if mineira:
        badge = f"🟠 Culinária mineira"
        if dish.get("prato_tipico"):
            badge += f" · {dish['prato_tipico']}"
        badge_class = "badge"
    else:
        badge = "⚪ Não identificada como culinária mineira"
        badge_class = "badge neutral"

    foods = dish.get("alimentos") or []
    chips = "".join(f'<span class="food-chip">{x}</span>' for x in foods[:12])

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="{badge_class}">{badge}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="food-description">{dish.get("descricao", "Refeição analisada")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'🍽️ <span class="portion">Porção estimada: <b>{dish.get("porcao_estimada", "Não informado")}</b></span>',
        unsafe_allow_html=True,
    )

    if foods:
        st.markdown(f'<div class="food-list">{chips}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        render_metric("Calorias", f"{round(dish['calorias'])}", "kcal")
    with cols[1]:
        render_metric("Carboidratos", f"{dish['carboidratos_g']:.1f}", "g")
    with cols[2]:
        render_metric("Proteínas", f"{dish['proteinas_g']:.1f}", "g")
    with cols[3]:
        render_metric("Gorduras", f"{dish['gorduras_g']:.1f}", "g")

    if dish.get("justificativa_classificacao"):
        st.markdown(
            f'<div style="margin-top:18px"><b>Por que essa classificação?</b><br>'
            f'<span style="color:#756b66">{dish["justificativa_classificacao"]}</span></div>',
            unsafe_allow_html=True,
        )

    foods = dish.get("alimentos", [])
    if foods:
        st.markdown("### 🍴 Detalhamento por alimento")
        for food in foods:
            if not isinstance(food, dict):
                continue
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #eadfd8;border-radius:16px;padding:15px;margin:8px 0">
              <b>{food.get("nome", "Alimento")}</b><br>
              <span style="color:#756b66;font-size:12px">{food.get("quantidade_estimada", "Porção não informada")}</span><br>
              <span style="font-size:12px">🔥 {float(food.get("calorias",0)):.0f} kcal · 🍚 {float(food.get("carboidratos_g",0)):.1f} g carboidratos · 🥩 {float(food.get("proteinas_g",0)):.1f} g proteínas · 🥑 {float(food.get("gorduras_g",0)):.1f} g gorduras · 🌿 {float(food.get("fibras_g",0)):.1f} g fibras</span>
            </div>
            """, unsafe_allow_html=True)

    sources = dish.get("fontes_nutrientes", {})
    st.markdown("### 🔎 De onde vêm os nutrientes?")
    source_tabs = st.tabs(["🍚 Carboidratos", "🥩 Proteínas", "🥑 Gorduras", "🌿 Fibras"])
    for tab, key, title in zip(source_tabs, ["carboidratos", "proteinas", "gorduras", "fibras"], ["Carboidratos", "Proteínas", "Gorduras", "Fibras"]):
        with tab:
            items = sources.get(key, [])
            if not items:
                st.info("A IA não conseguiu determinar uma fonte específica.")
            else:
                cols = st.columns(min(3, len(items)))
                for i, item in enumerate(items[:6]):
                    with cols[i % len(cols)]:
                        st.markdown(f"**{item.get('alimento','Não identificado')}** — {float(item.get('contribuicao_g',0)):.1f} g")
                        st.caption(item.get("explicacao", ""))

    if dish.get("insights"):
        st.markdown("### 💡 O que a IA percebeu")
        for insight in dish["insights"][:6]:
            st.markdown("• " + str(insight))

    if dish.get("observacao"):
        st.markdown(
            f'<div class="disclaimer">ℹ️ {dish["observacao"]} Os valores são aproximados e '
            'podem variar conforme ingredientes, quantidade e preparo.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">🍽️</div>
                <div class="brand-name">Power<span>Nutri</span> AI</div>
                <div class="brand-sub">Inteligência artificial para sua refeição</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Acesso à IA")
        api_key = get_openai_api_key()
        st.caption("✅ Chave da API configurada" if api_key else "⚠️ Configure OPENAI_API_KEY em Secrets.")

        st.markdown("---")
        st.markdown("### Menu")
        page = st.radio(
            "Escolha uma página",
            ["🏠 Analisar refeição", "📊 Histórico"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        history = st.session_state.history
        if history:
            st.caption(f"{len(history)} análise(s) nesta sessão")
            for item in history[:4]:
                desc = item["dish"].get("descricao", "Refeição")
                kcal = round(item["dish"].get("calorias", 0))
                st.caption(f"• {desc[:34]}… · {kcal} kcal")

        return api_key, page


# ----------------------------
# Páginas
# ----------------------------
def show_home(api_key: str):
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ANÁLISE NUTRICIONAL COM IA</div>
            <h1>Entenda o que tem<br>no seu <span>prato.</span></h1>
            <p>
                Fotografe uma refeição ou descreva o que comeu. A PowerNutri AI
                estima calorias e macronutrientes e identifica possíveis características
                da culinária mineira.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">📸</div>'
            '<div class="feature-title">Analise por foto</div>'
            '<div class="feature-text">Envie uma imagem do prato ou use a câmera do dispositivo.</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">🥗</div>'
            '<div class="feature-title">Veja os nutrientes</div>'
            '<div class="feature-text">Receba uma estimativa de calorias, carboidratos, proteínas e gorduras.</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">⛰️</div>'
            '<div class="feature-title">Cultura mineira</div>'
            '<div class="feature-text">Confira se há elementos característicos da culinária de Minas Gerais.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Nova análise</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Escolha uma forma de informar sua refeição.</div>',
        unsafe_allow_html=True,
    )

    tab_photo, tab_camera, tab_text = st.tabs(
        ["📁 Enviar foto", "📷 Tirar foto", "✍️ Descrever refeição"]
    )

    image = None
    description = ""
    description = ""

    with tab_photo:
        uploaded = st.file_uploader(
            "Selecione uma imagem",
            type=["jpg", "jpeg", "png", "webp"],
            help="Prefira uma foto de cima, com boa iluminação.",
        )
        if uploaded:
            image = Image.open(uploaded)

    with tab_camera:
        captured = st.camera_input("Fotografe seu prato")
        if captured:
            image = Image.open(captured)

    with tab_text:
        description = st.text_area(
            "O que você comeu?",
            placeholder="Ex.: 2 colheres de arroz, feijão tropeiro, couve refogada e um pedaço de frango assado.",
            height=140,
        )

    if image is not None:
        st.markdown("### Pré-visualização")
        st.image(image, width=430)

    if description.strip():
        st.markdown("### Descrição informada")
        st.info(description)

    ready = image is not None or description.strip()

    if ready:
        if st.button("✨ Analisar minha refeição", type="primary", use_container_width=True):
            if not api_key:
                st.error("Chave da API não configurada. Adicione OPENAI_API_KEY em Secrets no Streamlit Cloud.")
                return

            with st.spinner("A IA está analisando sua refeição..."):
                try:
                    client = OpenAI(api_key=api_key)
                    dish = analyze_dish(client, image=image, description=description)
                except Exception as exc:
                    st.error(f"Não foi possível concluir a análise: {exc}")
                    return

            if dish.get("erro"):
                st.error(dish["erro"])
                return

            st.markdown("### Resultado da análise")
            render_result(dish)

            st.session_state.history.insert(
                0,
                {
                    "dish": dish,
                    "image": image.copy() if image is not None else None,
                    "timestamp": datetime.now(),
                },
            )
    else:
        st.markdown(
            """
            <div class="empty">
                <div class="empty-icon">🍲</div>
                <div class="empty-title">Seu prato aparecerá aqui</div>
                <div class="empty-text">
                    Envie uma foto, tire uma foto ou descreva sua refeição para começar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_history():
    st.markdown('<div class="section-title">Histórico</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Veja as análises realizadas nesta sessão.</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.markdown(
            """
            <div class="empty">
                <div class="empty-icon">📊</div>
                <div class="empty-title">Nenhuma análise ainda</div>
                <div class="empty-text">Suas análises aparecerão aqui depois que você analisar uma refeição.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("🗑️ Limpar histórico", use_container_width=False):
        st.session_state.history = []
        st.rerun()

    for index, item in enumerate(history):
        dish = item["dish"]
        with st.container():
            left, right = st.columns([1, 3])
            with left:
                if item.get("image") is not None:
                    st.image(item["image"], use_container_width=True)
                else:
                    st.markdown("🍽️")
                st.caption(item["timestamp"].strftime("%d/%m/%Y · %H:%M"))
            with right:
                render_result(dish)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_style()

    if "history" not in st.session_state:
        st.session_state.history = []

    api_key, page = render_sidebar()

    if page == "🏠 Analisar refeição":
        show_home(api_key)
    else:
        show_history()

    st.markdown(
        '<div class="footer">PowerNutri AI· Estimativas geradas por inteligência artificial · '
        'Não substitui avaliação de nutricionista.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
