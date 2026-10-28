import json
import math
import os
import re
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_NAME = "Food Verse AI"
st.set_page_config(page_title=APP_NAME, page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
:root { --fv-ink:#16251f; --fv-green:#167d55; --fv-lime:#c9f27b; --fv-cream:#f7f8f2; }
.stApp { background: linear-gradient(135deg,#f8faf5 0%,#eef7ef 48%,#f8f5e9 100%); }
section[data-testid="stSidebar"] { background: #142b25; }
section[data-testid="stSidebar"] * { color: #f5fff7 !important; }
.hero { padding: 1.5rem 1.7rem; border-radius: 22px; color:#fff; background:linear-gradient(115deg,#123b30,#167d55 65%,#83a83a); margin-bottom:1rem; }
.hero h1 { font-size:2.2rem; margin:0; color:#fff; }
.hero p { margin:.4rem 0 0 0; color:#e8ffe9; }
.pill { display:inline-block; padding:.3rem .65rem; margin:.15rem; border-radius:999px; background:#e3f5d9; color:#145b3d; font-weight:600; font-size:.85rem; }
.small-note { color:#52645b; font-size:.88rem; }
div[data-testid="stMetric"] { background:#ffffffc9; border:1px solid #e0eadf; padding: .8rem; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🥗 Food Verse AI</h1>
  <p>Multimodal food intelligence for nutrition, freshness, kitchen safety and smarter meal planning.</p>
</div>
""", unsafe_allow_html=True)

# ---------- AI provider setup ----------
def get_provider():
    return st.session_state.get("ai_provider", "Demo mode")

def get_api_key():
    provider = get_provider()
    if provider == "Gemini":
        return st.session_state.get("gemini_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if provider == "Groq":
        return st.session_state.get("groq_key", "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    return ""

def call_vision(prompt: str, image: Image.Image) -> str:
    """Calls Gemini vision when configured; otherwise returns an explicit demo-mode response."""
    provider, key = get_provider(), get_api_key()
    if provider == "Gemini" and key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, image],
                config={"response_mime_type": "application/json"}
            )
            return response.text or "{}"
        except Exception as e:
            st.warning(f"Gemini request failed. Showing demo guidance instead. Details: {e}")
    elif provider == "Groq" and key:
        st.info("Groq is configured, but image understanding is not enabled in this starter build. Use Gemini for image analysis; Groq text generation can be added as a separate provider.")
    return ""

def call_text(prompt: str) -> str:
    """Calls Gemini text generation if configured; otherwise returns an empty string."""
    provider, key = get_provider(), get_api_key()
    if provider == "Gemini" and key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text or ""
        except Exception as e:
            st.warning(f"AI request failed. Using the local demo output. Details: {e}")
    elif provider == "Groq" and key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                temperature=0.6,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            st.warning(f"Groq request failed. Using the local demo output. Details: {e}")
    return ""

def parse_json(text: str):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None

def show_image(uploaded):
    if uploaded is None:
        return None
    try:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        return img
    except Exception:
        st.error("This image could not be opened. Please upload a JPG, PNG or WEBP image.")
        return None

def health_score(nutrition: dict[str, float]) -> int:
    # Transparent illustrative score; not a clinical or regulatory rating.
    score = 70
    score += min(nutrition.get("fiber", 0) * 2, 10)
    score += min(nutrition.get("protein", 0) * 0.35, 8)
    score -= min(nutrition.get("fat", 0) * 0.18, 10)
    score -= min(nutrition.get("carbs", 0) * 0.04, 8)
    return max(1, min(100, round(score)))

def nutrition_demo(dish: str, portion: float):
    # Approximate illustrative values per serving, scaled by portion multiplier.
    d = dish.lower()
    if any(x in d for x in ["salad", "sprout", "fruit"]):
        base = dict(calories=220, protein=8, carbs=32, fat=7, fiber=8)
    elif any(x in d for x in ["paneer", "tofu"]):
        base = dict(calories=390, protein=22, carbs=24, fat=22, fiber=5)
    elif any(x in d for x in ["rice", "biryani", "pulao"]):
        base = dict(calories=430, protein=12, carbs=66, fat=12, fiber=4)
    elif any(x in d for x in ["chicken", "fish", "egg"]):
        base = dict(calories=360, protein=32, carbs=18, fat=16, fiber=3)
    else:
        base = dict(calories=330, protein=14, carbs=42, fat=12, fiber=5)
    return {k: round(v * portion, 1) for k, v in base.items()}

def safe_json_prompt(task: str, details: str) -> str:
    return f"""You are a cautious food-science assistant. Analyze only visible evidence; do not claim certainty from an image alone.
Return valid JSON only. Task: {task}. {details}
If something cannot be reliably inferred, use null or state uncertainty. Avoid medical diagnosis and avoid guaranteeing food safety."""

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🍃 Food Verse AI")
    st.caption("Skill Wallet · Capstone Project 3")
    st.selectbox("AI provider", ["Demo mode", "Gemini", "Groq"], key="ai_provider")
    if st.session_state.ai_provider == "Gemini":
        st.text_input("Gemini API key", type="password", key="gemini_key", help="Used only for requests from this session. Prefer environment secrets for deployment.")
        st.caption("You can also set GEMINI_API_KEY in Streamlit secrets.")
    elif st.session_state.ai_provider == "Groq":
        st.text_input("Groq API key", type="password", key="groq_key", help="Used for text generation. Image analysis in this starter uses Gemini.")
        st.caption("You can also set GROQ_API_KEY in Streamlit secrets.")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🍽️ Nutrition Audit",
        "🍓 Freshness Check",
        "🧼 Kitchen Hygiene",
        "👩‍🍳 Recipe Assistant",
        "📅 Meal Planner",
    ])
    st.markdown("---")
    st.caption("Demo estimates are illustrative, not lab measurements or food-safety certification.")

# ---------- Overview ----------
if page == "🏠 Overview":
    st.subheader("Your kitchen intelligence dashboard")
    st.write("Analyze a meal, review visible freshness cues, inspect kitchen practices, cook with pantry ingredients, or plan a week of meals.")
    cols = st.columns(3)
    cards = [
        ("🍽️", "Nutrition", "Dish recognition and estimated macros"),
        ("🍓", "Freshness", "Visible quality cues and cautious storage advice"),
        ("🧼", "Hygiene", "Visual checklist for kitchen practices"),
        ("👩‍🍳", "Recipes", "Ingredient-aware recipe ideas"),
        ("📅", "Meal planning", "BMR/TDEE estimate and 7-day plan"),
        ("🔒", "Privacy", "No external database; data is session-based"),
    ]
    for i, (emoji, title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"### {emoji} {title}")
            st.write(desc)
    st.info("Start in Demo mode to explore the interface without an API key. Choose Gemini in the sidebar for image-based AI analysis.")
    st.markdown("**Important:** a photo cannot reliably determine exact nutrient quantities, internal spoilage, pathogen presence, or full regulatory compliance. Treat results as decision support only.")

# ---------- Nutrition Audit ----------
elif page == "🍽️ Nutrition Audit":
    st.subheader("Food Recognition & Nutrition Audit")
    uploaded = st.file_uploader("Upload a meal or dish photo", type=["jpg", "jpeg", "png", "webp"], key="nutrition_img")
    portion = st.slider("Estimated portion multiplier", 0.5, 2.5, 1.0, 0.25)
    img = show_image(uploaded)
    if st.button("Analyze meal", type="primary", disabled=img is None):
        prompt = safe_json_prompt("food recognition and nutrition estimation",
            """Return keys: dish_name, cuisine, ingredients (array), portion_description, calories_kcal, protein_g, carbohydrates_g, fats_g, fiber_g, allergens (array), uncertainty_notes (array).
Nutrition values should be plausible estimates for the visible portion, not falsely precise. Mention common allergens only when ingredients are visible or strongly inferred.""")
        with st.spinner("Reviewing visible food features..."):
            raw = call_vision(prompt, img)
            result = parse_json(raw)
        if result:
            st.session_state["nutrition_result"] = result
        else:
            st.session_state["nutrition_result"] = {
                "dish_name": "Sample mixed meal", "cuisine": "Not confirmed",
                "ingredients": ["Visible ingredients require confirmation"],
                "portion_description": "One pictured serving",
                "calories_kcal": nutrition_demo("mixed meal", portion)["calories"],
                "protein_g": nutrition_demo("mixed meal", portion)["protein"],
                "carbohydrates_g": nutrition_demo("mixed meal", portion)["carbs"],
                "fats_g": nutrition_demo("mixed meal", portion)["fat"],
                "fiber_g": nutrition_demo("mixed meal", portion)["fiber"],
                "allergens": ["Check ingredient labels and preparation details"],
                "uncertainty_notes": ["Demo estimate only; identify the dish and ingredients manually for better illustrative values."]
            }
            st.session_state["nutrition_demo"] = True
    result = st.session_state.get("nutrition_result")
    if result:
        st.markdown(f"### {result.get('dish_name') or 'Dish analysis'}")
        st.write(f"**Cuisine:** {result.get('cuisine') or 'Uncertain'}  ·  **Portion:** {result.get('portion_description') or 'Not estimated'}")
        cols = st.columns(4)
        for col, label, key, unit in [
            (cols[0],"Calories","calories_kcal","kcal"), (cols[1],"Protein","protein_g","g"),
            (cols[2],"Carbohydrates","carbohydrates_g","g"), (cols[3],"Fat","fats_g","g")]:
            val = result.get(key)
            col.metric(label, f"{val} {unit}" if val is not None else "Unknown")
        fiber = result.get("fiber_g")
        st.metric("Fiber", f"{fiber} g" if fiber is not None else "Unknown")
        macro = {
            "Protein (g)": result.get("protein_g") or 0,
            "Carbohydrates (g)": result.get("carbohydrates_g") or 0,
            "Fat (g)": result.get("fats_g") or 0,
            "Fiber (g)": fiber or 0
        }
        st.plotly_chart(px.bar(x=list(macro.keys()), y=list(macro.values()), labels={"x":"Nutrient","y":"Estimated grams"}, title="Estimated nutrient breakdown"), use_container_width=True)
        nutrition = {"protein": macro["Protein (g)"], "carbs": macro["Carbohydrates (g)"], "fat": macro["Fat (g)"], "fiber": macro["Fiber (g)"]}
        score = health_score(nutrition)
        st.markdown(f'<span class="pill">Illustrative health score: {score}/100</span>', unsafe_allow_html=True)
        st.caption("This score is a simple demonstration heuristic, not a validated nutrition standard. A balanced diet depends on the full meal and individual needs.")
        st.write("**Ingredients detected / suggested**")
        st.write(", ".join(result.get("ingredients") or ["Not identified"]))
        st.write("**Allergen checks**")
        for item in result.get("allergens") or ["No allergens confirmed; verify ingredients and cross-contact"]:
            st.warning(str(item))
        for note in result.get("uncertainty_notes") or []:
            st.caption(f"Note: {note}")

# ---------- Freshness ----------
elif page == "🍓 Freshness Check":
    st.subheader("Food Freshness & Spoilage Diagnostic")
    uploaded = st.file_uploader("Upload a photo of fruit, vegetables, meat or dairy", type=["jpg","jpeg","png","webp"], key="fresh_img")
    category = st.selectbox("Food category", ["Fruit", "Vegetable", "Meat / poultry", "Fish / seafood", "Dairy", "Other"])
    storage = st.selectbox("Storage history", ["Unknown", "Refrigerated continuously", "Room temperature", "Frozen", "Recently purchased"])
    img = show_image(uploaded)
    if st.button("Assess visible freshness cues", type="primary", disabled=img is None):
        prompt = safe_json_prompt("visual freshness assessment",
            f"""Food category: {category}. Report storage history supplied by user: {storage}.
Return keys: item_guess, visible_cues (array), stage (Fresh|Moderately Fresh|Spoiled|Uncertain), advice (Safe to consume|Consume soon|Discard|Cannot determine from image), storage_tips (array), limitations (array).
Do not declare meat, seafood or dairy safe based on appearance alone. If mold, unusual discoloration or uncertain high-risk food is shown, recommend caution. Explicitly note that smell, temperature history and pathogens cannot be verified from an image.""")
        with st.spinner("Checking visible cues..."):
            result = parse_json(call_vision(prompt, img))
        if result:
            st.session_state["fresh_result"] = result
        else:
            st.session_state["fresh_result"] = {
                "item_guess": category, "visible_cues": ["Demo mode: no AI vision analysis was run."],
                "stage": "Uncertain", "advice": "Cannot determine from image",
                "storage_tips": ["Keep perishable foods refrigerated at 4°C or below.", "Separate raw meat and seafood from ready-to-eat foods.", "Use clean, covered containers and follow package dates."],
                "limitations": ["A photo cannot confirm pathogen absence or internal spoilage. When in doubt, discard high-risk food."]
            }
    result = st.session_state.get("fresh_result")
    if result:
        st.markdown(f"### Assessment: {result.get('stage','Uncertain')}")
        st.write(f"**Item guess:** {result.get('item_guess','Not identified')}")
        st.write(f"**Guidance:** {result.get('advice','Cannot determine from image')}")
        st.write("**Visible cues**")
        for cue in result.get("visible_cues") or []: st.write(f"- {cue}")
        st.write("**Storage tips**")
        for tip in result.get("storage_tips") or []: st.write(f"- {tip}")
        for note in result.get("limitations") or []: st.warning(note)
        st.caption("This is not a food-safety test. Do not taste food to check whether it is spoiled.")

# ---------- Kitchen hygiene ----------
elif page == "🧼 Kitchen Hygiene":
    st.subheader("Kitchen Hygiene & Food Safety Compliance")
    st.write("Upload a kitchen image or a short video. Videos are sampled at a few intervals to keep processing lightweight.")
    media_kind = st.radio("Media type", ["Image", "Video"], horizontal=True)
    uploaded = st.file_uploader("Upload kitchen media", type=["jpg","jpeg","png","webp"] if media_kind=="Image" else ["mp4","mov","avi"], key="hygiene_media")
    frames = []
    if uploaded:
        if media_kind == "Image":
            im = show_image(uploaded)
            if im: frames = [im]
        else:
            st.video(uploaded)
            try:
                import cv2
                import tempfile
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue()); temp_path = tmp.name
                cap = cv2.VideoCapture(temp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = cap.get(cv2.CAP_PROP_FPS) or 1
                if total > 0:
                    sample_indices = sorted(set([0, total//3, (2*total)//3, total-1]))
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        if ok:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(Image.fromarray(frame))
                cap.release()
                os.unlink(temp_path)
                if frames:
                    st.image(frames, caption=[f"Sample frame {i+1}" for i in range(len(frames))], use_container_width=True)
                else:
                    st.error("Could not extract frames. Try a shorter MP4 video.")
            except Exception as e:
                st.error(f"Video processing needs OpenCV and a supported codec: {e}")
    if st.button("Review hygiene indicators", type="primary", disabled=not frames):
        observations = []
        with st.spinner("Reviewing sampled frame(s)..."):
            for i, frame in enumerate(frames[:4]):
                prompt = safe_json_prompt("visual kitchen hygiene review",
                    """Return keys: hair_covering (Compliant|Concern|Not visible), gloves (Compliant|Concern|Not visible), visible_cleanliness (Good|Concern|Not assessable), raw_ready_to_eat_separation (Good|Concern|Not visible), violations (array), limitations (array).
Only report visible evidence. Gloves do not replace handwashing. Do not infer unseen procedures.""")
                result = parse_json(call_vision(prompt, frame))
                if result:
                    observations.append({"frame": i+1, **result})
                else:
                    observations.append({"frame": i+1, "hair_covering":"Not assessed in demo mode","gloves":"Not assessed in demo mode",
                        "visible_cleanliness":"Not assessable","raw_ready_to_eat_separation":"Not visible",
                        "violations":["Demo mode: connect Gemini to analyze frames."],
                        "limitations":["This is not a formal inspection or certification. Check temperatures, cleaning records and procedures in person."]})
        st.session_state["hygiene_result"] = observations
    results = st.session_state.get("hygiene_result")
    if results:
        # Avoid implying a validated compliance score when no reliable AI observations exist.
        all_violations = []
        for r in results:
            st.markdown(f"**Sample frame {r['frame']}**")
            a,b,c = st.columns(3)
            a.metric("Hair covering", str(r.get("hair_covering","Not visible")))
            b.metric("Glove use", str(r.get("gloves","Not visible")))
            c.metric("Cleanliness", str(r.get("visible_cleanliness","Not assessable")))
            st.write("Raw / ready-to-eat separation:", r.get("raw_ready_to_eat_separation","Not visible"))
            for v in r.get("violations") or []:
                all_violations.append(v); st.warning(v)
            for n in r.get("limitations") or []: st.caption(n)
        st.info("A defensible compliance percentage requires a defined checklist, weighted criteria and human verification. This starter reports observations rather than presenting an unvalidated score as an official rating.")

# ---------- Recipe assistant ----------
elif page == "👩‍🍳 Recipe Assistant":
    st.subheader("Smart Kitchen Recipe Assistant")
    ingredients = st.text_area("What ingredients do you have?", placeholder="e.g. rice, tomatoes, onion, paneer, spinach, yogurt")
    diet = st.selectbox("Dietary preference", ["No restriction", "Vegetarian", "Vegan", "High protein", "Low sodium"])
    time_limit = st.slider("Maximum cooking time (minutes)", 10, 90, 30, 5)
    servings = st.number_input("Servings", min_value=1, max_value=12, value=2)
    avoid = st.text_input("Ingredients to avoid (optional)", placeholder="e.g. peanuts, dairy")
    if st.button("Generate recipe ideas", type="primary", disabled=not ingredients.strip()):
        prompt = f"""Create 3 practical recipes using these pantry ingredients: {ingredients}.
Diet: {diet}. Maximum time: {time_limit} minutes. Servings: {servings}. Avoid: {avoid or 'none stated'}.
For each recipe include title, short description, estimated time, difficulty, ingredients with quantities, numbered steps, substitutions, and a brief nutrition note. Do not invent medical claims. Mention if an ingredient is optional or needs to be purchased."""
        answer = call_text(prompt)
        if answer:
            st.session_state["recipes"] = answer
        else:
            ing = [x.strip() for x in ingredients.split(",") if x.strip()]
            st.session_state["recipes"] = (
                "### Pantry skillet bowl\n"
                f"**Servings:** {servings} · **Time:** about {min(time_limit,25)} minutes · **Difficulty:** Easy\n\n"
                "**Ingredients:** " + ", ".join(ing[:8]) + "; cooking oil, salt and spices as available.\n\n"
                "**Steps:**\n1. Wash and prepare the ingredients.\n2. Cook any grains or legumes that need longer cooking first.\n3. Sauté aromatics and firm vegetables in a pan.\n4. Add the remaining ingredients and cook until fully heated and appropriate for the food type.\n5. Taste, season, and serve.\n\n"
                "**Substitutions:** Use a similar vegetable or protein if an ingredient is missing. For vegan meals, replace dairy with an unsweetened plant alternative.\n\n"
                "**Demo mode:** This generic recipe template is not AI-generated. Choose Gemini or Groq for tailored ideas."
            )
    if st.session_state.get("recipes"):
        st.markdown(st.session_state["recipes"])

# ---------- Meal planner ----------
elif page == "📅 Meal Planner":
    st.subheader("Personalized Meal Planner")
    st.caption("BMR/TDEE are estimates for adults and should not replace individualized medical or dietetic advice.")
    with st.form("profile_form"):
        c1,c2,c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 100, 25)
        sex = c2.selectbox("Sex used by equation", ["Male", "Female"])
        height = c3.number_input("Height (cm)", 120.0, 230.0, 170.0, 1.0)
        c4,c5,c6 = st.columns(3)
        weight = c4.number_input("Weight (kg)", 30.0, 250.0, 70.0, 0.5)
        activity = c5.selectbox("Activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
        goal = c6.selectbox("Goal", ["Maintenance", "Fat loss", "Muscle gain"])
        diet = st.selectbox("Diet pattern", ["No restriction", "Vegetarian", "Vegan", "Keto", "Diabetic-friendly"])
        allergies = st.text_input("Allergies / foods to avoid", placeholder="e.g. peanuts, shellfish")
        submit = st.form_submit_button("Calculate targets & build week", type="primary")
    if submit:
        # Mifflin-St Jeor estimate
        bmr = 10*weight + 6.25*height - 5*age + (5 if sex == "Male" else -161)
        factor = {"Sedentary":1.2,"Lightly active":1.375,"Moderately active":1.55,"Very active":1.725,"Extra active":1.9}[activity]
        tdee = bmr * factor
        target = tdee + (-350 if goal == "Fat loss" else 250 if goal == "Muscle gain" else 0)
        target = max(1200 if sex == "Female" else 1500, target)
        st.session_state["meal_targets"] = {"bmr":round(bmr),"tdee":round(tdee),"target":round(target),"diet":diet,"goal":goal,"allergies":allergies}
        prompt = f"""Create a varied 7-day meal schedule for an adult. Estimated target: about {round(target)} kcal/day. Diet: {diet}; goal: {goal}; foods to avoid/allergies: {allergies or 'none stated'}.
Include breakfast, lunch, dinner and one snack each day, simple ingredients commonly available in India, approximate calories per meal, and a consolidated grocery list with approximate quantities. Avoid presenting as medical advice. Treat allergies strictly and state that packaged ingredient labels/cross-contact must be checked."""
        plan = call_text(prompt)
        if plan:
            st.session_state["meal_plan"] = plan
        else:
            st.session_state["meal_plan"] = None
    targets = st.session_state.get("meal_targets")
    if targets:
        c1,c2,c3 = st.columns(3)
        c1.metric("Estimated BMR", f"{targets['bmr']} kcal/day")
        c2.metric("Estimated TDEE", f"{targets['tdee']} kcal/day")
        c3.metric("Illustrative target", f"{targets['target']} kcal/day")
        st.caption("Mifflin–St Jeor BMR estimate × activity multiplier, with a modest goal adjustment. Individual needs vary.")
        if st.session_state.get("meal_plan"):
            st.markdown(st.session_state["meal_plan"])
        else:
            st.warning("Demo mode: below is a simple repeating template, not a personalized nutrition prescription.")
            if targets["diet"] == "Vegan":
                meals = ["Oats with soy milk and fruit","Rajma with rice and salad","Tofu and vegetable stir-fry","Roasted chana"]
            elif targets["diet"] == "Vegetarian":
                meals = ["Oats with milk and fruit","Dal, rice and salad","Paneer/tofu with roti and vegetables","Yogurt or roasted chana"]
            elif targets["diet"] == "Keto":
                meals = ["Eggs or tofu with low-carb vegetables","Paneer/tofu salad with olive oil","Vegetable and protein skillet","Nuts in a measured portion"]
            else:
                meals = ["Oats with fruit and yogurt","Dal or lean protein with roti/rice and vegetables","Protein with vegetables and a measured grain portion","Fruit, yogurt or roasted chana"]
            rows = []
            for d in range(7):
                rows.append({"Day":f"Day {d+1}","Breakfast":meals[0],"Lunch":meals[1],"Dinner":meals[2],"Snack":meals[3]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("**Starter grocery list**")
            st.write("Oats, seasonal fruit, vegetables, lentils/beans, rice or whole-wheat roti, a suitable protein, and optional yogurt or plant milk. Adjust quantities to serving size and verify all allergy labels.")
