# Food Verse AI — Multimodal Food Intelligence Agent

A Streamlit capstone starter covering food recognition/nutrition estimates, visible freshness cues, kitchen hygiene review, recipe assistance, and meal planning.

## Run locally

Python 3.10–3.12 is recommended for the smoothest compatibility with computer-vision packages.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Select **Demo mode** in the sidebar to explore without an API key.

## Configure Gemini

1. Create an API key in Google AI Studio.
2. In the sidebar, select **Gemini** and paste the key into the password field, or set `GEMINI_API_KEY` as an environment variable.
3. For Streamlit Community Cloud, add `GEMINI_API_KEY = "your-key"` to the app's Secrets settings (TOML format).

The code uses the Google Gen AI SDK and `gemini-2.5-flash`. Model availability can vary by account/region. Never commit API keys to Git.

## Configure Groq

Select **Groq** and enter a Groq API key. Text-based recipe and meal-plan generation uses a Llama model. Image analysis currently requires Gemini; the app clearly falls back to demo mode if vision analysis is unavailable.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. In Streamlit Community Cloud, create an app pointing to `app.py`.
3. Add API keys through the app's Secrets interface if using a provider.
4. Deploy.

No external database is required. Uploaded media is processed in the active app session; avoid uploading sensitive images. Streamlit hosting resources and upload limits apply.

## What is implemented

- Five interactive modules and overview dashboard
- Gemini image-analysis path for food and hygiene prompts
- Gemini/Groq text generation for recipes and meal plans
- Local demo fallbacks with explicit labels
- Nutrition macro chart, transparent illustrative score, and BMR/TDEE calculator
- Sampled video frames via OpenCV
- Session-state results and exception handling

## Important limitations before production use

- **Nutrition:** Image-based portions and nutrient quantities are estimates. The demo score is a heuristic, not a validated clinical rating.
- **Freshness:** Images cannot detect pathogens, internal spoilage, temperature abuse, or reliably establish that food is safe. Do not taste questionable food.
- **Kitchen hygiene:** A handful of video frames cannot certify compliance. PPE and separation checks need human review, defined scoring rules, and validation.
- **Meal planning:** Calorie formulas are estimates for adults and are not medical nutrition therapy. The starter template is not a substitute for a registered dietitian, especially for diabetes, pregnancy, eating disorders, or other clinical needs.
- **Security:** API keys typed into the sidebar are session inputs. Prefer Streamlit Secrets or environment variables in deployments. Do not log secrets.
- **Production readiness:** This is a capstone MVP, not a validated food-safety product. Add tests, rate limiting, image-size controls, monitoring, privacy disclosures, and model evaluation before public release.
