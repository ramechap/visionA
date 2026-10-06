# import base64
# from io import BytesIO
# from PIL import Image
# import streamlit as st
# from openai import OpenAI

# # 1. Setup the Web Page look
# st.set_page_config(page_title="Hackathon Vision Portal", layout="centered")
# st.title("📸 AI Image Recognition Hub")
# st.write("Upload an image to recognize text, objects, and details instantly.")

# # 2. Securely get your OpenRouter API Key
# # (We will set this up in Hugging Face settings later so it stays hidden)
# api_key = st.secrets.get("OPENROUTER_API_KEY")

# if not api_key:
#     st.warning("⚠️ API Key missing. Please configure OPENROUTER_API_KEY in your server settings.")
# else:
#     # Connect to the cloud AI server
#     client = OpenAI(
#         base_url="https://openrouter.ai",
#         api_key=api_key
#     )

#     # 3. Create the Image Upload Button
#     uploaded_file = st.file_uploader("Upload a photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Open and display the uploaded image on the screen
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Your Uploaded Photo", use_container_width=True)
        
#         # Convert the picture to a text string (Base64) so it can travel over the internet safely
#         buffered = BytesIO()
#         image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        

#         # 4. Create the "Process" button
#         # 4. Create the "Process" button
#         if st.button("Analyze Image 🚀"):
#             with st.spinner("The AI is analyzing your image in the cloud..."):
#                 try:
#                     # Explicitly formatting OpenRouter requirements
#                     response = client.chat.completions.create(
#                         model="deepseek/deepseek-v4.1-flash", 
#                         extra_headers={
#                             "HTTP-Referer": "https://streamlit.app", # Required by OpenRouter
#                             "X-Title": "Hackathon App Prototype",     # Required by OpenRouter
#                         },
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [
#                                     {"type": "text", "text": "Perform advanced image recognition. List all key objects, read any text present, and summarize what is happening in the scene."},
#                                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
#                                 ]
#                             }
#                         ]
#                     )
                    
#                     # --- SAFE CHECKING FOR STRUCT TYPE ---
#                     if isinstance(response, str):
#                         st.error("⚠️ The server sent an invalid text response. Check if your API key or model string has a typo.")
#                         st.code(response[:500]) # Prints snippet to prevent page clutter
#                     elif hasattr(response, 'choices') and response.choices:
#                         st.success("Analysis Complete!")
#                         st.markdown("### 📊 AI Recognition Results")
#                         st.write(response.choices[0].message.content)
#                     else:
#                         st.error("Received unexpected format from API. Please try again.")
                        
#                 except Exception as e:
#                     st.error(f"Something went wrong during the cloud call: {e}")


# import io
# import json
# import time

# import numpy as np
# import streamlit as st
# from PIL import Image, ImageDraw
# from transformers import pipeline

# st.set_page_config(page_title="AI Vision Scanner Pro", page_icon="📸", layout="wide")

# st.markdown("""
#     <style>
#     .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;
#         background-color: #FF9D00; color: white; height: 48px; border: none; }
#     .stButton>button:hover { background-color: #E08900; color: white; }
#     h1, .subtitle { text-align: center; }
#     </style>
#     """, unsafe_allow_html=True)

# st.title("📸 AI Vision Scanner Pro")
# st.markdown("<p class='subtitle'>Free, open-source image intelligence: captioning, object detection, OCR, "
#             "classification and custom-label search, all running locally with Hugging Face models.</p>",
#             unsafe_allow_html=True)

# # ---------- Lazy-loaded models (each loads only when first used, then stays cached) ----------


# @st.cache_resource(show_spinner=False)
# def get_captioner():
#     from transformers import BlipProcessor, BlipForConditionalGeneration
#     name = "Salesforce/blip-image-captioning-base"
#     processor = BlipProcessor.from_pretrained(name)
#     model = BlipForConditionalGeneration.from_pretrained(name)
#     model.eval()
#     return processor, model

# def generate_caption(img):
#     import torch
#     processor, model = get_captioner()
#     inputs = processor(images=img, return_tensors="pt")
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=40)
#     return processor.decode(out[0], skip_special_tokens=True)

# @st.cache_resource(show_spinner=False)
# def get_detector():
#     return pipeline("object-detection", model="hustvl/yolos-tiny")

# @st.cache_resource(show_spinner=False)
# def get_classifier():
#     return pipeline("image-classification", model="google/vit-base-patch16-224")

# @st.cache_resource(show_spinner=False)
# def get_zero_shot():
#     return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# @st.cache_resource(show_spinner=False)
# def get_ocr(lang_codes):
#     import easyocr
#     return easyocr.Reader(list(lang_codes), gpu=False)

# # ---------- Helpers ----------
# PALETTE = ["#FF9D00", "#00B4D8", "#E63946", "#2A9D8F", "#9B5DE5", "#F15BB5", "#80B918"]

# def dominant_colors(img, n=5):
#     q = img.resize((100, 100)).quantize(colors=n, method=Image.Quantize.MEDIANCUT)
#     pal = q.getpalette()[: n * 3]
#     out = []
#     for count, idx in sorted(q.getcolors(), reverse=True):
#         r, g, b = pal[idx * 3: idx * 3 + 3]
#         out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "percent": round(count / 100, 1)})
#     return out

# def draw_detections(img, dets):
#     out = img.copy()
#     d = ImageDraw.Draw(out)
#     labels = sorted({x["label"] for x in dets})
#     for x in dets:
#         c = PALETTE[labels.index(x["label"]) % len(PALETTE)]
#         b = x["box"]
#         d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=c, width=4)
#         d.text((b["xmin"] + 5, b["ymin"] + 5), f'{x["label"]} {x["score"]:.0%}', fill=c)
#     return out

# def draw_ocr(img, items):
#     out = img.copy()
#     d = ImageDraw.Draw(out)
#     for it in items:
#         pts = [tuple(p) for p in it["box"]]
#         d.polygon(pts, outline="#00B4D8")
#     return out

# def to_png_bytes(img):
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     return buf.getvalue()

# # ---------- Sidebar controls ----------
# with st.sidebar:
#     st.header("⚙️ Analysis Options")
#     do_caption = st.checkbox("📝 Scene description (captioning)", True)
#     do_detect = st.checkbox("🔍 Object detection", True)
#     do_classify = st.checkbox("🎯 Classification (top 5)", True)
#     do_ocr = st.checkbox("🔤 Read text in image (OCR)", False)
#     do_custom = st.checkbox("🏷️ Custom label search (CLIP)", False)
#     do_colors = st.checkbox("🎨 Dominant colors", True)

#     threshold = st.slider("Detection confidence", 0.1, 0.95, 0.5, 0.05)
#     ocr_lang = st.multiselect("OCR languages", ["en", "ne", "hi", "fr", "de", "es"], default=["en"])
#     custom_labels = st.text_input("Custom labels (comma-separated)",
#                                   "dog, cat, car, person, food, building")

#     st.caption("Tip: models load the first time you use them. If the app runs out of memory, "
#                "turn off some options or free memory below.")
#     if st.button("🧹 Free memory"):
#         st.cache_resource.clear()
#         st.success("Models unloaded.")

# # ---------- Main ----------
# uploaded = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

# if uploaded:
#     image = Image.open(uploaded).convert("RGB")
#     image.thumbnail((1024, 1024))  # keeps things fast and light on memory

#     left, right = st.columns([1, 1])
#     left.image(image, caption=f"{uploaded.name} ({image.width}×{image.height})", use_container_width=True)

#     with right:
#         st.write("Choose options in the sidebar, then run the scan.")
#         run = st.button("Run Full Analysis 🚀")

#     if run:
#         R = {"file": uploaded.name, "timings_sec": {}}
#         t_all = time.time()
#         try:
#             if do_caption:
#                 with st.spinner("Describing the scene..."):
#                     t = time.time()
#                     R["caption"] = generate_caption(image)
#                     R["timings_sec"]["caption"] = round(time.time() - t, 2)

#             if do_classify:
#                 with st.spinner("Classifying..."):
#                     t = time.time()
#                     R["classes"] = [{"label": c["label"], "score": round(float(c["score"]), 4)}
#                                     for c in get_classifier()(image, top_k=5)]
#                     R["timings_sec"]["classification"] = round(time.time() - t, 2)

#             if do_detect:
#                 with st.spinner("Detecting objects..."):
#                     t = time.time()
#                     dets = [d for d in get_detector()(image) if d["score"] >= threshold]
#                     R["objects"] = [{"label": d["label"], "score": round(float(d["score"]), 4),
#                                      "box": d["box"]} for d in dets]
#                     R["objects_img"] = to_png_bytes(draw_detections(image, dets))
#                     R["timings_sec"]["detection"] = round(time.time() - t, 2)

#             if do_ocr:
#                 with st.spinner("Reading text (first run downloads OCR models)..."):
#                     t = time.time()
#                     reader = get_ocr(tuple(ocr_lang or ["en"]))
#                     raw = reader.readtext(np.array(image))
#                     R["ocr"] = [{"text": txt, "confidence": round(float(conf), 3),
#                                  "box": [[int(x), int(y)] for x, y in box]} for box, txt, conf in raw]
#                     R["ocr_img"] = to_png_bytes(draw_ocr(image, R["ocr"]))
#                     R["timings_sec"]["ocr"] = round(time.time() - t, 2)

#             if do_custom:
#                 labels = [l.strip() for l in custom_labels.split(",") if l.strip()]
#                 if labels:
#                     with st.spinner("Matching custom labels..."):
#                         t = time.time()
#                         res = get_zero_shot()(image, candidate_labels=labels)
#                         R["custom"] = [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in res]
#                         R["timings_sec"]["custom_labels"] = round(time.time() - t, 2)

#             if do_colors:
#                 R["colors"] = dominant_colors(image)

#             R["timings_sec"]["total"] = round(time.time() - t_all, 2)
#             st.session_state["results"] = R
#         except Exception as e:
#             st.error(f"Something went wrong: {e}")
#             st.info("If this is a memory error, turn off OCR / custom labels, click 'Free memory', and retry.")

#     # ---------- Results (stored in session so downloads don't wipe them) ----------
#     R = st.session_state.get("results")
#     if R and R.get("file") == uploaded.name:
#         st.success(f"Analysis complete in {R['timings_sec'].get('total', 0)}s")
#         tabs = st.tabs(["📝 Summary", "🔍 Objects", "🔤 Text (OCR)", "🏷️ Custom Labels", "🎨 Colors", "📥 Report"])

#         with tabs[0]:
#             if "caption" in R:
#                 st.markdown(f"### {R['caption'].capitalize()}")
#             m1, m2, m3 = st.columns(3)
#             m1.metric("Objects found", len(R.get("objects", [])))
#             m2.metric("Text blocks", len(R.get("ocr", [])))
#             m3.metric("Time (s)", R["timings_sec"].get("total", 0))
#             if "classes" in R:
#                 st.markdown("**Top predictions**")
#                 for c in R["classes"]:
#                     st.progress(min(max(c["score"], 0.0), 1.0), text=f"{c['label']}: {c['score']:.1%}")

#         with tabs[1]:
#             if "objects_img" in R:
#                 st.image(R["objects_img"], use_container_width=True)
#                 if R["objects"]:
#                     counts = {}
#                     for o in R["objects"]:
#                         counts[o["label"]] = counts.get(o["label"], 0) + 1
#                     st.table([{"Object": k, "Count": v} for k, v in counts.items()])
#                     st.download_button("⬇️ Download annotated image", R["objects_img"],
#                                        "detections.png", "image/png")
#                 else:
#                     st.info("No objects above the threshold. Lower the slider in the sidebar.")
#             else:
#                 st.info("Enable 'Object detection' in the sidebar.")

#         with tabs[2]:
#             if "ocr" in R:
#                 if R["ocr"]:
#                     st.image(R["ocr_img"], use_container_width=True)
#                     full_text = "\n".join(o["text"] for o in R["ocr"])
#                     st.text_area("Extracted text", full_text, height=200)
#                     st.download_button("⬇️ Download text", full_text, "extracted_text.txt")
#                 else:
#                     st.info("No text detected in this image.")
#             else:
#                 st.info("Enable 'Read text in image (OCR)' in the sidebar.")

#         with tabs[3]:
#             if "custom" in R:
#                 for c in R["custom"]:
#                     st.progress(min(max(c["score"], 0.0), 1.0), text=f"{c['label']}: {c['score']:.1%}")
#             else:
#                 st.info("Enable 'Custom label search' and type your own labels in the sidebar.")

#         with tabs[4]:
#             if "colors" in R:
#                 cols = st.columns(len(R["colors"]))
#                 for col, c in zip(cols, R["colors"]):
#                     col.markdown(
#                         f"<div style='background:{c['hex']};height:70px;border-radius:8px'></div>"
#                         f"<p style='text-align:center'>{c['hex']}<br>{c['percent']}%</p>",
#                         unsafe_allow_html=True)
#             else:
#                 st.info("Enable 'Dominant colors' in the sidebar.")

#         with tabs[5]:
#             report = {k: v for k, v in R.items() if not k.endswith("_img")}
#             st.json(report)
#             st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2),
#                                "vision_report.json", "application/json")




import io
import json
import re
import time

import streamlit as st
from PIL import Image, ImageDraw
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gemini Vision Scanner",
    page_icon="📸",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #FF9D00;
        color: white;
        height: 48px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #E08900;
        color: white;
    }

    h1, .subtitle {
        text-align: center;
    }

    .model-box {
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(128,128,128,0.1);
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("📸 Gemini Vision Scanner")

st.markdown(
    """
    <p class='subtitle'>
    Scene understanding, object detection, text reading and visual Q&A,
    powered by Google Gemini.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API KEY
# ============================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.warning(
        "⚠️ Add GEMINI_API_KEY in Streamlit Cloud → Settings → Secrets. "
        "Get your API key from Google AI Studio."
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# FALLBACK MODELS
# ============================================================

# These are only fallbacks.
# Models discovered from Google's API are preferred.
FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


# ============================================================
# COLORS FOR OBJECT BOXES
# ============================================================

PALETTE = [
    "#FF9D00",
    "#00B4D8",
    "#E63946",
    "#2A9D8F",
    "#9B5DE5",
    "#F15BB5",
    "#80B918",
]


# ============================================================
# DISCOVER MODELS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def discover_models(key):
    """
    Ask Google which Gemini models this API key can access.

    Only models that appear to support generateContent are included.
    """

    try:
        found = []

        discovery_client = genai.Client(api_key=key)

        for model in discovery_client.models.list():

            name = getattr(model, "name", "")
            actions = getattr(model, "supported_actions", None) or []

            name = name.replace("models/", "")

            # Only Gemini models
            if not name.startswith("gemini"):
                continue

            # Exclude image/audio/live/embedding models
            excluded = (
                "image",
                "tts",
                "live",
                "audio",
                "embedding",
            )

            if any(x in name.lower() for x in excluded):
                continue

            # If supported_actions exists, require generateContent
            if actions and "generateContent" not in actions:
                continue

            found.append(name)

        # Prefer flash models first.
        flash_models = [
            name for name in found
            if "flash" in name.lower()
        ]

        other_models = [
            name for name in found
            if "flash" not in name.lower()
        ]

        return sorted(flash_models, reverse=True) + sorted(
            other_models,
            reverse=True,
        )

    except Exception:
        return []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    custom_model = st.text_input(
        "Model name (optional override)",
        "",
        placeholder="e.g. gemini-3.8-flash",
    )

    language = st.selectbox(
        "Response language",
        [
            "English",
            "Nepali",
            "Hindi",
            "French",
            "Spanish",
            "German",
        ],
    )

    detail = st.radio(
        "Detail level",
        [
            "Quick",
            "Detailed",
        ],
        index=1,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # LIST MODELS BUTTON
    # --------------------------------------------------------

    if st.button("🔎 List models available to my key"):

        try:

            available = discover_models(api_key)

            st.session_state["available"] = available

            if available:
                st.success(
                    f"Found {len(available)} model(s)."
                )
            else:
                st.warning(
                    "No compatible Gemini models were found."
                )

        except Exception as e:

            st.error(
                f"Could not list models: {e}"
            )

    # --------------------------------------------------------
    # SHOW AVAILABLE MODELS
    # --------------------------------------------------------

    if st.session_state.get("available"):

        st.caption(
            "Models your API key can access:"
        )

        st.code(
            "\n".join(
                st.session_state["available"]
            )
        )

    st.markdown("---")

    st.caption(
        "⚠️ On Google's free tier, prompts may be used to "
        "improve Google products. Don't upload sensitive images."
    )


# ============================================================
# MODEL CANDIDATES
# ============================================================

def model_candidates():
    """
    Build an ordered list of models.

    Priority:
    1. User override
    2. Models discovered from the API
    3. Hard-coded fallbacks
    """

    override = []

    if custom_model.strip():
        override = [
            custom_model.strip()
        ]

    discovered = discover_models(api_key)

    candidates = (
        override
        + discovered
        + FALLBACK_MODELS
    )

    # Remove duplicates while preserving order
    seen = set()
    ordered = []

    for name in candidates:

        if not name:
            continue

        if name in seen:
            continue

        seen.add(name)
        ordered.append(name)

    return ordered


# ============================================================
# IMAGE CONVERSION
# ============================================================

def image_part(img):
    """
    Convert PIL image into Gemini-compatible image Part.
    """

    buf = io.BytesIO()

    img.save(
        buf,
        format="JPEG",
        quality=90,
    )

    return types.Part.from_bytes(
        data=buf.getvalue(),
        mime_type="image/jpeg",
    )


# ============================================================
# GEMINI API CALL
# ============================================================

def call_gemini(contents, as_json=False):
    """
    Try available Gemini models.

    503 / UNAVAILABLE:
        Temporary overload.
        Retry up to 3 times.

    429 / RESOURCE_EXHAUSTED:
        Quota/rate limit.
        DO NOT waste time retrying the same model.

    Other errors:
        Move to the next model.
    """

    cfg = types.GenerateContentConfig(
        response_mime_type=(
            "application/json"
            if as_json
            else "text/plain"
        ),
        temperature=0.2,
    )

    errors = []

    status_box = st.empty()

    candidates = model_candidates()

    if not candidates:

        raise RuntimeError(
            "No Gemini models were found for this API key."
        )

    for name in candidates:

        for attempt in range(3):

            try:

                resp = client.models.generate_content(
                    model=name,
                    contents=contents,
                    config=cfg,
                )

                status_box.empty()

                if resp.text:

                    return resp.text, name

                errors.append(
                    f"{name}: empty response"
                )

                break

            except Exception as e:

                msg = str(e)

                upper_msg = msg.upper()

                # =================================================
                # 429 / QUOTA
                # =================================================

                if (
                    "429" in msg
                    or "RESOURCE_EXHAUSTED" in upper_msg
                    or "QUOTA" in upper_msg
                ):

                    errors.append(
                        f"{name}: quota/rate limit exhausted"
                    )

                    # DO NOT retry.
                    break

                # =================================================
                # 503 / TEMPORARY UNAVAILABLE
                # =================================================

                if (
                    "503" in msg
                    or "UNAVAILABLE" in upper_msg
                    or "500" in msg
                ):

                    if attempt < 2:

                        wait = 3 * (attempt + 1)

                        status_box.warning(
                            f"{name} is temporarily busy. "
                            f"Retrying in {wait}s "
                            f"(attempt {attempt + 2}/3)..."
                        )

                        time.sleep(wait)

                        continue

                # =================================================
                # OTHER ERROR
                # =================================================

                errors.append(
                    f"{name}: {msg[:250]}"
                )

                break

    status_box.empty()

    raise RuntimeError(
        "No Gemini model could process this request.\n\n"
        + "\n".join(errors)
    )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text):
    """
    Clean markdown code fences and parse JSON.
    """

    text = text.strip()

    # Remove ```json ... ```
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return json.loads(text)


# ============================================================
# DRAW OBJECT BOXES
# ============================================================

def draw_boxes(img, objects):

    out = img.copy()

    draw = ImageDraw.Draw(out)

    width, height = out.size

    labels = sorted(
        {
            str(
                obj.get(
                    "label",
                    "",
                )
            )
            for obj in objects
        }
    )

    for obj in objects:

        box = obj.get("box_2d")

        if not box or len(box) != 4:
            continue

        try:

            ymin, xmin, ymax, xmax = box

            x0 = xmin / 1000 * width
            y0 = ymin / 1000 * height
            x1 = xmax / 1000 * width
            y1 = ymax / 1000 * height

        except Exception:
            continue

        label = str(
            obj.get(
                "label",
                "",
            )
        )

        if label in labels:

            color = PALETTE[
                labels.index(label)
                % len(PALETTE)
            ]

        else:

            color = PALETTE[0]

        draw.rectangle(
            [
                x0,
                y0,
                x1,
                y1,
            ],
            outline=color,
            width=4,
        )

        draw.text(
            (
                x0 + 5,
                y0 + 5,
            ),
            label,
            fill=color,
        )

    return out


# ============================================================
# ANALYSIS PROMPT
# ============================================================

ANALYSIS_PROMPT = """
You are an expert computer-vision analyst.

Analyze the image and return ONLY valid JSON.

The JSON must contain exactly these keys:

{
  "caption": "one-sentence description of the scene",

  "summary": "{depth} description of the context, setting, mood and activity",

  "objects": [
    {
      "label": "object name",
      "box_2d": [ymin, xmin, ymax, xmax],
      "confidence": "high|medium|low"
    }
  ],

  "text_found": [
    "every piece of readable text, verbatim, in its original language"
  ],

  "colors": [
    "3-5 dominant colors as plain names"
  ],

  "tags": [
    "8-12 short keywords"
  ]
}

Rules:

1. box_2d values must be integers from 0 to 1000.

2. box_2d format is:
   [ymin, xmin, ymax, xmax]

3. Coordinates are normalized to 0-1000.

4. List up to 20 of the most important objects.

5. Write caption, summary, tags and colors in {lang}.

6. Keep text_found in the original language.

7. Extract every piece of readable text you can identify.

8. If there is no readable text, use an empty list.

9. Return ONLY JSON.

10. Do not include markdown fences.

11. Do not add explanations outside the JSON.
"""


# ============================================================
# MAIN APP
# ============================================================

uploaded = st.file_uploader(
    "Upload an image (JPG, PNG, JPEG)",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

        image.thumbnail(
            (
                1536,
                1536,
            )
        )

    except Exception as e:

        st.error(
            f"Could not open image: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # IMAGE / CONTROLS
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.image(
            image,
            caption=uploaded.name,
            use_container_width=True,
        )

    with right:

        run = st.button(
            "🚀 Run Full Analysis"
        )

        st.markdown("---")

        question = st.text_input(
            "💬 Or ask a question about this image",
            placeholder=(
                "How many people are wearing hats?"
            ),
        )

        ask = st.button(
            "🔎 Ask Gemini"
        )

    # ========================================================
    # FULL ANALYSIS
    # ========================================================

    if run:

        with st.spinner(
            "Gemini is analyzing your image..."
        ):

            try:

                prompt = ANALYSIS_PROMPT.format(
                    depth=(
                        "2-3 sentence"
                        if detail == "Quick"
                        else "detailed 4-6 sentence"
                    ),
                    lang=language,
                )

                text, used_model = call_gemini(
                    [
                        image_part(image),
                        prompt,
                    ],
                    as_json=True,
                )

                data = parse_json(text)

                # Store model used
                data["_model"] = used_model

                # Store result
                st.session_state[
                    "gem_result"
                ] = (
                    uploaded.name,
                    data,
                )

            except json.JSONDecodeError:

                st.error(
                    "Gemini returned invalid JSON. "
                    "Please try again."
                )

            except Exception as e:

                error_text = str(e)

                st.error(
                    f"Analysis failed: {error_text}"
                )

                # ------------------------------------------------
                # Friendly explanation
                # ------------------------------------------------

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "quota" in error_text.lower()
                ):

                    st.warning(
                        "⚠️ Your Gemini API quota/rate limit "
                        "appears to be exhausted. "
                        "Changing the image or retrying immediately "
                        "usually will not fix a quota error."
                    )

                    st.info(
                        "Check your Google AI Studio/API project "
                        "quota and billing settings, or use an "
                        "API key from a project with available quota."
                    )

                elif (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    st.warning(
                        "⚠️ Gemini is temporarily overloaded. "
                        "The app already retried the request. "
                        "Please try again later."
                    )

                else:

                    st.info(
                        "Check your API key, model selection, "
                        "Google API availability, and request limits."
                    )

                # Show models that were attempted
                st.caption(
                    "Models considered: "
                    + ", ".join(
                        model_candidates()
                    )
                )

    # ========================================================
    # VISUAL QUESTION ANSWERING
    # ========================================================

    if ask and question.strip():

        with st.spinner(
            "Gemini is thinking..."
        ):

            try:

                answer, used_model = call_gemini(
                    [
                        image_part(image),
                        (
                            f"Answer in {language}. "
                            f"Question about this image: "
                            f"{question}"
                        ),
                    ]
                )

                st.session_state[
                    "gem_answer"
                ] = (
                    uploaded.name,
                    question,
                    answer,
                    used_model,
                )

            except Exception as e:

                error_text = str(e)

                st.error(
                    f"Question failed: {error_text}"
                )

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    st.warning(
                        "Gemini API quota/rate limit is exhausted. "
                        "Check your Google API quota or billing."
                    )

                elif (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    st.warning(
                        "Gemini is temporarily unavailable. "
                        "Please try again later."
                    )

    # ========================================================
    # ANALYSIS RESULTS
    # ========================================================

    saved = st.session_state.get(
        "gem_result"
    )

    if (
        saved
        and saved[0] == uploaded.name
    ):

        data = saved[1]

        st.success(
            "Analysis complete "
            f"(model: {data.get('_model', 'unknown')})"
        )

        tabs = st.tabs(
            [
                "📝 Summary",
                "🔍 Objects",
                "🔤 Text",
                "🏷️ Tags & Colors",
                "📥 Report",
            ]
        )

        # ====================================================
        # SUMMARY
        # ====================================================

        with tabs[0]:

            st.markdown(
                f"### {data.get('caption', '')}"
            )

            st.write(
                data.get(
                    "summary",
                    "",
                )
            )

        # ====================================================
        # OBJECTS
        # ====================================================

        with tabs[1]:

            objects = data.get(
                "objects",
                [],
            )

            if objects:

                st.image(
                    draw_boxes(
                        image,
                        objects,
                    ),
                    use_container_width=True,
                )

                table_data = []

                for obj in objects:

                    table_data.append(
                        {
                            "Object": obj.get(
                                "label",
                                "",
                            ),
                            "Confidence": obj.get(
                                "confidence",
                                "",
                            ),
                        }
                    )

                st.table(
                    table_data
                )

            else:

                st.info(
                    "No objects detected."
                )

        # ====================================================
        # TEXT
        # ====================================================

        with tabs[2]:

            texts = data.get(
                "text_found",
                [],
            )

            if texts:

                joined = "\n".join(
                    str(text)
                    for text in texts
                )

                st.text_area(
                    "Text found in image",
                    joined,
                    height=200,
                )

                st.download_button(
                    "⬇️ Download text",
                    joined,
                    "extracted_text.txt",
                    "text/plain",
                )

            else:

                st.info(
                    "No readable text found."
                )

        # ====================================================
        # TAGS / COLORS
        # ====================================================

        with tabs[3]:

            tags = data.get(
                "tags",
                [],
            )

            colors = data.get(
                "colors",
                [],
            )

            st.markdown(
                "**Tags:** "
                + " · ".join(
                    f"`{tag}`"
                    for tag in tags
                )
            )

            st.markdown(
                "**Colors:** "
                + ", ".join(
                    str(color)
                    for color in colors
                )
            )

        # ====================================================
        # JSON REPORT
        # ====================================================

        with tabs[4]:

            st.json(
                data
            )

            json_report = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                "⬇️ Download JSON report",
                json_report,
                "gemini_report.json",
                "application/json",
            )

    # ========================================================
    # Q&A RESULT
    # ========================================================

    qa = st.session_state.get(
        "gem_answer"
    )

    if (
        qa
        and qa[0] == uploaded.name
    ):

        st.markdown(
            "### 💬 Q&A"
        )

        st.markdown(
            f"**Q:** {qa[1]}"
        )

        st.markdown(
            f"**A:** {qa[2]}"
        )

        st.caption(
            f"model: {qa[3]}"
        )
