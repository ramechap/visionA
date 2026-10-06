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


import io
import json
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from transformers import pipeline

st.set_page_config(page_title="AI Vision Scanner Pro", page_icon="📸", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold;
        background-color: #FF9D00; color: white; height: 48px; border: none; }
    .stButton>button:hover { background-color: #E08900; color: white; }
    h1, .subtitle { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 AI Vision Scanner Pro")
st.markdown("<p class='subtitle'>Free, open-source image intelligence: captioning, object detection, OCR, "
            "classification and custom-label search, all running locally with Hugging Face models.</p>",
            unsafe_allow_html=True)

# ---------- Lazy-loaded models (each loads only when first used, then stays cached) ----------
@st.cache_resource(show_spinner=False)
def get_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource(show_spinner=False)
def get_detector():
    return pipeline("object-detection", model="hustvl/yolos-tiny")

@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

@st.cache_resource(show_spinner=False)
def get_zero_shot():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

@st.cache_resource(show_spinner=False)
def get_ocr(lang_codes):
    import easyocr
    return easyocr.Reader(list(lang_codes), gpu=False)

# ---------- Helpers ----------
PALETTE = ["#FF9D00", "#00B4D8", "#E63946", "#2A9D8F", "#9B5DE5", "#F15BB5", "#80B918"]

def dominant_colors(img, n=5):
    q = img.resize((100, 100)).quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[: n * 3]
    out = []
    for count, idx in sorted(q.getcolors(), reverse=True):
        r, g, b = pal[idx * 3: idx * 3 + 3]
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "percent": round(count / 100, 1)})
    return out

def draw_detections(img, dets):
    out = img.copy()
    d = ImageDraw.Draw(out)
    labels = sorted({x["label"] for x in dets})
    for x in dets:
        c = PALETTE[labels.index(x["label"]) % len(PALETTE)]
        b = x["box"]
        d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=c, width=4)
        d.text((b["xmin"] + 5, b["ymin"] + 5), f'{x["label"]} {x["score"]:.0%}', fill=c)
    return out

def draw_ocr(img, items):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for it in items:
        pts = [tuple(p) for p in it["box"]]
        d.polygon(pts, outline="#00B4D8")
    return out

def to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Analysis Options")
    do_caption = st.checkbox("📝 Scene description (captioning)", True)
    do_detect = st.checkbox("🔍 Object detection", True)
    do_classify = st.checkbox("🎯 Classification (top 5)", True)
    do_ocr = st.checkbox("🔤 Read text in image (OCR)", False)
    do_custom = st.checkbox("🏷️ Custom label search (CLIP)", False)
    do_colors = st.checkbox("🎨 Dominant colors", True)

    threshold = st.slider("Detection confidence", 0.1, 0.95, 0.5, 0.05)
    ocr_lang = st.multiselect("OCR languages", ["en", "ne", "hi", "fr", "de", "es"], default=["en"])
    custom_labels = st.text_input("Custom labels (comma-separated)",
                                  "dog, cat, car, person, food, building")

    st.caption("Tip: models load the first time you use them. If the app runs out of memory, "
               "turn off some options or free memory below.")
    if st.button("🧹 Free memory"):
        st.cache_resource.clear()
        st.success("Models unloaded.")

# ---------- Main ----------
uploaded = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image.thumbnail((1024, 1024))  # keeps things fast and light on memory

    left, right = st.columns([1, 1])
    left.image(image, caption=f"{uploaded.name} ({image.width}×{image.height})", use_container_width=True)

    with right:
        st.write("Choose options in the sidebar, then run the scan.")
        run = st.button("Run Full Analysis 🚀")

    if run:
        R = {"file": uploaded.name, "timings_sec": {}}
        t_all = time.time()
        try:
            if do_caption:
                with st.spinner("Describing the scene..."):
                    t = time.time()
                    R["caption"] = get_captioner()(image, max_new_tokens=40)[0]["generated_text"]
                    R["timings_sec"]["caption"] = round(time.time() - t, 2)

            if do_classify:
                with st.spinner("Classifying..."):
                    t = time.time()
                    R["classes"] = [{"label": c["label"], "score": round(float(c["score"]), 4)}
                                    for c in get_classifier()(image, top_k=5)]
                    R["timings_sec"]["classification"] = round(time.time() - t, 2)

            if do_detect:
                with st.spinner("Detecting objects..."):
                    t = time.time()
                    dets = [d for d in get_detector()(image) if d["score"] >= threshold]
                    R["objects"] = [{"label": d["label"], "score": round(float(d["score"]), 4),
                                     "box": d["box"]} for d in dets]
                    R["objects_img"] = to_png_bytes(draw_detections(image, dets))
                    R["timings_sec"]["detection"] = round(time.time() - t, 2)

            if do_ocr:
                with st.spinner("Reading text (first run downloads OCR models)..."):
                    t = time.time()
                    reader = get_ocr(tuple(ocr_lang or ["en"]))
                    raw = reader.readtext(np.array(image))
                    R["ocr"] = [{"text": txt, "confidence": round(float(conf), 3),
                                 "box": [[int(x), int(y)] for x, y in box]} for box, txt, conf in raw]
                    R["ocr_img"] = to_png_bytes(draw_ocr(image, R["ocr"]))
                    R["timings_sec"]["ocr"] = round(time.time() - t, 2)

            if do_custom:
                labels = [l.strip() for l in custom_labels.split(",") if l.strip()]
                if labels:
                    with st.spinner("Matching custom labels..."):
                        t = time.time()
                        res = get_zero_shot()(image, candidate_labels=labels)
                        R["custom"] = [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in res]
                        R["timings_sec"]["custom_labels"] = round(time.time() - t, 2)

            if do_colors:
                R["colors"] = dominant_colors(image)

            R["timings_sec"]["total"] = round(time.time() - t_all, 2)
            st.session_state["results"] = R
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("If this is a memory error, turn off OCR / custom labels, click 'Free memory', and retry.")

    # ---------- Results (stored in session so downloads don't wipe them) ----------
    R = st.session_state.get("results")
    if R and R.get("file") == uploaded.name:
        st.success(f"Analysis complete in {R['timings_sec'].get('total', 0)}s")
        tabs = st.tabs(["📝 Summary", "🔍 Objects", "🔤 Text (OCR)", "🏷️ Custom Labels", "🎨 Colors", "📥 Report"])

        with tabs[0]:
            if "caption" in R:
                st.markdown(f"### {R['caption'].capitalize()}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects found", len(R.get("objects", [])))
            m2.metric("Text blocks", len(R.get("ocr", [])))
            m3.metric("Time (s)", R["timings_sec"].get("total", 0))
            if "classes" in R:
                st.markdown("**Top predictions**")
                for c in R["classes"]:
                    st.progress(min(max(c["score"], 0.0), 1.0), text=f"{c['label']}: {c['score']:.1%}")

        with tabs[1]:
            if "objects_img" in R:
                st.image(R["objects_img"], use_container_width=True)
                if R["objects"]:
                    counts = {}
                    for o in R["objects"]:
                        counts[o["label"]] = counts.get(o["label"], 0) + 1
                    st.table([{"Object": k, "Count": v} for k, v in counts.items()])
                    st.download_button("⬇️ Download annotated image", R["objects_img"],
                                       "detections.png", "image/png")
                else:
                    st.info("No objects above the threshold. Lower the slider in the sidebar.")
            else:
                st.info("Enable 'Object detection' in the sidebar.")

        with tabs[2]:
            if "ocr" in R:
                if R["ocr"]:
                    st.image(R["ocr_img"], use_container_width=True)
                    full_text = "\n".join(o["text"] for o in R["ocr"])
                    st.text_area("Extracted text", full_text, height=200)
                    st.download_button("⬇️ Download text", full_text, "extracted_text.txt")
                else:
                    st.info("No text detected in this image.")
            else:
                st.info("Enable 'Read text in image (OCR)' in the sidebar.")

        with tabs[3]:
            if "custom" in R:
                for c in R["custom"]:
                    st.progress(min(max(c["score"], 0.0), 1.0), text=f"{c['label']}: {c['score']:.1%}")
            else:
                st.info("Enable 'Custom label search' and type your own labels in the sidebar.")

        with tabs[4]:
            if "colors" in R:
                cols = st.columns(len(R["colors"]))
                for col, c in zip(cols, R["colors"]):
                    col.markdown(
                        f"<div style='background:{c['hex']};height:70px;border-radius:8px'></div>"
                        f"<p style='text-align:center'>{c['hex']}<br>{c['percent']}%</p>",
                        unsafe_allow_html=True)
            else:
                st.info("Enable 'Dominant colors' in the sidebar.")

        with tabs[5]:
            report = {k: v for k, v in R.items() if not k.endswith("_img")}
            st.json(report)
            st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2),
                               "vision_report.json", "application/json")
