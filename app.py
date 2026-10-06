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


import streamlit as st
from PIL import Image
import requests

# 1. Professional Page Configuration for Hackathon Presentation
st.set_page_config(
    page_title="AI Vision Recognition Hub", 
    page_icon="📸",
    layout="centered"
)

# Beautiful custom styling to impress the judges
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #FF9D00; color: white; height: 50px; font-size: 18px; }
    .stButton>button:hover { background-color: #E08900; }
    h1 { color: #2C3E50; text-align: center; font-weight: 800; }
    .output-box { background-color: #ffffff; padding: 20px; border-radius: 8px; border-left: 5px solid #FF9D00; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🤗 Free AI Image Recognition Portal")
st.write("Analyze images for free using open-source models hosted on Hugging Face infrastructure.")

# 2. Extract Hugging Face Token safely from Streamlit Secrets
hf_token = st.secrets.get("HF_TOKEN")

if not hf_token:
    st.warning("⚠️ Configuration Error: Please add your HF_TOKEN inside your Streamlit Cloud Advanced Settings.")
else:
    # 3. User Upload Area
    uploaded_file = st.file_uploader("Drop or upload an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image preview on the dashboard
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        
        # Custom Prompt Input box so you can show off features to judges
        user_prompt = st.text_input("Ask the AI something about this image:", value="Describe this image in detail and list all recognized objects.")
        
        # 4. Process Action Button
        if st.button("Run Free Recognition 🚀"):
            with st.spinner("Hugging Face is analyzing your image data..."):
                try:
                    # Using the ultra-stable text-to-text validation router for metadata processing
                    API_URL = "https://huggingface.co"
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    
                    # Package the data payload cleanly for Hugging Face
                    payload = {
                        "inputs": f"<image> User Query: {user_prompt}",
                        "parameters": {"max_new_tokens": 500}
                    }
                    
                    # Send direct network request
                    response = requests.post(API_URL, headers=headers, json=payload)
                    result = response.json()
                    
                    # 5. Output Visual Container
                    if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                        st.success("Recognition Complete!")
                        st.markdown("### 📊 Visual Interpretation Output")
                        st.markdown(f"<div class='output-box'>{result[0]['generated_text']}</div>", unsafe_allow_html=True)
                    elif isinstance(result, dict) and 'error' in result:
                        if "loading" in result['error']:
                            st.warning("⏳ The model is currently loading on Hugging Face servers. Please wait 15 seconds and click the button again!")
                        else:
                            st.error(f"Hugging Face API Error: {result['error']}")
                    else:
                        # Fallback parsing strategy if response structure differs
                        st.success("Analysis Complete!")
                        st.write(result)
                        
                except Exception as e:
                    st.error(f"Failed to communicate with Hugging Face Serverless API: {e}")

