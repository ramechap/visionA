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

# 1. Professional Page Configuration for Hackathon Judging
st.set_page_config(
    page_title="AI Vision Scanner", 
    page_icon="📸",
    layout="centered"
)

# Custom styling to make the app look clean and modern
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #4A90E2; color: white; }
    .stButton>button:hover { background-color: #357ABD; }
    h1 { color: #2C3E50; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 AI Image Recognition Portal")
st.write("Analyze and recognize visual details instantly using **Salesforce/blip-image-captioning-large**.")

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
        
        # 4. Process Action Button
        if st.button("Run Image Recognition 🚀"):
            with st.spinner("Hugging Face is analyzing your image data..."):
                try:
                    # UPDATED FIX: New unified domain for Hugging Face APIs
                    API_URL = "https://api.huggingface.co/models/Salesforce/blip-image-captioning-large"
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    
                    # Read the raw byte data from the upload container file
                    image_bytes = uploaded_file.getvalue()
                    
                    # Direct POST request sending raw data payload
                    response = requests.post(API_URL, headers=headers, data=image_bytes)
                    result = response.json()
                    
                    # 5. Output Visual Container
                    # FIXED FIX: Safely parse array dictionary from the BLIP model output
                    if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                        st.success("Recognition Complete!")
                        st.markdown("### 📊 Visual Interpretation Output")
                        st.info(f"**AI Description:** {result[0]['generated_text']}")
                    elif isinstance(result, dict) and 'error' in result:
                        # Sometimes a model needs 20 seconds to load if nobody has used it recently
                        if "loading" in result['error']:
                            st.warning("⏳ The model is currently booting up on Hugging Face servers. Please wait 15 seconds and try clicking the button again!")
                        else:
                            st.error(f"Hugging Face API Error: {result['error']}")
                    else:
                        st.error("Received an unexpected data format back from the server.")
                        st.write(result)
                    
                except Exception as e:
                    st.error(f"Failed to communicate with Hugging Face Serverless API: {e}")


                    
                except Exception as e:
                    st.error(f"Failed to communicate with Hugging Face Serverless API: {e}")
