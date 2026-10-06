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
from openai import OpenAI
import base64
from io import BytesIO

# 1. Professional Page Configuration for Hackathon Submission
st.set_page_config(
    page_title="AI Vision Recognition Scanner", 
    page_icon="📸",
    layout="centered"
)

# Clean, professional CSS branding for your judging panel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #FF9D00; color: white; height: 48px; }
    .stButton>button:hover { background-color: #E08900; }
    h1 { color: #2C3E50; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 Open-Source Image Recognition Portal")
st.write("Analyze visual data seamlessly via Hugging Face's official unified router.")

# 2. Extract Hugging Face Token safely from Streamlit Cloud Secrets
hf_token = st.secrets.get("HF_TOKEN")

if not hf_token:
    st.warning("⚠️ Configuration Error: Please ensure your hf_... access token is saved under HF_TOKEN inside Streamlit Cloud Settings.")
else:
    # Initialize connection using the new official Hugging Face OpenAI-compatible gateway
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token
    )
        
    # 3. Simple Image File Uploader Component
    uploaded_file = st.file_uploader("Drop or upload an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display image layout preview
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_container_width=True)
        
        # 4. Trigger Analysis Execution
        if st.button("Run Image Analysis 🚀"):
            with st.spinner("Processing image details on Hugging Face infrastructure..."):
                try:
                    # Convert image file to a safe base64 text data block
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    # Connect to a flagship vision model hosted on Hugging Face's ecosystem
                    response = client.chat.completions.create(
                        model="meta-llama/Llama-3.2-11B-Vision-Instruct:novita",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Perform exhaustive image recognition. Itemize all key objects detected, interpret any written text, and summarize the context of the scene."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                                ]
                            }
                        ],
                        max_tokens=500
                    )
                    
                    # 5. Output Visual Container Presentation
                    st.success("Recognition Complete!")
                    st.markdown("### 📊 Visual Interpretation Output")
                    st.write(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Failed to communicate with Hugging Face Router API: {e}")

