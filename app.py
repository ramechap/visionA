import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from openai import OpenAI

# 1. Setup the Web Page look
st.set_page_config(page_title="Hackathon Vision Portal", layout="centered")
st.title("📸 AI Image Recognition Hub")
st.write("Upload an image to recognize text, objects, and details instantly.")

# 2. Securely get your OpenRouter API Key
# (We will set this up in Hugging Face settings later so it stays hidden)
api_key = st.secrets.get("OPENROUTER_API_KEY")

if not api_key:
    st.warning("⚠️ API Key missing. Please configure OPENROUTER_API_KEY in your server settings.")
else:
    # Connect to the cloud AI server
    client = OpenAI(
        base_url="https://openrouter.ai",
        api_key=api_key
    )

    # 3. Create the Image Upload Button
    uploaded_file = st.file_uploader("Upload a photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image on the screen
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Photo", use_container_width=True)
        
        # Convert the picture to a text string (Base64) so it can travel over the internet safely
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        

        # 4. Create the "Process" button
        # 4. Create the "Process" button
        if st.button("Analyze Image 🚀"):
            with st.spinner("The AI is analyzing your image in the cloud..."):
                try:
                    # Send the image data to DeepSeek
                    response = client.chat.completions.create(
                        model="deepseek/deepseek-v4.1-flash", 
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Perform advanced image recognition. List all key objects, read any text present, and summarize what is happening in the scene."},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                                ]
                            }
                        ]
                    )
                    
                    # --- SAFE CHECKING ---
                    # If it returns a string error or a dictionary instead of an object
                    if isinstance(response, str):
                        st.error("⚠️ OpenRouter returned an error string instead of data:")
                        st.code(response)
                    elif hasattr(response, 'choices') and response.choices:
                        # Show results on the web screen if successful
                        st.success("Analysis Complete!")
                        st.markdown("### 📊 AI Recognition Results")
                        st.write(response.choices[0].message.content)
                    elif isinstance(response, dict) and 'choices' in response:
                        st.success("Analysis Complete!")
                        st.markdown("### 📊 AI Recognition Results")
                        st.write(response['choices'][0]['message']['content'])
                    else:
                        st.error("Unexpected response layout received:")
                        st.write(response)
                    
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
