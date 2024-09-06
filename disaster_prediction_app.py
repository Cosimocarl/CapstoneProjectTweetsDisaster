import streamlit as st
from sklearn.pipeline import Pipeline
import joblib
from PIL import Image
import datetime
import time

# Load the best model (replace 'best_pipeline.pkl' with your actual model filename)
model = joblib.load('best_pipeline.pkl')

# Function to make predictions
def predict_disaster(text):
    prediction = model.predict([text])
    return prediction[0] == 1

# Set up the Streamlit app
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Welcome!", "About us", "Disaster Prediction", "Thank You!"])

    if selection == "Welcome!":
        # Home page with title, cover image, date, and time
        st.markdown(
            """
            <style>
            .title-font {{
                font-size: 100px;  /* Increased font size */
                color: #FF4B4B;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 40px;  /* Added space between title and image */
            }}
            .time-font {{
                font-size: 20px;
                color: #333;
                text-align: center;
                font-family: 'Arial', sans-serif;
                margin-top: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='title-font'><h2>Disaster Prediction Project</div>", unsafe_allow_html=True)
        
        # Cover image with space above
        cover_image = Image.open("images/Logo.jpg")  # Replace with your cover image filename
        st.image(cover_image, use_column_width=True)

        # Display date and time
        st.markdown(f"<div class='time-font'>Date: {datetime.datetime.now().strftime('%B %d, %Y')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='time-font'>Current Time: {time.strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

    elif selection == "About us":
        # About Us page with your and your colleague's photos and descriptions
        st.markdown(
            """
            <style>
            .title-font {{
                font-size: 200px;  /* Increased the font size */
                color: #FF4B4B;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                font-weight: bold;
                margin-top: 20px;
            }}
            .name-font {{
                font-size: 30px;
                color: #1F77B4;
                text-align: center;
                margin-top: 20px;
            }}
            .description-font {{
                font-size: 18px;
                color: #333;
                text-align: center;
                margin-top: 10px;
                margin-bottom: 40px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Add the title "Team members"
        st.markdown("<div class='title-font'><h2>Rocking predictions members</div>", unsafe_allow_html=True)

        # Your details
        st.image("images/Carlofoto.jpg", caption="Cosimo Carlo Canova", width=300)  # Replace with your image filename
        st.markdown("<div class='description-font'>Economist</div>", unsafe_allow_html=True)

        '---------------------------------------------------------'
        # Your colleague's details
        st.image("images/Joaofoto.jpg", caption="Joao de Faria Junior", width=300)  # Replace with your image filename
        st.markdown("<div class='description-font'>Web developer/Finance</div>", unsafe_allow_html=True)

    elif selection == "Disaster Prediction":
        # Disaster Prediction page with the app functionality
        st.markdown(
            """
            <style>
            .title-font {{
                font-size: 50px;
                color: #1F77B4;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 30px;
            }}
            .command-font {{
                font-size: 25px;
                color: #1F77B4;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .alert-font {{
                font-size: 35px;
                color: white;
                text-align: center;
                font-weight: bold;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # Title for Disaster Prediction Page
        st.markdown("<div class='title-font'><h2>Disaster Prediction</div>", unsafe_allow_html=True)

        # Text input prompt
        st.markdown("<div class='command-font'>Please enter a sentence to predict if it's a disaster:</div>", unsafe_allow_html=True)

        # Text input for prediction
        user_input = st.text_area("Enter your sentence here:")

        if st.button("Predict"):
            if user_input:
                is_disaster = predict_disaster(user_input)
                if is_disaster:
                    st.markdown(
                        """
                        <style>
                        .stApp {{
                            background-color: #FF4B4B;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("<div class='alert-font'>😱 It's a disaster! Run away! 😱</div>", unsafe_allow_html=True)
                    # Show a disaster image (replace 'disaster_image.jpg' with your actual image file)
                    disaster_image = Image.open("images/Becarefull.jpg")
                    st.image(disaster_image, caption="Disaster detected!", use_column_width=True)
                else:
                    st.markdown(
                        """
                        <style>
                        .stApp {{
                            background-color: #28a745;
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("<div class='alert-font'>😊 No disaster detected. 😊</div>", unsafe_allow_html=True)
                    # Show a safe image (replace 'safe_image.jpg' with your actual image file)
                    safe_image = Image.open("images/Safe!.jpg")
                    st.image(safe_image, caption="All is well!", use_column_width=True)
            else:
                st.warning("Please enter a sentence to make a prediction.")

    elif selection == "Thank You!":
        # Thank You page
        st.markdown(
            """
            <style>
            .title-font {{
                font-size: 50px;
                color: #FF4B4B;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 30px;
            }}
            .thank-you-font {{
                font-size: 25px;
                color: #333;
                text-align: center;
                font-family: 'Arial', sans-serif;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<div style='text-align:center;'markdown='1' class='thank-you-font'><h2>Thank you for your attention and for using our application!</div>", unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class='thank-you-font'>
            -----------------------------------------------------------------------------------
            📧 Email: joaodefariajunior1@gmail.com<br>
            🔗 LinkedIn: [Joao de Faria Junior](https://www.linkedin.com/in/joao-de-faria-jr-04b3a5174/?originalSubdomain=de)<br>
            -----------------------------------------------------------------------------------
            📧 Email: carloo_96@hotmail.it<br>
            🔗 LinkedIn: [Cosimo Carlo Canova](https://www.linkedin.com/in/cosimo-carlo-canova-812608232/)<br>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()