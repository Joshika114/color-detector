# 🎨 Assistive Color Detector with Perceptra Bot

A Streamlit-based utility tool for designers, artists, accessibility testers, and developers who need to detect and interpret colors from images.  
Upload an image, click on any pixel, and instantly view its RGB, HEX, mood interpretation, nearest named color, and color-blindness simulation.  
Now enhanced with **Perceptra Bot**, an integrated assistant that answers color-related queries inside the same app.

---

## ✨ Features

- Upload and display an image
- Click anywhere on the image to detect the color of that exact pixel
- Shows:
  - Raw RGB values
  - HEX value
  - Closest named color from dataset (`colors.csv`)
  - Mood & color-psychology meaning
  - Audio announcement of detected color (optional)
- Color-blindness simulation for:
  - Protanopia
  - Deuteranopia
  - Tritanopia
- Top-N closest color matches (ΔE with CIEDE2000)
- Integrated **Perceptra Bot** to assist with color questions inside the UI

---

## Requirements.txt
streamlit
pandas
numpy
opencv-python
colormath
streamlit-image-coordinates
pyttsx3

---

## 📂 Project Structure

color-detector/
│── app.py # Main Streamlit app
│── colors.csv # Color dataset with names & HEX
│── perspectra_bot.py # Embedded interactive bot module
│── requirements.txt # Dependencies
│── README.md # Project documentation

---

## 🚀 Running the Application

Make sure required packages are installed (as per requirements.txt) and then run:
## streamlit run app.py

This will automatically open the web UI at:
http://localhost:8501

---

## 🤝 Contribution

Pull requests, issues, and enhancements are welcome.
For major changes, open an issue to discuss the proposed update.
