import streamlit as st
import colorsys
import re
import math
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

st.markdown(
    """
    <style>
    /* Floating square chat bot icon */
    .chatbot-icon {
        position: fixed;
        bottom: 25px;
        right: 25px;
        width: 80px;
        height: 80px;
        border-radius: 20px;
        background-color: #25D366; /* WhatsApp green */
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s ease;
    }
    .chatbot-icon:hover {
        transform: scale(1.05);
    }
    .chatbot-icon img {
        width: 45px;
        height: 45px;
    }

    /* Chat window */
    .chat-window {
        position: fixed;
        bottom: 120px;
        right: 25px;
        width: 320px;
        max-height: 400px;
        background-color: #fff;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        overflow-y: auto;
        z-index: 1001;
        display: none;
        flex-direction: column;
        padding: 15px;
    }

    /* WhatsApp-style message bubbles */
    .user-msg {
        background-color: #dcf8c6;
        align-self: flex-end;
        color: #000;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 15px 15px 0 15px;
        max-width: 80%;
        word-wrap: break-word;
    }
    .bot-msg {
        background-color: #f1f0f0;
        align-self: flex-start;
        color: #000;
        padding: 10px 15px;
        margin: 5px;
        border-radius: 15px 15px 15px 0;
        max-width: 80%;
        word-wrap: break-word;
    }

    .input-row {
        display: flex;
        margin-top: 10px;
    }
    .input-row input {
        flex-grow: 1;
        padding: 8px;
        border-radius: 15px;
        border: 1px solid #ccc;
    }
    .input-row button {
        background-color: #25D366;
        color: white;
        border: none;
        padding: 8px 15px;
        margin-left: 8px;
        border-radius: 15px;
        cursor: pointer;
    }
    </style>

    <div class="chatbot-icon" onclick="toggleChat()">
        <img src="https://cdn-icons-png.flaticon.com/512/220/220236.png" alt="Bot">
    </div>

    <div id="chat-window" class="chat-window">
        <div id="chat-container"></div>
        <div class="input-row">
            <input id="user-input" type="text" placeholder="Ask about color...">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
    function toggleChat() {
        var chat = document.getElementById("chat-window");
        chat.style.display = chat.style.display === "none" || chat.style.display === "" ? "flex" : "none";
    }
    function sendMessage() {
        var input = document.getElementById("user-input");
        var msg = input.value.trim();
        if (msg === "") return;
        window.parent.postMessage({type: 'user_message', text: msg}, '*');
        input.value = "";
    }
    </script>
    """,
    unsafe_allow_html=True
)

def hex_to_rgb(hex_code):
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def complementary_color(hex_code):
    r, g, b = hex_to_rgb(hex_code)
    comp_rgb = (255 - r, 255 - g, 255 - b)
    return rgb_to_hex(*comp_rgb)

def color_temperature_and_mood(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = h * 360
    if s < 0.2 and l > 0.8:
        return "White → purity & simplicity"
    elif s < 0.2 and l < 0.2:
        return "Black → power & mystery"
    elif s < 0.2:
        return "Gray → balance & neutrality"
    if hue < 15 or hue >= 345:
        return "Red → energy & passion"
    elif 15 <= hue < 45:
        return "Orange → enthusiasm & creativity"
    elif 45 <= hue < 75:
        return "Yellow → optimism & cheerfulness"
    elif 75 <= hue < 165:
        return "Green → growth & harmony"
    elif 165 <= hue < 195:
        return "Cyan → clarity & refreshment"
    elif 195 <= hue < 255:
        return "Blue → calm & trust"
    elif 255 <= hue < 285:
        return "Indigo → intuition & sophistication"
    elif 285 <= hue < 345:
        return "Purple/Pink → imagination, love & compassion"
    else:
        return "Neutral / balanced color"

def get_bot_response(message):
    msg = message.lower().strip()
    
    # HEX -> RGB
    if re.match(r'#?[0-9a-f]{6}$', msg):
        if not msg.startswith('#'):
            msg = '#' + msg
        rgb = hex_to_rgb(msg)
        comp = complementary_color(msg)
        mood = color_temperature_and_mood(rgb)
        return f"🎨 HEX: {msg}\n🟢 RGB: {rgb}\n✨ Mood: {mood}\n💠 Complementary: {comp}"

    # RGB -> HEX
    rgb_match = re.findall(r'\d+', msg)
    if len(rgb_match) == 3:
        try:
            r, g, b = map(int, rgb_match)
            if all(0 <= v <= 255 for v in (r, g, b)):
                hex_val = rgb_to_hex(r, g, b)
                comp = complementary_color(hex_val)
                mood = color_temperature_and_mood((r, g, b))
                return f"🟢 RGB: ({r}, {g}, {b})\n🎨 HEX: {hex_val}\n✨ Mood: {mood}\n💠 Complementary: {comp}"
        except:
            pass


    if "hello" in msg or "hi" in msg:
        return "👋 Hey there! I'm *Perceptra*, your color assistant. Ask me about any color in HEX or RGB format!"
    if "help" in msg:
        return "🧠 You can ask me like:\n- HEX to RGB → `#ff5733`\n- RGB to HEX → `rgb(255,87,51)`\n- Mood or complementary color → `What is the mood of #00ffcc?`"
    if "complementary" in msg:
        hex_in_msg = re.search(r'#?[0-9a-f]{6}', msg)
        if hex_in_msg:
            color = hex_in_msg.group()
            if not color.startswith('#'):
                color = '#' + color
            return f"💠 Complementary of {color} is {complementary_color(color)}"
    if "mood" in msg or "temperature" in msg:
        hex_in_msg = re.search(r'#?[0-9a-f]{6}', msg)
        if hex_in_msg:
            color = hex_in_msg.group()
            if not color.startswith('#'):
                color = '#' + color
            rgb = hex_to_rgb(color)
            return f"✨ Mood of {color}: {color_temperature_and_mood(rgb)}"
    return "🤖 Sorry, I didn’t understand that. Try asking about HEX, RGB, mood, or complementary color!"


st.session_state.setdefault("chat_history", [])

message_placeholder = st.empty()

def render_chat():
    chat_html = ""
    for entry in st.session_state.chat_history:
        if entry["sender"] == "user":
            chat_html += f'<div class="user-msg">{entry["text"]}</div>'
        else:
            chat_html += f'<div class="bot-msg">{entry["text"]}</div>'
    message_placeholder.markdown(f"<div class='chat-window'>{chat_html}</div>", unsafe_allow_html=True)

render_chat()

st.markdown(
    """
    <script>
    window.addEventListener('message', (event) => {
        if (event.data.type === 'user_message') {
            const msg = event.data.text;
            window.parent.postMessage({type: 'streamlit_message', text: msg}, '*');
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)

if "streamlit_message" in st.session_state:
    msg = st.session_state.streamlit_message
    st.session_state.chat_history.append({"sender": "user", "text": msg})
    response = get_bot_response(msg)
    st.session_state.chat_history.append({"sender": "bot", "text": response})
    del st.session_state.streamlit_message
    render_chat()
