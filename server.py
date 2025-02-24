from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyCgCcrekPoayfH8Wm8b9prXskaAs4Wsa2M")

app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    try:
        response = genai.GenerativeModel('gemini-1.5-pro').generate_content(user_message)
        bot_reply = response.text
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on port 5001
