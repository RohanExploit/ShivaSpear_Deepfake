from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained Logistic Regression model
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def verify_fact(news_text):
    """Predict whether the given news text is real or fake."""
    text_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(text_vectorized)
    return "Real News" if prediction[0] == 1 else "Fake News"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        news_text = request.form.get("news_text")  # Capture user input
        if news_text:
            prediction_text = verify_fact(news_text)

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)

