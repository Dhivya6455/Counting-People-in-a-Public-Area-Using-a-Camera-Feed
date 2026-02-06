from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route("/start")
def start():
    # start people count script
    subprocess.Popen(["python", "main.py"])
    return jsonify({"status": "People counting started"})

@app.route("/count")
def count():
    # return count (example)
    return jsonify({"people": 12})

if __name__ == "__main__":
    app.run(debug=True)
