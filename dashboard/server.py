from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

current_state = {
    "s1": 0,
    "s2": 0,
    "s3": 0,
    "s4": 0,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/3d")
def dashboard3d():
    return render_template("index3d.html")


@app.route("/3d-fk")
def dashboard3d_fk():
    return render_template("index3d_fk.html")


@socketio.on("connect")
def handle_connect():
    emit("update", current_state)


def update_state(s1, s2, s3, s4):
    global current_state

    current_state = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
    }
    socketio.emit("update", current_state)
