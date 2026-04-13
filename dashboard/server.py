import sys

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import calibration as calibration_store

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = None

current_state = {
    "s1": 0,
    "s2": calibration_store.DEFAULT_CALIBRATION["s2_smin"],
    "s3": calibration_store.DEFAULT_CALIBRATION["s3_center"],
    "s4": calibration_store.DEFAULT_CALIBRATION["s4_center"],
    "manual_override": False,
    "manual_state": False,
    "gripper_mode": "AUTO",
}


@app.context_processor
def inject_shared_dashboard_state():
    return {
        "default_calibration": calibration_store.DEFAULT_CALIBRATION,
        "initial_calibration": calibration_store.get_calibration(),
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


@app.route("/3d-ik")
def dashboard3d_ik():
    return render_template("index3d_ik.html")


@app.route("/pick-sim")
def pick_sim():
    return render_template("index_pick_sim.html")


@app.route("/tune")
def tune_dashboard():
    return render_template("calibration.html")


def set_model(active_model):
    global model
    model = active_model


def resolve_model():
    global model

    if model is not None:
        return model

    main_module = sys.modules.get("__main__")
    active_model = getattr(main_module, "model", None)
    if active_model is not None:
        model = active_model

    return model


def get_gripper_state(active_model=None):
    active_model = active_model or resolve_model()

    if active_model is None:
        manual_override = bool(current_state.get("manual_override", False))
        manual_state = bool(current_state.get("manual_state", False))
    else:
        manual_override = bool(getattr(active_model, "manual_override", False))
        manual_state = bool(getattr(active_model, "manual_state", False))

    return {
        "manual_override": manual_override,
        "manual_state": manual_state,
        "gripper_mode": "MANUAL" if manual_override else "AUTO",
    }


def sync_gripper_state(active_model=None, emit_update=False):
    global current_state

    gripper_state = get_gripper_state(active_model)
    current_state.update(gripper_state)
    socketio.emit("gripper_mode", gripper_state)

    if emit_update:
        socketio.emit("update", current_state)


@socketio.on("connect")
def handle_connect():
    current_state.update(get_gripper_state())
    emit("update", current_state)
    emit("gripper_mode", get_gripper_state())
    emit("calibration_update", calibration_store.get_calibration())


@socketio.on("toggle_gripper")
def toggle_gripper():
    global model

    model = resolve_model()
    if model is not None:
        model.manual_override = True
        model.manual_state = not bool(getattr(model, "manual_state", False))
        current_state["s1"] = 100 if model.manual_state else 0
    else:
        current_state["manual_override"] = True
        current_state["manual_state"] = not bool(current_state.get("manual_state", False))
        current_state["s1"] = 100 if current_state["manual_state"] else 0

    sync_gripper_state(model, emit_update=True)


@socketio.on("auto_gripper")
def auto_gripper():
    global model

    model = resolve_model()
    if model is not None:
        model.manual_override = False
        resolve_gripper_servo = getattr(model, "resolve_gripper_servo", None)
        if callable(resolve_gripper_servo):
            current_state["s1"] = int(resolve_gripper_servo())
    else:
        current_state["manual_override"] = False

    sync_gripper_state(model, emit_update=True)


@socketio.on("update_calibration")
def update_calibration(data):
    updated_calibration = calibration_store.update_calibration_values(data)
    socketio.emit("calibration_update", updated_calibration)


@socketio.on("request_calibration")
def send_calibration():
    emit("calibration_update", calibration_store.get_calibration())


@socketio.on("save_calibration")
def save_calibration():
    calibration_store.save_calibration_values()


@socketio.on("reset_calibration")
def reset_calibration():
    updated_calibration = calibration_store.reset_calibration_values()
    socketio.emit("calibration_update", updated_calibration)


@socketio.on("restore_calibration")
def restore_calibration():
    updated_calibration = calibration_store.restore_calibration_values()
    socketio.emit("calibration_update", updated_calibration)


def update_state(s1, s2, s3, s4):
    global current_state

    current_state = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        **get_gripper_state(),
    }
    socketio.emit("update", current_state)
