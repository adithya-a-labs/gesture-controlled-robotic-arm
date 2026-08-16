# Gesture-Controlled Robotic Arm

> A real-time human-to-robot control prototype that converts monocular upper-body and hand landmarks into calibrated, constrained commands for a servo-driven manipulator, while broadcasting the same command state to browser-based kinematic visualizations.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](requirements.txt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white)](vision/camera.py)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?logo=google&logoColor=white)](vision/handtracking.py)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)](vision/gesture_model.py)
[![ESP32](https://img.shields.io/badge/ESP32-Arduino%20%2F%20C%2B%2B-E7352C?logo=espressif&logoColor=white)](controller%20code/esp32-control-code/esp32-control-code.ino)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](dashboard/server.py)
[![Flask-SocketIO](https://img.shields.io/badge/Flask--SocketIO-5.3-010101?logo=socketdotio&logoColor=white)](dashboard/server.py)
[![Three.js](https://img.shields.io/badge/Three.js-r152-000000?logo=threedotjs&logoColor=white)](dashboard/templates/index3d_fk.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Human–Robot Interaction · Computer Vision · Robotic Manipulation · Embodied AI Infrastructure · Real-Time Control**

> **Research question:** How can human motion and intent be converted into stable, bounded, real-time physical robot motion when perception is noisy and human and robot kinematics do not match?

This repository documents a working research prototype, not an autonomous robot. MediaPipe provides learned visual perception; the current action policy is explicit geometry, calibration, filtering, and threshold logic. Inverse kinematics, browser simulations, and alternative mappings are maintained as experiments. Object reasoning, planning, learned policies, and autonomy are research directions—not implemented capabilities.

<details>
<summary><strong>Navigate this technical overview</strong></summary>

- [Project status](#project-status)
- [System architecture](#system-architecture)
- [Runtime software architecture](#runtime-software-architecture)
- [Vision and geometric control](#vision-and-geometric-control)
- [Human-to-robot calibration](#human-to-robot-calibration)
- [Signal conditioning](#signal-conditioning-and-control-stability)
- [ESP32 hardware boundary](#esp32-hardware-boundary)
- [Visual model and forward kinematics](#synchronized-visual-model-and-forward-kinematics)
- [Experimental control paths](#experimental-control-paths)
- [Installation and operation](#installation)
- [Evaluation framework](#evaluation-framework)
- [Research roadmap](#research-roadmap)
- [Safety and limitations](#safety-and-known-limitations)

</details>

---

## Project status

| Maturity | What exists in this repository |
| --- | --- |
| **Implemented** | Webcam acquisition; MediaPipe pose and single-hand tracking; right-arm, torso, and pinch geometry; live calibration; temporal filtering; rate limiting; serial commands; direct ESP32 PWM control; 2D/3D command-state dashboards; manual/automatic gripper modes; Fusion 360 archives |
| **Experimental** | Alternative vector-based shoulder mapping; planar two-link IK prototypes; browser-only target IK visualizer; telemetry-derived IK diagnostic view; scripted pick-and-place visualization |
| **Planned / research direction** | Measured latency and accuracy evaluation; feedback sensing; physical link-parameter identification; 3D IK; collision-aware trajectories; depth and object perception; visual servoing; shared autonomy; imitation learning; VLA-based task planning; ROS/ROS 2 integration |

## Why this problem matters

Traditional teleoperation usually begins with a joystick, button panel, or pre-authored trajectory. This project asks whether a person's own motion can become the control interface.

Landmark detection is only the first stage. A physical manipulator cannot safely consume raw image coordinates: observations move with camera perspective, individual landmarks disappear, joint estimates jitter, human anatomy has different ranges and neutral positions from the mechanism, and servos convert every numerical fluctuation into torque. The engineering problem is therefore a complete perception-to-action chain:

~~~text
human intent
  → noisy visual observation
  → geometric state
  → human-to-robot calibration
  → temporal conditioning and constraints
  → transport
  → actuator output
  → physical behavior
~~~

The prototype is useful as a compact study in human–robot interaction, calibration, physical-system uncertainty, observability, and the boundary between learned perception and deterministic control.

---

## System architecture

~~~mermaid
flowchart LR
    H["Human upper-body motion<br/>and pinch intent"] --> C["Webcam<br/>OpenCV camera 0"]
    C --> P["MediaPipe Tasks<br/>PoseLandmarker + HandLandmarker"]
    P --> G["2D geometry<br/>right arm, hip center, pinch distance"]
    G --> K["Calibration + constraints<br/>human space → servo space"]
    K --> F["Signal conditioning<br/>fallback, smoothing, rate limits"]
    F --> S["Command state<br/>S1, S2, S3, S4"]

    S --> T["CSV over USB serial<br/>S4,S3,S2,S1 at 115200 baud"]
    T --> E["ESP32 LEDC controller<br/>50 Hz, 16-bit PWM"]
    E --> R["Physical mechanism<br/>base + shoulder + elbow + gripper"]

    S --> W["Flask-SocketIO<br/>update event"]
    W --> D["2D telemetry +<br/>Three.js visualizations"]
    W <--> Q["Live calibration and<br/>gripper controls"]

    classDef perception fill:#123b4a,stroke:#57c7d4,color:#fff
    classDef control fill:#49351c,stroke:#e8ad55,color:#fff
    classDef physical fill:#40252b,stroke:#df7180,color:#fff
    classDef web fill:#243650,stroke:#7da7e8,color:#fff
    class P,G perception
    class K,F,S control
    class T,E,R physical
    class W,D,Q web
~~~

### Data representation through the pipeline

~~~mermaid
flowchart TD
    A["Mirrored BGR camera frame"] --> B["RGB MediaPipe Image"]
    B --> C["Normalized 2D landmarks<br/>(x, y, visibility)"]
    C --> D["Geometry<br/>elbow angle · shoulder angle<br/>hip-center offset · pinch distance"]
    D --> E["Calibrated targets<br/>(s1, s2, s3, s4)"]
    E --> F["Validated, smoothed,<br/>step-limited integer angles"]
    F --> G["Serial line<br/>s4,s3,s2,s1 newline"]
    G --> H["16-bit LEDC duty values"]
    H --> I["50 Hz servo PWM"]
    I --> J["Physical joint motion"]

    F --> K["Socket.IO update object<br/>{s1, s2, s3, s4, mode}"]
    K --> L["2D / Three.js command-state view"]
~~~

The order changes at the transport boundary:

| Layer | Order | Meaning |
| --- | --- | --- |
| Python model return and dashboard state | <code>s1, s2, s3, s4</code> | gripper, elbow, shoulder, base |
| Serial packet | <code>s4,s3,s2,s1\n</code> | base, shoulder, elbow, gripper |

Example packet: <code>90,60,45,10\n</code>.

---

## Runtime software architecture

<code>main.py</code> is the implemented end-to-end entrypoint. It constructs the camera, MediaPipe tracker, gesture model, optional serial connection, and dashboard server, then coordinates three concurrent activities:

- a camera thread continuously replaces a shared latest frame;
- a processing thread copies that frame, runs both landmarkers, computes servo targets, publishes dashboard state, and optionally writes serial data;
- the main thread renders landmarks and servo telemetry in an OpenCV window, handles <kbd>Esc</kbd>, and performs shutdown.

Flask-SocketIO runs in its own daemon thread on <code>0.0.0.0:5000</code>. The shared frame, results, and servo state are module globals; there is no queue, timestamp synchronization, or explicit lock between camera and processing loops. Each consumer uses a frame copy, and processing favors the latest observation rather than preserving every frame.

~~~mermaid
sequenceDiagram
    actor U as Operator
    participant C as Camera thread
    participant P as Processing thread
    participant M as GestureModel
    participant W as Flask-SocketIO
    participant E as ESP32
    participant V as OpenCV main thread

    loop while running
        C->>C: capture and mirror frame
        C-->>P: shared latest_frame
        P->>P: PoseLandmarker + HandLandmarker
        P->>M: pose_result, hand_result
        M->>M: geometry, calibration, filtering
        M-->>P: (s1, s2, s3, s4)
        P->>W: update_state(...)
        W-->>U: Socket.IO "update"
        opt serial connection available
            P->>E: "s4,s3,s2,s1\n"
        end
        P-->>V: shared landmarks + servo state
        V-->>U: annotated camera window
    end
    U->>V: Esc
    V->>C: stop and release resources
    V->>E: close serial port
~~~

The loops are not scheduled at explicit fixed rates. Camera capture, MediaPipe inference, Socket.IO emission, serial output, and browser rendering proceed at their own available rates. This is a practical prototype architecture, but it also creates a useful future research problem: instrumenting and controlling a multi-rate perception–action system.

---

## Vision and geometric control

### Acquisition and inference

<code>vision/camera.py</code> opens OpenCV camera index <code>0</code> and mirrors each frame horizontally. <code>vision/handtracking.py</code> creates two MediaPipe Tasks models in video mode:

- <code>pose_landmarker.task</code> for body pose;
- <code>hand_landmarker.task</code> for at most one hand.

The implementation uses separate pose and hand landmarkers despite the class name <code>HolisticTracker</code>. MediaPipe is the learned perception component. Everything downstream in the primary pipeline is deterministic.

### Anatomical inputs

| Control signal | MediaPipe landmarks | Computation |
| --- | --- | --- |
| S2 elbow | right shoulder 12, elbow 14, wrist 16 | planar three-point angle at the elbow |
| S3 shoulder | right shoulder 12 and elbow 14 | angle between upper-arm direction and an image-space upward reference |
| S4 base | left hip 23 and right hip 24 | horizontal midpoint relative to image center |
| S1 gripper | thumb tip 4, index tip 8; index MCP 5 and wrist 0 validated | thumb–index tip distance with hysteresis |

Both shoulders are also resolved as part of the cached pose tuple, but the primary mapping does not use shoulder-to-shoulder orientation for base control. Base motion follows the horizontal location of the hip midpoint in the camera image.

### Joint-angle geometry

For three valid image-plane points \(\mathbf{a}\), \(\mathbf{b}\), and \(\mathbf{c}\), the controller evaluates:

$$
\theta =
\cos^{-1}\left(
\frac{(\mathbf{a}-\mathbf{b})\cdot(\mathbf{c}-\mathbf{b})}
{\lVert\mathbf{a}-\mathbf{b}\rVert\,\lVert\mathbf{c}-\mathbf{b}\rVert}
\right)
$$

The cosine is clipped to \([-1,1]\), and a zero or non-finite denominator produces an invalid result rather than an actuator command.

- **Elbow:** \(\theta_e = \angle(\text{shoulder},\text{elbow},\text{wrist})\).
- **Shoulder:** \(\theta_s = \angle(\text{elbow},\text{shoulder},\text{vertical reference})\), where the reference is 0.2 normalized image units above the shoulder.
- **Base:** \(o = 2((x_{hip,L}+x_{hip,R})/2 - 0.5)\). Values with \(|o| < 0.05\) are set to zero.

These are monocular, 2D image-space relationships. They are not recovered 3D joint poses, and perspective or out-of-plane motion can change the inferred geometry.

### Invalid observations and state retention

The control path treats perception output defensively:

- pose points require finite <code>x</code>/<code>y</code> and visibility of at least <code>0.5</code>;
- an invalid point reuses its last valid value;
- if a complete valid pose has never been established, the last complete servo output is retained;
- invalid joint geometry retains that joint's previous command;
- hand loss retains the latched gripper state;
- final values are finite-checked, rounded, and clipped.

This fallback strategy favors continuity. It does not estimate uncertainty or impose a timeout that returns the arm to a safe pose after prolonged tracking loss; that remains a safety and control improvement.

---

## Human-to-robot calibration

Human joint angle is not servo angle. The mechanism has different ranges, neutral positions, linkage geometry, mounting directions, and safe travel. In this repository, calibration is part of the control algorithm rather than a cosmetic configuration layer.

~~~mermaid
flowchart LR
    H["Human / image-space<br/>measurement"] --> V["Finite + visibility<br/>validation"]
    V --> C["Clamp to assumed<br/>human input interval"]
    C --> M["Linear mapping into<br/>calibrated servo interval"]
    M --> S["Smoothing +<br/>maximum step"]
    S --> J["Final joint-specific<br/>mechanical bounds"]
    J --> O["Servo target"]
~~~

### Current calibration contract

Defaults come from <code>calibration.py</code>:

| Parameter | Default | Runtime meaning | Live tuning |
| --- | ---: | --- | --- |
| <code>s2_hmin</code> / <code>s2_hmax</code> | 60° / 180° | elbow human-angle interval | fixed by sanitizer and UI |
| <code>s2_smin</code> / <code>s2_smax</code> | 20° / 75° | elbow servo interval | adjustable within 20°–75° |
| <code>s3_center</code> / <code>s3_min</code> | 55° / 55° | shoulder neutral/lower bound | fixed |
| <code>s3_max</code> | 85° | shoulder upper bound | adjustable from 55°–85° |
| <code>s4_center</code> | 90° | base center | adjustable from 0°–180° |
| <code>s4_range</code> | 60° | total base travel around center | adjustable from 2°–180°, clipped to 0°–180° |
| <code>pinch_threshold</code> | 0.04 | close threshold in normalized image coordinates | adjustable |
| <code>release_threshold</code> | 0.07 | open threshold | adjustable; sanitized to be no lower than pinch |

The primary mappings are:

$$
\theta_e \in [60^\circ,180^\circ]
\longrightarrow
s_2 \in [s2_{min},s2_{max}]
$$

$$
\theta_s \in [30^\circ,150^\circ]
\longrightarrow
s_3 \in [s3_{center},s3_{max}]
$$

$$
o \in [-1,1]
\longrightarrow
s_4 \in
\left[s4_{center}-\frac{s4_{range}}{2},
s4_{center}+\frac{s4_{range}}{2}\right]
$$

All servo intervals are additionally constrained to 0°–180°.

### Live tuning workflow

Open <http://localhost:5000/tune> while <code>main.py</code> is running. Slider changes emit <code>update_calibration</code>; the server sanitizes them under a re-entrant lock and broadcasts <code>calibration_update</code> to connected views. The gesture model reads a fresh calibration snapshot on every processing pass.

The panel supports:

- **Save** — copy the current live values to an in-memory snapshot;
- **Restore** — return to that in-memory snapshot;
- **Reset** — return to source-code defaults.

Calibration is **not persisted to disk**. Restarting Python restores <code>DEFAULT_CALIBRATION</code>. This is appropriate for rapid experiments but should become a versioned, persistent profile system before repeatable quantitative studies.

---

## Signal conditioning and control stability

Raw landmarks should never be treated as motor commands. The primary <code>GestureModel</code> uses several layers of conditioning:

### Exponential smoothing

For elbow and shoulder:

$$
y_t = \alpha x_t + (1-\alpha)y_{t-1}
$$

The smoothing factor is adaptive:

| Target change from prior command | \(\alpha\) | Behavior |
| --- | ---: | --- |
| ≤ 10° | 0.3 | stronger jitter rejection |
| > 10° | 0.7 | faster response to deliberate motion |

After smoothing, both joints are limited to a maximum 5° command step per processing update.

Base control uses fixed \(\alpha=0.3\), a 0.05 normalized-coordinate center deadband, and a maximum 3° step per processing update. The gripper is a binary 10°/100° command and is stabilized by stateful hysteresis rather than numerical smoothing.

### Conditioning order

~~~text
geometric target
  → finite-number fallback
  → calibrated range mapping
  → exponential smoothing
  → maximum angular step
  → calibrated range clip
  → integer servo command
~~~

The tradeoff is fundamental: stronger filtering reduces visible and physical jitter but adds lag; weaker filtering feels responsive but transmits more perception noise. Because limits are per processing update rather than per second, their physical slew rate depends on inference throughput. Time-based rate limits are a natural next control improvement.

### Gripper hysteresis

The gripper measures normalized 2D distance between thumb tip and index fingertip:

~~~text
tip distance < pinch threshold (0.04 default)
    → latch closed → S1 = 100°

tip distance > release threshold (0.07 default)
    → latch open → S1 = 10°

between thresholds, or hand unavailable
    → retain previous state
~~~

Separate close and release boundaries prevent repeated open/close transitions when the observed distance fluctuates near one threshold. The index-MCP-to-wrist distance is checked for validity, but the current implementation does not divide tip distance by that reference; pinch thresholds therefore remain sensitive to camera scale.

Dashboards can switch the gripper to manual mode and toggle its state through Socket.IO, then return control to automatic pinch detection.

---

## Control contract and physical mechanism

The repository controls **three positioning joints plus one gripper actuator**. Calling all four channels “4-DOF” can be ambiguous, so this README describes the actual actuation explicitly.

| Channel | Robot function | Human control signal | Default command range |
| --- | --- | --- | ---: |
| S4 | base yaw | horizontal hip-center displacement | 60°–120° |
| S3 | shoulder pitch | right upper-arm angle to image vertical | 55°–85° |
| S2 | elbow flexion/extension | right elbow angle | 20°–75° |
| S1 | gripper open/close | thumb–index pinch latch | 10° open / 100° closed |

The <code>cad-model/</code> directory contains Autodesk Fusion 360 <code>.f3z</code> archives for the arm assembly, gripper, and total assembly, plus a [static Autodesk viewer link](https://a360.co/4sxEt1Y). Archive metadata references component designs named MG996R and MG90S, but the repository does not provide a definitive as-built bill of materials. Servo model, torque, supply, link length, payload, mass, and verified mechanical travel should therefore be treated as undocumented until measured and recorded.

### ESP32 hardware boundary

~~~mermaid
flowchart LR
    P["Python<br/>bounded integer angles"] -->|USB serial<br/>115200 baud| X["ESP32 parser<br/>sscanf: S4,S3,S2,S1"]
    X --> D["angleToDuty<br/>0°–180° → 1638–8192"]
    D --> L["LEDC<br/>50 Hz · 16 bit"]
    L --> B["GPIO 27<br/>S4 base"]
    L --> Sh["GPIO 14<br/>S3 shoulder"]
    L --> El["GPIO 12<br/>S2 elbow"]
    L --> Gr["GPIO 25<br/>S1 gripper"]
~~~

The production firmware is <code>controller code/esp32-control-code/esp32-control-code.ino</code>. It:

- starts serial at 115200 baud;
- attaches four ESP32 LEDC outputs at 50 Hz with 16-bit resolution;
- reads one newline-terminated CSV packet;
- parses four integers with <code>sscanf</code>;
- maps 0°–180° to duty counts 1638–8192 (approximately 0.5–2.5 ms at 50 Hz);
- writes all four outputs immediately and echoes the applied tuple.

The production firmware performs **no smoothing, slew limiting, or input constraining**. Those protections currently live in Python, and the firmware trusts the packet. Invalid or out-of-range external senders can therefore produce unsafe PWM values. Firmware-side bounds and a command watchdog are high-priority safety improvements.

The sketches under <code>esp32-test codes/</code> are isolated bring-up utilities for all servos, the gripper, and slow single-servo motion. Their command formats and pins are not the production protocol; use the controller firmware above for <code>main.py</code>.

---

## Synchronized visual model and forward kinematics

The server publishes the post-conditioned command state to every connected browser:

~~~mermaid
flowchart TD
    C["Conditioned command state<br/>S1, S2, S3, S4"] --> H["Serial branch"]
    H --> E["ESP32"] --> R["Physical robot"]
    C --> W["Socket.IO update event"]
    W --> T["2D dashboard"]
    W --> F["Three.js FK view"]
    W --> I["IK-oriented diagnostics"]
    Q["Calibration updates"] --> C
    Q --> T
    Q --> F
    Q --> I
~~~

This fork makes internal control state observable without needing the physical arm. It helps distinguish:

- perception/mapping errors, visible in both command state and visualization;
- serial/electrical/mechanical errors, which can appear only on the physical branch;
- calibration inconsistencies, which update across all connected dashboards.

Strictly, the browser model is a **command-state digital twin**, not a feedback-validated digital twin: there are no encoders or other joint sensors reporting actual physical pose.

### FK representation

The <code>/3d-fk</code> page maps servo values back into display-space shoulder and elbow angles, then applies a hierarchical Three.js transform:

~~~text
base frame (S4 yaw)
  → shoulder frame (S3 pitch)
  → L1 = 3 display units
  → elbow frame (S2 relative pitch)
  → L2 = 3 display units
  → wrist / gripper world position
~~~

For the planar display chain:

$$
x = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2)
$$

$$
y = L_1\sin\theta_1 + L_2\sin(\theta_1+\theta_2)
$$

The values \(L_1=L_2=3\) are visualization units defined in the dashboard, not verified physical dimensions from CAD. The renderer uses a simplified arm geometry rather than loading the Fusion 360 assembly.

### Dashboard inventory

All standard routes are served by <code>dashboard/server.py</code>:

| URL | Status | Data / interaction | Engineering use |
| --- | --- | --- | --- |
| <code>/</code> | implemented | live S1–S4 state, 2D canvas arm, manual/auto gripper | compact telemetry and mapping check |
| <code>/3d</code> | implemented | live S1–S4 Three.js arm | general command-state visualization |
| <code>/3d-fk</code> | implemented | calibrated joint mapping, fixed-link chain, wrist world position | FK and transform debugging |
| <code>/tune</code> | implemented | live calibration sliders, save/restore/reset | mechanical tuning and repeatable experiments within one process |
| <code>/3d-ik</code> | experimental diagnostic | reconstructs an XY target from live servo-derived angles, then interpolates the same joint targets | IK-oriented telemetry visualization; not an independent target solver |
| <code>/3d-semi-ik</code> | experimental, browser-only | X/Y sliders and planar two-link solver | reachable-workspace and IK branch exploration; does not drive hardware |
| <code>/pick-sim</code> | experimental, scripted | predefined servo-state phases and an attached cube | visualization concept only; not perception- or planner-driven |

Running <code>main_vector_experimental.py</code> adds <code>/3d-vector</code>, a Three.js view paired with the alternative vector shoulder mapping. Three.js r152 and the Socket.IO browser client are loaded from public CDNs, so those dashboards require network access unless the libraries are vendored locally.

---

## Experimental control paths

### Experimental: planar inverse kinematics

<code>main_ik_experimental.py</code> is a separate, deliberately experimental entrypoint. It:

1. tracks right shoulder, elbow, and wrist with <code>pose_landmarker_full.task</code>;
2. treats the shoulder-to-wrist displacement in normalized image coordinates as a planar target;
3. solves a two-link cosine-law IK problem with \(L_1=L_2=0.3\);
4. maps the solved shoulder and elbow angles into the current S3/S2 calibration bounds;
5. overlays the angles and mapped values on the camera frame.

~~~mermaid
flowchart LR
    P["Right shoulder + wrist<br/>normalized image coordinates"] --> T["Planar target<br/>dx, dy"]
    T --> R["Reach clamp<br/>ε to L1 + L2 − ε"]
    R --> I["Two-link analytical IK"]
    I --> A["θ1 shoulder<br/>θ2 elbow"]
    A --> C["Calibration-aware map<br/>S3, S2"]
    C --> O["On-screen debug overlay"]
    O -. "not connected" .-> X["Serial / physical robot"]
~~~

The solver clamps unreachable distances to avoid invalid <code>acos</code> inputs. It remains a monocular 2D experiment: the tracked elbow is displayed but not used by the solver, link lengths are nominal image-space values, branch selection is fixed, and the entrypoint does not send serial commands or publish dashboard state.

The related <code>ik_pipeline/semi_ik.py</code> and browser <code>/3d-semi-ik</code> use the same planar analytical structure with \(L_1=L_2=3\) arbitrary units. These prototypes establish the mathematics and UI needed for target-based control, but production IK still requires identified link dimensions, coordinate-frame calibration, joint-limit-aware branch selection, temporal control, and hardware validation.

### Experimental: vector shoulder mapping

<code>main_vector_experimental.py</code> replaces the primary shoulder angle with a blend of:

- upper-arm direction from <code>atan2</code>; and
- vertical shoulder-to-elbow displacement.

It broadcasts results to dashboards, including the extra <code>/3d-vector</code> route, but it has no serial transport. This keeps mapping experiments isolated from the implemented hardware path.

### Gesture notebooks

<code>gesturedetection-trial/</code> preserves MediaPipe notebooks used during exploration, including earlier hand-gesture and incremental servo-control ideas. They are research history rather than dependencies of <code>main.py</code>.

---

## Technology stack

| Layer | Verified technologies | Role |
| --- | --- | --- |
| Perception | Python, OpenCV, MediaPipe Tasks, NumPy | image acquisition, landmark inference, geometry, numerical validation |
| Host control | Python, PySerial, threading | calibration, conditioning, command formatting, concurrent runtime |
| Web observability | Flask, Flask-SocketIO, Eventlet, HTML/CSS/JavaScript | routes, telemetry events, live tuning, operator controls |
| Kinematic visualization | Three.js | simplified 3D arm transforms, FK/IK diagnostics, scripted simulation |
| Embedded control | ESP32, Arduino/C++, LEDC PWM | serial parsing and four-channel 50 Hz servo output |
| Mechanical design | Autodesk Fusion 360 archives | arm, gripper, and combined assembly design files |

---

## Repository map

~~~text
gesture-controlled-robotic-arm/
├── main.py                         # primary threaded vision → dashboard/serial runtime
├── calibration.py                  # synchronized calibration defaults and sanitization
├── main_ik_experimental.py         # camera-based planar IK overlay experiment
├── main_vector_experimental.py     # alternative shoulder-vector experiment
├── requirements.txt                # pinned Python runtime dependencies
├── pose_landmarker*.task           # MediaPipe pose model assets
├── hand_landmarker.task            # MediaPipe hand model asset
├── vision/
│   ├── camera.py                   # mirrored OpenCV acquisition
│   ├── handtracking.py             # MediaPipe pose + one-hand tracking and drawing
│   ├── gesture_model.py            # primary geometry, calibration, filtering, constraints
│   └── gesture_model_vector.py     # experimental direction-based shoulder model
├── dashboard/
│   ├── server.py                   # Flask routes, Socket.IO state, calibration events
│   └── templates/
│       ├── index.html              # 2D command-state dashboard
│       ├── index3d.html            # general Three.js view
│       ├── index3d_fk.html         # fixed-link FK view and debug telemetry
│       ├── calibration.html        # live tuning panel
│       ├── index3d_ik.html         # telemetry-derived IK diagnostic
│       ├── index3d_semi_ik.html    # standalone X/Y IK visualizer
│       ├── index3d_vector_fk.html  # experimental vector-mapping view
│       └── index_pick_sim.html     # scripted pick-and-place visualization
├── ik_pipeline/
│   ├── tracker.py                  # experimental right-arm PoseLandmarker wrapper
│   ├── ik_model.py                 # analytical planar two-link IK
│   ├── controller.py               # IK angle → calibrated S3/S2 mapping
│   ├── semi_ik.py                  # alternate planar solver
│   ├── ik_controller.py            # shoulder-relative target adapter
│   ├── utils.py                    # clamp and distance helpers
│   └── test_ik.py                  # interactive console experiment
├── controller code/
│   └── esp32-control-code/
│       └── esp32-control-code.ino  # production CSV → LEDC firmware
├── esp32-test codes/               # servo bring-up sketches, not production protocol
├── cad-model/
│   ├── assemblyv3.f3z              # Fusion 360 arm assembly
│   ├── Gripperv3.f3z               # Fusion 360 gripper assembly
│   ├── totalassembly.f3z           # combined Fusion 360 assembly
│   └── model.txt                   # Autodesk web-view link
└── gesturedetection-trial/         # exploratory MediaPipe notebooks and model copies
~~~

---

## Installation

The repository does not record a tested Python version. Use a Python release compatible with the pinned packages in <code>requirements.txt</code>.

~~~bash
git clone https://github.com/adithya-a-labs/gesture-controlled-robotic-arm.git
cd gesture-controlled-robotic-arm

python -m venv .venv
~~~

Activate the environment:

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bat
:: Windows Command Prompt
.\.venv\Scripts\activate.bat
~~~

~~~bash
# macOS / Linux
source .venv/bin/activate
~~~

Install dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

The pinned stack is OpenCV, MediaPipe, NumPy, Flask, Flask-SocketIO, Eventlet, and PySerial. Model assets are already stored at the repository root.

## Run without the physical arm

Hardware-independent development is built into <code>main.py</code>.

1. Edit the source constant:

   ~~~python
   USE_SERIAL = False
   ~~~

2. Start the application from the repository root:

   ~~~bash
   python main.py
   ~~~

3. Open <http://localhost:5000/3d-fk> and <http://localhost:5000/tune>.
4. Press <kbd>Esc</kbd> in the OpenCV camera window to stop.

With serial disabled, computed packets are printed as <code>[DEV MODE]</code>. If <code>USE_SERIAL</code> is true but <code>COM5</code> cannot be opened, the current code also falls back to this hardware-disabled behavior after reporting the connection error.

This mode supports perception work, calibration, dashboard testing, and algorithm debugging without energizing the mechanism. A webcam and desktop display are still required by the primary entrypoint.

## Run with the physical robot

### 1. Flash the production firmware

Use:

<code>controller code/esp32-control-code/esp32-control-code.ino</code>

The active sketch uses the ESP32 Arduino core's LEDC API and does not require the <code>ESP32Servo</code> library used by some test sketches.

### 2. Wire signal and power

| Channel | Function | ESP32 signal pin |
| --- | --- | ---: |
| S4 | base | GPIO 27 |
| S3 | shoulder | GPIO 14 |
| S2 | elbow | GPIO 12 |
| S1 | gripper | GPIO 25 |

- power servos from an external regulated 5 V supply, **not** from the ESP32;
- connect servo-supply ground and ESP32 ground;
- preserve the existing project guidance of at least 2 A supply capacity, then size the supply and wiring for the measured simultaneous stall current of the actual servos;
- verify the mechanism can move through configured ranges without collision before enabling live control.

### 3. Configure the host

In <code>main.py</code>:

~~~python
USE_SERIAL = True
ser = serial.Serial("COM5", 115200, timeout=1)
~~~

Replace <code>COM5</code> with the port assigned to the ESP32 on your system. Port selection is currently a source edit; there is no CLI flag or environment variable.

### 4. Start conservatively

1. Place the arm in a clear workspace.
2. Power the ESP32 and external servo supply.
3. Run <code>python main.py</code>.
4. Confirm telemetry in <code>/3d-fk</code> before making large gestures.
5. Tune only within collision-free ranges at <code>/tune</code>.
6. Keep an immediate means of removing servo power available.

---

## Engineering challenges and design decisions

| Challenge | Current design response | Remaining limitation |
| --- | --- | --- |
| Landmark jitter becomes actuator jitter | exponential smoothing and per-update step limits | limits are not time-normalized or quantitatively tuned |
| Human and robot kinematics differ | explicit per-joint range mapping and live calibration | no camera-to-robot extrinsic calibration or identified physical model |
| Pose landmarks disappear | visibility gate, previous-point cache, previous-command fallback | no dropout timer or commanded safe state |
| Pinch distance oscillates | separate close/release thresholds and a latched state | distance is not scale-normalized |
| Hardware slows and raises the risk of debugging | <code>USE_SERIAL = False</code>, automatic serial fallback, visual state branch | the primary entrypoint still requires camera and GUI |
| Internal state is difficult to inspect | Socket.IO telemetry, 2D/3D views, FK readouts, live tuning | view represents commands, not measured joint pose |
| Components operate at different rates | latest-frame shared-state architecture | no bounded queues, timestamps, latency budget, or deterministic scheduler |
| Bad numerical geometry can propagate | finite checks, cosine clipping, range clipping, broad last-output fallback | broad exception handling can hide the cause of failures |
| Host-side safety can be bypassed | calibrated Python bounds | firmware lacks bounds, watchdog, and safe timeout |

---

## Engineering and research learnings

1. **Perception output is not actuator-ready.** Landmark coordinates are observations with noise and missing data; a physical command needs validation, state, filtering, and limits.

2. **Calibration is control logic.** A correct human joint estimate can still produce unusable robot motion unless input ranges, servo neutral positions, mounting directions, and mechanical travel are reconciled.

3. **Stability and responsiveness are coupled.** The adaptive smoothing rule makes deliberate large changes more responsive while filtering small changes more strongly, but its behavior must eventually be evaluated against measured latency and variance.

4. **Human imitation is a mapping problem, not angle copying.** The current system deliberately compresses human elbow and shoulder motion into much narrower servo intervals.

5. **Physical AI software must be defensive.** NaNs, degenerate vectors, low-visibility points, camera dropouts, and serial failure are normal system conditions, not edge cases.

6. **Observability changes the speed of robotics work.** A shared command-state view and live calibration interface make it possible to inspect perception/control behavior before blaming mechanics or electronics.

7. **A visualization is not feedback.** Broadcasting the intended servo state is valuable, but it cannot prove that the physical arm reached that state. Sensors are required to close that loop.

8. **Real-time robotics is a systems problem.** Camera acquisition, inference, calibration, browser networking, serial transport, PWM, power integrity, and mechanism constraints all shape the final behavior.

---

## Embodied AI interpretation

The current system is embodied because visual observations are transformed into actions on a physical mechanism:

~~~text
human intent
  → learned visual perception
  → explicit geometric state extraction
  → calibrated deterministic action
  → physical embodiment
~~~

It is not autonomous and does not presently contain a learned control policy, world model, task planner, object detector, language model, or action feedback loop. Its research value for embodied AI is infrastructural: it establishes a perception–action path, a human demonstration interface, kinematic experiments, live system observability, and a physical platform on which richer agents could later be evaluated.

A future autonomous loop would add several stages:

~~~text
perception
  → world and robot state
  → uncertainty-aware reasoning / planning
  → constrained action
  → physical observation
  → adaptation
~~~

Gesture teleoperation may also become a data source: synchronized perception, operator intent, command trajectories, and future measured robot state could form demonstrations for imitation learning or shared-autonomy research.

---

## Evaluation framework

No benchmark results or quantitative performance claims are stored in the repository. The following is a proposed evaluation plan, not reported performance.

| Metric to measure | Instrumentation | Why it matters |
| --- | --- | --- |
| end-to-end latency | timestamp camera capture, inference completion, serial write, and measured motion onset | perceived responsiveness |
| servo-command variance at static pose | log each S1–S4 stream during held gestures | jitter and filtering quality |
| step response and settling time | issue controlled gesture/target changes and record commands plus physical pose | stability–latency tradeoff |
| human-angle → command repeatability | repeat calibrated poses across users and camera distances | mapping robustness |
| physical joint-angle error | add encoder or external motion-capture reference | command-state versus physical-state accuracy |
| end-effector error | identify physical link frames and compare target with measured tool position | FK/IK validity |
| landmark-dropout recovery | induce occlusion and measure hold duration and recovery transient | perception fault handling |
| pinch transition consistency | repeat close/open trials across scale and lighting | gripper reliability |
| serial and watchdog behavior | inject malformed, stale, and out-of-range packets safely | embedded fault tolerance |

A useful experimental record should include camera resolution and placement, compute hardware, package versions, servo models, supply voltage/current, mechanical load, calibration profile, processing rate, and repeated-trial distributions—not only averages.

---

## Research roadmap

~~~mermaid
flowchart LR
    A["1 · Gesture<br/>teleoperation"] --> B["2 · Persistent calibration<br/>+ instrumentation"]
    B --> C["3 · Measured robot state<br/>+ safety watchdogs"]
    C --> D["4 · Identified FK<br/>+ constrained 3D IK"]
    D --> E["5 · Depth, objects<br/>+ visual servoing"]
    E --> F["6 · Trajectory planning<br/>+ collision constraints"]
    F --> G["7 · Demonstration data<br/>+ shared autonomy"]
    G --> H["8 · Language-conditioned<br/>task planning"]
    H --> I["9 · Evaluated autonomous<br/>manipulation"]
~~~

1. **Harden teleoperation.** Add configuration-driven ports, clean thread shutdown, time-based filters, explicit failure telemetry, and automated tests.
2. **Make experiments reproducible.** Persist named calibration profiles and add timestamped logging plus benchmark protocols.
3. **Close the joint loop.** Add measured joint state where hardware allows, firmware bounds, a serial watchdog, and a defined loss-of-tracking policy.
4. **Identify the mechanism.** Record physical link dimensions and frames, validate FK against the arm, then implement joint-limit-aware 3D IK.
5. **Add environment perception.** Introduce depth and object pose estimation, then investigate closed-loop visual servoing.
6. **Plan safe motion.** Add trajectory generation, velocity/acceleration limits, workspace constraints, and collision checking.
7. **Move toward shared autonomy.** Use gesture control for demonstrations while an assistive controller handles constraints and low-level stabilization.
8. **Investigate intelligent task interfaces.** Evaluate language or vision-language-action models only after state estimation, safety, and measurement are reliable.
9. **Evaluate autonomy.** Test pick-and-place and other manipulation tasks with declared metrics, baselines, and failure analysis. ROS/ROS 2 may become useful at this integration stage; it is not currently present.

### Open research questions

- How accurately can 2D human arm geometry transfer to a mechanism with different kinematics and narrower ranges?
- Which filter minimizes physical jitter without increasing perceived teleoperation latency?
- How should pose-estimation confidence and dropout duration propagate into actuator commands?
- Can pinch intent be made robust to user scale and camera distance without sacrificing responsiveness?
- When does monocular control become insufficient for manipulation accuracy?
- How closely does commanded state match physical joint and end-effector state under load?
- Which calibration parameters should be user-specific, mechanism-specific, or learned online?
- Can the command-state visualization become a validated simulator once physical geometry is identified?
- How should control authority be shared between a human operator and an autonomous constraint/planning layer?
- Can trajectories collected through gesture control become useful imitation-learning demonstrations?

---

## Safety and known limitations

> This is an experimental prototype. The MIT license provides the software without warranty; repository-level constraints are not a substitute for an emergency stop, correct power design, or mechanical risk assessment.

- Never power multiple servos from the ESP32 regulator or logic pin.
- Use a correctly sized external regulated supply and a common ground.
- Verify polarity, GPIO assignment, servo pulse range, and unobstructed travel before enabling output.
- Begin with conservative calibration limits and no payload.
- Keep people, cables, and fragile objects outside the swept volume.
- Provide a rapid, physical means to remove actuator power.
- The production firmware currently lacks input bounds, timeout, and watchdog behavior.
- Python holds the last command on invalid perception; it does not automatically return to a safe pose after extended tracking loss.
- There is no collision detection, torque sensing, encoder feedback, or certified safety system.
- The model uses monocular 2D landmarks and does not estimate metric depth.
- Dashboard link lengths and geometry are illustrative and are not validated against CAD dimensions.

---

## Contributors

Attribution and contact information from the original project documentation are preserved:

| Contributor | GitHub | Contact |
| --- | --- | --- |
| Adithya A | [@adithya-a-labs](https://github.com/adithya-a-labs) | [adithya.a.builds@gmail.com](mailto:adithya.a.builds@gmail.com) |
| Rohan Skaria | [@skariarohan](https://github.com/skariarohan) | [rohsk12@gmail.com](mailto:rohsk12@gmail.com) |
| Aryan Sajan Nair | [@Ar27-25](https://github.com/Ar27-25) | [aryansajannair@gmail.com](mailto:aryansajannair@gmail.com) |
| Medicherla Satya Kalyana Bhairava Mukesh | [@Satya2508](https://github.com/Satya2508) | [n5517301mukhesh@gmail.com](mailto:n5517301mukhesh@gmail.com) |

Issues, experiment reports, calibration findings, and collaboration proposals are welcome through the repository's [issue tracker](https://github.com/adithya-a-labs/gesture-controlled-robotic-arm/issues).

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Adithya A, Rohan Skaria, Aryan Nair, and M Satya.
