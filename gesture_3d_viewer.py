import threading
import time
import math
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLUT import *
from pyrr import Matrix44, Vector3

# Single-file Gesture-Controlled 3D Object Viewer
# Contains: HandTracker, GestureInterpreter, OBJLoader, OpenGLRenderer, main

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class HandTracker:
    """MediaPipe Hands wrapper with moving-average smoothing."""
    def __init__(self, max_num_hands=1, smoothing_window=5, detection_conf=0.6, tracking_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         model_complexity=1,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=tracking_conf)
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
        self.lock = threading.Lock()
        self.frame_size = (640, 480)

    def update_frame_size(self, w, h):
        with self.lock:
            self.frame_size = (w, h)

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        lm = None
        h, w = frame_bgr.shape[:2]
        self.update_frame_size(w, h)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand = results.multi_hand_landmarks[0]
            pts = []
            for l in hand.landmark:
                pts.append([l.x * w, l.y * h, l.z * w])
            lm = np.array(pts, dtype=np.float32)
        with self.lock:
            self.history.append(lm)
        return self.get_smoothed()

    def get_smoothed(self):
        with self.lock:
            valid = [h for h in self.history if h is not None]
        if not valid:
            return None
        arr = np.stack(valid, axis=0)
        mean = np.mean(arr, axis=0)
        return mean


class GestureInterpreter:
    """Interprets gestures from landmarks and provides control signals."""
    def __init__(self, frame_size=(640, 480)):
        self.frame_size = frame_size
        self.prev_wrist = deque(maxlen=8)
        self.last_swipe_time = 0
        self.swipe_cooldown = 0.6
        self.open_palm_since = None
        self.freeze_since = None
        self.pinch_since = None
        self.pinch_active = False
        self.open_palm_required = 1.0
        self.freeze_required = 0.3
        self.swipe_threshold = max(60, frame_size[0] * 0.08)
        # Smoothed outputs
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self.alpha = 0.25

    def update_frame_size(self, w, h):
        self.frame_size = (w, h)
        self.swipe_threshold = max(60, w * 0.08)

    @staticmethod
    def dist(a, b):
        return np.linalg.norm(a - b)

    def finger_count(self, lm):
        if lm is None:
            return 0
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        count = 0
        for t, p in zip(tips, pips):
            if lm[t][1] < lm[p][1] - 8:
                count += 1
        return count

    def is_pinch(self, lm):
        if lm is None:
            self.pinch_active = False
            return False, 0.0
        thumb = lm[4][:2]
        index = lm[8][:2]
        d = self.dist(thumb, index)
        is_pinched = d < max(30, self.frame_size[0] * 0.045)
        if is_pinched:
            if not self.pinch_active:
                self.pinch_since = time.time()
                self.pinch_active = True
        else:
            self.pinch_active = False
            self.pinch_since = None
        return is_pinched, d

    def is_open_palm(self, lm):
        if lm is None:
            self.open_palm_since = None
            return False
        tips = [8, 12, 16, 20]
        extended = 0
        for t in tips:
            if lm[t][1] < lm[t - 2][1] - 8:
                extended += 1
        is_open = extended >= 4
        now = time.time()
        if is_open:
            if self.open_palm_since is None:
                self.open_palm_since = now
            elif now - self.open_palm_since >= self.open_palm_required:
                return True
        else:
            self.open_palm_since = None
        return False

    def is_fist(self, lm):
        if lm is None:
            self.freeze_since = None
            return False
        wrist = lm[0][:2]
        tips = [4, 8, 12, 16, 20]
        close = 0
        for t in tips:
            if self.dist(lm[t][:2], wrist) < max(30, self.frame_size[0] * 0.06):
                close += 1
        now = time.time()
        is_fist = close >= 4
        if is_fist:
            if self.freeze_since is None:
                self.freeze_since = now
            elif now - self.freeze_since >= self.freeze_required:
                return True
        else:
            self.freeze_since = None
        return False

    def detect_swipe(self, lm):
        if lm is None:
            self.prev_wrist.append(None)
            return None
        wx = lm[0][0]
        t = time.time()
        self.prev_wrist.append((wx, t))
        pts = [p for p in self.prev_wrist if p is not None]
        if len(pts) >= 3:
            start_x, start_t = pts[0]
            end_x, end_t = pts[-1]
            dx = end_x - start_x
            dt = end_t - start_t
            if dt > 0 and abs(dx) > self.swipe_threshold and (time.time() - self.last_swipe_time) > self.swipe_cooldown:
                self.last_swipe_time = time.time()
                return 'LEFT' if dx < 0 else 'RIGHT'
        return None

    def interpret(self, lm):
        gesture = 'NONE'
        pinch, pinch_dist = self.is_pinch(lm)
        swipe = self.detect_swipe(lm)
        if swipe:
            gesture = f'SWIPE_{swipe}'
        if pinch:
            gesture = 'PINCH'
        if self.is_fist(lm):
            gesture = 'FREEZE'
        if self.is_open_palm(lm):
            gesture = 'RESET'

        # Rotation mapping: track index fingertip relative to center
        rot_yaw = 0.0
        rot_pitch = 0.0
        if lm is not None:
            idx = lm[8][:2]
            w, h = self.frame_size
            cx, cy = w / 2.0, h / 2.0
            dx = (idx[0] - cx) / w
            dy = (idx[1] - cy) / h
            rot_yaw = -dx * 180.0
            rot_pitch = -dy * 180.0

        # Apply exponential smoothing
        self.yaw = self.alpha * rot_yaw + (1 - self.alpha) * self.yaw
        self.pitch = self.alpha * rot_pitch + (1 - self.alpha) * self.pitch

        # Zoom mapping from pinch distance
        if pinch:
            # map larger pinch (small d) -> zoom in
            zoom_target = np.interp(pinch_dist, [20, self.frame_size[0] * 0.5], [2.0, 0.4])
        else:
            zoom_target = 1.0
        self.zoom = self.alpha * zoom_target + (1 - self.alpha) * self.zoom

        count = self.finger_count(lm)
        return {
            'gesture': gesture,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'zoom': self.zoom,
            'finger_count': count,
            'pinch_dist': pinch_dist if lm is not None else 0.0
        }


class OBJLoader:
    """Simple OBJ loader that returns vertices, normals and optionally uvs."""
    def __init__(self):
        pass

    def load(self, path):
        verts = []
        norms = []
        uvs = []
        faces = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vn':
                    norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    uvs.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    face = []
                    for v in parts[1:]:
                        vals = v.split('/')
                        vi = int(vals[0]) - 1 if vals[0] else None
                        ti = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                        ni = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                        face.append((vi, ti, ni))
                    faces.append(face)

        # Build flat arrays (triangulate quads/polygons)
        vertex_data = []
        normals_data = []
        uv_data = []
        for face in faces:
            # triangulate fan
            for i in range(1, len(face) - 1):
                for idx in (0, i, i + 1):
                    vi, ti, ni = face[idx]
                    v = verts[vi] if vi is not None else [0.0, 0.0, 0.0]
                    vertex_data.extend(v)
                    if ni is not None and ni < len(norms):
                        normals_data.extend(norms[ni])
                    else:
                        normals_data.extend([0.0, 0.0, 0.0])
                    if ti is not None and ti < len(uvs):
                        uv_data.extend(uvs[ti])
                    else:
                        uv_data.extend([0.0, 0.0])

        vertices = np.array(vertex_data, dtype=np.float32)
        normals = np.array(normals_data, dtype=np.float32) if normals_data else None
        uvs = np.array(uv_data, dtype=np.float32) if uv_data else None
        return vertices, normals, uvs


def create_cube():
    # positions and normals
    # 8 vertices, 36 indices (12 triangles)
    positions = [
        -1, -1, -1,
        1, -1, -1,
        1, 1, -1,
        -1, 1, -1,
        -1, -1, 1,
        1, -1, 1,
        1, 1, 1,
        -1, 1, 1,
    ]
    # indices for triangles
    indices = [
        0,1,2, 2,3,0,
        4,5,6, 6,7,4,
        0,4,7, 7,3,0,
        1,5,6, 6,2,1,
        3,2,6, 6,7,3,
        0,1,5, 5,4,0
    ]
    # generate normals per face in expanded vertex list
    vertex_data = []
    for i in range(0, len(indices), 3):
        ia, ib, ic = indices[i], indices[i+1], indices[i+2]
        a = np.array(positions[3*ia:3*ia+3])
        b = np.array(positions[3*ib:3*ib+3])
        c = np.array(positions[3*ic:3*ic+3])
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) != 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0.0, 0.0, 1.0])
        for idx in (ia, ib, ic):
            v = positions[3*idx:3*idx+3]
            vertex_data.extend(v)
            vertex_data.extend(normal.tolist())
    return np.array(vertex_data, dtype=np.float32)


def create_low_poly_human():
    # Very simple low-poly human: stacked boxes (head, torso, legs)
    # We'll create vertices for three cubes scaled and translated
    def cube_at(center, scale):
        cx, cy, cz = center
        sx, sy, sz = scale
        # create 8 corners
        corners = []
        for dx in (-sx/2, sx/2):
            for dy in (-sy/2, sy/2):
                for dz in (-sz/2, sz/2):
                    corners.append((cx+dx, cy+dy, cz+dz))
        # indices same as cube
        idx_map = list(range(len(corners)))
        positions = [coord for c in corners for coord in c]
        indices = [
            0,1,3, 3,2,0,
            4,5,7, 7,6,4,
            0,4,6, 6,2,0,
            1,5,7, 7,3,1,
            2,3,7, 7,6,2,
            0,1,5, 5,4,0
        ]
        vertex_data = []
        for i in range(0, len(indices), 3):
            ia, ib, ic = indices[i], indices[i+1], indices[i+2]
            a = np.array(positions[3*ia:3*ia+3])
            b = np.array(positions[3*ib:3*ib+3])
            c = np.array(positions[3*ic:3*ic+3])
            normal = np.cross(b - a, c - a)
            if np.linalg.norm(normal) != 0:
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0.0,0.0,1.0])
            for idx in (ia, ib, ic):
                v = positions[3*idx:3*idx+3]
                vertex_data.extend(v)
                vertex_data.extend(normal.tolist())
        return vertex_data

    data = []
    # torso
    data.extend(cube_at((0,0,0), (0.6,1.0,0.3)))
    # head
    data.extend(cube_at((0,0.9,0), (0.4,0.4,0.4)))
    # left leg
    data.extend(cube_at((-0.2,-1.0,0), (0.25,0.6,0.25)))
    # right leg
    data.extend(cube_at((0.2,-1.0,0), (0.25,0.6,0.25)))
    # left arm
    data.extend(cube_at((-0.6,0.1,0), (0.2,0.8,0.2)))
    # right arm
    data.extend(cube_at((0.6,0.1,0), (0.2,0.8,0.2)))
    return np.array(data, dtype=np.float32)


VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragPos;
out vec3 fragNormal;

void main() {
    fragPos = vec3(model * vec4(position, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(fragPos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 fragPos;
in vec3 fragNormal;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform vec3 lightColor;

void main() {
    float ambientStrength = 0.25;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""


class OpenGLRenderer:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
        self.width = width
        self.height = height
        self.program = None
        self.models = []  # list of dicts: {name, vao, vbo, vertex_count, color, mode}
        self.current_model = 0
        self.rotation = [0.0, 0.0, 0.0]
        self.zoom = 1.0
        self.lock = threading.Lock()
        self.init_done = False

    def init(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Gesture 3D Viewer")
        glEnable(GL_DEPTH_TEST)
        self.program = compileProgram(compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER),
                                      compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER))
        self.init_done = True

    def build_vao(self, vertex_array):
        # vertex_array is [x,y,z, nx,ny,nz, ...]
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        # normal
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(12))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao, vbo

    def add_model(self, name, vertex_array, color=(0.8, 0.5, 0.3), mode=GL_TRIANGLES):
        vao, vbo = self.build_vao(vertex_array)
        vertex_count = int(len(vertex_array) / 6)
        self.models.append({'name': name, 'vao': vao, 'vbo': vbo, 'count': vertex_count, 'color': color, 'mode': mode})

    def add_model_from_obj(self, name, vertices, normals=None, color=(0.7,0.7,0.7)):
        # vertices flattened xyz, normals flattened
        if normals is None:
            # build simple normals
            normals = np.zeros_like(vertices)
        # interleave
        verts = vertices.reshape(-1,3)
        norms = normals.reshape(-1,3) if normals is not None else np.zeros_like(verts)
        inter = np.empty((len(verts), 6), dtype=np.float32)
        inter[:,0:3] = verts
        inter[:,3:6] = norms
        arr = inter.flatten()
        self.add_model(name, arr, color=color)

    def set_current_model(self, idx):
        with self.lock:
            self.current_model = idx % len(self.models)

    def next_model(self):
        with self.lock:
            self.current_model = (self.current_model + 1) % len(self.models)

    def prev_model(self):
        with self.lock:
            self.current_model = (self.current_model - 1) % len(self.models)

    def set_transform(self, yaw, pitch, zoom):
        with self.lock:
            # Degrees
            self.rotation[1] = yaw
            self.rotation[0] = pitch
            self.zoom = np.clip(zoom, 0.2, 3.0)

    def reset_view(self):
        with self.lock:
            self.rotation = [0.0, 0.0, 0.0]
            self.zoom = 1.0

    def display(self):
        glClearColor(0.12, 0.12, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)

        # camera
        eye = Vector3([0.0, 0.0, 4.0 * self.zoom])
        center = Vector3([0.0, 0.0, 0.0])
        up = Vector3([0.0, 1.0, 0.0])
        view = Matrix44.look_at(eye, center, up)
        projection = Matrix44.perspective_projection(45.0, float(self.width)/self.height, 0.1, 100.0)

        # light and view
        light_pos = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        view_pos = np.array(eye, dtype=np.float32)

        loc_model = glGetUniformLocation(self.program, 'model')
        loc_view = glGetUniformLocation(self.program, 'view')
        loc_proj = glGetUniformLocation(self.program, 'projection')
        loc_light = glGetUniformLocation(self.program, 'lightPos')
        loc_viewpos = glGetUniformLocation(self.program, 'viewPos')
        loc_objcol = glGetUniformLocation(self.program, 'objectColor')
        loc_lightcol = glGetUniformLocation(self.program, 'lightColor')

        glUniformMatrix4fv(loc_view, 1, GL_FALSE, view.astype('float32').flatten())
        glUniformMatrix4fv(loc_proj, 1, GL_FALSE, projection.astype('float32').flatten())
        glUniform3fv(loc_light, 1, light_pos)
        glUniform3fv(loc_viewpos, 1, view_pos)
        glUniform3fv(loc_lightcol, 1, np.array([1.0,1.0,1.0], dtype=np.float32))

        with self.lock:
            model_info = self.models[self.current_model]
            yaw = self.rotation[1]
            pitch = self.rotation[0]
            zoom = self.zoom

        model = Matrix44.from_x_rotation(math.radians(pitch)) * Matrix44.from_y_rotation(math.radians(yaw)) * Matrix44.from_scale([0.8,0.8,0.8])

        glUniformMatrix4fv(loc_model, 1, GL_FALSE, model.astype('float32').flatten())
        glUniform3fv(loc_objcol, 1, np.array(model_info['color'], dtype=np.float32))

        glBindVertexArray(model_info['vao'])
        glDrawArrays(model_info['mode'], 0, model_info['count'])
        glBindVertexArray(0)

        glutSwapBuffers()

    def idle(self):
        glutPostRedisplay()


def main():
    # Paths and global data
    target_folder = r"C:\Users\Tuba Khan\Downloads\3D object"

    # Shared state
    shared = {
        'landmarks': None,
        'gesture': 'NONE',
        'yaw': 0.0,
        'pitch': 0.0,
        'zoom': 1.0,
        'model_index': 0,
        'freeze': False,
        'reset': False,
        'fps': 0.0,
        'frame': None
    }

    hand_tracker = HandTracker(max_num_hands=1, smoothing_window=6)
    gesture_interp = GestureInterpreter(frame_size=hand_tracker.frame_size)
    renderer = OpenGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Build models
    cube = create_cube()
    human = create_low_poly_human()

    # Initialize OpenGL (GLUT) and add models
    renderer.init()
    renderer.add_model('Cube', cube, color=(0.2, 0.6, 0.9))
    renderer.add_model('LowPolyHuman', human, color=(0.9, 0.6, 0.4))
    # Add a builtin teapot using GLUT as a fallback drawn via arrays? We'll emulate teapot by using GLUT directly when selected.

    # Additional support: attempt to load OBJ files in folder named model1.obj/model2.obj if present
    loader = OBJLoader()
    try:
        import os
        obj_candidates = [os.path.join(target_folder, n) for n in os.listdir(target_folder) if n.lower().endswith('.obj')]
        for i, p in enumerate(obj_candidates[:3]):
            verts, norms, uvs = loader.load(p)
            if verts is not None and len(verts) > 0:
                renderer.add_model(f'OBJ_{i}', verts, color=(0.7,0.7,0.7))
    except Exception:
        pass

    # Camera capture thread
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    running = True

    def capture_loop():
        nonlocal running
        last_t = time.time()
        frames = 0
        while running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            sm = hand_tracker.process(frame)
            gesture_interp.update_frame_size(*hand_tracker.frame_size)
            info = gesture_interp.interpret(sm)

            # Update shared state
            shared['landmarks'] = sm
            shared['gesture'] = info['gesture']
            shared['yaw'] = info['yaw']
            shared['pitch'] = info['pitch']
            shared['zoom'] = info['zoom']
            shared['finger_count'] = info['finger_count']
            shared['pinch_dist'] = info['pinch_dist']
            shared['frame'] = frame

            # Actions: switch model, reset, freeze
            if info['gesture'].startswith('SWIPE_'):
                if info['gesture'] == 'SWIPE_LEFT':
                    renderer.prev_model()
                else:
                    renderer.next_model()
            if info['gesture'] == 'RESET':
                renderer.reset_view()
            if info['gesture'] == 'FREEZE':
                shared['freeze'] = True
            else:
                shared['freeze'] = False

            # Update renderer transform if not frozen
            if not shared['freeze']:
                renderer.set_transform(shared['yaw'], shared['pitch'], shared['zoom'])

            # Overlay UI on frame
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            # Text lines
            lines = [f"Model: {renderer.models[renderer.current_model]['name']}",
                     f"Zoom: {shared['zoom']:.2f}",
                     f"Yaw: {shared['yaw']:.1f} Pitch: {shared['pitch']:.1f}",
                     f"Gesture: {shared['gesture']}"]
            y0 = 30
            for i, ln in enumerate(lines):
                cv2.putText(overlay, ln, (12, y0 + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Draw landmarks
            if sm is not None:
                for (x,y,z) in sm:
                    cv2.circle(overlay, (int(x), int(y)), 4, (0,255,0), -1)

            cv2.imshow('Camera Overlay', overlay)

            frames += 1
            if time.time() - last_t >= 1.0:
                shared['fps'] = frames / (time.time() - last_t)
                frames = 0
                last_t = time.time()

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    # Start capture thread
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # GLUT callbacks
    def display_cb():
        renderer.display()

    def idle_cb():
        renderer.idle()

    glutDisplayFunc(display_cb)
    glutIdleFunc(idle_cb)

    try:
        glutMainLoop()
    except Exception:
        # When GLUT main loop ends, stop capture
        pass

    # cleanup
    running = False


if __name__ == '__main__':
    main()
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLUT import *
from pyrr import Matrix44, Vector3
import sys
import math

# ------------------------- HandTracker -------------------------
class HandTracker:
    def __init__(self, max_num_hands=1, smoothing_window=5, detection_conf=0.6, tracking_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=max_num_hands,
                                         model_complexity=1,
                                         min_detection_confidence=detection_conf,
                                         min_tracking_confidence=tracking_conf)
        self.smoothing_window = smoothing_window
        self.landmark_history = deque(maxlen=smoothing_window)
        self.frame_size = (640, 480)

    def update_frame_size(self, w, h):
        self.frame_size = (w, h)

    def process(self, frame):
        # frame expected BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        landmarks = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # convert normalized to pixel
            h, w = frame.shape[:2]
            self.update_frame_size(w, h)
            pts = []
            for lm in hand.landmark:
                pts.append((lm.x * w, lm.y * h, lm.z * w))
            landmarks = np.array(pts)
            self.landmark_history.append(landmarks)
        else:
            # no hand: append None to maintain timing
            self.landmark_history.append(None)
        return self.get_smoothed_landmarks()

    def get_smoothed_landmarks(self):
        # moving average over history
        valid = [h for h in self.landmark_history if h is not None]
        if not valid:
            return None
        arr = np.stack(valid, axis=0)
        mean = np.nanmean(arr, axis=0)
        return mean

# ------------------------- GestureInterpreter -------------------------
class GestureInterpreter:
    def __init__(self, frame_size=(640,480)):
        self.frame_size = frame_size
        self.prev_wrist_x = deque(maxlen=8)
        self.prev_time = time.time()
        self.swipe_cooldown = 0.6
        self.last_swipe_time = 0
        self.open_palm_since = None
        self.freeze_since = None
        self.pinch_since = None
        self.pinch_active = False
        self.swipe_threshold = 80  # pixels
        self.open_palm_required = 1.0
        self.freeze_required = 0.3
        self.alpha = 0.35  # exponential smoothing for gestures

        # smoothing states
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.smoothed_zoom = 1.0

    def update_frame_size(self, w, h):
        self.frame_size = (w, h)

    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def finger_count(self, lm):
        # lm: Nx3 landmarks in pixel coords
        if lm is None:
            return 0
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        count = 0
        for t, p in zip(tips, pips):
            if lm[t][1] < lm[p][1] - 10:  # tip higher (y smaller)
                count += 1
        return count

    def is_pinch(self, lm):
        if lm is None:
            self.pinch_active = False
            return False, 0.0
        thumb = lm[4][:2]
        index = lm[8][:2]
        d = self.distance(thumb, index)
        diagonal = math.hypot(*self.frame_size)
        norm = d / diagonal
        is_pinched = d < max(30, self.frame_size[0]*0.05)
        if is_pinched:
            if not self.pinch_active:
                self.pinch_since = time.time()
                self.pinch_active = True
        else:
            self.pinch_active = False
            self.pinch_since = None
        return is_pinched, d

    def is_open_palm(self, lm):
        if lm is None:
            self.open_palm_since = None
            return False
        # check fingers extended and spread
        tips = [8, 12, 16, 20]
        wrist = lm[0][:2]
        extended = 0
        for t in tips:
            if lm[t][1] < lm[t-2][1] - 8:
                extended += 1
        is_open = extended >= 4
        now = time.time()
        if is_open:
            if self.open_palm_since is None:
                self.open_palm_since = now
            elif now - self.open_palm_since >= self.open_palm_required:
                return True
        else:
            self.open_palm_since = None
        return False

    def is_fist(self, lm):
        if lm is None:
            self.freeze_since = None
            return False
        # fist: tips close to wrist
        wrist = lm[0][:2]
        tips = [4,8,12,16,20]
        close = 0
        for t in tips:
            if self.distance(lm[t][:2], wrist) < max(30, self.frame_size[0]*0.06):
                close += 1
        now = time.time()
        is_fist = close >= 4
        if is_fist:
            if self.freeze_since is None:
                self.freeze_since = now
            elif now - self.freeze_since >= self.freeze_required:
                return True
        else:
            self.freeze_since = None
        return False

    def detect_swipe(self, lm):
        if lm is None:
            self.prev_wrist_x.append(None)
            return None
        wrist_x = lm[0][0]
        t = time.time()
        self.prev_wrist_x.append((wrist_x, t))
        # check start vs end
        if len(self.prev_wrist_x) >= 6:
            pts = [p for p in self.prev_wrist_x if p is not None]
            if len(pts) < 3:
                return None
            start_x, start_t = pts[0]
            end_x, end_t = pts[-1]
            dx = end_x - start_x
            dt = end_t - start_t
            if dt > 0 and abs(dx) > self.swipe_threshold and (time.time() - self.last_swipe_time) > self.swipe_cooldown:
                self.last_swipe_time = time.time()
                return 'LEFT' if dx < 0 else 'RIGHT'
        return None

    def interpret(self, lm):
        gesture = 'NONE'
        pinch, pinch_dist = self.is_pinch(lm)
        if pinch:
            gesture = 'PINCH'
        swipe = self.detect_swipe(lm)
        if swipe:
            gesture = f'SWIPE_{swipe}'
        if self.is_fist(lm):
            gesture = 'FREEZE'
        if self.is_open_palm(lm):
            gesture = 'RESET'
        count = self.finger_count(lm)
        return {
            'gesture': gesture,
            'pinch_dist': pinch_dist,
            'finger_count': count,
            'swipe': swipe
        }

# ------------------------- OBJLoader -------------------------
class OBJLoader:
    def __init__(self, obj_text=None):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        if obj_text:
            self.load_from_string(obj_text)

    def load_from_string(self, text):
        verts = []
        norms = []
        texs = []
        faces = []
        for line in text.splitlines():
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'vn':
                norms.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'vt':
                texs.append(tuple(map(float, parts[1:3])))
            elif parts[0] == 'f':
                face = []
                for v in parts[1:]:
                    vals = v.split('/')
                    v_idx = int(vals[0]) - 1 if vals[0] else None
                    vt_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    vn_idx = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                    face.append((v_idx, vt_idx, vn_idx))
                faces.append(face)
        # expand to triangles
        positions = []
        normals = []
        for f in faces:
            if len(f) < 3:
                continue
            # triangulate fan
            for i in range(1, len(f)-1):
                for idx in (0, i, i+1):
                    v_idx, vt_idx, vn_idx = f[idx]
                    positions.append(verts[v_idx])
                    if vn_idx is not None and vn_idx < len(norms):
                        normals.append(norms[vn_idx])
                    else:
                        normals.append((0.0, 0.0, 0.0))
        self.vertices = np.array(positions, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)

    def to_vbo(self):
        # interleave positions and normals
        if len(self.vertices) == 0:
            return np.array([], dtype=np.float32)
        data = np.hstack([self.vertices, self.normals])
        return data.flatten().astype(np.float32)

# ------------------------- Simple Models (embedded OBJ strings) -------------------------
CUBE_OBJ = '''
# Cube
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
vn 0 0 -1
vn 0 0 1
vn 0 -1 0
vn 0 1 0
vn -1 0 0
vn 1 0 0
f 1//1 2//1 3//1 4//1
f 5//2 8//2 7//2 6//2
f 1//3 5//3 6//3 2//3
f 2//6 6//6 7//6 3//6
f 3//4 7//4 8//4 4//4
f 5//5 1//5 4//5 8//5
'''

# Low-poly "teapot-like" pseudo-model (small dome)
TEAPOT_OBJ = '''
# Low-poly teapot-like dome
v 0 1 0
v -0.7 0.3 -0.7
v 0.7 0.3 -0.7
v 0.7 0.3 0.7
v -0.7 0.3 0.7
v 0 0.0 0
vn 0 1 0
f 1 2 3
f 1 3 4
f 1 4 5
f 1 5 2
f 2 6 3
f 3 6 4
f 4 6 5
f 5 6 2
'''

# Low-poly human (very simple)
HUMAN_OBJ = '''
# Low-poly human
v 0 1.2 0
v -0.2 0.6 0
v 0.2 0.6 0
v -0.3 -0.4 0
v 0.3 -0.4 0
v 0 0.3 0.2
v 0 0.3 -0.2
vn 0 0 1
f 1 2 3
f 2 4 5
f 3 5 4
f 2 6 3
f 3 7 2
'''

# ------------------------- OpenGLRenderer -------------------------
VERTEX_SHADER = '''
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 vNormal;
out vec3 vPos;
void main(){
    vNormal = mat3(transpose(inverse(model))) * normal;
    vPos = vec3(model * vec4(position,1.0));
    gl_Position = projection * view * model * vec4(position,1.0);
}
'''

FRAGMENT_SHADER = '''
#version 330
in vec3 vNormal;
in vec3 vPos;
out vec4 outColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
void main(){
    vec3 ambient = 0.25 * objectColor;
    vec3 norm = normalize(vNormal);
    vec3 lightDir = normalize(lightPos - vPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * objectColor;
    vec3 color = ambient + diffuse;
    outColor = vec4(color, 1.0);
}
'''

class Model:
    def __init__(self, name, vbo_data):
        self.name = name
        self.vbo_data = vbo_data
        self.vertex_count = 0
        self.vao = None
        self.vbo = None
        self.setup_buffers()

    def setup_buffers(self):
        if self.vbo_data.size == 0:
            return
        # each vertex: 3 pos + 3 normal
        self.vertex_count = int(len(self.vbo_data) // 6)
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vbo_data.nbytes, self.vbo_data, GL_STATIC_DRAW)
        # positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        # normals
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

class OpenGLRenderer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.program = None
        self.models = []
        self.current_model = 0
        self.model_matrix = Matrix44.identity()
        self.view_matrix = Matrix44.look_at(Vector3([0.0, 0.5, 4.0]), Vector3([0.0, 0.2, 0.0]), Vector3([0.0, 1.0, 0.0]))
        self.projection = Matrix44.perspective_projection(45.0, width/float(height), 0.1, 100.0)
        self.last_time = time.time()
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self.frozen = False
        self.setup_gl()

    def setup_gl(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(100, 100)
        self.window = glutCreateWindow(b"Gesture 3D Viewer - OpenGL")
        glEnable(GL_DEPTH_TEST)
        self.program = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    def add_model(self, model: Model):
        self.models.append(model)

    def set_transform(self, yaw, pitch, zoom):
        if not self.frozen:
            self.yaw = yaw
            self.pitch = pitch
            self.zoom = zoom

    def toggle_freeze(self, freeze=True):
        self.frozen = freeze

    def next_model(self):
        if not self.models:
            return
        self.current_model = (self.current_model + 1) % len(self.models)

    def prev_model(self):
        if not self.models:
            return
        self.current_model = (self.current_model - 1) % len(self.models)

    def reset_view(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0

    def display(self):
        glClearColor(0.1, 0.12, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        # camera/proj
        model = Matrix44.from_x_rotation(math.radians(self.pitch)) * Matrix44.from_y_rotation(math.radians(self.yaw)) * Matrix44.from_scale([self.zoom, self.zoom, self.zoom])
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'model'), 1, GL_FALSE, model.astype('float32'))
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'view'), 1, GL_FALSE, self.view_matrix.astype('float32'))
        glUniformMatrix4fv(glGetUniformLocation(self.program, 'projection'), 1, GL_FALSE, self.projection.astype('float32'))
        # lighting
        glUniform3f(glGetUniformLocation(self.program, 'lightPos'), 2.0, 4.0, 2.0)
        glUniform3f(glGetUniformLocation(self.program, 'viewPos'), 0.0, 0.5, 4.0)
        glUniform3f(glGetUniformLocation(self.program, 'objectColor'), 0.7, 0.5, 0.3)
        # draw model
        if self.models:
            model_obj = self.models[self.current_model]
            if model_obj.vao is not None:
                glBindVertexArray(model_obj.vao)
                glDrawArrays(GL_TRIANGLES, 0, model_obj.vertex_count)
                glBindVertexArray(0)
        glUseProgram(0)
        glutSwapBuffers()

# ------------------------- Main App Integration -------------------------
def main():
    # Setup camera and trackers
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    ret, frame = cap.read()
    if not ret:
        print('Camera read failed')
        return

    h, w = frame.shape[:2]

    hand_tracker = HandTracker(max_num_hands=1, smoothing_window=5)
    gesture = GestureInterpreter(frame_size=(w,h))

    # Setup renderer
    renderer = OpenGLRenderer(width=800, height=600)
    # Load models
    cube = OBJLoader(CUBE_OBJ)
    teapot = OBJLoader(TEAPOT_OBJ)
    human = OBJLoader(HUMAN_OBJ)
    m1 = Model('Cube', cube.to_vbo())
    m2 = Model('Teapot', teapot.to_vbo())
    m3 = Model('Human', human.to_vbo())
    renderer.add_model(m1)
    renderer.add_model(m2)
    renderer.add_model(m3)

    # Shared state for GLUT callback
    state = {
        'yaw': 0.0,
        'pitch': 0.0,
        'zoom': 1.0,
        'gesture_text': 'NONE',
        'model_name': renderer.models[renderer.current_model].name,
        'frozen': False
    }

    # Smoothing
    yaw_s = 0.0
    pitch_s = 0.0
    zoom_s = 1.0
    alpha = 0.18

    last_time = time.time()

    def idle():
        nonlocal yaw_s, pitch_s, zoom_s, last_time
        # read frame
        ret, frame = cap.read()
        if not ret:
            return
        small = cv2.flip(frame, 1)
        lm = hand_tracker.process(small)
        gesture.update_frame_size(small.shape[1], small.shape[0])
        interp = gesture.interpret(lm)
        g = interp['gesture']
        state['gesture_text'] = g
        # freeze logic
        if g == 'FREEZE':
            renderer.toggle_freeze(True)
            state['frozen'] = True
        elif g == 'RESET':
            renderer.reset_view()
            state['frozen'] = False
            renderer.toggle_freeze(False)
        else:
            if g != 'PINCH' and g != 'NONE':
                # handle swipes
                if interp['swipe'] == 'LEFT':
                    renderer.prev_model()
                    state['model_name'] = renderer.models[renderer.current_model].name
                elif interp['swipe'] == 'RIGHT':
                    renderer.next_model()
                    state['model_name'] = renderer.models[renderer.current_model].name
            if g != 'FREEZE':
                renderer.toggle_freeze(False)
                state['frozen'] = False
        # rotation from index finger
        if lm is not None:
            index = lm[8][:2]
            cx, cy = small.shape[1]/2, small.shape[0]/2
            dx = (index[0] - cx) / cx
            dy = (index[1] - cy) / cy
            target_yaw = dx * 90.0  # degrees
            target_pitch = -dy * 45.0
        else:
            target_yaw = renderer.yaw
            target_pitch = renderer.pitch
        # zoom from pinch
        is_pinched, pinch_dist = gesture.is_pinch(lm)
        if is_pinched:
            # map pinch distance to zoom
            diag = math.hypot(small.shape[1], small.shape[0])
            z = np.clip(1.5 - (pinch_dist / diag) * 3.0, 0.4, 2.5)
            target_zoom = z
        else:
            target_zoom = renderer.zoom
        # exponential smoothing
        yaw_s = alpha * target_yaw + (1 - alpha) * yaw_s
        pitch_s = alpha * target_pitch + (1 - alpha) * pitch_s
        zoom_s = alpha * target_zoom + (1 - alpha) * zoom_s
        renderer.set_transform(yaw_s, pitch_s, zoom_s)
        # overlay UI on camera feed
        overlay = small.copy()
        cv2.putText(overlay, f'Model: {state["model_name"]}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(overlay, f'Gesture: {state["gesture_text"]}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
        cv2.putText(overlay, f'Zoom: {renderer.zoom:.2f}', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,255,180), 2)
        cv2.putText(overlay, f'Yaw: {renderer.yaw:.1f}', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
        cv2.putText(overlay, f'Pitch: {renderer.pitch:.1f}', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
        fps = int(1.0 / max(1e-3, time.time() - last_time))
        last_time = time.time()
        cv2.putText(overlay, f'FPS: {fps}', (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,200), 2)
        # draw landmarks
        if lm is not None:
            for (x,y,z) in lm:
                cv2.circle(overlay, (int(x), int(y)), 3, (0,255,0), -1)
        cv2.imshow('Camera', overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # cleanup and exit
            cap.release()
            cv2.destroyAllWindows()
            glutDestroyWindow(renderer.window)
            sys.exit(0)
        # trigger redisplay
        glutPostRedisplay()

    # Register glut callbacks
    glutDisplayFunc(renderer.display)
    glutIdleFunc(idle)

    # Start
    print('Starting viewer. Press q in camera window to quit.')
    glutMainLoop()

if __name__ == '__main__':
    main()
