import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const root = document.querySelector("#viewer");
const loading = document.querySelector("#loading");
const loadingLabel = document.querySelector("#loading-label");
const loadingValue = document.querySelector("#loading-value");
const loadingProgress = document.querySelector("#loading-progress");
const enterButton = document.querySelector("#enter");
const errorPanel = document.querySelector("#error");
const retryButton = document.querySelector("#retry");
const lockState = document.querySelector("#lock-state");
const fpsLabel = document.querySelector("#fps");
const joystick = document.querySelector("#joystick");
const joystickKnob = document.querySelector("#joystick-knob");
const altitude = document.querySelector("#altitude");
const riseButton = document.querySelector("#rise");
const descendButton = document.querySelector("#descend");

const isTouchLayout = window.matchMedia(
  "(pointer: coarse), (max-width: 700px)",
).matches;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f12);

const camera = new THREE.PerspectiveCamera(55, 1, 0.005, 40);
camera.position.set(0.05, 0.24, 1.62);
camera.lookAt(0.058, -0.044, -0.069);
camera.rotation.order = "YXZ";

let yaw = camera.rotation.y;
let pitch = camera.rotation.x;
let ready = false;
let active = false;
let lookPointerId = null;
let lookX = 0;
let lookY = 0;
let verticalAxis = 0;
let joystickPointerId = null;
const movement = { forward: 0, strafe: 0 };
const keys = new Set();

const renderer = new THREE.WebGLRenderer({
  antialias: false,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setPixelRatio(
  Math.min(window.devicePixelRatio, isTouchLayout ? 1.25 : 1.75),
);
renderer.domElement.tabIndex = 0;
renderer.domElement.setAttribute(
  "aria-label",
  "Interactive Gaussian splat scene",
);
root.prepend(renderer.domElement);

const spark = new SparkRenderer({
  renderer,
  blurAmount: 0.3,
  sortRadial: false,
  focalAdjustment: 1.0,
});
scene.add(spark);

const clock = new THREE.Clock();
const moveVector = new THREE.Vector3();
const fpsWindow = [];

function setLoadProgress(event) {
  if (!event.lengthComputable || event.total <= 0) {
    loadingLabel.textContent = "Decoding scene";
    loadingValue.textContent = "SPZ";
    loadingProgress.style.width = "64%";
    return;
  }
  const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
  loadingValue.textContent = `${percent}%`;
  loadingProgress.style.width = `${percent}%`;
}

const splat = new SplatMesh({
  url: "./kitchen-7000.spz",
  onProgress: setLoadProgress,
});
scene.add(splat);

async function initialize() {
  try {
    await splat.initialized;
    ready = true;
    loading.hidden = true;
    enterButton.hidden = false;
    lockState.textContent = "READY";
    if (isTouchLayout) {
      joystick.hidden = false;
      altitude.hidden = false;
      enterButton.hidden = true;
      setActive(true);
    }
  } catch (error) {
    console.error(error);
    loading.hidden = true;
    enterButton.hidden = true;
    errorPanel.hidden = false;
    lockState.textContent = "ERROR";
  }
}

function setActive(nextActive) {
  active = nextActive && ready;
  root.classList.toggle("is-active", active);
  lockState.textContent = active ? "FLY" : "READY";
  enterButton.hidden = active || !ready || isTouchLayout;
}

function enterDesktop() {
  if (!ready || isTouchLayout) return;
  renderer.domElement.requestPointerLock();
}

enterButton.addEventListener("click", enterDesktop);
renderer.domElement.addEventListener("click", () => {
  if (ready && !active && !isTouchLayout) enterDesktop();
});

document.addEventListener("pointerlockchange", () => {
  setActive(document.pointerLockElement === renderer.domElement);
});

document.addEventListener("mousemove", (event) => {
  if (!active || isTouchLayout) return;
  yaw -= event.movementX * 0.0018;
  pitch -= event.movementY * 0.0018;
  pitch = THREE.MathUtils.clamp(pitch, -Math.PI * 0.48, Math.PI * 0.48);
});

window.addEventListener("keydown", (event) => {
  keys.add(event.code);
  if (
    ["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(
      event.code,
    )
  ) {
    event.preventDefault();
  }
});

window.addEventListener("keyup", (event) => keys.delete(event.code));
window.addEventListener("blur", () => keys.clear());

function keyboardAxis(positive, negative) {
  return Number(keys.has(positive)) - Number(keys.has(negative));
}

function updateCamera(delta) {
  camera.rotation.set(pitch, yaw, 0, "YXZ");
  const keyboardForward =
    keyboardAxis("KeyW", "KeyS") + keyboardAxis("ArrowUp", "ArrowDown");
  const keyboardStrafe =
    keyboardAxis("KeyD", "KeyA") + keyboardAxis("ArrowRight", "ArrowLeft");
  const keyboardVertical =
    Number(keys.has("Space") || keys.has("KeyE")) -
    Number(keys.has("KeyQ") || keys.has("KeyC"));
  const speed =
    (keys.has("ShiftLeft") || keys.has("ShiftRight") ? 0.82 : 0.28) * delta;

  moveVector.set(
    THREE.MathUtils.clamp(keyboardStrafe + movement.strafe, -1, 1),
    0,
    -THREE.MathUtils.clamp(keyboardForward + movement.forward, -1, 1),
  );
  if (moveVector.lengthSq() > 1) moveVector.normalize();
  moveVector.applyQuaternion(camera.quaternion).multiplyScalar(speed);
  camera.position.add(moveVector);
  camera.position.y +=
    THREE.MathUtils.clamp(keyboardVertical + verticalAxis, -1, 1) * speed;
}

function updateFps(now) {
  fpsWindow.push(now);
  while (fpsWindow.length && fpsWindow[0] < now - 1000) fpsWindow.shift();
  if (fpsWindow.length > 1) fpsLabel.textContent = `${fpsWindow.length} FPS`;
}

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function updateJoystick(event) {
  const bounds = joystick.getBoundingClientRect();
  const centerX = bounds.left + bounds.width / 2;
  const centerY = bounds.top + bounds.height / 2;
  const maxRadius = 34;
  let dx = event.clientX - centerX;
  let dy = event.clientY - centerY;
  const distance = Math.hypot(dx, dy);
  if (distance > maxRadius) {
    dx = (dx / distance) * maxRadius;
    dy = (dy / distance) * maxRadius;
  }
  joystickKnob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
  movement.strafe = dx / maxRadius;
  movement.forward = -dy / maxRadius;
}

joystick.addEventListener("pointerdown", (event) => {
  joystickPointerId = event.pointerId;
  joystick.setPointerCapture(event.pointerId);
  updateJoystick(event);
});
joystick.addEventListener("pointermove", (event) => {
  if (event.pointerId === joystickPointerId) updateJoystick(event);
});

function resetJoystick(event) {
  if (event.pointerId !== joystickPointerId) return;
  joystickPointerId = null;
  movement.forward = 0;
  movement.strafe = 0;
  joystickKnob.style.transform = "translate(-50%, -50%)";
}

joystick.addEventListener("pointerup", resetJoystick);
joystick.addEventListener("pointercancel", resetJoystick);

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (!isTouchLayout || !ready || event.clientX < window.innerWidth * 0.38)
    return;
  lookPointerId = event.pointerId;
  lookX = event.clientX;
  lookY = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (event.pointerId !== lookPointerId) return;
  const dx = event.clientX - lookX;
  const dy = event.clientY - lookY;
  lookX = event.clientX;
  lookY = event.clientY;
  yaw -= dx * 0.005;
  pitch -= dy * 0.005;
  pitch = THREE.MathUtils.clamp(pitch, -Math.PI * 0.48, Math.PI * 0.48);
});

function endLook(event) {
  if (event.pointerId === lookPointerId) lookPointerId = null;
}

renderer.domElement.addEventListener("pointerup", endLook);
renderer.domElement.addEventListener("pointercancel", endLook);

function bindAltitude(button, direction) {
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    verticalAxis = direction;
  });
  const stop = () => {
    if (verticalAxis === direction) verticalAxis = 0;
  };
  button.addEventListener("pointerup", stop);
  button.addEventListener("pointercancel", stop);
}

bindAltitude(riseButton, 1);
bindAltitude(descendButton, -1);

retryButton.addEventListener("click", () => window.location.reload());
window.addEventListener("resize", resize);

resize();
renderer.setAnimationLoop((time) => {
  const delta = Math.min(clock.getDelta(), 0.05);
  if (active) updateCamera(delta);
  renderer.render(scene, camera);
  updateFps(time);
});

initialize();
