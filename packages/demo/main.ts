// Native autodiff remains in the operator; asynchronous rendering lives here.
import { defaultDevice, init, jit, numpy as np, tree } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector, project, reconstruct } from "../src/index.js";
import { createImageMetrics } from "./metrics.js";

const geometry = new FanbeamGeometry({
  resolution: 64,
  source_distance: 256,
  det_distance: 128,
  det_count: 80,
  det_spacing: 1.3,
  ray_sampling: 2,
});
const angles = Array.from({ length: 60 }, (_, index) => (index * Math.PI) / 30);

/**
 * Create a simplified Shepp-Logan-style phantom.
 *
 * @param size - Image size (pixels).
 * @returns Flattened phantom image data.
 */
function createPhantom(size: number): Float32Array {
  const data = new Float32Array(size * size);

  // Ellipse parameters: [centerX, centerY, semiAxisA, semiAxisB, value]
  // Pixel-centre coordinates relative to the image side, spanning (-0.5, 0.5).
  const ellipses: [number, number, number, number, number][] = [
    [0.0, 0.0, 0.45, 0.35, 1.0], // Outer body
    [0.0, -0.015, 0.41, 0.28, -0.8], // Inner cavity
    [0.22, 0.0, 0.12, 0.21, -0.2], // Right structure
    [-0.22, 0.0, 0.16, 0.21, -0.2], // Left structure
    [0.0, 0.25, 0.046, 0.046, 0.3], // Top feature
    [0.0, -0.25, 0.046, 0.046, 0.3], // Bottom feature
    [0.06, -0.08, 0.024, 0.03, 0.2], // Small detail 1
    [-0.06, -0.08, 0.024, 0.03, 0.2], // Small detail 2
  ];

  for (const [ecx, ecy, ea, eb, ev] of ellipses) {
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const dx = ((col + 0.5) / size - 0.5 - ecx) / ea;
        const dy = ((row + 0.5) / size - 0.5 - ecy) / eb;
        if (dx * dx + dy * dy <= 1) {
          data[row * size + col] += ev;
        }
      }
    }
  }

  // Clamp to non-negative
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.max(0, data[i]);
  }

  return data;
}

// ── Canvas Rendering ─────────────────────────────────────────────────────────

/**
 * Render a Float32Array as a grayscale image onto a canvas.
 *
 * @param canvasId - DOM ID of the canvas element.
 * @param data - Image data.
 * @param width - Image width.
 * @param height - Image height.
 * @param maxVal - Maximum value for normalization.
 */
function renderToCanvas(
  canvasId: string,
  data: Float32Array | Float64Array,
  width: number,
  height: number,
  maxVal?: number,
): void {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  canvas.width = width;
  canvas.height = height;
  const imageData = ctx.createImageData(width, height);

  if (maxVal == null) {
    maxVal = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > maxVal) maxVal = data[i];
    }
  }
  if (maxVal === 0) maxVal = 1;

  for (let i = 0; i < data.length; i++) {
    const v = Math.min(255, Math.max(0, Math.floor((data[i] / maxVal) * 255)));
    imageData.data[i * 4] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

type ImageErrorSample = { step: number; percent: number };

function renderImageError(
  canvas: HTMLCanvasElement,
  samples: ImageErrorSample[],
  steps: number,
  maxError: number,
): void {
  const ctx = canvas.getContext("2d")!;
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  if (samples.length === 0) return;

  const padding = 4;
  ctx.strokeStyle = getComputedStyle(canvas).color;
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let x = 0;
  let y = 0;
  for (const { step, percent } of samples) {
    x = padding + (width - 2 * padding) * (step / steps);
    y = padding + (height - 2 * padding) * (1 - percent / maxError);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fill();
}

async function main(): Promise<void> {
  const status = document.getElementById("status")!;
  const progress = document.getElementById("progress") as HTMLProgressElement;
  const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
  const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
  const iterations = document.getElementById("iterations") as HTMLSelectElement;
  const tvWeight = document.getElementById("tvWeight") as HTMLSelectElement;
  const delaySlider = document.getElementById("delaySlider") as HTMLInputElement;
  const delayValue = document.getElementById("delayValue")!;
  const imageError = document.getElementById("imageError") as HTMLOutputElement;
  const imageSsim = document.getElementById("imageSsim") as HTMLOutputElement;
  const imagePsnr = document.getElementById("imagePsnr") as HTMLOutputElement;
  const imageVariation = document.getElementById("imageVariation") as HTMLOutputElement;
  const imageErrorChart = document.getElementById("imageErrorChart") as HTMLCanvasElement;
  const imageErrorMax = document.getElementById("imageErrorMax")!;
  const imageErrorSteps = document.getElementById("imageErrorSteps")!;
  delaySlider.addEventListener("input", () => {
    delayValue.textContent = `${delaySlider.value} ms`;
  });
  status.textContent = "Initializing jax-js…";
  const available = await init("wasm", "webgpu");
  defaultDevice(available.includes("webgpu") ? "webgpu" : "wasm");
  const n = geometry.resolution;
  const phantom = createPhantom(n);
  const phantomMax = Math.max(...phantom);
  const measure = createImageMetrics(phantom, n);
  renderToCanvas("phantom", phantom, n, n);
  status.textContent = "Ready — click Start";
  startBtn.disabled = false;
  let controller: AbortController | undefined;
  stopBtn.addEventListener("click", () => controller?.abort());

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    iterations.disabled = true;
    tvWeight.disabled = true;
    const steps = Number(iterations.value);
    const tv_weight = Number(tvWeight.value);
    const errorSamples: ImageErrorSample[] = [];
    let maxError = 100;
    controller = new AbortController();
    const signal = controller.signal;
    const projector = new FanbeamProjector(geometry, angles);
    const forward = jit((image: np.Array, state: typeof projector.state) =>
      project(image, state, geometry.ray_step),
    );
    let measured: np.Array | undefined;
    progress.value = 0;
    for (const output of [imageError, imageSsim, imagePsnr, imageVariation]) output.value = "—";
    imageErrorMax.textContent = `${maxError}%`;
    imageErrorSteps.textContent = `Step ${steps}`;
    renderImageError(imageErrorChart, errorSamples, steps, maxError);
    renderToCanvas("reconstruction", new Float32Array(n * n), n, n);
    try {
      status.textContent = "Computing forward projection…";
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      signal.throwIfAborted();
      measured = forward(np.array(phantom).reshape([n, n]), tree.ref(projector.state));
      renderToCanvas(
        "sinogram",
        (await measured.ref.data()) as Float32Array,
        geometry.det_count,
        angles.length,
      );
      progress.value = 10;
      let losses = "";
      const result = await reconstruct(projector, measured, {
        steps,
        lr: 0.03,
        tv_weight,
        view_chunk_size: 8,
        report_every: 5,
        signal,
        on_progress: async ({ image, step, loss, objective }) => {
          losses = `projection loss ${loss.toPrecision(4)}`;
          if (tv_weight > 0) losses += ` · objective ${objective.toPrecision(4)}`;
          const { relativeL2: percent, psnr, ssim, flatVariation } = measure(image);
          errorSamples.push({ step, percent });
          maxError = Math.max(maxError, Math.ceil(percent / 10) * 10);
          imageError.value = `${percent.toFixed(1)}%`;
          imageSsim.value = ssim.toFixed(4);
          imagePsnr.value = `${psnr === Infinity ? "∞" : psnr.toFixed(2)} dB`;
          imageVariation.value = flatVariation === undefined ? "—" : `${flatVariation.toFixed(2)}%`;
          imageErrorMax.textContent = `${maxError}%`;
          renderImageError(imageErrorChart, errorSamples, steps, maxError);
          renderToCanvas("reconstruction", image, n, n, phantomMax);
          progress.value = 10 + (90 * step) / steps;
          status.textContent = `Adam step ${step} / ${steps} · ${losses}`;
          await new Promise<void>((resolve) => setTimeout(resolve, Number(delaySlider.value)));
        },
      });
      result.dispose();
      status.textContent = `Done · ${losses}`;
    } catch (error) {
      if (signal.aborted) status.textContent = "Stopped — click Start to run again";
      else {
        console.error(error);
        status.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
      }
    } finally {
      measured?.dispose();
      forward.dispose();
      projector.dispose();
      controller = undefined;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      iterations.disabled = false;
      tvWeight.disabled = false;
    }
  });
}

main().catch((error: Error) => {
  console.error(error);
  document.getElementById("status")!.textContent = `Error: ${error.message}`;
});
