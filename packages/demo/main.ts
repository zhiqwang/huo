// Native autodiff remains in the operator; asynchronous rendering lives here.
import { defaultDevice, init, jit, numpy as np, tree } from "@jax-js/jax";
import { FanbeamGeometry, FanbeamProjector, project, reconstruct } from "../src/index.js";

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

async function main(): Promise<void> {
  const status = document.getElementById("status")!;
  const progress = document.getElementById("progress") as HTMLProgressElement;
  const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
  const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
  const delaySlider = document.getElementById("delaySlider") as HTMLInputElement;
  const delayValue = document.getElementById("delayValue")!;
  delaySlider.addEventListener("input", () => {
    delayValue.textContent = `${delaySlider.value} ms`;
  });
  status.textContent = "Initializing jax-js…";
  const available = await init("wasm", "webgpu");
  defaultDevice(available.includes("webgpu") ? "webgpu" : "wasm");
  const n = geometry.resolution;
  const phantom = createPhantom(n);
  const phantomMax = Math.max(...phantom);
  renderToCanvas("phantom", phantom, n, n);
  status.textContent = "Ready — click Start";
  startBtn.disabled = false;
  let controller: AbortController | undefined;
  stopBtn.addEventListener("click", () => controller?.abort());

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    controller = new AbortController();
    const signal = controller.signal;
    const projector = new FanbeamProjector(geometry, angles);
    const forward = jit((image: np.Array, state: typeof projector.state) =>
      project(image, state, geometry.ray_step),
    );
    let measured: np.Array | undefined;
    progress.value = 0;
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
      let final_loss = 0;
      const result = await reconstruct(projector, measured, {
        steps: 120,
        lr: 0.03,
        view_chunk_size: 8,
        report_every: 5,
        signal,
        on_progress: async ({ image, step, steps, loss }) => {
          final_loss = loss;
          renderToCanvas("reconstruction", image, n, n, phantomMax);
          progress.value = 10 + (90 * step) / steps;
          status.textContent = `Adam step ${step} / ${steps} · loss ${loss.toPrecision(4)}`;
          await new Promise<void>((resolve) => setTimeout(resolve, Number(delaySlider.value)));
        },
      });
      result.dispose();
      status.textContent = `Done · projection loss ${final_loss.toPrecision(4)}`;
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
    }
  });
}

main().catch((error: Error) => {
  console.error(error);
  document.getElementById("status")!.textContent = `Error: ${error.message}`;
});
