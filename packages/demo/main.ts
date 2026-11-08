// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.
//
// Demo: CT forward projection (Radon transform) and ART reconstruction with
// interactive, frame-by-frame visualisation.
//
// Generates a phantom image, forward-projects it angle-by-angle to create a
// sinogram (with live canvas updates), then reconstructs the image using the
// ART algorithm — again with live canvas updates.
//
// Terminology follows the torch-radon convention:
//   forward()       → Radon transform  (image → sinogram)
//   backprojection() → uniform backfill (sinogram → image)
//
// Ported from tools/projection.py.

import { init, numpy } from "@jax-js/jax";
import { RadonFanbeam } from "../src/art.js";

const np = numpy;

// ── Scanner geometry: radians, pixel units, [angle, detector] sinograms ────────

const radon = new RadonFanbeam({
  resolution: 128,
  angles: Array.from({ length: 180 }, (_, index) => index * Math.PI / 90),
  source_distance: 872,
  det_distance: 194.66666666666666,
  det_count: 200,
  det_spacing: 0.8,
  ray_sampling: 2,
});

// ── Phantom Generation ───────────────────────────────────────────────────────

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
  data: Float32Array,
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


// ── Helpers ───────────────────────────────────────────────────────────────────

/** Read the current frame-delay value from the slider (ms). */
function getDelay(): number {
  return Number((document.getElementById("delaySlider") as HTMLInputElement).value);
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const status = document.getElementById("status")!;
  const progress = document.getElementById("progress") as HTMLProgressElement;
  const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
  const delaySlider = document.getElementById("delaySlider") as HTMLInputElement;
  const delayValue = document.getElementById("delayValue")!;

  // Keep the delay label in sync with the slider
  delaySlider.addEventListener("input", () => {
    delayValue.textContent = `${delaySlider.value} ms`;
  });

  // Initialize jax-js backend
  status.textContent = "Initializing jax-js…";
  await init();

  const N = radon.geometry.resolution;

  // Generate phantom
  status.textContent = "Generating phantom…";
  const phantomData = createPhantom(N);
  let phantomMax = 0;
  for (let i = 0; i < phantomData.length; i++) {
    if (phantomData[i] > phantomMax) phantomMax = phantomData[i];
  }
  renderToCanvas("phantom", phantomData, N, N);

  const num_angles = radon.angles.length;

  status.textContent = "Ready — click Start";
  startBtn.disabled = false;

  // ── Run button handler ────────────────────────────────────────────────────
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;

    // ── Phase 1: Forward projection (Radon transform), frame by frame ─────
    status.textContent = "Forward projection (angle 0)…";
    progress.value = 0;

    const phantomImg = np.array(phantomData).reshape([N, N]);
    const sinogram = await radon.forward(phantomImg, {
      on_progress: async (sinogram_data, angle_index) => {
        renderToCanvas("sinogram", sinogram_data, radon.geometry.det_count, num_angles);
        progress.value = ((angle_index + 1) / num_angles) * 50;
        status.textContent = `Forward projection: angle ${angle_index + 1} / ${num_angles}`;
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      },
      delay: getDelay(),
    });
    phantomImg.dispose();

    // ── Phase 2: ART reconstruction, frame by frame ───────────────────────
    status.textContent = "Reconstructing (ART)…";
    let iterCount = 0;

    const result = await radon.art(sinogram, {
      on_progress: async (image_data, _angle_index) => {
        iterCount++;
        if (iterCount % 5 === 0 || iterCount === num_angles) {
          renderToCanvas("reconstruction", image_data, N, N, phantomMax * 1.2);
          progress.value = 50 + (iterCount / num_angles) * 50;
          status.textContent = `Reconstruction: iteration ${iterCount} / ${num_angles}`;
          await new Promise<void>((resolve) => setTimeout(resolve, 0));
        }
      },
      delay: getDelay(),
    });
    sinogram.dispose();

    const resultData = await result.data();
    renderToCanvas("reconstruction", resultData as Float32Array, N, N, phantomMax * 1.2);
    progress.value = 100;
    status.textContent = "Done!";
    startBtn.disabled = false;
  });
}

main().catch((err: Error) => {
  console.error(err);
  document.getElementById("status")!.textContent = `Error: ${err.message}`;
});
