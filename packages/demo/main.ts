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
//   backprojection() → inverse Radon   (sinogram → image)
//
// Ported from tools/projection.py.

import { init, numpy } from "@jax-js/jax";
import { art, scan, type CTParam } from "../src/art.js";

const np = numpy;

// ── CT Scanner Parameters (matching the Python defaults) ─────────────────────

const param: CTParam = {
  imgPixels: 128, // Reduced from 512 for interactive performance
  imgLen: 144, // Diameter of the FOV (mm)
  detrNum: 200, // Number of detector elements (det_count)
  detrLen: 180, // Detector panel length (mm)
  latSampling: 2, // Lateral sampling grid multiplier
  sdd: 1200, // Source-to-detector distance (mm)
  sod: 981, // Source-to-object distance (mm)
  rotateStep: 2, // Rotation step (degrees); 360/2 = 180 angles
};

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
  // Coordinates are relative to image center, normalized to [0, 1].
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
        const dx = (col / size - 0.5 - ecx) / ea;
        const dy = (row / size - 0.5 - ecy) / eb;
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

// ── Gantry Coordinate Computation (ported from tools/projection.py) ──────────

interface GantryCoordinates {
  gantryCoordX: numpy.Array;
  gantryCoordY: numpy.Array;
  imgEnd: number;
  detrEnd: number;
  angles: number[];
}

/**
 * Compute gantry ray-tracing coordinates for fan-beam CT geometry.
 *
 * Sets up the fan-beam source / detector geometry that is passed to both the
 * forward projection (Radon transform) and back-projection steps.
 *
 * @param param - CT geometry parameters.
 * @returns Computed gantry coordinates and derived quantities.
 */
function computeGantryCoordinates(param: CTParam): GantryCoordinates {
  const imgStep = param.imgLen / param.imgPixels;
  const imgEnd = (param.imgLen - imgStep) / 2;
  const detrStep = param.detrLen / param.detrNum;
  const detrEnd = (param.detrLen - detrStep) / 2;

  // View angles
  const angles: number[] = [];
  for (let a = 0; a < 360; a += param.rotateStep) {
    angles.push(a);
  }

  // X-ray source is on the top vertical axis at (0, -sod)
  const srcCoordY = -param.sod;

  // Detector coordinates
  const detrCoordX = new Float32Array(param.detrNum);
  const detrCoordY = new Float32Array(param.detrNum);
  for (let i = 0; i < param.detrNum; i++) {
    detrCoordX[i] = -detrEnd + (i * (2 * detrEnd)) / (param.detrNum - 1);
    detrCoordY[i] = param.sdd - param.sod;
  }

  // Lateral grid along each ray
  const latEnd = param.imgLen / 2;
  const latSteps = param.latSampling * param.imgPixels + 1;
  const latGrid = new Float32Array(latSteps);
  for (let i = 0; i < latSteps; i++) {
    latGrid[i] = -latEnd + (i * (2 * latEnd)) / (latSteps - 1);
    latGrid[i] -= srcCoordY; // Move origin up to X-ray source
  }

  // Adjust detector Y coordinates
  for (let i = 0; i < param.detrNum; i++) {
    detrCoordY[i] -= srcCoordY;
  }

  // Fan-beam distance for each detector element
  const fand = new Float32Array(param.detrNum);
  for (let i = 0; i < param.detrNum; i++) {
    fand[i] = Math.sqrt(detrCoordX[i] ** 2 + detrCoordY[i] ** 2);
  }

  // Gantry coordinates [latSteps, detrNum]
  const gantryCoordXData = new Float32Array(latSteps * param.detrNum);
  const gantryCoordYData = new Float32Array(latSteps * param.detrNum);
  for (let k = 0; k < latSteps; k++) {
    for (let j = 0; j < param.detrNum; j++) {
      gantryCoordXData[k * param.detrNum + j] = (latGrid[k] * detrCoordX[j]) / fand[j];
      gantryCoordYData[k * param.detrNum + j] =
        (latGrid[k] * detrCoordY[j]) / fand[j] + srcCoordY;
    }
  }

  // Normalize to [-1, 1] by image extent
  for (let i = 0; i < gantryCoordXData.length; i++) {
    gantryCoordXData[i] /= imgEnd;
    gantryCoordYData[i] /= imgEnd;
  }

  const gantryCoordX = np.array(gantryCoordXData).reshape([latSteps, param.detrNum]);
  const gantryCoordY = np.array(gantryCoordYData).reshape([latSteps, param.detrNum]);

  return { gantryCoordX, gantryCoordY, imgEnd, detrEnd, angles };
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

  const P = param.imgPixels;

  // Generate phantom
  status.textContent = "Generating phantom…";
  const phantomData = createPhantom(P);
  let phantomMax = 0;
  for (let i = 0; i < phantomData.length; i++) {
    if (phantomData[i] > phantomMax) phantomMax = phantomData[i];
  }
  renderToCanvas("phantom", phantomData, P, P);

  // Compute gantry geometry
  status.textContent = "Computing fan-beam geometry…";
  const { gantryCoordX, gantryCoordY, imgEnd, detrEnd, angles } =
    computeGantryCoordinates(param);
  const numAngles = angles.length;

  status.textContent = "Ready — click Start";
  startBtn.disabled = false;

  // ── Run button handler ────────────────────────────────────────────────────
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;

    // ── Phase 1: Forward projection (Radon transform), frame by frame ─────
    status.textContent = "Forward projection (angle 0)…";
    progress.value = 0;

    const phantomImg = np.array(phantomData).reshape([P, P]);
    const sinogram = await scan(
      phantomImg,
      gantryCoordX,
      gantryCoordY,
      angles,
      param,
      // onProgress — render the sinogram as it is being built
      async (sinoData, detCount, numAnglesTotal, angleIdx) => {
        renderToCanvas("sinogram", sinoData, numAnglesTotal, detCount);
        progress.value = ((angleIdx + 1) / numAnglesTotal) * 50; // first half
        status.textContent = `Forward projection: angle ${angleIdx + 1} / ${numAnglesTotal}`;
        // Yield to the browser for rendering
        await new Promise<void>((r) => setTimeout(r, 0));
      },
      getDelay(),
    );
    phantomImg.dispose();

    // ── Phase 2: ART reconstruction, frame by frame ───────────────────────
    status.textContent = "Reconstructing (ART)…";
    let iterCount = 0;

    const result = await art(
      sinogram,
      imgEnd,
      detrEnd,
      gantryCoordX,
      gantryCoordY,
      angles,
      param,
      async (imgData, size, _angleIdx) => {
        iterCount++;
        // Update canvas periodically to show reconstruction progress
        if (iterCount % 5 === 0 || iterCount === numAngles) {
          renderToCanvas("reconstruction", imgData, size, size, phantomMax * 1.2);
          progress.value = 50 + (iterCount / numAngles) * 50; // second half
          status.textContent = `Reconstruction: iteration ${iterCount} / ${numAngles}`;
          // Yield to the browser for rendering
          await new Promise<void>((r) => setTimeout(r, 0));
        }
      },
      getDelay(),
    );

    const resultData = await result.data();
    renderToCanvas("reconstruction", resultData as Float32Array, P, P, phantomMax * 1.2);
    progress.value = 100;
    status.textContent = "Done!";
    startBtn.disabled = false;
  });
}

main().catch((err: Error) => {
  console.error(err);
  document.getElementById("status")!.textContent = `Error: ${err.message}`;
});
