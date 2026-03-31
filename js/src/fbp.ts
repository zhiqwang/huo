// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.
//
// Filtered Back-Projection (FBP) for parallel beam CT geometry.
//
// Implements the classical FBP algorithm:
//   1. Apply the Ram-Lak (ramp) filter to each sinogram projection.
//   2. Back-project the filtered projections using parallel beam geometry.
//
// For the parallel beam forward projection (image → sinogram), reuse the
// existing `forward()` / `scan()` from art.ts with parallel beam gantry
// coordinates.
//
// Reference:
//   - https://astra-toolbox.com/docs/geom2d.html#parallel
//   - https://en.wikipedia.org/wiki/Radon_transform#Filtered_back-projection

import { numpy } from "@jax-js/jax";
import type { CTParam } from "./art.js";

const np = numpy;

// ── FFT utilities ────────────────────────────────────────────────────────────

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * The length of both arrays must be a power of two.
 *
 * @param re - Real parts (modified in place).
 * @param im - Imaginary parts (modified in place).
 */
function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len *= 2) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);

    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j;
        const b = i + j + len / 2;
        const tRe = curRe * re[b] - curIm * im[b];
        const tIm = curRe * im[b] + curIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }
}

/**
 * In-place inverse FFT (via conjugate-then-FFT trick).
 *
 * @param re - Real parts (modified in place).
 * @param im - Imaginary parts (modified in place).
 */
function ifft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  for (let i = 0; i < n; i++) {
    re[i] /= n;
    im[i] = -im[i] / n;
  }
}

// ── Ramp Filter ──────────────────────────────────────────────────────────────

/**
 * Apply the Ram-Lak (ramp) filter to every column of a sinogram.
 *
 * Each projection (column) is zero-padded to the next power of two,
 * multiplied by |freq| in the frequency domain, and transformed back.
 *
 * @param sinoData - Sinogram in row-major [detrNum, numAngles] layout.
 * @param detrNum - Number of detector elements (rows).
 * @param numAngles - Number of projection angles (columns).
 * @param detrLen - Detector panel length (mm).
 * @returns Filtered sinogram in the same layout.
 */
export function rampFilter(
  sinoData: Float32Array,
  detrNum: number,
  numAngles: number,
  detrLen: number,
): Float32Array {
  const detrStep = detrLen / detrNum;

  // Pad to next power of 2
  let padLen = 1;
  while (padLen < 2 * detrNum) padLen *= 2;

  // Build ramp filter: |k / (padLen * detrStep)|
  const ramp = new Float32Array(padLen);
  for (let k = 0; k < padLen; k++) {
    let freq: number;
    if (k <= padLen / 2) {
      freq = k / (padLen * detrStep);
    } else {
      freq = (k - padLen) / (padLen * detrStep);
    }
    ramp[k] = Math.abs(freq);
  }

  const result = new Float32Array(detrNum * numAngles);

  for (let a = 0; a < numAngles; a++) {
    // Copy and zero-pad this projection
    const re = new Float32Array(padLen);
    const im = new Float32Array(padLen);
    for (let j = 0; j < detrNum; j++) {
      re[j] = sinoData[j * numAngles + a];
    }

    // FFT → multiply by ramp → IFFT
    fft(re, im);
    for (let k = 0; k < padLen; k++) {
      re[k] *= ramp[k];
      im[k] *= ramp[k];
    }
    ifft(re, im);

    // Store back (only the first detrNum elements)
    for (let j = 0; j < detrNum; j++) {
      result[j * numAngles + a] = re[j];
    }
  }

  return result;
}

// ── Parallel Beam Back-Projection ────────────────────────────────────────────

/**
 * 1D linear interpolation for parallel beam back-projection.
 * Maps normalized positions in [-1, 1] to data indices [0, N-1].
 */
function linearInterp(
  data: Float32Array,
  N: number,
  positions: Float32Array,
  count: number,
): Float32Array {
  const result = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const pos = (positions[i] + 1) * 0.5 * (N - 1);
    const i0 = Math.floor(pos);
    const w = pos - i0;

    const ci0 = Math.max(0, Math.min(N - 1, i0));
    const ci1 = Math.max(0, Math.min(N - 1, i0 + 1));

    result[i] = data[ci0] * (1 - w) + data[ci1] * w;
  }

  return result;
}

/**
 * Parallel beam back-projection for a single angle.
 *
 * For each pixel (x, y) the detector position is:
 *   t = x · cos(θ) + y · sin(θ)
 *
 * This is simpler than fan-beam back-projection because all rays are
 * parallel (no magnification correction).
 *
 * @param sinogramData - Filtered sinogram column for one angle [detrNum].
 * @param imgEnd - Image coordinate boundary (half the FOV extent).
 * @param detEnd - Detector coordinate boundary (half the panel extent).
 * @param angle - Projection angle in degrees.
 * @param param - CT geometry parameters.
 * @returns Back-projected image [imgPixels, imgPixels].
 */
export async function parallelBackprojection(
  sinogramData: Float32Array,
  imgEnd: number,
  detEnd: number,
  angle: number,
  param: CTParam,
): Promise<numpy.Array> {
  const angleRad = (angle * Math.PI) / 180;
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);
  const P = param.imgPixels;

  // For each pixel, compute its detector coordinate
  const detPositions = new Float32Array(P * P);

  for (let row = 0; row < P; row++) {
    for (let col = 0; col < P; col++) {
      // Normalized pixel coordinates in [-1, 1]
      const x = -1 + (2 * col) / (P - 1);
      const y = -1 + (2 * row) / (P - 1);

      // Parallel beam: detector position = rotated x, scaled to detector space
      const rotX = cosA * x + sinA * y;
      detPositions[row * P + col] = (rotX * imgEnd) / detEnd;
    }
  }

  // Interpolate sinogram at computed detector positions
  const imgData = linearInterp(sinogramData, param.detrNum, detPositions, P * P);

  return np.array(imgData).reshape([P, P]);
}

// ── FBP Reconstruction ───────────────────────────────────────────────────────

/** Callback invoked after each angle during FBP reconstruction. */
export type FbpProgressCallback = (
  imgData: Float32Array,
  imgSize: number,
  angleIdx: number,
) => Promise<void>;

/**
 * Filtered Back-Projection (FBP) for parallel beam CT geometry.
 *
 * Applies the Ram-Lak (ramp) filter to each projection, then back-projects
 * all filtered projections using the parallel beam geometry.
 *
 * Reference: https://astra-toolbox.com/docs/geom2d.html#parallel
 *
 * @param sinogram - Measured sinogram [detrNum, numAngles].
 * @param imgEnd - Image coordinate boundary (half the FOV extent).
 * @param detEnd - Detector coordinate boundary (half the panel extent).
 * @param angles - Array of projection angles in degrees (typically [0, 180)).
 * @param param - CT geometry parameters.
 * @param onProgress - Optional async callback for visualisation.
 * @param delay - Minimum delay in ms between consecutive angles.
 * @returns Reconstructed image [imgPixels, imgPixels].
 */
export async function fbp(
  sinogram: numpy.Array,
  imgEnd: number,
  detEnd: number,
  angles: number[],
  param: CTParam,
  onProgress?: FbpProgressCallback,
  delay = 0,
): Promise<numpy.Array> {
  const numAngles = angles.length;
  const P = param.imgPixels;
  const angleStepRad = (param.rotateStep * Math.PI) / 180;

  // Read sinogram data
  const fullSinoData = await sinogram.ref.data();

  // Apply ramp filter
  const filteredSino = rampFilter(
    fullSinoData as Float32Array,
    param.detrNum,
    numAngles,
    param.detrLen,
  );

  // Initialize reconstruction to zeros
  const imgData = new Float32Array(P * P);

  // Back-project all filtered projections
  for (let i = 0; i < numAngles; i++) {
    const angle = angles[i];

    // Extract filtered sinogram column
    const sinoCol = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      sinoCol[j] = filteredSino[j * numAngles + i];
    }

    // Back-project
    const bp = await parallelBackprojection(sinoCol, imgEnd, detEnd, angle, param);
    const bpData = await bp.data();

    // Accumulate with angular weight
    for (let k = 0; k < P * P; k++) {
      imgData[k] += (bpData[k] as number) * angleStepRad;
    }

    if (onProgress) {
      await onProgress(imgData, P, i);
    }

    if (delay > 0) {
      await new Promise<void>((r) => setTimeout(r, delay));
    }
  }

  return np.array(imgData).reshape([P, P]);
}
