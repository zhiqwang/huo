// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.
//
// Algebraic Reconstruction Technique (ART) for CT image reconstruction.
// Ported from PyTorch (huo/art.py) to jax-js.
//
// Terminology follows the torch-radon convention:
//   - forward()        : Radon transform (image → sinogram), one angle at a time
//   - backprojection()  : back-projection (sinogram → image), one angle at a time
//   - scan()           : full forward projection over all angles (image → complete sinogram)
//   - art()            : iterative ART reconstruction (sinogram → image)
//
// References:
//   - https://torch-radon.readthedocs.io/en/latest/
//   - https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique
//   - https://github.com/ekzhang/jax-js

import { numpy } from "@jax-js/jax";

const np = numpy;

/** CT geometry parameters used by the forward / back-projection routines.
 *
 * Immutable configuration inspired by ``RaysCfg`` in
 * [torch-radon](https://github.com/matteo-ronchetti/torch-radon).
 */
export class RaysCfg {
  /** Number of image pixels along each axis. */
  readonly imgPixels: number;
  /** Diameter of the field of view (mm). */
  readonly imgLen: number;
  /** Number of detector elements. */
  readonly detrNum: number;
  /** Detector panel length (mm). */
  readonly detrLen: number;
  /** Lateral sampling grid multiplier. */
  readonly latSampling: number;
  /** Source-to-detector distance (mm). */
  readonly sdd: number;
  /** Source-to-object distance (mm). */
  readonly sod: number;
  /** Rotation step in degrees. */
  readonly rotateStep: number;

  constructor(cfg: {
    imgPixels: number;
    imgLen: number;
    detrNum: number;
    detrLen: number;
    latSampling: number;
    sdd: number;
    sod: number;
    rotateStep: number;
  }) {
    this.imgPixels = cfg.imgPixels;
    this.imgLen = cfg.imgLen;
    this.detrNum = cfg.detrNum;
    this.detrLen = cfg.detrLen;
    this.latSampling = cfg.latSampling;
    this.sdd = cfg.sdd;
    this.sod = cfg.sod;
    this.rotateStep = cfg.rotateStep;
  }
}

/**
 * Bilinear interpolation on a 2D image.
 * Equivalent to PyTorch's F.grid_sample with align_corners=True.
 *
 * @param imgData - Flattened image data of size H*W.
 * @param H - Image height.
 * @param W - Image width.
 * @param gridX - X coordinates in [-1, 1] (width axis).
 * @param gridY - Y coordinates in [-1, 1] (height axis).
 * @returns Interpolated values with the same length as gridX.
 */
function bilinearSample(
  imgData: Float32Array,
  H: number,
  W: number,
  gridX: Float32Array,
  gridY: Float32Array,
): Float32Array {
  const N = gridX.length;
  const result = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    // Convert normalized [-1, 1] to pixel coordinates
    const px = (gridX[i] + 1) * 0.5 * (W - 1);
    const py = (gridY[i] + 1) * 0.5 * (H - 1);

    const x0 = Math.floor(px);
    const y0 = Math.floor(py);

    const wx = px - x0;
    const wy = py - y0;

    // Clamp indices to valid range
    const cx0 = Math.max(0, Math.min(W - 1, x0));
    const cy0 = Math.max(0, Math.min(H - 1, y0));
    const cx1 = Math.max(0, Math.min(W - 1, x0 + 1));
    const cy1 = Math.max(0, Math.min(H - 1, y0 + 1));

    result[i] =
      imgData[cy0 * W + cx0] * (1 - wy) * (1 - wx) +
      imgData[cy0 * W + cx1] * (1 - wy) * wx +
      imgData[cy1 * W + cx0] * wy * (1 - wx) +
      imgData[cy1 * W + cx1] * wy * wx;
  }

  return result;
}

/**
 * 1D linear interpolation for back-projection.
 * Maps normalized positions in [-1, 1] to data indices [0, N-1].
 *
 * @param data - 1D data array to interpolate.
 * @param N - Length of data.
 * @param positions - Normalized positions in [-1, 1].
 * @param count - Number of positions.
 * @returns Interpolated values.
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
 * Fisher-Yates shuffle to generate a random permutation of [0, n).
 *
 * @param n - Length of the permutation.
 * @returns Shuffled indices.
 */
function randperm(n: number): number[] {
  const arr = globalThis.Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Forward projection (Radon transform) for a single angle.
 *
 * Computes the line integrals of the image along the ray paths defined by the
 * fan-beam geometry at the given view angle. The gantry coordinates are rotated,
 * the image is sampled via bilinear interpolation along each ray, and the
 * integrals are summed across the lateral sampling direction to produce one
 * column of the sinogram.
 *
 * Corresponds to `Radon.forward()` / `RadonFanbeam.forward()` in torch-radon,
 * restricted to a single projection angle.
 *
 * @param img - 2D volume image [imgPixels, imgPixels].
 * @param gantryCoordX - Gantry X coordinates [latSteps, detCount].
 * @param gantryCoordY - Gantry Y coordinates [latSteps, detCount].
 * @param angle - Projection angle in degrees.
 * @param param - CT geometry parameters.
 * @returns Sinogram column for this angle [detCount].
 */
export async function forward(
  img: numpy.Array,
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angle: number,
  param: RaysCfg,
): Promise<numpy.Array> {
  const angleRad = (angle * Math.PI) / 180;
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);

  // Read raw data from jax-js arrays (.ref keeps the originals alive)
  const [imgData, gxData, gyData] = await Promise.all([
    img.ref.data(),
    gantryCoordX.ref.data(),
    gantryCoordY.ref.data(),
  ]);

  // Rotate gantry coordinates counter-clockwise by the projection angle
  const N = gxData.length;
  const rotX = new Float32Array(N);
  const rotY = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    rotX[i] = (gxData[i] as number) * cosA - (gyData[i] as number) * sinA;
    rotY[i] = (gxData[i] as number) * sinA + (gyData[i] as number) * cosA;
  }

  // Sample image at rotated coordinates via bilinear interpolation
  const interp = bilinearSample(
    imgData as Float32Array,
    param.imgPixels,
    param.imgPixels,
    rotX,
    rotY,
  );

  // Sum along the lateral (ray) direction to compute line integrals
  const latSteps = param.latSampling * param.imgPixels + 1;
  const latStep = param.imgLen / param.imgPixels / param.latSampling;
  const sino = new Float32Array(param.detrNum);

  for (let j = 0; j < param.detrNum; j++) {
    let sum = 0;
    for (let k = 0; k < latSteps; k++) {
      sum += interp[k * param.detrNum + j];
    }
    sino[j] = sum * latStep;
  }

  return np.array(sino);
}

/**
 * Back-projection for a single angle.
 *
 * For each pixel in the reconstruction grid, computes which detector element
 * it maps to under the given projection angle (fan-beam geometry), then
 * interpolates the sinogram value at that detector position and distributes
 * it back into the image.
 *
 * Corresponds to `Radon.backprojection()` in torch-radon, restricted to a
 * single projection angle.
 *
 * @param sinogramData - Sinogram column for one angle [detCount].
 * @param imgEnd - Image coordinate boundary (half the FOV extent).
 * @param detEnd - Detector coordinate boundary (half the panel extent).
 * @param angle - Projection angle in degrees.
 * @param param - CT geometry parameters.
 * @returns Back-projected image [imgPixels, imgPixels].
 */
export async function backprojection(
  sinogramData: Float32Array,
  imgEnd: number,
  detEnd: number,
  angle: number,
  param: RaysCfg,
): Promise<numpy.Array> {
  const angleRad = (angle * Math.PI) / 180;
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);
  const P = param.imgPixels;

  // For each image pixel, compute its detector coordinate after rotation.
  //
  // Rotation matrix:
  //   rotX = cos*x + sin*y
  //   rotY = -sin*x + cos*y
  //
  // Fan-beam detector mapping (normalized to [-1, 1]):
  //   detCoord = sdd * rotX / (rotY + sod / imgEnd) / detEnd
  const detPositions = new Float32Array(P * P);

  for (let row = 0; row < P; row++) {
    for (let col = 0; col < P; col++) {
      // Normalized pixel coordinates in [-1, 1]
      const x = -1 + (2 * col) / (P - 1);
      const y = -1 + (2 * row) / (P - 1);

      const rotX = cosA * x + sinA * y;
      const rotY = -sinA * x + cosA * y;

      const adjustedY = rotY + param.sod / imgEnd;
      detPositions[row * P + col] = (param.sdd * rotX) / adjustedY / detEnd;
    }
  }

  // Interpolate sinogram at computed detector positions
  const imgData = linearInterp(sinogramData, param.detrNum, detPositions, P * P);

  return np.array(imgData).reshape([P, P]);
}

/** Callback invoked after each angle during forward projection (scan). */
export type ScanProgressCallback = (
  sinogramData: Float32Array,
  detCount: number,
  numAngles: number,
  angleIdx: number,
) => Promise<void>;

/** Callback invoked after each iteration during ART reconstruction. */
export type ArtProgressCallback = (
  imgData: Float32Array,
  imgSize: number,
  angleIdx: number,
) => Promise<void>;

/**
 * Full forward projection (Radon transform) over all angles.
 *
 * Generates a complete sinogram by running `forward()` at each projection
 * angle.  Corresponds to calling `RadonFanbeam.forward(image)` in torch-radon.
 *
 * @param img - Input volume image [imgPixels, imgPixels].
 * @param gantryCoordX - Gantry X coordinates [latSteps, detCount].
 * @param gantryCoordY - Gantry Y coordinates [latSteps, detCount].
 * @param angles - Array of projection angles in degrees.
 * @param param - CT geometry parameters.
 * @param onProgress - Optional async callback invoked after each angle.
 * @param delay - Minimum delay in ms between consecutive angles.
 *   Useful for slowing down the visualisation so each projection frame is visible.
 * @returns Sinogram [detCount, numAngles].
 */
export async function scan(
  img: numpy.Array,
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angles: number[],
  param: RaysCfg,
  onProgress?: ScanProgressCallback,
  delay = 0,
): Promise<numpy.Array> {
  const numAngles = angles.length;
  // Store sinogram in [detCount, numAngles] layout (one column per angle)
  const sinoData = new Float32Array(param.detrNum * numAngles);

  for (let i = 0; i < numAngles; i++) {
    const sino = await forward(img, gantryCoordX, gantryCoordY, angles[i], param);
    // data() returns the typed array and disposes the jax-js array
    const data = await sino.data();
    for (let j = 0; j < param.detrNum; j++) {
      sinoData[j * numAngles + i] = data[j] as number;
    }

    if (onProgress) {
      await onProgress(sinoData, param.detrNum, numAngles, i);
    }

    if (delay > 0) {
      await new Promise<void>((r) => setTimeout(r, delay));
    }
  }

  return np.array(sinoData).reshape([param.detrNum, numAngles]);
}

/**
 * Algebraic Reconstruction Technique (ART).
 *
 * Iteratively reconstructs a CT image from sinogram data by processing one
 * projection angle at a time in random order (Kaczmarz method).  For each angle:
 *   1. Forward-project the current estimate to predict the sinogram column.
 *   2. Compute the residual (measured − predicted).
 *   3. Back-project the residual to update the image estimate.
 *   4. Clamp negative values to zero.
 *
 * This is a classical iterative CT reconstruction algorithm.  See also:
 *   - https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique
 *   - torch-radon documentation for forward / backprojection terminology.
 *
 * @param sinogram - Measured sinogram [detCount, numAngles].
 * @param imgEnd - Image coordinate boundary (half the FOV extent).
 * @param detEnd - Detector coordinate boundary (half the panel extent).
 * @param gantryCoordX - Gantry X coordinates [latSteps, detCount].
 * @param gantryCoordY - Gantry Y coordinates [latSteps, detCount].
 * @param angles - Array of projection angles in degrees.
 * @param param - CT geometry parameters.
 * @param onProgress - Optional async callback for visualisation.
 * @param delay - Minimum delay in ms between consecutive angles.
 *   Useful for slowing down the visualisation so each reconstruction frame is visible.
 * @returns Reconstructed image [imgPixels, imgPixels].
 */
export async function art(
  sinogram: numpy.Array,
  imgEnd: number,
  detEnd: number,
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angles: number[],
  param: RaysCfg,
  onProgress?: ArtProgressCallback,
  delay = 0,
): Promise<numpy.Array> {
  const numAngles = angles.length;
  const P = param.imgPixels;

  // Initialize reconstruction to zeros
  const imgData = new Float32Array(P * P);

  // Read sinogram data (.ref keeps the original alive)
  const fullSinoData = await sinogram.ref.data();

  // Process angles in random order (Kaczmarz method)
  const indices = randperm(numAngles);

  for (const idx of indices) {
    const angle = angles[idx];

    // Create jax-js array for current image estimate
    const img = np.array(imgData).reshape([P, P]);

    // Forward-project current estimate
    const fp = await forward(img, gantryCoordX, gantryCoordY, angle, param);
    const fpData = await fp.data();

    // Extract measured sinogram column for this angle
    const sinoCol = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      sinoCol[j] = fullSinoData[j * numAngles + idx] as number;
    }

    // Compute residual: (measured − predicted) / imgLen
    const resData = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      resData[j] = (sinoCol[j] - (fpData[j] as number)) / param.imgLen;
    }

    // Back-project residual into image space
    const bp = await backprojection(resData, imgEnd, detEnd, angle, param);
    const bpData = await bp.data();

    // Update image estimate: img += back-projection, then clamp to [0, +∞)
    for (let i = 0; i < P * P; i++) {
      imgData[i] = Math.max(0, imgData[i] + (bpData[i] as number));
    }

    // Dispose the temporary image array
    img.dispose();

    // Notify progress for visualisation
    if (onProgress) {
      await onProgress(imgData, P, idx);
    }

    if (delay > 0) {
      await new Promise<void>((r) => setTimeout(r, delay));
    }
  }

  return np.array(imgData).reshape([P, P]);
}
