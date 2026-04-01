// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.
//
// Algebraic Reconstruction Technique (ART) for CT image reconstruction.
// Ported from PyTorch (huo/radon.py) to jax-js.
//
// Terminology follows the torch-radon convention:
//   - forward()  : Radon transform (image → sinogram), one angle at a time
//   - scan()     : full forward projection over all angles (image → complete sinogram)
//   - art()      : iterative ART reconstruction (sinogram → image)
//
// Backprojection is computed automatically via jax-js `grad()`, mirroring
// the autograd approach used in the Python `RadonFanbeam` class.
//
// References:
//   - https://torch-radon.readthedocs.io/en/latest/
//   - https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique
//   - https://github.com/ekzhang/jax-js

import { numpy, grad, lax } from "@jax-js/jax";

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
 * Bilinear interpolation on a 2D image using jax-js operations.
 * Equivalent to PyTorch's F.grid_sample with align_corners=True.
 *
 * All operations go through the jax-js computation graph so that
 * `grad()` can differentiate through this function.
 *
 * @param img - Image array of shape [H, W].
 * @param H - Image height.
 * @param W - Image width.
 * @param gridX - X coordinates in [-1, 1] (width axis), flattened [N].
 * @param gridY - Y coordinates in [-1, 1] (height axis), flattened [N].
 * @returns Interpolated values [N].
 */
function bilinearSample(
  img: numpy.Array,
  H: number,
  W: number,
  gridX: numpy.Array,
  gridY: numpy.Array,
): numpy.Array {
  // Convert normalized [-1, 1] to pixel coordinates
  const px = np.multiply(np.add(gridX, 1), 0.5 * (W - 1));
  const py = np.multiply(np.add(gridY, 1), 0.5 * (H - 1));

  // Integer parts (used for indexing; gradient does not flow through indices)
  const x0f = lax.stopGradient(np.floor(px.ref));
  const y0f = lax.stopGradient(np.floor(py.ref));

  // Fractional parts (interpolation weights; gradient flows through here)
  const wx = np.subtract(px, x0f.ref);
  const wy = np.subtract(py, y0f.ref);

  // Clamp integer indices to valid range and cast to int32
  const x0 = np.clip(x0f.ref, 0, W - 1).astype(np.int32);
  const x1 = np.clip(np.add(x0f, 1), 0, W - 1).astype(np.int32);
  const y0 = np.clip(y0f.ref, 0, H - 1).astype(np.int32);
  const y1 = np.clip(np.add(y0f, 1), 0, H - 1).astype(np.int32);

  // Flat indices into the image: idx = y * W + x
  const imgFlat = img.flatten();
  const idx00 = np.add(np.multiply(y0.ref, W), x0.ref);
  const idx01 = np.add(np.multiply(y0, W), x1.ref);
  const idx10 = np.add(np.multiply(y1.ref, W), x0);
  const idx11 = np.add(np.multiply(y1, W), x1);

  // Gather the four corner values (gradient scatters back through take)
  const v00 = np.take(imgFlat.ref, idx00);
  const v01 = np.take(imgFlat.ref, idx01);
  const v10 = np.take(imgFlat.ref, idx10);
  const v11 = np.take(imgFlat, idx11);

  // Bilinear interpolation weights
  const oneMinusWx = np.subtract(1, wx.ref);
  const oneMinusWy = np.subtract(1, wy.ref);
  const w00 = np.multiply(oneMinusWy.ref, oneMinusWx.ref);
  const w01 = np.multiply(oneMinusWy, wx.ref);
  const w10 = np.multiply(wy.ref, oneMinusWx);
  const w11 = np.multiply(wy, wx);

  // Weighted sum of corner values
  return np.add(
    np.add(np.multiply(v00, w00), np.multiply(v01, w01)),
    np.add(np.multiply(v10, w10), np.multiply(v11, w11)),
  );
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
 * Differentiable forward projection for a single angle (internal).
 *
 * All computation goes through jax-js operations so that `grad()` can
 * differentiate through this function with respect to `img`.
 *
 * @param img - 2D volume image [imgPixels, imgPixels].
 * @param rotX - Pre-rotated gantry X coordinates [latSteps * detCount].
 * @param rotY - Pre-rotated gantry Y coordinates [latSteps * detCount].
 * @param param - CT geometry parameters.
 * @returns Sinogram column for this angle [detCount].
 */
function _forwardAngle(
  img: numpy.Array,
  rotX: numpy.Array,
  rotY: numpy.Array,
  param: RaysCfg,
): numpy.Array {
  const P = param.imgPixels;
  const latSteps = param.latSampling * P + 1;
  const latStep = param.imgLen / P / param.latSampling;

  // Sample image at rotated gantry coordinates via differentiable bilinear interpolation
  const interp = bilinearSample(img, P, P, rotX, rotY);

  // Reshape to [latSteps, detrNum] and sum along the lateral (ray) direction
  return interp.reshape([latSteps, param.detrNum]).sum(0).mul(latStep);
}

/**
 * Rotate gantry coordinates by a projection angle.
 *
 * @param gantryCoordX - Gantry X coordinates [latSteps, detCount].
 * @param gantryCoordY - Gantry Y coordinates [latSteps, detCount].
 * @param angle - Projection angle in degrees.
 * @returns Rotated and flattened coordinates [rotX, rotY] each of shape [latSteps * detCount].
 */
function _rotateGantryCoords(
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angle: number,
): [numpy.Array, numpy.Array] {
  const angleRad = (angle * Math.PI) / 180;
  const cosA = Math.cos(angleRad);
  const sinA = Math.sin(angleRad);

  // Rotate counter-clockwise by the projection angle:
  //   rotX = gx * cos - gy * sin
  //   rotY = gx * sin + gy * cos
  const gxFlat = gantryCoordX.flatten();
  const gyFlat = gantryCoordY.flatten();

  const rotX = np.subtract(
    np.multiply(gxFlat.ref, cosA),
    np.multiply(gyFlat.ref, sinA),
  );
  const rotY = np.add(
    np.multiply(gxFlat, sinA),
    np.multiply(gyFlat, cosA),
  );

  return [rotX, rotY];
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
 * This function is differentiable: `grad()` can compute the backprojection
 * (adjoint) automatically.
 *
 * Corresponds to `RadonFanbeam._forward_angle()` in the Python implementation.
 *
 * @param img - 2D volume image [imgPixels, imgPixels].
 * @param gantryCoordX - Gantry X coordinates [latSteps, detCount].
 * @param gantryCoordY - Gantry Y coordinates [latSteps, detCount].
 * @param angle - Projection angle in degrees.
 * @param param - CT geometry parameters.
 * @returns Sinogram column for this angle [detCount].
 */
export function forward(
  img: numpy.Array,
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angle: number,
  param: RaysCfg,
): numpy.Array {
  const [rotX, rotY] = _rotateGantryCoords(gantryCoordX, gantryCoordY, angle);
  return _forwardAngle(img, rotX, rotY, param);
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
    const sino = forward(img.ref, gantryCoordX.ref, gantryCoordY.ref, angles[i], param);
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
 *   2. Compute the squared-error loss between predicted and measured.
 *   3. Use `grad()` to compute the gradient (which is the backprojection
 *      of the residual), mirroring the autograd approach in the Python
 *      `RadonFanbeam` class.
 *   4. Update the image estimate and clamp negative values to zero.
 *
 * This is a classical iterative CT reconstruction algorithm.  See also:
 *   - https://en.wikipedia.org/wiki/Algebraic_reconstruction_technique
 *   - torch-radon documentation for forward / backprojection terminology.
 *
 * @param sinogram - Measured sinogram [detCount, numAngles].
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
  gantryCoordX: numpy.Array,
  gantryCoordY: numpy.Array,
  angles: number[],
  param: RaysCfg,
  onProgress?: ArtProgressCallback,
  delay = 0,
): Promise<numpy.Array> {
  const numAngles = angles.length;
  const P = param.imgPixels;

  // Read sinogram data (.ref keeps the original alive)
  const fullSinoData = await sinogram.ref.data();

  // Initialize reconstruction to zeros
  let img = np.zeros([P, P]);

  // Process angles in random order (Kaczmarz method)
  const indices = randperm(numAngles);

  for (const idx of indices) {
    const angle = angles[idx];

    // Extract measured sinogram column for this angle
    const sinoCol = new Float32Array(param.detrNum);
    for (let j = 0; j < param.detrNum; j++) {
      sinoCol[j] = fullSinoData[j * numAngles + idx] as number;
    }
    const measured = np.array(sinoCol);

    // Pre-compute rotated gantry coordinates for this angle
    const [rotX, rotY] = _rotateGantryCoords(
      gantryCoordX.ref,
      gantryCoordY.ref,
      angle,
    );

    // Compute gradient of the squared-error loss via grad().
    //
    // loss(img) = 0.5 * sum((forward(img) - measured)^2)
    // grad(loss)(img) = J^T @ (forward(img) - measured)
    //
    // The ART update is: img -= grad / imgLen, then clamp.
    // This is equivalent to: img += J^T @ (measured - forward(img)) / imgLen
    const gradFn = grad((imgArg: numpy.Array) => {
      const predicted = _forwardAngle(imgArg, rotX.ref, rotY.ref, param);
      const diff = np.subtract(predicted, measured.ref);
      return np.sum(np.square(diff)).mul(0.5);
    });
    const gradImg = gradFn(img.ref);

    // Dispose pre-computed arrays
    rotX.dispose();
    rotY.dispose();
    measured.dispose();

    // ART update: img = max(0, img - grad / imgLen)
    img = np.maximum(np.subtract(img, np.multiply(gradImg, 1.0 / param.imgLen)), 0);

    // Notify progress for visualisation
    if (onProgress) {
      const imgData = await img.ref.data();
      await onProgress(imgData as Float32Array, P, idx);
    }

    if (delay > 0) {
      await new Promise<void>((r) => setTimeout(r, delay));
    }
  }

  // Dispose borrowed arrays
  sinogram.dispose();
  gantryCoordX.dispose();
  gantryCoordY.dispose();

  return img;
}
