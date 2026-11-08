// Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import { numpy } from "@jax-js/jax";
import { FanbeamGeometry, compute_ray_coordinates, type FanbeamParameters } from "./geometry.js";
export { FanbeamGeometry } from "./geometry.js";

const np = numpy;

/** A callback receives the current buffer and the original angle index. */
export interface ProgressOptions {
  on_progress?: (data: Float32Array, angle_index: number) => void | Promise<void>;
  delay?: number;
}

/** Bilinear sampling with align_corners=True and zero-valued outside neighbours. */
function bilinear_sample(
  image: Float32Array, resolution: number, x: number, y: number,
): number {
  const col = (x + 1) * (resolution - 1) / 2;
  const row = (y + 1) * (resolution - 1) / 2;
  const col0 = Math.floor(col), row0 = Math.floor(row);
  const wx = col - col0, wy = row - row0;
  let value = 0;
  for (let r = 0; r < 2; r++) {
    if (row0 + r < 0 || row0 + r >= resolution) continue;
    for (let c = 0; c < 2; c++) {
      if (col0 + c < 0 || col0 + c >= resolution) continue;
      value += image[(row0 + r) * resolution + col0 + c]
        * (r ? wy : 1 - wy) * (c ? wx : 1 - wx);
    }
  }
  return value;
}

/** Detector interpolation retains partial weights at the two edges. */
function linear_sample(projection: Float32Array, index: number): number {
  const i = Math.floor(index), weight = index - i;
  const left = i >= 0 && i < projection.length ? projection[i] : 0;
  const right = i + 1 >= 0 && i + 1 < projection.length ? projection[i + 1] : 0;
  return left * (1 - weight) + right * weight;
}

/** Per-angle forward operator; angle is radians and distances are pixels. */
export function forward_angle(
  image: Float32Array, ray_x: Float32Array, ray_y: Float32Array,
  angle: number, geometry: FanbeamGeometry,
): Float32Array {
  const c = Math.cos(angle), s = Math.sin(angle);
  const projection = new Float32Array(geometry.det_count);
  for (let i = 0; i < geometry.det_count; i++) {
    let sum = 0;
    for (let k = 0; k < geometry.ray_sample_count; k++) {
      const index = k * geometry.det_count + i;
      // R_-angle maps scanner rays into image coordinates.
      const x = Math.fround(c * ray_x[index] + s * ray_y[index]);
      const y = Math.fround(-s * ray_x[index] + c * ray_y[index]);
      sum += Math.fround(bilinear_sample(image, geometry.resolution, x, y));
    }
    projection[i] = geometry.ray_step * sum;
  }
  return projection;
}

/** Uniform residual backfill, the B operator in the mathematical article. */
export function backprojection_angle(
  projection: Float32Array, angle: number, geometry: FanbeamGeometry,
): Float32Array {
  const c = Math.cos(angle), s = Math.sin(angle);
  const image = new Float32Array(geometry.resolution ** 2);
  for (let row = 0; row < geometry.resolution; row++) {
    for (let col = 0; col < geometry.resolution; col++) {
      const x = col - geometry.pixel_extent, y = row - geometry.pixel_extent;
      // R_angle maps image pixels into scanner coordinates.
      const x_rot = c * x - s * y, y_rot = s * x + c * y;
      const denominator = geometry.source_distance - y_rot;
      if (denominator <= 0) {
        throw new RangeError("All image pixel centres must be in front of the source at this angle");
      }
      const u = geometry.source_detector_distance * x_rot / denominator;
      const index = u / geometry.det_spacing + (geometry.det_count - 1) / 2;
      image[row * geometry.resolution + col] = linear_sample(projection, index);
    }
  }
  return image;
}

function random_order(count: number): number[] {
  const indices = Array.from({ length: count }, (_, index) => index);
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices;
}

async function report_progress(options: ProgressOptions, data: Float32Array, index: number) {
  await options.on_progress?.(data, index);
  if (options.delay && options.delay > 0) {
    await new Promise<void>((resolve) => setTimeout(resolve, options.delay));
  }
}

async function read_array(value: numpy.Array, shape: number[], name: string): Promise<Float32Array> {
  if (value.shape.length !== shape.length || value.shape.some((size, i) => size !== shape[i])) {
    throw new RangeError(`${name} must have shape [${shape}], got [${value.shape}]`);
  }
  const data = await value.ref.data();
  if (!(data instanceof Float32Array) || !data.every(Number.isFinite)) {
    throw new TypeError(`${name} must contain finite float32 values`);
  }
  return data;
}

/**
 * A fixed fan-beam scanner with the same fields and methods as Python.
 * Angles are radians; distances are pixels; sinograms have shape [angle, detector].
 * The zero-angle source is (0, source_distance), the detector plane is
 * y=-det_distance, and the gantry rotates by R_-angle.
 */
export class RadonFanbeam {
  readonly geometry: FanbeamGeometry;
  readonly angles: readonly number[];
  private readonly ray_x: Float32Array;
  private readonly ray_y: Float32Array;

  constructor(parameters: FanbeamParameters & { angles: ArrayLike<number> }) {
    this.geometry = new FanbeamGeometry(parameters);
    const angles = Array.from(parameters.angles);
    if (!angles.length || !angles.every(Number.isFinite)) {
      throw new RangeError("angles must be a nonempty finite sequence in radians");
    }
    this.angles = Object.freeze(angles);
    const { ray_x, ray_y } = compute_ray_coordinates(this.geometry);
    this.ray_x = ray_x;
    this.ray_y = ray_y;
    Object.freeze(this);
  }

  async forward_angle(image: numpy.Array, angle: number): Promise<numpy.Array> {
    if (!Number.isFinite(angle)) throw new RangeError("angle must be finite (in radians)");
    const n = this.geometry.resolution;
    const data = await read_array(image, [n, n], "image");
    return np.array(forward_angle(data, this.ray_x, this.ray_y, angle, this.geometry));
  }

  async backprojection_angle(projection: numpy.Array, angle: number): Promise<numpy.Array> {
    if (!Number.isFinite(angle)) throw new RangeError("angle must be finite (in radians)");
    const data = await read_array(projection, [this.geometry.det_count], "projection");
    const n = this.geometry.resolution;
    return np.array(backprojection_angle(data, angle, this.geometry)).reshape([n, n]);
  }

  /** Project to [angles.length, det_count], reporting after each angle. */
  async forward(image: numpy.Array, options: ProgressOptions = {}): Promise<numpy.Array> {
    const n = this.geometry.resolution, m = this.geometry.det_count;
    const data = await read_array(image, [n, n], "image");
    const sinogram = new Float32Array(this.angles.length * m);
    for (let index = 0; index < this.angles.length; index++) {
      sinogram.set(forward_angle(data, this.ray_x, this.ray_y, this.angles[index], this.geometry), index * m);
      await report_progress(options, sinogram, index);
    }
    return np.array(sinogram).reshape([this.angles.length, m]);
  }

  /** Sum uniform backfill over all rows of [angles.length, det_count]. */
  async backprojection(sinogram: numpy.Array): Promise<numpy.Array> {
    const n = this.geometry.resolution, m = this.geometry.det_count;
    const data = await read_array(sinogram, [this.angles.length, m], "sinogram");
    const image = new Float32Array(n * n);
    for (let index = 0; index < this.angles.length; index++) {
      const correction = backprojection_angle(
        data.subarray(index * m, (index + 1) * m), this.angles[index], this.geometry,
      );
      for (let i = 0; i < image.length; i++) image[i] += correction[i];
    }
    return np.array(image).reshape([n, n]);
  }

  /** One randomly ordered residual sweep from zero, with nonnegative clipping. */
  async art(sinogram: numpy.Array, options: ProgressOptions = {}): Promise<numpy.Array> {
    const n = this.geometry.resolution, m = this.geometry.det_count;
    const data = await read_array(sinogram, [this.angles.length, m], "sinogram");
    const image = new Float32Array(n * n);
    for (const index of random_order(this.angles.length)) {
      const predicted = forward_angle(image, this.ray_x, this.ray_y, this.angles[index], this.geometry);
      const residual = new Float32Array(m);
      for (let i = 0; i < m; i++) residual[i] = (data[index * m + i] - predicted[i]) / n;
      const correction = backprojection_angle(residual, this.angles[index], this.geometry);
      for (let i = 0; i < image.length; i++) image[i] = Math.max(0, Math.fround(image[i] + correction[i]));
      await report_progress(options, image, index);
    }
    return np.array(image).reshape([n, n]);
  }
}
