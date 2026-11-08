/** Shared fan-beam parameters. Distances are in pixels; image spacing is one. */
export interface FanbeamParameters {
  resolution: number;
  source_distance: number;
  det_distance: number;
  det_count: number;
  det_spacing: number;
  ray_sampling?: number;
}

/** Immutable geometry, with exactly the same field names as Python. */
export class FanbeamGeometry {
  readonly resolution: number;
  readonly source_distance: number;
  readonly det_distance: number;
  readonly det_count: number;
  readonly det_spacing: number;
  readonly ray_sampling: number;

  constructor(parameters: FanbeamParameters) {
    this.resolution = parameters.resolution;
    this.source_distance = parameters.source_distance;
    this.det_distance = parameters.det_distance;
    this.det_count = parameters.det_count;
    this.det_spacing = parameters.det_spacing;
    this.ray_sampling = parameters.ray_sampling ?? 2;
    for (const [name, minimum] of [["resolution", 2], ["det_count", 2], ["ray_sampling", 1]] as const) {
      if (!Number.isInteger(this[name]) || this[name] < minimum) {
        throw new RangeError(`${name} must be an integer >= ${minimum}`);
      }
    }
    for (const name of ["source_distance", "det_distance", "det_spacing"] as const) {
      if (!Number.isFinite(this[name]) || this[name] <= 0) {
        throw new RangeError(`${name} must be positive and finite (in pixels)`);
      }
    }
    if (this.source_distance <= this.resolution / 2) {
      throw new RangeError("source_distance must exceed resolution/2 so the ray window stays in front of the source");
    }
    Object.freeze(this);
  }

  get source_detector_distance(): number { return this.source_distance + this.det_distance; }
  get pixel_extent(): number { return (this.resolution - 1) / 2; }
  get det_extent(): number { return (this.det_count - 1) * this.det_spacing / 2; }
  get ray_sample_count(): number { return this.ray_sampling * this.resolution + 1; }
  get ray_step(): number { return 1 / this.ray_sampling; }
}

/** Normalized zero-angle ray coordinates [ray_sample_count, det_count]. */
export function compute_ray_coordinates(geometry: FanbeamGeometry): {
  ray_x: Float32Array;
  ray_y: Float32Array;
} {
  const ray_x = new Float32Array(geometry.ray_sample_count * geometry.det_count);
  const ray_y = new Float32Array(ray_x.length);
  for (let k = 0; k < geometry.ray_sample_count; k++) {
    const t = geometry.source_distance - geometry.resolution / 2 + k * geometry.ray_step;
    for (let i = 0; i < geometry.det_count; i++) {
      const u = (i - (geometry.det_count - 1) / 2) * geometry.det_spacing;
      const distance = Math.sqrt(u * u + geometry.source_detector_distance ** 2);
      const index = k * geometry.det_count + i;
      ray_x[index] = t * u / distance / geometry.pixel_extent;
      ray_y[index] = (geometry.source_distance - t * geometry.source_detector_distance / distance)
        / geometry.pixel_extent;
    }
  }
  return { ray_x, ray_y };
}
