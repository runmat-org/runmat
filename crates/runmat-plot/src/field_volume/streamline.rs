//! Streamlines of a vector field: deterministic seeds on a plane across the mean flow, traced
//! both ways with classical RK4 on the field direction (arc-length steps), the field evaluated
//! with each element's own value or basis at every stage. Each step is also taken as two half
//! steps; the difference estimates the step's error (Richardson, /15 for fourth order), the
//! half-step result is kept, and a line's estimated error is the sum over its steps.

use super::{
    add_scaled, cross, norm, normalize, sub, FieldPolyline, Reduction, SolvedField,
    TetrahedronLocator,
};

#[derive(Debug, Clone, PartialEq)]
pub struct StreamlineOptions {
    /// Step length in metres.
    pub step: f64,
    pub max_steps: usize,
    /// Seed grid rows and columns on the seed plane (points outside the mesh are dropped).
    pub seed_grid: usize,
    /// Lines stop where the magnitude falls below this fraction of the field's largest.
    pub min_relative_magnitude: f64,
}

impl StreamlineOptions {
    /// A quarter of the mean element edge per step, enough steps to cross the mesh four times.
    pub fn for_mesh(locator: &TetrahedronLocator<'_>) -> Self {
        let mesh = locator.mesh();
        let step = mesh.mean_edge() / 4.0;
        let (min, max) = mesh.bounds();
        let diagonal = norm(sub(max, min));
        Self {
            step,
            max_steps: ((4.0 * diagonal / step).ceil() as usize).clamp(100, 20_000),
            seed_grid: 12,
            min_relative_magnitude: 1.0e-3,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Streamlines {
    pub lines: Vec<FieldPolyline>,
    pub seeds: usize,
    /// Largest estimated accumulated integration error of a line, in metres.
    pub integration_error_m: f64,
    /// Largest curved-element deviation from the straight-sided tetrahedra used to locate points.
    pub geometric_error_m: f64,
}

/// Streamlines of the vector `field` (real part; the instantaneous field at `phase` for a
/// phasor), colored by `color` (usually its magnitude).
pub fn streamlines(
    locator: &TetrahedronLocator<'_>,
    field: &SolvedField,
    color: Reduction,
    phase_radians: f64,
    options: &StreamlineOptions,
) -> Streamlines {
    let mesh = locator.mesh();
    let mut result = Streamlines::default();
    if field.components != 3 {
        return result;
    }
    let (s, c) = phase_radians.sin_cos();
    let direction_at = |p: [f64; 3]| -> Option<([f64; 3], f64, usize)> {
        let located = locator.locate(p)?;
        let value = field.at(mesh, located.element, located.weights);
        let v = [0, 1, 2].map(|k| value.real[k] * c - value.imaginary[k] * s);
        let magnitude = norm(v);
        Some((
            v.map(|x| x / magnitude.max(f64::MIN_POSITIVE)),
            color.apply(&value),
            located.element,
        ))
        .filter(|_| magnitude > 0.0)
    };
    let floor = field.largest_magnitude() * options.min_relative_magnitude;
    let speed_at = |p: [f64; 3]| -> Option<[f64; 3]> {
        let located = locator.locate(p)?;
        let value = field.at(mesh, located.element, located.weights);
        let v = [0, 1, 2].map(|k| value.real[k] * c - value.imaginary[k] * s);
        (norm(v) > floor).then(|| normalize(v))
    };
    let seeds = seed_plane(locator, field, phase_radians, options.seed_grid);
    result.seeds = seeds.len();
    for seed in seeds {
        let Some((_, value, element)) = direction_at(seed) else {
            continue;
        };
        result.geometric_error_m = result
            .geometric_error_m
            .max(mesh.curvature_deviation(element));
        let mut error = 0.0;
        let mut traced = [Vec::new(), Vec::new()];
        for (half, sign) in traced.iter_mut().zip([-1.0, 1.0]) {
            let mut p = seed;
            for _ in 0..options.max_steps {
                let h = sign * options.step;
                let Some(full) = rk4(p, h, &speed_at) else {
                    break;
                };
                let Some(middle) = rk4(p, 0.5 * h, &speed_at) else {
                    break;
                };
                let Some(next) = rk4(middle, 0.5 * h, &speed_at) else {
                    break;
                };
                let Some((_, value, element)) = direction_at(next) else {
                    break;
                };
                error += norm(sub(full, next)) / 15.0;
                result.geometric_error_m = result
                    .geometric_error_m
                    .max(mesh.curvature_deviation(element));
                half.push((next, value));
                p = next;
            }
        }
        let [backward, forward] = traced;
        // Backward points run from the seed outward; the line reads start to end.
        let (points, values): (Vec<[f64; 3]>, Vec<f64>) = backward
            .into_iter()
            .rev()
            .chain(std::iter::once((seed, value)))
            .chain(forward)
            .unzip();
        if points.len() < 2 {
            continue;
        }
        result.integration_error_m = result.integration_error_m.max(error);
        result.lines.push(FieldPolyline {
            points: points.into_iter().map(|p| p.map(|v| v as f32)).collect(),
            values,
        });
    }
    result
}

fn rk4(p: [f64; 3], h: f64, direction: &dyn Fn([f64; 3]) -> Option<[f64; 3]>) -> Option<[f64; 3]> {
    let k1 = direction(p)?;
    let k2 = direction(add_scaled(p, k1, 0.5 * h))?;
    let k3 = direction(add_scaled(p, k2, 0.5 * h))?;
    let k4 = direction(add_scaled(p, k3, h))?;
    let d = [0, 1, 2].map(|a| (k1[a] + 2.0 * k2[a] + 2.0 * k3[a] + k4[a]) / 6.0);
    Some(add_scaled(p, d, h))
}

/// Seeds on a `grid` × `grid` lattice over the plane through the mesh's volume centroid
/// normal to the volume-weighted mean field direction (across a pipe or channel), keeping the
/// points inside the mesh.
fn seed_plane(
    locator: &TetrahedronLocator<'_>,
    field: &SolvedField,
    phase_radians: f64,
    grid: usize,
) -> Vec<[f64; 3]> {
    let mesh = locator.mesh();
    let (s, c) = phase_radians.sin_cos();
    let mut centroid = [0.0; 3];
    let mut mean = [0.0; 3];
    let mut total = 0.0;
    for element in 0..mesh.element_count() {
        let volume = mesh.element_volume(element);
        let value = field.at(mesh, element, [0.25; 4]);
        let center = mesh.point(element, [0.25; 4]);
        for axis in 0..3 {
            centroid[axis] += volume * center[axis];
            mean[axis] += volume * (value.real[axis] * c - value.imaginary[axis] * s);
        }
        total += volume;
    }
    if total <= 0.0 {
        return Vec::new();
    }
    let centroid = centroid.map(|v| v / total);
    // A circulating field (a vortex, flux around a conductor) has no mean direction: the plane
    // then lies across the field where it is strongest, which contains the axis it turns about.
    let normal = if norm(mean) > 1.0e-9 * total * field.largest_magnitude() {
        normalize(mean)
    } else {
        let strongest = (0..mesh.element_count())
            .map(|element| {
                let value = field.at(mesh, element, [0.25; 4]);
                [0, 1, 2].map(|k| value.real[k] * c - value.imaginary[k] * s)
            })
            .max_by(|a, b| norm(*a).total_cmp(&norm(*b)));
        match strongest {
            Some(direction) if norm(direction) > 0.0 => normalize(direction),
            _ => return Vec::new(),
        }
    };
    let helper = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(normal, helper));
    let v = cross(normal, u);
    let (min, max) = mesh.bounds();
    let reach = norm(sub(max, min));
    let mut seeds = Vec::new();
    for row in 0..grid {
        for column in 0..grid {
            let a = ((row as f64 + 0.5) / grid as f64 - 0.5) * reach;
            let b = ((column as f64 + 0.5) / grid as f64 - 0.5) * reach;
            let p = add_scaled(add_scaled(centroid, u, a), v, b);
            if locator.locate(p).is_some() {
                seeds.push(p);
            }
        }
    }
    // A coarse lattice can miss a thin cross-section: refine until it holds some seeds.
    if seeds.is_empty() && grid < 96 {
        return seed_plane(locator, field, phase_radians, grid * 2);
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::{TetMesh, TetOrder};
    use super::*;

    /// A box [0, 4] x [0, 1] x [0, 1] of Kuhn cubes.
    fn channel() -> TetMesh {
        let cube = unit_cube(TetOrder::Linear);
        let mut positions = Vec::new();
        let mut elements = Vec::new();
        for shift in 0..4 {
            let base = positions.len() as u32;
            positions.extend(
                super::super::test_support::positions(&cube)
                    .iter()
                    .map(|p| [p[0] + shift as f64, p[1], p[2]]),
            );
            for element in 0..cube.element_count() {
                elements.extend(cube.element_nodes(element).iter().map(|n| n + base));
            }
        }
        // Weld coincident nodes so neighbouring cubes share faces.
        let mut welded: Vec<[f64; 3]> = Vec::new();
        let mut map = Vec::new();
        for p in &positions {
            let found = welded.iter().position(|q| norm(sub(*q, *p)) < 1e-12);
            map.push(found.unwrap_or_else(|| {
                welded.push(*p);
                welded.len() - 1
            }) as u32);
        }
        let elements = elements.into_iter().map(|n| map[n as usize]).collect();
        TetMesh::new(welded, TetOrder::Linear, elements).unwrap()
    }

    #[test]
    fn uniform_flow_gives_straight_lines_through_the_whole_channel() {
        let mesh = channel();
        let field = nodal(&mesh, |_| vec![1.0, 0.0, 0.0]);
        let locator = TetrahedronLocator::new(&mesh);
        let options = StreamlineOptions {
            step: 0.05,
            max_steps: 400,
            seed_grid: 6,
            min_relative_magnitude: 1e-3,
        };
        let result = streamlines(&locator, &field, Reduction::Magnitude, 0.0, &options);
        assert!(result.seeds > 0);
        assert_eq!(result.lines.len(), result.seeds);
        for line in &result.lines {
            let first = line.points.first().unwrap();
            let last = line.points.last().unwrap();
            assert!(first[0] < 0.06 && last[0] > 3.94, "{first:?} .. {last:?}");
            assert!(line
                .points
                .iter()
                .all(|p| (p[1] - first[1]).abs() < 1e-5 && (p[2] - first[2]).abs() < 1e-5));
            assert!(line.values.iter().all(|v| (v - 1.0).abs() < 1e-12));
        }
        assert!(result.integration_error_m < 1e-9);
    }

    #[test]
    fn rotating_flow_stays_on_its_circle_within_the_reported_error() {
        let mesh = unit_cube(TetOrder::Quadratic);
        // A quadratic field is reproduced exactly by Tet10: v = (-(y - 1/2), x - 1/2, 0).
        let field = nodal(&mesh, |p| vec![-(p[1] - 0.5), p[0] - 0.5, 0.0]);
        let locator = TetrahedronLocator::new(&mesh);
        let options = StreamlineOptions {
            step: 0.02,
            max_steps: 60,
            seed_grid: 8,
            min_relative_magnitude: 1e-3,
        };
        let result = streamlines(&locator, &field, Reduction::Magnitude, 0.0, &options);
        assert!(!result.lines.is_empty());
        for line in &result.lines {
            let radius = |p: &[f32; 3]| (p[0] as f64 - 0.5).hypot(p[1] as f64 - 0.5);
            let r0 = radius(&line.points[0]);
            for p in &line.points {
                assert!(
                    (radius(p) - r0).abs() < 1e-5 + result.integration_error_m * 10.0,
                    "radius drift"
                );
            }
        }
    }
}
