//! Triangulation helpers shared by MATLAB-facing builtins.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct Delaunay2d {
    pub triangles: Vec<[usize; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriangulationError {
    NonFinitePoint,
}

impl std::fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinitePoint => write!(f, "points must be finite"),
        }
    }
}

impl std::error::Error for TriangulationError {}

pub fn delaunay_2d(points: &[[f64; 2]]) -> Result<Delaunay2d, TriangulationError> {
    let mut unique_points = Vec::with_capacity(points.len());
    let mut original_indices = Vec::with_capacity(points.len());
    let mut seen = HashMap::with_capacity(points.len());
    for (idx, point) in points.iter().enumerate() {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(TriangulationError::NonFinitePoint);
        }
        let key = (stable_float_key(point[0]), stable_float_key(point[1]));
        if seen.contains_key(&key) {
            continue;
        }
        seen.insert(key, unique_points.len());
        unique_points.push(delaunator::Point {
            x: point[0],
            y: point[1],
        });
        original_indices.push(idx);
    }

    if unique_points.len() < 3 {
        return Ok(Delaunay2d {
            triangles: Vec::new(),
        });
    }

    let triangulation = delaunator::triangulate(&unique_points);
    let mut triangles = Vec::with_capacity(triangulation.triangles.len() / 3);
    for tri in triangulation.triangles.chunks_exact(3) {
        triangles.push([
            original_indices[tri[0]],
            original_indices[tri[1]],
            original_indices[tri[2]],
        ]);
    }
    Ok(Delaunay2d { triangles })
}

fn stable_float_key(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

pub fn boundary_edges(triangles: &[[usize; 3]]) -> Vec<[usize; 2]> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::with_capacity(triangles.len() * 3);
    for tri in triangles {
        for [a, b] in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
            let edge = if a <= b { (a, b) } else { (b, a) };
            *counts.entry(edge).or_insert(0) += 1;
        }
    }
    let mut edges = counts
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some([edge.0, edge.1]))
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges
}

pub fn nearest_neighbor_indices(points: &[[f64; 2]], queries: &[[f64; 2]]) -> Vec<Option<usize>> {
    queries
        .iter()
        .map(|query| {
            let mut best = None;
            let mut best_distance = f64::INFINITY;
            for (idx, point) in points.iter().enumerate() {
                let dx = query[0] - point[0];
                let dy = query[1] - point[1];
                let distance = dx.mul_add(dx, dy * dy);
                if distance < best_distance {
                    best_distance = distance;
                    best = Some(idx);
                }
            }
            best
        })
        .collect()
}

pub fn point_locations(
    points: &[[f64; 2]],
    triangles: &[[usize; 3]],
    queries: &[[f64; 2]],
) -> Vec<(Option<usize>, [f64; 3])> {
    queries
        .iter()
        .map(|query| {
            for (idx, tri) in triangles.iter().enumerate() {
                let bary = barycentric(points[tri[0]], points[tri[1]], points[tri[2]], *query);
                if bary
                    .iter()
                    .all(|value| *value >= -1.0e-12 && *value <= 1.0 + 1.0e-12)
                {
                    return (Some(idx), bary);
                }
            }
            (None, [f64::NAN; 3])
        })
        .collect()
}

fn barycentric(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> [f64; 3] {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let denominator = v0[0] * v1[1] - v1[0] * v0[1];
    if denominator.abs() <= f64::EPSILON {
        return [f64::NAN; 3];
    }
    let v = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator;
    let w = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator;
    [1.0 - v - w, v, w]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangulates_square_as_two_one_based_ready_faces() {
        let mesh = delaunay_2d(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unwrap();
        assert_eq!(mesh.triangles.len(), 2);
        assert!(mesh
            .triangles
            .iter()
            .all(|tri| tri.iter().all(|idx| *idx < 4)));
    }

    #[test]
    fn boundary_edges_ignore_shared_interior_edge() {
        let edges = boundary_edges(&[[0, 1, 2], [1, 3, 2]]);
        assert_eq!(edges, vec![[0, 1], [0, 2], [1, 3], [2, 3]]);
    }

    #[test]
    fn point_locations_return_triangle_and_barycentric_coordinates() {
        let locations = point_locations(
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            &[[0, 1, 2]],
            &[[0.25, 0.25], [2.0, 2.0]],
        );
        assert_eq!(locations[0].0, Some(0));
        assert!((locations[0].1[0] - 0.5).abs() < 1.0e-12);
        assert_eq!(locations[1].0, None);
        assert!(locations[1].1[0].is_nan());
    }
}
