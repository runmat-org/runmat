//! Constrained distance to a triangle: interior plane projection plus all three edges.
use super::*;
pub(super) fn closest(
    vertices: [[f64; 3]; 3],
    query: [f64; 3],
) -> Result<([f64; 3], f64), BoundarySearchError> {
    let ab = subtract(vertices[1], vertices[0]);
    let ac = subtract(vertices[2], vertices[0]);
    let n = cross(ab, ac);
    let squared = dot(n, n);
    if !squared.is_finite() || squared <= 0.0 {
        return Err(BoundarySearchError::Resolution);
    }
    let ap = subtract(query, vertices[0]);
    let u = dot(cross(ap, ac), n) / squared;
    let v = dot(cross(ab, ap), n) / squared;
    let mut best = ([1.0, 0.0, 0.0], norm(ap));
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        let b = [1.0 - u - v, u, v];
        let position = std::array::from_fn(|i| (0..3).map(|j| vertices[j][i] * b[j]).sum());
        best = (b, norm(subtract(query, position)));
    }
    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        let edge = subtract(vertices[b], vertices[a]);
        let denominator = dot(edge, edge);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(BoundarySearchError::Resolution);
        }
        let t = (dot(subtract(query, vertices[a]), edge) / denominator).clamp(0.0, 1.0);
        let mut weights = [0.0; 3];
        weights[a] = 1.0 - t;
        weights[b] = t;
        let position = std::array::from_fn(|i| vertices[a][i] + t * edge[i]);
        let distance = norm(subtract(query, position));
        if distance < best.1 {
            best = (weights, distance);
        }
    }
    if !best.1.is_finite() {
        return Err(BoundarySearchError::Resolution);
    }
    Ok(best)
}
