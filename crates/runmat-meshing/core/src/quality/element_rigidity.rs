//! The element rigidity test that linear-elastic assembly applies to every element block, and the
//! smallest tetrahedron shape quality at which a straight element still passes it.
//!
//! Assembly proves its operator positive definite element by element (`runmat-analysis-fea`,
//! `algebra::elastic_rigidity`): every element block must annihilate the element's six rigid
//! motions and be positive definite on their orthogonal complement with a roundoff-relative
//! margin, `c·(d+1)·ε·tr(A)` (Rump's floating-point Cholesky criterion). The test lives here so
//! the mesher can derive its volume quality floor from the same arithmetic: an element that the
//! mesher's floor admits is one that assembly can factor.
//!
//! A flat tetrahedron's smallest complement energy relative to ‖K‖ falls with its scaled
//! Jacobian `s`: about `s²` for Tet4 and about `s⁴` for Tet10, whose midside nodes add a nearly
//! free mode across a flat element. At the volume's default `s = 0.001` a straight Tet4 keeps a
//! margin of about 1e-7 ‖K‖, while a Tet10 sliver falls to about 1e-14 ‖K‖, below the ~1.7e-13
//! ‖K‖ assembly requires. [`assembly_minimum_scaled_jacobian`] finds, per element order, the
//! scaled Jacobian of the flattest canonical sliver that still passes.
use std::cmp::Ordering;

use crate::{ElementOrder, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

/// Multiplier `c` of the roundoff-relative element margin `c·(d+1)·ε·tr(A)`: covers the rounding
/// of forming the projected block and of its Cholesky factorization (Rump's criterion uses
/// `γ_{d+1} ≈ (d+1)ε` for the factorization alone).
pub const ELEMENT_RIGIDITY_MARGIN_ROUNDING_FACTOR: f64 = 3.0;
/// Largest admitted `‖Kₑ U‖_F / ‖Kₑ‖_F` for the orthonormal rigid-motion basis `U`.
pub const RIGID_MODE_RESIDUAL_RATIO: f64 = 1.0e-8;
/// The most nearly incompressible isotropic material the derived quality floor covers. The floor
/// grows without bound as Poisson's ratio approaches 0.5; elements of a material beyond this
/// bound are still checked one by one at assembly, which names any element it refuses.
pub const ASSEMBLY_FLOOR_POISSON_RATIO: f64 = 0.45;

/// Why an element block fails the rigidity test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ElementRigidityShortfall {
    /// The element's nodes do not span the six rigid motions.
    DegenerateGeometry,
    /// The block does not annihilate rigid motions to [`RIGID_MODE_RESIDUAL_RATIO`].
    RigidMotionEnergy { residual_ratio: f64 },
    /// The block is not positive definite on the rigid-motion complement beyond rounding.
    EnergyMargin {
        /// The required roundoff-relative margin over `‖Kₑ‖_F`.
        margin_ratio: f64,
        /// Bisection estimate of the projected block's smallest eigenvalue over `‖Kₑ‖_F`.
        smallest_ratio: f64,
    },
}

/// The roundoff-relative margin subtracted before the element Cholesky test.
fn element_energy_margin(dimension: usize, trace: f64) -> f64 {
    let gamma = ELEMENT_RIGIDITY_MARGIN_ROUNDING_FACTOR * (dimension as f64 + 1.0) * f64::EPSILON;
    gamma / (1.0 - gamma) * trace
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Row `axis` of the rigid basis at point `q`: translations along x, y and z, then rotations
/// about x, y and z.
fn rigid_row(q: [f64; 3], axis: usize) -> [f64; 6] {
    let mut row = [0.0; 6];
    row[axis] = 1.0;
    let rotations = [[0.0, -q[2], q[1]], [q[2], 0.0, -q[0]], [-q[1], q[0], 0.0]];
    for (r, rotation) in rotations.iter().enumerate() {
        row[3 + r] = rotation[axis];
    }
    row
}

/// Lower Cholesky of a row-major symmetric `n × n` matrix; false when a pivot is not positive.
fn cholesky_in_place(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut pivot = a[j * n + j];
        for k in 0..j {
            pivot -= a[j * n + k] * a[j * n + k];
        }
        if pivot.partial_cmp(&0.0) != Some(Ordering::Greater) || !pivot.is_finite() {
            return false;
        }
        let pivot = pivot.sqrt();
        a[j * n + j] = pivot;
        for i in j + 1..n {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = value / pivot;
        }
    }
    true
}

/// The element test: rigid motions are (numerically) null and the row-major block `k` over the
/// element's `3n` nodal translations is positive definite on their orthogonal complement with
/// the roundoff-relative margin.
pub fn element_rigidity(points: &[[f64; 3]], k: &[f64]) -> Result<(), ElementRigidityShortfall> {
    let n = points.len();
    let d = 3 * n;
    let mut center = [0.0; 3];
    for p in points {
        for a in 0..3 {
            center[a] += p[a] / n as f64;
        }
    }
    let scale = points
        .iter()
        .map(|p| norm(&[p[0] - center[0], p[1] - center[1], p[2] - center[2]]))
        .fold(0.0, f64::max);
    if scale.partial_cmp(&0.0) != Some(Ordering::Greater) || !scale.is_finite() {
        return Err(ElementRigidityShortfall::DegenerateGeometry);
    }
    // Rigid basis U (d × 6, column-major), orthonormalized by two Gram–Schmidt passes.
    let mut u = vec![0.0; d * 6];
    for (node, p) in points.iter().enumerate() {
        let q = [0, 1, 2].map(|a| (p[a] - center[a]) / scale);
        for axis in 0..3 {
            let row = rigid_row(q, axis);
            for c in 0..6 {
                u[c * d + 3 * node + axis] = row[c];
            }
        }
    }
    for c in 0..6 {
        let original = norm(&u[c * d..(c + 1) * d]);
        for _ in 0..2 {
            for p in 0..c {
                let dot = (0..d).map(|i| u[c * d + i] * u[p * d + i]).sum::<f64>();
                for i in 0..d {
                    u[c * d + i] -= dot * u[p * d + i];
                }
            }
        }
        let length = norm(&u[c * d..(c + 1) * d]);
        if length.partial_cmp(&(1.0e-6 * original)) != Some(Ordering::Greater) {
            return Err(ElementRigidityShortfall::DegenerateGeometry);
        }
        for i in 0..d {
            u[c * d + i] /= length;
        }
    }
    let k_norm = norm(k);
    if k_norm.partial_cmp(&0.0) != Some(Ordering::Greater) || !k_norm.is_finite() {
        return Err(ElementRigidityShortfall::EnergyMargin {
            margin_ratio: f64::NAN,
            smallest_ratio: f64::NAN,
        });
    }
    // KU (d × 6, column-major).
    let mut ku = vec![0.0; d * 6];
    for c in 0..6 {
        for i in 0..d {
            ku[c * d + i] = (0..d).map(|j| k[i * d + j] * u[c * d + j]).sum();
        }
    }
    let residual = norm(&ku) / k_norm;
    if !matches!(
        residual.partial_cmp(&RIGID_MODE_RESIDUAL_RATIO),
        Some(Ordering::Less | Ordering::Equal)
    ) {
        return Err(ElementRigidityShortfall::RigidMotionEnergy {
            residual_ratio: residual,
        });
    }
    // A = P K P + ‖K‖ U Uᵀ with P = I − U Uᵀ; A ≻ margin ‖K‖ iff the complement block is.
    // UᵀKU (6 × 6).
    let mut utku = [0.0; 36];
    for a in 0..6 {
        for b in 0..6 {
            utku[a * 6 + b] = (0..d).map(|i| u[a * d + i] * ku[b * d + i]).sum();
        }
    }
    let mut matrix = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            // (I − UUᵀ) K (I − UUᵀ) = K − KUUᵀ − UUᵀK + U(UᵀKU)Uᵀ.
            let mut value = 0.5 * (k[i * d + j] + k[j * d + i]);
            for a in 0..6 {
                value -= 0.5 * (ku[a * d + i] * u[a * d + j] + ku[a * d + j] * u[a * d + i]);
                value -= 0.5 * (u[a * d + i] * ku[a * d + j] + u[a * d + j] * ku[a * d + i]);
                for b in 0..6 {
                    value += u[a * d + i] * utku[a * 6 + b] * u[b * d + j];
                }
                value += k_norm * u[a * d + i] * u[a * d + j];
            }
            matrix[i * d + j] = value;
        }
    }
    let trace = (0..d).map(|i| matrix[i * d + i]).sum::<f64>();
    let margin = element_energy_margin(d, trace.abs());
    let succeeds = |shift: f64| {
        let mut shifted = matrix.clone();
        for i in 0..d {
            shifted[i * d + i] -= shift;
        }
        cholesky_in_place(&mut shifted, d)
    };
    if !succeeds(margin) {
        // Diagnostics only: bisect the largest shift that still factors (≈ λ_min of the block).
        let (mut low, mut high) = (-k_norm, margin);
        for _ in 0..60 {
            let middle = 0.5 * (low + high);
            if succeeds(middle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return Err(ElementRigidityShortfall::EnergyMargin {
            margin_ratio: margin / k_norm,
            smallest_ratio: low / k_norm,
        });
    }
    Ok(())
}

/// Scaled Jacobian of a tetrahedron: the smallest corner triple product over the product of its
/// three edge lengths, times √2 so a regular tetrahedron scores one.
pub fn tetrahedron_scaled_jacobian(corners: &[[f64; 3]; 4]) -> f64 {
    let mut smallest = f64::INFINITY;
    for corner in 0..4 {
        let o = corners[corner];
        let mut edges = [[0.0; 3]; 3];
        let mut slot = 0;
        for (other, p) in corners.iter().enumerate() {
            if other != corner {
                edges[slot] = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
                slot += 1;
            }
        }
        let [a, b, c] = edges;
        let triple = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        smallest = smallest.min(triple.abs() / (norm(&a) * norm(&b) * norm(&c)));
    }
    smallest * std::f64::consts::SQRT_2
}

/// Isotropic stiffness of a straight element over its nodal translations, row-major, from the
/// gradient moments `G[i,a,j,b] = ∫ ∂ₐNᵢ ∂ᵦNⱼ dV`: `λ G[i,a,j,b] + μ G[i,b,j,a] + μ δₐᵦ Σc G[i,c,j,c]`.
fn isotropic_stiffness(
    nodes: usize,
    moment: impl Fn(usize, usize, usize, usize) -> f64,
) -> Vec<f64> {
    let nu = ASSEMBLY_FLOOR_POISSON_RATIO;
    let lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 1.0 / (2.0 * (1.0 + nu));
    let d = 3 * nodes;
    let mut k = vec![0.0; d * d];
    for i in 0..nodes {
        for j in 0..nodes {
            let trace = (0..3).map(|c| moment(i, c, j, c)).sum::<f64>();
            for a in 0..3 {
                for b in 0..3 {
                    let diagonal = if a == b { mu * trace } else { 0.0 };
                    k[(3 * i + a) * d + 3 * j + b] =
                        lambda * moment(i, a, j, b) + mu * moment(i, b, j, a) + diagonal;
                }
            }
        }
    }
    k
}

/// The canonical sliver of height `h`: two crossing unit diagonals `h` apart, the flattening
/// that drives a tetrahedron's stiffness margin down fastest at a given scaled Jacobian.
/// Positively oriented, so the Tet10 rule certifies it.
fn canonical_sliver(height: f64) -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, height],
        [1.0, 0.0, height],
    ]
}

/// Whether a straight element of `order` on `corners` passes assembly's rigidity test for the
/// reference material.
fn straight_element_passes(order: ElementOrder, corners: [[f64; 3]; 4]) -> bool {
    match order {
        ElementOrder::Tet4 => {
            // Barycentric gradients are constant: G[i,a,j,b] = V gᵢₐ gⱼᵦ.
            let e = [1, 2, 3].map(|r| [0, 1, 2].map(|a| corners[r][a] - corners[0][a]));
            let det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
                - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
                + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
            if det.abs().partial_cmp(&0.0) != Some(Ordering::Greater) {
                return false;
            }
            // Rows of the inverse of [e₁ e₂ e₃]ᵀ are the gradients of nodes 1..3.
            let cross = |p: [f64; 3], q: [f64; 3]| {
                [
                    p[1] * q[2] - p[2] * q[1],
                    p[2] * q[0] - p[0] * q[2],
                    p[0] * q[1] - p[1] * q[0],
                ]
            };
            let g1 = cross(e[1], e[2]).map(|v| v / det);
            let g2 = cross(e[2], e[0]).map(|v| v / det);
            let g3 = cross(e[0], e[1]).map(|v| v / det);
            let g0 = [0, 1, 2].map(|a| -(g1[a] + g2[a] + g3[a]));
            let gradients = [g0, g1, g2, g3];
            let volume = det.abs() / 6.0;
            let k = isotropic_stiffness(4, |i, a, j, b| volume * gradients[i][a] * gradients[j][b]);
            element_rigidity(&corners, &k).is_ok()
        }
        ElementOrder::Tet10 => {
            let mut nodes = [[0.0; 3]; 10];
            nodes[..4].copy_from_slice(&corners);
            for (edge, [a, b]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
                nodes[4 + edge] =
                    [0, 1, 2].map(|axis| 0.5 * (corners[*a][axis] + corners[*b][axis]));
            }
            let Ok(moments) = crate::tetrahedron10_stiffness_moments(&nodes) else {
                return false;
            };
            let k = isotropic_stiffness(10, |i, a, j, b| moments.moment(i, a, j, b));
            element_rigidity(&nodes, &k).is_ok()
        }
    }
}

/// The smallest scaled Jacobian at which a straight element of `order` passes assembly's
/// rigidity test for an isotropic material with Poisson's ratio up to
/// [`ASSEMBLY_FLOOR_POISSON_RATIO`]: the scaled Jacobian of the flattest canonical sliver that
/// passes, found by bisection on the sliver's height. About 3.2e-3 for Tet10 and below 1e-6 for
/// Tet4. A volume whose elements all meet this floor publishes no element that assembly refuses
/// for its shape.
pub fn assembly_minimum_scaled_jacobian(order: ElementOrder) -> f64 {
    static TET4: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    static TET10: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let cell = match order {
        ElementOrder::Tet4 => &TET4,
        ElementOrder::Tet10 => &TET10,
    };
    *cell.get_or_init(|| {
        // The unit-height sliver passes for both orders; a vanishing one fails.
        let (mut fails, mut passes) = (0.0_f64, 1.0_f64);
        for _ in 0..60 {
            let middle = 0.5 * (fails + passes);
            if straight_element_passes(order, canonical_sliver(middle)) {
                passes = middle;
            } else {
                fails = middle;
            }
        }
        tetrahedron_scaled_jacobian(&canonical_sliver(passes))
    })
}

#[cfg(test)]
mod tests;
