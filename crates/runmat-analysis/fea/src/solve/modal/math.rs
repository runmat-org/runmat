use crate::algebra::{apply_m, AssembledRealSystem};

use super::MODAL_MASS_ORTHOGONALIZATION_PASSES;

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

pub(super) fn normalize_mass(system: &AssembledRealSystem, vector: &mut [f64]) {
    let mv = apply_m(system, vector);
    let mass_norm = dot(vector, &mv).abs().sqrt().max(1.0e-12);
    for value in vector.iter_mut() {
        *value /= mass_norm;
    }
}

pub(super) fn mass_inner_product(system: &AssembledRealSystem, left: &[f64], right: &[f64]) -> f64 {
    dot(left, &apply_m(system, right))
}

pub(super) fn orthonormalize_mass(
    system: &AssembledRealSystem,
    vector: &mut [f64],
    basis: &[Vec<f64>],
) {
    for mode in basis {
        let mv = apply_m(system, mode);
        let projection = dot(vector, &mv);
        for (value, base) in vector.iter_mut().zip(mode.iter()) {
            *value -= projection * *base;
        }
    }
}

pub(super) fn orthonormalize_mass_block(system: &AssembledRealSystem, vectors: &mut Vec<Vec<f64>>) {
    let mut basis = Vec::with_capacity(vectors.len());
    for mut vector in vectors.drain(..) {
        // Two passes materially improve orthogonality for clustered eigenspaces.
        for _ in 0..MODAL_MASS_ORTHOGONALIZATION_PASSES {
            orthonormalize_mass(system, &mut vector, &basis);
        }
        let norm_squared = mass_inner_product(system, &vector, &vector);
        if norm_squared.is_finite() && norm_squared > 1.0e-24 {
            normalize_mass(system, &mut vector);
            basis.push(vector);
        }
    }
    *vectors = basis;
}
