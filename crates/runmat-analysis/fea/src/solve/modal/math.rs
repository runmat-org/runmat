use crate::algebra::{apply_m, AssembledRealSystem};

use super::MODAL_MASS_ORTHOGONALIZATION_PASSES;

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
}

pub(super) fn normalize_mass(system: &AssembledRealSystem, vector: &mut [f64]) {
    let mv = apply_m(system, vector);
    let generalized_mass = runmat_types::physical::Mass::new(dot(vector, &mv));
    let reference = crate::pipeline::modal_basis::ModalBasisCoordinates::canonical_reference_mass();
    let mass_norm = generalized_mass
        .ratio_to(reference)
        .canonical_value()
        .abs()
        .sqrt()
        .max(1.0e-12);
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

#[cfg(test)]
mod physical_normalization_tests {
    use super::*;
    use crate::algebra::{
        CanonicalRealOperators, MatrixProperties, OperatorRole, RealSparseMatrix,
    };

    #[test]
    fn one_kilogram_reference_is_invariant_under_eigenvector_rescaling() {
        let matrix = |values, role| {
            RealSparseMatrix::diagonal(values, role, MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE)
                .unwrap()
        };
        let operators = CanonicalRealOperators::try_new(
            matrix(vec![8.0, 27.0], OperatorRole::Stiffness),
            matrix(vec![2.0, 3.0], OperatorRole::Mass),
            matrix(vec![0.0, 0.0], OperatorRole::Damping),
        )
        .unwrap();
        let system = AssembledRealSystem::try_new(operators, vec![false; 2], vec![0.0; 2]).unwrap();
        let mut canonical = [1.0, 2.0];
        let mut rescaled = [37.0, 74.0];
        normalize_mass(&system, &mut canonical);
        normalize_mass(&system, &mut rescaled);
        for (left, right) in canonical.into_iter().zip(rescaled) {
            assert!((left - right).abs() < 1.0e-14);
        }
        let generalized_mass =
            runmat_types::physical::Mass::new(mass_inner_product(&system, &canonical, &canonical));
        assert!(
            (generalized_mass
                .ratio_to(
                    crate::pipeline::modal_basis::ModalBasisCoordinates::canonical_reference_mass()
                )
                .canonical_value()
                - 1.0)
                .abs()
                < 1.0e-14
        );
    }
}
