//! The current vector supplies an energy-minimizing start along its own span.
//! CG still solves the original Kx=Mv equation with its original stopping policy.
use super::{apply_k, dot, AssembledRealSystem};

pub(super) fn rayleigh_start(
    system: &AssembledRealSystem,
    vector: &[f64],
    mass_vector: &[f64],
) -> Option<Vec<f64>> {
    let mut start = apply_k(system, vector);
    let stiffness_energy = dot(vector, &start);
    let mass_energy = dot(vector, mass_vector);
    if !stiffness_energy.is_finite()
        || stiffness_energy <= 0.0
        || !mass_energy.is_finite()
        || mass_energy <= 0.0
    {
        return None;
    }
    let inverse_rayleigh = mass_energy / stiffness_energy;
    if !inverse_rayleigh.is_finite() || inverse_rayleigh <= 0.0 {
        return None;
    }
    for (value, source) in start.iter_mut().zip(vector) {
        *value = inverse_rayleigh * source;
    }
    start.iter().all(|value| value.is_finite()).then_some(start)
}

#[cfg(test)]
mod tests {
    use super::super::tests::diagonal_system;
    use super::*;

    #[test]
    fn an_original_generalized_eigenvector_starts_at_its_exact_inverse() {
        let system = diagonal_system(vec![4., 27.], vec![2., 3.]);
        let vector = [2., 0.];
        let rhs = super::super::apply_m(&system, &vector);
        let start = rayleigh_start(&system, &vector, &rhs).unwrap();
        assert_eq!(start, vec![1., 0.]);
        assert_eq!(apply_k(&system, &start), rhs);
    }

    #[test]
    fn a_mixed_vector_reduces_inverse_error_in_the_stiffness_energy_norm() {
        let system = diagonal_system(vec![4., 27.], vec![2., 3.]);
        let vector = [1., 1.];
        let rhs = super::super::apply_m(&system, &vector);
        let start = rayleigh_start(&system, &vector, &rhs).unwrap();
        let exact = [0.5, 1. / 9.];
        let energy = |candidate: &[f64]| {
            candidate
                .iter()
                .zip(exact)
                .zip([4., 27.])
                .map(|((value, exact), stiffness)| stiffness * (value - exact).powi(2))
                .sum::<f64>()
        };
        assert!(energy(&start) < energy(&[0., 0.]));
        // It is only a starting guess: the mixed-vector equation is not solved.
        assert_ne!(apply_k(&system, &start), rhs);
    }
}
