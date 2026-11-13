use thiserror::Error;

use super::{dense::solve_square, ModalReducedSystem};

#[derive(Clone, Debug, PartialEq)]
pub struct ModalHarmonicLoad {
    pub real: runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<f64>>,
    pub imaginary: runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModalHarmonicResponse {
    pub angular_frequency_rad_s: f64,
    pub displacement_real:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    pub displacement_imaginary:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    pub residual_norm: f64,
    pub relative_residual: f64,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModalHarmonicError {
    #[error("modal harmonic inputs have inconsistent dimensions")]
    DimensionMismatch,
    #[error("modal harmonic inputs contain a nonfinite value")]
    NonFinite,
    #[error("modal harmonic dynamic stiffness is singular")]
    Singular,
    #[error("modal harmonic solve was cancelled")]
    Cancelled,
    #[error("modal harmonic numerical usage: {0}")]
    Usage(runmat_simulation_usage::LedgerError),
}

/// Solve `(K - omega^2 M + i omega C) q = f` without discarding modal coupling.
pub fn solve_modal_harmonic(
    system: &ModalReducedSystem,
    angular_frequency_rad_s: f64,
    load: &ModalHarmonicLoad,
) -> Result<ModalHarmonicResponse, ModalHarmonicError> {
    solve_operators(
        DenseHarmonicOperators {
            size: system.mode_count(),
            stiffness: system.stiffness().canonical_value(),
            mass: system.mass().canonical_value(),
            damping: system.damping().canonical_value(),
        },
        angular_frequency_rad_s,
        load,
    )
}

#[derive(Clone, Copy)]
struct DenseHarmonicOperators<'a> {
    size: usize,
    stiffness: &'a [f64],
    mass: &'a [f64],
    damping: &'a [f64],
}

fn solve_operators(
    operators: DenseHarmonicOperators<'_>,
    angular_frequency_rad_s: f64,
    load: &ModalHarmonicLoad,
) -> Result<ModalHarmonicResponse, ModalHarmonicError> {
    let size = operators.size;
    if load.real.canonical_values().len() != size || load.imaginary.canonical_values().len() != size
    {
        return Err(ModalHarmonicError::DimensionMismatch);
    }
    let matrix_size = size
        .checked_mul(size)
        .ok_or(ModalHarmonicError::DimensionMismatch)?;
    if size == 0
        || operators.stiffness.len() != matrix_size
        || operators.mass.len() != matrix_size
        || operators.damping.len() != matrix_size
    {
        return Err(ModalHarmonicError::DimensionMismatch);
    }
    if !angular_frequency_rad_s.is_finite()
        || angular_frequency_rad_s <= 0.0
        || operators
            .stiffness
            .iter()
            .chain(operators.mass)
            .chain(operators.damping)
            .any(|value| !value.is_finite())
        || load
            .real
            .canonical_values()
            .iter()
            .chain(load.imaginary.canonical_values())
            .any(|value| !value.is_finite())
    {
        return Err(ModalHarmonicError::NonFinite);
    }
    if crate::is_fea_cancelled() {
        return Err(ModalHarmonicError::Cancelled);
    }

    let block_size = size * 2;
    let mut block = vec![0.0; block_size * block_size];
    for row in 0..size {
        for column in 0..size {
            let index = row * size + column;
            let real = operators.stiffness[index]
                - angular_frequency_rad_s.powi(2) * operators.mass[index];
            let imaginary = angular_frequency_rad_s * operators.damping[index];
            block[row * block_size + column] = real;
            block[row * block_size + size + column] = -imaginary;
            block[(size + row) * block_size + column] = imaginary;
            block[(size + row) * block_size + size + column] = real;
        }
    }
    let mut right_hand_side = load.real.canonical_values().clone();
    right_hand_side.extend_from_slice(load.imaginary.canonical_values());
    let solution =
        solve_square(&block, &right_hand_side, block_size).map_err(|error| match error {
            super::dense::DenseSolveError::DimensionMismatch => {
                ModalHarmonicError::DimensionMismatch
            }
            super::dense::DenseSolveError::NonFinite => ModalHarmonicError::NonFinite,
            super::dense::DenseSolveError::Singular => ModalHarmonicError::Singular,
            super::dense::DenseSolveError::Usage(error) => ModalHarmonicError::Usage(error),
        })?;
    let displacement_real = solution[..size].to_vec();
    let displacement_imaginary = solution[size..].to_vec();
    let (residual_norm, relative_residual) =
        residual_metrics(&block, &solution, &right_hand_side, block_size);
    Ok(ModalHarmonicResponse {
        angular_frequency_rad_s,
        displacement_real: runmat_types::physical::Quantity::try_new(displacement_real)
            .map_err(|_| ModalHarmonicError::NonFinite)?,
        displacement_imaginary: runmat_types::physical::Quantity::try_new(displacement_imaginary)
            .map_err(|_| ModalHarmonicError::NonFinite)?,
        residual_norm,
        relative_residual,
    })
}

fn residual_metrics(matrix: &[f64], solution: &[f64], load: &[f64], size: usize) -> (f64, f64) {
    let applied = (0..size)
        .map(|row| {
            (0..size)
                .map(|column| matrix[row * size + column] * solution[column])
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let residual_norm = load
        .iter()
        .zip(&applied)
        .fold(0.0_f64, |norm, (load, applied)| norm.hypot(load - applied));
    let load_norm = load.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));
    let applied_norm = applied
        .iter()
        .fold(0.0_f64, |norm, value| norm.hypot(*value));
    (
        residual_norm,
        residual_norm / (load_norm + applied_norm).max(f64::MIN_POSITIVE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve_dense_modal(
        stiffness: &[f64],
        mass: &[f64],
        damping: &[f64],
        omega: f64,
        load: ModalHarmonicLoad,
    ) -> ModalHarmonicResponse {
        solve_operators(
            DenseHarmonicOperators {
                size: load.real.canonical_values().len(),
                stiffness,
                mass,
                damping,
            },
            omega,
            &load,
        )
        .unwrap()
    }

    #[test]
    fn damped_single_mode_matches_closed_form_complex_response() {
        let response = solve_dense_modal(
            &[4.0],
            &[1.0],
            &[0.5],
            2.0,
            ModalHarmonicLoad {
                real: runmat_types::physical::Quantity::new(vec![1.0]),
                imaginary: runmat_types::physical::Quantity::new(vec![0.0]),
            },
        );
        assert!(response.displacement_real.canonical_values()[0].abs() <= 1.0e-14);
        assert!((response.displacement_imaginary.canonical_values()[0] + 1.0).abs() <= 1.0e-14);
        assert!(response.relative_residual <= 1.0e-15);
    }

    #[test]
    fn coupled_damping_transfers_harmonic_response_between_modes() {
        let response = solve_dense_modal(
            &[4.0, 0.0, 0.0, 9.0],
            &[1.0, 0.0, 0.0, 1.0],
            &[0.2, 0.1, 0.1, 0.3],
            1.5,
            ModalHarmonicLoad {
                real: runmat_types::physical::Quantity::new(vec![1.0, 0.0]),
                imaginary: runmat_types::physical::Quantity::new(vec![0.0, 0.0]),
            },
        );
        assert!(response.displacement_real.canonical_values()[1].abs() > 1.0e-4);
        assert!(response.displacement_imaginary.canonical_values()[1].abs() > 1.0e-4);
        assert!(response.relative_residual <= 1.0e-14);
    }
}
