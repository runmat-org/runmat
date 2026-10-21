use thiserror::Error;

use crate::{
    algebra::spmv,
    assembly::AssemblySummary,
    execution::{ComplexStructuralBalance, StructuralComplexVector},
    solve::reaction::recover_structural_reaction,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicPhysicalState {
    pub displacement: StructuralComplexVector,
    pub reaction: StructuralComplexVector,
    pub balance: ComplexStructuralBalance,
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum HarmonicBalanceError {
    #[error("harmonic physical-state inputs have inconsistent dimensions")]
    DimensionMismatch,
    #[error("harmonic physical-state inputs contain a nonfinite value or invalid frequency")]
    NonFinite,
    #[error("harmonic structural reaction recovery failed")]
    ReactionRecovery,
}

/// Reconstruct the exact full-order phasor balance for one structural response.
///
/// Internal angular frequency is explicit here. Public plans and result metadata remain in Hz;
/// the frequency-task adapter owns the sole Hz-to-rad/s conversion before calling the solver and
/// this evaluator.
pub fn evaluate_harmonic_physical_state(
    summary: &AssemblySummary,
    angular_frequency_rad_s: f64,
    displacement_real: &[f64],
    displacement_imaginary: &[f64],
    applied_load_real: &[f64],
    applied_load_imaginary: &[f64],
) -> Result<HarmonicPhysicalState, HarmonicBalanceError> {
    let dof_count = summary.dof_count;
    if displacement_real.len() != dof_count
        || displacement_imaginary.len() != dof_count
        || applied_load_real.len() != dof_count
        || applied_load_imaginary.len() != dof_count
    {
        return Err(HarmonicBalanceError::DimensionMismatch);
    }
    if !angular_frequency_rad_s.is_finite()
        || angular_frequency_rad_s <= 0.0
        || displacement_real
            .iter()
            .chain(displacement_imaginary)
            .chain(applied_load_real)
            .chain(applied_load_imaginary)
            .any(|value| !value.is_finite())
    {
        return Err(HarmonicBalanceError::NonFinite);
    }
    let operators = summary.operator.operators();
    let mass_real = apply(operators.mass(), displacement_real)?;
    let mass_imaginary = apply(operators.mass(), displacement_imaginary)?;
    let damping_real = apply(operators.damping(), displacement_real)?;
    let damping_imaginary = apply(operators.damping(), displacement_imaginary)?;
    let elastic_real = apply(operators.stiffness(), displacement_real)?;
    let elastic_imaginary = apply(operators.stiffness(), displacement_imaginary)?;
    let omega_squared = angular_frequency_rad_s * angular_frequency_rad_s;
    let inertia = complex_scaled(&mass_real, &mass_imaginary, -omega_squared, 0.0)?;
    let damping = complex_scaled(
        &damping_real,
        &damping_imaginary,
        0.0,
        angular_frequency_rad_s,
    )?;
    let elastic = StructuralComplexVector::new(elastic_real, elastic_imaginary)
        .map_err(|_| HarmonicBalanceError::NonFinite)?;
    let applied =
        StructuralComplexVector::new(applied_load_real.to_vec(), applied_load_imaginary.to_vec())
            .map_err(|_| HarmonicBalanceError::NonFinite)?;
    let balance = ComplexStructuralBalance::new(applied, inertia, damping, elastic)
        .map_err(|_| HarmonicBalanceError::NonFinite)?;
    let reaction_real = recover_structural_reaction(summary, balance.equilibrium_residual().real())
        .map_err(|_| HarmonicBalanceError::ReactionRecovery)?;
    let reaction_imaginary =
        recover_structural_reaction(summary, balance.equilibrium_residual().imaginary())
            .map_err(|_| HarmonicBalanceError::ReactionRecovery)?;
    let reaction = StructuralComplexVector::new(reaction_real, reaction_imaginary)
        .map_err(|_| HarmonicBalanceError::NonFinite)?;
    let displacement =
        StructuralComplexVector::new(displacement_real.to_vec(), displacement_imaginary.to_vec())
            .map_err(|_| HarmonicBalanceError::NonFinite)?;
    Ok(HarmonicPhysicalState {
        displacement,
        reaction,
        balance,
    })
}

fn apply(
    operator: &crate::algebra::RealSparseMatrix,
    value: &[f64],
) -> Result<Vec<f64>, HarmonicBalanceError> {
    spmv(operator, value).map_err(|_| HarmonicBalanceError::DimensionMismatch)
}

/// `(a + ib) * (real + i imaginary)`.
fn complex_scaled(
    real: &[f64],
    imaginary: &[f64],
    a: f64,
    b: f64,
) -> Result<StructuralComplexVector, HarmonicBalanceError> {
    if real.len() != imaginary.len() {
        return Err(HarmonicBalanceError::DimensionMismatch);
    }
    StructuralComplexVector::new(
        real.iter()
            .zip(imaginary)
            .map(|(real, imaginary)| a * real - b * imaginary)
            .collect(),
        real.iter()
            .zip(imaginary)
            .map(|(real, imaginary)| b * real + a * imaginary)
            .collect(),
    )
    .map_err(|_| HarmonicBalanceError::NonFinite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::StructuralFormulationRequirement,
        excitation::ExcitationSamplePoint,
        fixtures::FixtureId,
        solve::dynamics::{solve_direct_harmonic, DirectHarmonicPolicy},
    };

    #[test]
    fn direct_harmonic_solution_closes_full_physical_equilibrium_after_reactions() {
        let model = crate::fixtures::fixture_model(FixtureId::CantileverLinearStatic);
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &excitations,
            None,
            StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        let imaginary_load = vec![0.0; summary.dof_count];
        let omega = 0.5;
        let response = solve_direct_harmonic(
            &summary,
            &imaginary_load,
            omega,
            DirectHarmonicPolicy {
                relative_tolerance: 1.0e-10,
                ..DirectHarmonicPolicy::default()
            },
        )
        .unwrap();
        let physical = evaluate_harmonic_physical_state(
            &summary,
            omega,
            &response.displacement_real,
            &response.displacement_imaginary,
            summary.operator.load(),
            &imaginary_load,
        )
        .unwrap();
        let (_, relative) = physical
            .balance
            .closure_metrics(&physical.reaction)
            .unwrap();
        assert!(relative <= 1.0e-10, "relative physical balance={relative}");
    }
}
