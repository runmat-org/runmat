//! Revalidate retained terminal physics and projections without advancing equilibrium.
use super::{map_assembly_error, map_error, recovery, NonlinearSolveOptions, NonlinearSolveResult};
use crate::{
    assembly::{nonlinear_solid::CanonicalNonlinearSolidBasis, AssemblySummary},
    solve::nonlinear::solid::SolidNewtonProblem,
    FeaRunError,
};
use runmat_analysis_core::AnalysisModel;

pub(crate) fn validate_original_terminal_result(
    model: &AnalysisModel,
    summary: &AssemblySummary,
    options: &NonlinearSolveOptions,
    result: &NonlinearSolveResult,
) -> Result<(), FeaRunError> {
    let mesh = options
        .solver_mesh
        .as_ref()
        .ok_or_else(|| FeaRunError::InvalidModel("original J2 validation lacks its mesh".into()))?;
    let basis =
        CanonicalNonlinearSolidBasis::prepare(model, mesh, summary).map_err(map_assembly_error)?;
    let problem =
        SolidNewtonProblem::prepare(&basis, summary, options.newton_policy()).map_err(map_error)?;
    problem
        .validate_accepted(&result.accepted_state)
        .map_err(map_error)?;
    let state = &result.accepted_state;
    let evaluation = basis
        .evaluate(&state.displacement, &state.constitutive)
        .map_err(|e| FeaRunError::Solve(e.to_string()))?;
    let fields = recovery::recover(&evaluation, &state.constitutive)?;
    let pairs = [
        (
            result.internal_force_snapshots.last(),
            evaluation.internal_force.as_slice(),
        ),
        (result.stress_snapshots.last(), fields.stress.as_slice()),
        (
            result.total_strain_snapshots.last(),
            fields.total_strain.as_slice(),
        ),
        (
            result.plastic_strain_snapshots.last(),
            fields.plastic.as_slice(),
        ),
        (
            result.equivalent_plastic_strain_snapshots.last(),
            fields.alpha.as_slice(),
        ),
        (
            result.von_mises_snapshots.last(),
            fields.von_mises.as_slice(),
        ),
    ];
    // Re-evaluation can take a return-map correction at machine roundoff. This
    // comparison is roundoff-only, independent of the caller's Newton tolerance.
    for (stored, physical) in pairs {
        let stored = stored
            .ok_or_else(|| FeaRunError::Solve("original J2 terminal projection absent".into()))?;
        let scale = stored
            .iter()
            .chain(physical)
            .fold(f64::MIN_POSITIVE, |scale, value| scale.max(value.abs()));
        let roundoff = 512.0 * f64::EPSILON * scale;
        if stored.len() != physical.len()
            || stored
                .iter()
                .zip(physical)
                .any(|(a, b)| !a.is_finite() || !b.is_finite() || (a - b).abs() > roundoff)
        {
            return Err(FeaRunError::Solve(
                "original J2 terminal projection differs from constitutive state".into(),
            ));
        }
    }
    Ok(())
}
