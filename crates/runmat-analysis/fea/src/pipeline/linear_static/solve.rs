use crate::{
    assembly::AssemblySummary,
    contracts::{FeaRunError, LinearStaticSolveOptions},
    diagnostics::FeaDiagnosticSeverity,
    progress::check_cancelled,
    solve::{
        backend::build_backend,
        linear::{solve_linear_system_with_execution, LinearSolveResult},
        recover_structural_static_balance, StructuralStaticBalance,
    },
};

pub(super) fn solve_static_system(
    summary: &AssemblySummary,
    options: &LinearStaticSolveOptions,
) -> Result<LinearSolveResult, FeaRunError> {
    let algebra_backend = build_backend(options.algebra_backend_kind);
    if options.initial_displacement.as_ref().is_some_and(|values| {
        values.len() != summary.dof_count || values.iter().any(|value| !value.is_finite())
    }) {
        return Err(FeaRunError::InvalidModel(format!(
            "initial displacement state must contain exactly {} finite solver DOFs",
            summary.dof_count
        )));
    }
    let result = solve_linear_system_with_execution(
        summary,
        options.preconditioner_kind,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
        options.initial_displacement.as_deref(),
        options.local_execution,
    )
    .map_err(|error| {
        if crate::is_fea_cancelled() {
            FeaRunError::Cancelled
        } else {
            FeaRunError::Solve(error.to_string())
        }
    })?;
    check_cancelled("fea.run_linear_static")?;
    if !result.converged {
        let reason = result
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.severity == FeaDiagnosticSeverity::Error)
            .map(|diagnostic| format!("{}: {}", diagnostic.code, diagnostic.message))
            .unwrap_or_else(|| {
                format!(
                    "linear static solve did not converge after {} iterations (residual norm {})",
                    result.iterations, result.residual_norm
                )
            });
        return Err(FeaRunError::Solve(reason));
    }
    Ok(result)
}

pub(super) fn recover_static_balance(
    summary: &AssemblySummary,
    displacement: &[f64],
    options: &LinearStaticSolveOptions,
) -> Result<StructuralStaticBalance, FeaRunError> {
    recover_structural_static_balance(summary, displacement, options.local_execution).map_err(
        |error| {
            if error.is_cancelled() {
                FeaRunError::Cancelled
            } else {
                FeaRunError::Assembly(error.to_string())
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{
            AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
            RealSparseMatrix,
        },
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{fixture_model, FixtureId},
    };

    #[test]
    fn invalid_exact_operator_stops_static_pipeline_before_field_recovery() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let excitations = realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let mut summary = assemble_structural_system(
            &model,
            &excitations,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let dofs = summary.dof_count;
        let operators = summary.operator.operators();
        summary.operator = AssembledRealSystem::try_new(
            CanonicalRealOperators::try_new(
                RealSparseMatrix::diagonal(
                    vec![-1.0; dofs],
                    OperatorRole::Stiffness,
                    MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
                )
                .unwrap(),
                operators.mass().clone(),
                operators.damping().clone(),
            )
            .unwrap(),
            vec![false; dofs],
            vec![1.0; dofs],
        )
        .unwrap();
        let error = solve_static_system(&summary, &LinearStaticSolveOptions::default())
            .expect_err("invalid exact operator must not yield a recoverable static solution");
        assert!(
            matches!(error, FeaRunError::Solve(message) if message.contains("FEA_INVALID_REAL_SYSTEM"))
        );
    }
}
