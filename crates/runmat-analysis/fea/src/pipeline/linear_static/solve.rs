use crate::{
    assembly::AssemblySummary,
    contracts::{FeaRunError, LinearStaticSolveOptions},
    diagnostics::FeaDiagnosticSeverity,
    planning::resource::SolverAdmissionResourcePolicy,
    progress::check_cancelled,
    solve::{
        backend::build_backend,
        linear::{solve_linear_system_with_execution_admitted, LinearSolveResult},
        recover_structural_static_balance, StructuralStaticBalance,
    },
};

pub(super) fn solve_static_system(
    summary: &AssemblySummary,
    options: &LinearStaticSolveOptions,
    admission: Option<&SolverAdmissionResourcePolicy>,
    planned_reduced_support: Option<&crate::algebra::SparsePattern>,
    elastic: Option<&mut crate::algebra::LinearPreparationReuse>,
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
    // The elastic rigidity path proves without a factor and always solves with AMG-PCG.
    let preconditioner = if elastic.is_some() {
        crate::solve::preconditioner::SpdPreconditionerKind::Amg
    } else {
        options.preconditioner_kind
    };
    let result = solve_linear_system_with_execution_admitted(
        summary,
        preconditioner,
        options.algebra_backend_kind,
        algebra_backend.as_ref(),
        options.initial_displacement.as_deref(),
        options.local_execution,
        admission,
        planned_reduced_support,
        elastic,
    )
    .map_err(|error| {
        if crate::is_fea_cancelled() {
            FeaRunError::Cancelled
        } else {
            FeaRunError::Solve(error.to_string())
        }
    })?;
    finish_static_solve(result)
}

pub(super) async fn solve_static_system_with_owner<O, F, A>(
    summary: &AssemblySummary,
    options: &LinearStaticSolveOptions,
    admission: Option<&SolverAdmissionResourcePolicy>,
    planned_reduced_support: Option<&crate::algebra::SparsePattern>,
    elastic: Option<&mut crate::algebra::LinearPreparationReuse>,
    before_capacity: O,
) -> Result<LinearSolveResult, A>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<FeaRunError>,
{
    use crate::algebra::coercivity::NumericalCapacityError;
    use crate::solve::linear::{
        solve_operator_system_with_settings_owned, ScopedLinearSolveSettings,
    };
    if options.initial_displacement.as_ref().is_some_and(|values| {
        values.len() != summary.dof_count || values.iter().any(|value| !value.is_finite())
    }) {
        return Err(FeaRunError::InvalidModel(format!(
            "initial displacement state must contain exactly {} finite solver DOFs",
            summary.dof_count
        ))
        .into());
    }
    let preconditioner = if elastic.is_some() {
        crate::solve::preconditioner::SpdPreconditionerKind::Amg
    } else {
        options.preconditioner_kind
    };
    if summary.affine_constraint_reduction.is_some() || summary.constraint_matrix.is_some() {
        let derived_reduction = summary
            .constraint_matrix
            .as_ref()
            .and_then(|matrix| matrix.affine_reduction().ok());
        let reduction = summary
            .affine_constraint_reduction
            .as_ref()
            .or(derived_reduction.as_ref())
            .ok_or_else(|| {
                FeaRunError::Solve(
                    "canonical structural constraint matrix has no affine reduction".into(),
                )
            })?;
        let kind = if elastic.is_some() {
            preconditioner
        } else {
            crate::solve::preconditioner::SpdPreconditionerKind::Auto
        };
        let mut settings =
            ScopedLinearSolveSettings::with_default_tolerance(options.local_execution, admission);
        settings.preparation = elastic;
        let view = crate::solve::linear::StructuralLinearSystemView {
            planned_reduced_support,
            ..crate::solve::linear::StructuralLinearSystemView::from(summary)
        };
        let result = crate::solve::solve_affine_constraint_system_with_kind_owned(
            &view,
            reduction,
            options.algebra_backend_kind,
            options.initial_displacement.as_deref(),
            crate::solve::AffineOwnedSolveOptions::new(settings, kind, false),
            before_capacity,
        )
        .await
        .map_err(|error| match error {
            NumericalCapacityError::Admission(error) => error,
            NumericalCapacityError::Numerical(crate::algebra::ScopedSpmvError::Usage(cause)) => {
                FeaRunError::Usage(cause).into()
            }
            NumericalCapacityError::Numerical(error) => {
                FeaRunError::Solve(error.to_string()).into()
            }
        })?;
        return finish_static_solve(result).map_err(Into::into);
    }
    let mut settings =
        ScopedLinearSolveSettings::with_default_tolerance(options.local_execution, admission);
    settings.preparation = elastic;
    let result = solve_operator_system_with_settings_owned(
        &summary.operator,
        preconditioner,
        options.algebra_backend_kind,
        options.initial_displacement.as_deref(),
        settings,
        before_capacity,
    )
    .await
    .map_err(|error| match error {
        NumericalCapacityError::Admission(error) => error,
        NumericalCapacityError::Numerical(error) => error.into(),
    })?;
    finish_static_solve(result).map_err(Into::into)
}

fn finish_static_solve(result: LinearSolveResult) -> Result<LinearSolveResult, FeaRunError> {
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
    recover_structural_static_balance(summary, displacement, options.local_execution)
        .map_err(|error| error.into_run_error())
}

pub(super) async fn recover_static_balance_with_owner<O, F, A>(
    summary: &AssemblySummary,
    displacement: &[f64],
    options: &LinearStaticSolveOptions,
    mut before: O,
) -> Result<StructuralStaticBalance, A>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<FeaRunError>,
{
    use crate::algebra::{
        capacity::PhaseCapacityOwner, coercivity::NumericalCapacityError, NumericalCapacityPhase,
    };
    let owner = PhaseCapacityOwner::new(&mut before, NumericalCapacityPhase::StaticBalance);
    crate::solve::recover_structural_static_balance_with_capacity(
        summary,
        displacement,
        options.local_execution,
        Some(&owner),
    )
    .await
    .map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error.into_run_error().into(),
        NumericalCapacityError::Admission(error) => error,
    })
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
        let error = solve_static_system(
            &summary,
            &LinearStaticSolveOptions::default(),
            None,
            None,
            None,
        )
        .expect_err("invalid exact operator must not yield a recoverable static solution");
        assert!(
            matches!(error, FeaRunError::Solve(message) if message.contains("FEA_INVALID_REAL_SYSTEM"))
        );
    }
}
