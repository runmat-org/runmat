use std::collections::BTreeMap;

use thiserror::Error;

use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use crate::{
    assembly::AssemblySummary,
    constraints::solve_constraint_multipliers_with_owner,
    planning::resource::{
        execute_scoped_slices_mut, LocalExecutionPlan, ScopedLocalExecutionError,
    },
};
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};

#[derive(Debug, Error)]
pub(crate) enum StructuralReactionError {
    #[error("structural reaction numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
    #[error("structural reaction recovery cancelled")]
    Cancelled,
    #[error("structural reaction residual has the wrong dimension or a nonfinite value")]
    InvalidResidual,
    #[error("constraint-row reactions could not be recovered from the structural residual")]
    Recovery,
    #[error("structural reaction local execution failed: {0}")]
    Scoped(#[from] ScopedLocalExecutionError<ReactionWorkError>),
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum ReactionWorkError {
    #[error("structural reaction recovery cancelled")]
    Cancelled,
}

pub(crate) fn recover_structural_reaction(
    summary: &AssemblySummary,
    residual: &[f64],
) -> Result<Vec<f64>, StructuralReactionError> {
    recover_structural_reaction_with_plan(summary, residual, None)
}

#[cfg(test)]
pub(crate) fn recover_structural_reaction_scoped(
    summary: &AssemblySummary,
    residual: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<Vec<f64>, StructuralReactionError> {
    recover_structural_reaction_with_plan(summary, residual, Some(local_execution))
}

fn recover_structural_reaction_with_plan(
    summary: &AssemblySummary,
    residual: &[f64],
    local_execution: Option<LocalExecutionPlan>,
) -> Result<Vec<f64>, StructuralReactionError> {
    use futures::FutureExt;
    recover_structural_reaction_with_capacity::<std::convert::Infallible>(
        summary,
        residual,
        local_execution,
        None,
    )
    .now_or_never()
    .expect("native reaction capacity is immediately ready")
    .map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error,
        NumericalCapacityError::Admission(never) => match never {},
    })
}

pub(crate) async fn recover_structural_reaction_with_capacity<A>(
    summary: &AssemblySummary,
    residual: &[f64],
    local_execution: Option<LocalExecutionPlan>,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<Vec<f64>, NumericalCapacityError<StructuralReactionError, A>> {
    if residual.len() != summary.dof_count || residual.iter().any(|value| !value.is_finite()) {
        return Err(StructuralReactionError::InvalidResidual.into());
    }
    if let Some(matrix) = summary.constraint_matrix.as_ref() {
        let mut rows = Vec::with_capacity(matrix.row_count());
        for row_index in 0..matrix.row_count() {
            let start = matrix.row_offsets[row_index];
            let end = matrix.row_offsets[row_index + 1];
            let row = matrix.column_indices[start..end]
                .iter()
                .copied()
                .zip(matrix.coefficients[start..end].iter().copied())
                .collect::<BTreeMap<_, _>>();
            rows.push(row);
        }
        let mut projection_work = OperationCount::ZERO;
        let mut scatter_work = OperationCount::ZERO;
        for row in &rows {
            let count = u64::try_from(row.len())
                .map_err(|_| StructuralReactionError::Accounting(LedgerError::CountOverflow))?;
            projection_work = projection_work
                .checked_add(
                    OperationCount::direct_real_dot(count)
                        .map_err(LedgerError::from)
                        .map_err(StructuralReactionError::from)?,
                )
                .map_err(LedgerError::from)
                .map_err(StructuralReactionError::from)?;
            scatter_work = scatter_work
                .checked_add(
                    OperationCount::from_operations(count)
                        .checked_mul(2)
                        .map_err(LedgerError::from)
                        .map_err(StructuralReactionError::from)?,
                )
                .map_err(LedgerError::from)
                .map_err(StructuralReactionError::from)?;
        }
        let projected_residual = measured(
            owner,
            UsageCategory::Solve,
            projection_work,
            || -> Result<Vec<f64>, StructuralReactionError> {
                Ok(if let Some(plan) = local_execution {
                    // Rows own disjoint scalar outputs. Each row's ordered sum is unchanged,
                    // and the existing projection vector is the only retained output.
                    let mut projected = vec![0.0; rows.len()];
                    let cancellation = crate::progress::cancellation_token();
                    execute_scoped_slices_mut(plan, &mut projected, |chunk, output| {
                        for (offset, value) in output.iter_mut().enumerate() {
                            if cancellation.is_cancelled() {
                                return Err(ReactionWorkError::Cancelled);
                            }
                            *value = rows[chunk.start() as usize + offset]
                                .iter()
                                .map(|(dof, coefficient)| coefficient * residual[*dof])
                                .sum();
                        }
                        Ok(())
                    })?;
                    projected
                } else {
                    rows.iter()
                        .map(|row| {
                            row.iter()
                                .map(|(dof, coefficient)| coefficient * residual[*dof])
                                .sum()
                        })
                        .collect()
                })
            },
        )
        .await?;
        let multipliers = solve_constraint_multipliers_with_owner(
            &rows,
            &projected_residual,
            |work| async move {
                match owner {
                    Some(owner) => owner.ensure_capacity(work).await,
                    None => Ok::<_, A>(()),
                }
            },
        )
        .await
        .map_err(|error| {
            error.map_numerical(|error| match error {
                crate::constraints::ConstraintMultiplierError::Accounting(cause) => {
                    StructuralReactionError::Accounting(cause)
                }
                crate::constraints::ConstraintMultiplierError::Cancelled => {
                    StructuralReactionError::Cancelled
                }
            })
        })?
        .ok_or(StructuralReactionError::Recovery)?;
        let reaction = measured(owner, UsageCategory::Solve, scatter_work, || {
            let mut reaction = vec![0.0; summary.dof_count];
            for (row, multiplier) in rows.iter().zip(multipliers) {
                for (dof, coefficient) in row {
                    reaction[*dof] += coefficient * multiplier;
                }
            }
            Ok::<_, StructuralReactionError>(reaction)
        })
        .await?;
        return Ok(reaction);
    }
    let constrained_dofs = summary.operator.constrained_dofs();
    if constrained_dofs.len() != residual.len() {
        return Err(StructuralReactionError::InvalidResidual.into());
    }
    if let Some(plan) = local_execution {
        // The admitted solve envelope already owns this one reaction vector.
        // Disjoint slices require no per-worker DOF-sized scratch or scatter.
        let mut reaction = vec![0.0; residual.len()];
        let cancellation = crate::progress::cancellation_token();
        execute_scoped_slices_mut(plan, &mut reaction, |chunk, output| {
            for (offset, value) in output.iter_mut().enumerate() {
                if cancellation.is_cancelled() {
                    return Err(ReactionWorkError::Cancelled);
                }
                let index = chunk.start() as usize + offset;
                if constrained_dofs[index] {
                    *value = residual[index];
                }
            }
            Ok(())
        })
        .map_err(StructuralReactionError::from)?;
        Ok(reaction)
    } else {
        Ok(residual
            .iter()
            .zip(constrained_dofs)
            .map(|(residual, constrained)| if *constrained { *residual } else { 0.0 })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        constraints::ResolvedConstraintMatrix,
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{fixture_model, FixtureId},
        planning::resource::{execute_scoped_chunks, LocalThreadCount},
    };

    // witness: FEA-SHARED-CONSTRAINTS-001/original_reaction_and_work @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: Independent constraint rows match the diagonal reaction law across workers.
    #[test]
    fn independent_constraint_rows_match_the_diagonal_reaction_law_across_workers() {
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
        let row_count = summary.dof_count.min(8);
        assert!(row_count >= 7);
        summary.constraint_matrix = Some(ResolvedConstraintMatrix {
            full_dof_count: summary.dof_count,
            row_offsets: (0..=row_count).collect(),
            column_indices: (0..row_count).collect(),
            coefficients: vec![1.0; row_count],
            right_hand_side: vec![0.0; row_count],
            row_identities: (0..row_count)
                .map(|row| runmat_execution::Digest::sha256(row.to_le_bytes()))
                .collect(),
            constraint_ids: (0..row_count).map(|row| format!("row_{row}")).collect(),
        });
        summary
            .constraint_matrix
            .as_ref()
            .unwrap()
            .validate()
            .unwrap();
        let residual = (0..summary.dof_count)
            .map(|dof| dof as f64 + 1.0)
            .collect::<Vec<_>>();
        let expected = (0..summary.dof_count)
            .map(|dof| if dof < row_count { residual[dof] } else { 0.0 })
            .collect::<Vec<_>>();
        assert_eq!(
            recover_structural_reaction(&summary, &residual).unwrap(),
            expected
        );
        for threads in [2, 3, 7] {
            let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    recover_structural_reaction_scoped(&summary, &residual, plan).unwrap(),
                    expected,
                );
            }
        }
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            recover_structural_reaction_scoped(&summary, &residual, LocalExecutionPlan::serial())
        });
        assert!(matches!(
            nested,
            Err(ScopedLocalExecutionError::Work {
                source: StructuralReactionError::Scoped(ScopedLocalExecutionError::NestedExecution),
                ..
            })
        ));
        let context = crate::progress::FeaProgressContext::default();
        context.cancellation.cancel();
        let _guard = crate::progress::replace_fea_progress_context(context);
        assert!(matches!(
            recover_structural_reaction_scoped(&summary, &residual, LocalExecutionPlan::serial()),
            Err(StructuralReactionError::Scoped(
                ScopedLocalExecutionError::Work {
                    source: ReactionWorkError::Cancelled,
                    ..
                }
            ))
        ));
    }
}
