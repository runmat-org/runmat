//! Physical equilibrium and work balance for a completed structural static solve.

use crate::algebra::{capacity::measured, coercivity::NumericalCapacityError};
use runmat_simulation_usage::{AsyncCapacityOwner, LedgerError, OperationCount, UsageCategory};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::{dot_scoped, spmv_scoped, vector_sub_scoped, ScopedSpmvError, SpmvWorkError},
    assembly::AssemblySummary,
    planning::resource::{LocalExecutionPlan, ScopedLocalExecutionError},
    solve::reaction::{
        recover_structural_reaction_with_capacity, ReactionWorkError, StructuralReactionError,
    },
};

const RELATIVE_TOLERANCE: f64 = 1.0e-8;
#[cfg(test)]
use super::reaction::recover_structural_reaction_scoped;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStaticBalanceEvidence {
    pub internal_work_joules: f64,
    pub applied_load_work_joules: f64,
    pub constraint_work_joules: f64,
    pub work_balance_residual_joules: f64,
    pub work_balance_relative_error: f64,
    pub equilibrium_residual_norm_newtons: f64,
    pub equilibrium_scale_newtons: f64,
    pub equilibrium_relative_error: f64,
    pub relative_tolerance: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructuralStaticBalance {
    pub evidence: StructuralStaticBalanceEvidence,
    pub reaction: Vec<f64>,
}

#[derive(Debug, Error)]
pub(crate) enum StructuralStaticBalanceError {
    #[error("structural balance numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
    #[error("structural static balance state has the wrong dimension or a nonfinite value")]
    InvalidState,
    #[error("structural static reaction recovery failed: {0}")]
    Reaction(#[from] StructuralReactionError),
    #[error("structural static balance local execution failed: {0}")]
    Scoped(#[from] ScopedSpmvError),
    #[error("structural static balance evidence is inconsistent")]
    InvalidEvidence,
}

impl StructuralStaticBalanceError {
    pub(crate) fn into_run_error(self) -> crate::contracts::FeaRunError {
        match self {
            Self::Accounting(cause)
            | Self::Scoped(ScopedSpmvError::Usage(cause))
            | Self::Reaction(StructuralReactionError::Accounting(cause)) => {
                crate::contracts::FeaRunError::Usage(cause)
            }
            error if error.is_cancelled() => crate::contracts::FeaRunError::Cancelled,
            other => crate::contracts::FeaRunError::Assembly(other.to_string()),
        }
    }
    pub(crate) fn is_cancelled(&self) -> bool {
        matches!(
            self,
            Self::Reaction(StructuralReactionError::Cancelled)
                | Self::Scoped(ScopedSpmvError::Execution(
                    ScopedLocalExecutionError::Work {
                        source: SpmvWorkError::Cancelled,
                        ..
                    }
                ))
                | Self::Reaction(StructuralReactionError::Scoped(
                    ScopedLocalExecutionError::Work {
                        source: ReactionWorkError::Cancelled,
                        ..
                    }
                ))
        )
    }
}

impl StructuralStaticBalanceEvidence {
    pub(crate) fn validate(&self) -> Result<(), StructuralStaticBalanceError> {
        let values = [
            self.internal_work_joules,
            self.applied_load_work_joules,
            self.constraint_work_joules,
            self.work_balance_residual_joules,
            self.work_balance_relative_error,
            self.equilibrium_residual_norm_newtons,
            self.equilibrium_scale_newtons,
            self.equilibrium_relative_error,
            self.relative_tolerance,
        ];
        let work_residual =
            self.internal_work_joules - self.applied_load_work_joules - self.constraint_work_joules;
        let work_scale = self
            .internal_work_joules
            .abs()
            .max(self.applied_load_work_joules.abs())
            .max(self.constraint_work_joules.abs());
        let accepted = self.work_balance_relative_error <= self.relative_tolerance
            && self.equilibrium_relative_error <= self.relative_tolerance;
        if values.iter().any(|value| !value.is_finite())
            || self.work_balance_relative_error < 0.0
            || self.equilibrium_residual_norm_newtons < 0.0
            || self.equilibrium_scale_newtons < 0.0
            || self.equilibrium_relative_error < 0.0
            || self.relative_tolerance <= 0.0
            || (work_residual - self.work_balance_residual_joules).abs()
                > work_scale * f64::EPSILON * 16.0
            || (relative_error(work_residual.abs(), work_scale) - self.work_balance_relative_error)
                .abs()
                > f64::EPSILON * 16.0
            || (relative_error(
                self.equilibrium_residual_norm_newtons,
                self.equilibrium_scale_newtons,
            ) - self.equilibrium_relative_error)
                .abs()
                > f64::EPSILON * 16.0
            || self.accepted != accepted
        {
            return Err(StructuralStaticBalanceError::InvalidEvidence);
        }
        Ok(())
    }
}

pub(crate) fn recover_structural_static_balance(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
) -> Result<StructuralStaticBalance, StructuralStaticBalanceError> {
    use futures::FutureExt;
    recover_structural_static_balance_with_capacity::<std::convert::Infallible>(
        summary,
        displacement,
        local_execution,
        None,
    )
    .now_or_never()
    .expect("native balance capacity is immediately ready")
    .map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error,
        NumericalCapacityError::Admission(never) => match never {},
    })
}

pub(crate) async fn recover_structural_static_balance_with_capacity<A>(
    summary: &AssemblySummary,
    displacement: &[f64],
    local_execution: LocalExecutionPlan,
    owner: Option<&dyn AsyncCapacityOwner<A>>,
) -> Result<StructuralStaticBalance, NumericalCapacityError<StructuralStaticBalanceError, A>> {
    if displacement.len() != summary.dof_count
        || summary.physical_load.len() != summary.dof_count
        || displacement.iter().any(|value| !value.is_finite())
        || summary.physical_load.iter().any(|value| !value.is_finite())
    {
        return Err(StructuralStaticBalanceError::InvalidState.into());
    }
    // The structural solve plan already charges whole-vector scratch and scoped
    // CSR output. Reuse it here; there is no second pool or per-worker full vector.
    let matrix = summary.operator.operators().stiffness();
    let mut spmv_work = OperationCount::ZERO;
    for row in 0..matrix.rows() {
        spmv_work = spmv_work
            .checked_add(dot_work(matrix.pattern().row_range(row).len())?)
            .map_err(LedgerError::from)
            .map_err(StructuralStaticBalanceError::from)?;
    }
    let length = count(displacement.len())?;
    let dot_work = dot_work(displacement.len())?;
    let norm_work = dot_work
        .checked_add(OperationCount::from_operations(1))
        .map_err(LedgerError::from)
        .map_err(StructuralStaticBalanceError::from)?;
    let internal_force = measured(owner, UsageCategory::Solve, spmv_work, || {
        spmv_scoped(
            summary.operator.operators().stiffness(),
            displacement,
            local_execution,
        )
        .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let equilibrium_residual = measured(owner, UsageCategory::Solve, length, || {
        vector_sub_scoped(&internal_force, &summary.physical_load, local_execution)
            .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let reaction = recover_structural_reaction_with_capacity(
        summary,
        &equilibrium_residual,
        Some(local_execution),
        owner,
    )
    .await
    .map_err(|error| error.map_numerical(StructuralStaticBalanceError::from))?;
    let free_residual = measured(owner, UsageCategory::Solve, length, || {
        vector_sub_scoped(&equilibrium_residual, &reaction, local_execution)
            .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let equilibrium_residual_norm_newtons =
        measured(owner, UsageCategory::Solve, norm_work, || {
            norm(&free_residual, local_execution).map_err(StructuralStaticBalanceError::from)
        })
        .await?;
    let internal_norm = measured(owner, UsageCategory::Solve, norm_work, || {
        norm(&internal_force, local_execution).map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let load_norm = measured(owner, UsageCategory::Solve, norm_work, || {
        norm(&summary.physical_load, local_execution).map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let reaction_norm = measured(owner, UsageCategory::Solve, norm_work, || {
        norm(&reaction, local_execution).map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let equilibrium_scale_newtons = internal_norm.max(load_norm).max(reaction_norm);
    let equilibrium_relative_error = measured(
        owner,
        UsageCategory::Solve,
        OperationCount::from_operations(u64::from(equilibrium_scale_newtons != 0.0)),
        || {
            Ok::<_, StructuralStaticBalanceError>(relative_error(
                equilibrium_residual_norm_newtons,
                equilibrium_scale_newtons,
            ))
        },
    )
    .await?;
    let internal_work_joules = measured(owner, UsageCategory::Solve, dot_work, || {
        dot_scoped(displacement, &internal_force, local_execution)
            .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let applied_load_work_joules = measured(owner, UsageCategory::Solve, dot_work, || {
        dot_scoped(displacement, &summary.physical_load, local_execution)
            .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let constraint_work_joules = measured(owner, UsageCategory::Solve, dot_work, || {
        dot_scoped(displacement, &reaction, local_execution)
            .map_err(StructuralStaticBalanceError::from)
    })
    .await?;
    let work_balance_residual_joules = measured(
        owner,
        UsageCategory::Solve,
        OperationCount::from_operations(2),
        || {
            Ok::<_, StructuralStaticBalanceError>(
                internal_work_joules - applied_load_work_joules - constraint_work_joules,
            )
        },
    )
    .await?;
    let work_scale = internal_work_joules
        .abs()
        .max(applied_load_work_joules.abs())
        .max(constraint_work_joules.abs());
    let work_balance_relative_error = measured(
        owner,
        UsageCategory::Solve,
        OperationCount::from_operations(u64::from(work_scale != 0.0)),
        || {
            Ok::<_, StructuralStaticBalanceError>(relative_error(
                work_balance_residual_joules.abs(),
                work_scale,
            ))
        },
    )
    .await?;
    let accepted = work_balance_relative_error <= RELATIVE_TOLERANCE
        && equilibrium_relative_error <= RELATIVE_TOLERANCE;
    let evidence = StructuralStaticBalanceEvidence {
        internal_work_joules,
        applied_load_work_joules,
        constraint_work_joules,
        work_balance_residual_joules,
        work_balance_relative_error,
        equilibrium_residual_norm_newtons,
        equilibrium_scale_newtons,
        equilibrium_relative_error,
        relative_tolerance: RELATIVE_TOLERANCE,
        accepted,
    };
    evidence.validate()?;
    Ok(StructuralStaticBalance { evidence, reaction })
}

fn count(length: usize) -> Result<OperationCount, StructuralStaticBalanceError> {
    Ok(OperationCount::from_operations(
        u64::try_from(length).map_err(|_| LedgerError::CountOverflow)?,
    ))
}
fn dot_work(length: usize) -> Result<OperationCount, StructuralStaticBalanceError> {
    OperationCount::direct_real_dot(count(length)?.operations())
        .map_err(LedgerError::from)
        .map_err(StructuralStaticBalanceError::from)
}

fn norm(values: &[f64], local_execution: LocalExecutionPlan) -> Result<f64, ScopedSpmvError> {
    Ok(dot_scoped(values, values, local_execution)?.sqrt())
}

fn relative_error(residual: f64, scale: f64) -> f64 {
    if scale == 0.0 {
        if residual == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        residual / scale
    }
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        AnalysisConstraint, CoordinateFrame, Displacement, PrescribedStructuralDofs,
        PrescribedValue, RegionTarget, StructuralConstraint, StructuralConstraintKind,
        StructuralTranslationDof, TranslationDofValue,
    };

    use super::*;
    use crate::{
        assembly::{
            assemble_structural_system, dofs::StructuralDofKind, StructuralFormulationRequirement,
        },
        constraints::{CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintResult},
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{
            fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
            FixtureId,
        },
        planning::resource::{execute_scoped_chunks, LocalThreadCount},
        solve::{
            backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
            linear::solve_linear_system,
            preconditioner::SpdPreconditionerKind,
        },
    };

    #[test]
    fn static_balance_evidence_uses_physical_scale_even_below_one() {
        assert_eq!(relative_error(0.0, 0.0), 0.0);
        assert_eq!(relative_error(5.0e-13, 1.0e-12), 0.5);

        let evidence = StructuralStaticBalanceEvidence {
            internal_work_joules: 1.0e-12,
            applied_load_work_joules: 0.0,
            constraint_work_joules: 0.0,
            work_balance_residual_joules: 1.0e-12,
            work_balance_relative_error: 1.0,
            equilibrium_residual_norm_newtons: 5.0e-13,
            equilibrium_scale_newtons: 1.0e-12,
            equilibrium_relative_error: 0.5,
            relative_tolerance: RELATIVE_TOLERANCE,
            accepted: false,
        };
        evidence.validate().unwrap();

        let mut understated_work = evidence.clone();
        understated_work.work_balance_relative_error = 1.0e-12;
        assert!(understated_work.validate().is_err());

        let mut understated_equilibrium = evidence;
        understated_equilibrium.equilibrium_relative_error = 5.0e-13;
        assert!(understated_equilibrium.validate().is_err());
    }

    #[test]
    fn tiny_physical_moment_does_not_hide_an_incorrect_static_solution() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "tiny_tip_moment".into(),
                region_id: "node:2".into(),
                kind: FixtureExcitationKind::Moment([0.0, 0.0, 1.0e-9]),
            }],
        );
        let excitations = realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = assemble_structural_system(
            &model,
            &excitations,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let solve = solve_linear_system(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        )
        .unwrap();
        assert!(solve.converged, "{:?}", solve.diagnostics);
        let correct = recover_structural_static_balance(
            &summary,
            &solve.solution,
            LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert!(correct.evidence.accepted, "{:?}", correct.evidence);
        assert!(correct.evidence.equilibrium_scale_newtons < 1.0);
        assert!(correct.evidence.internal_work_joules.abs() < 1.0);

        let half_solution = solve
            .solution
            .iter()
            .map(|value| value * 0.5)
            .collect::<Vec<_>>();
        let incorrect = recover_structural_static_balance(
            &summary,
            &half_solution,
            LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert!(!incorrect.evidence.accepted, "{:?}", incorrect.evidence);
        assert!(incorrect.evidence.equilibrium_relative_error > 0.1);
    }

    // witness: FEA-SHARED-CONSTRAINTS-001/original_reaction_and_work @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
    //   boundary: A prescribed beam extension closes reaction and work against the closed form.
    #[test]
    fn prescribed_beam_extension_closes_reaction_and_work_against_closed_form() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        replace_fixture_excitations(&mut model, Vec::new());
        let displacement_m = 1.0e-3;
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "prescribed_tip_extension".into(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point("node:2"),
                    translations: vec![TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: PrescribedValue::Constant(Displacement::new(displacement_m)),
                    }],
                    rotations: Vec::new(),
                }),
            }));
        let excitations = realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = assemble_structural_system(
            &model,
            &excitations,
            None,
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        assert!(summary.constraint_matrix.is_none());
        assert!(summary.physical_load.iter().all(|value| *value == 0.0));

        let solve = solve_linear_system(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
        )
        .unwrap();
        assert!(solve.converged, "{:?}", solve.diagnostics);
        let balance = recover_structural_static_balance(
            &summary,
            &solve.solution,
            LocalExecutionPlan::serial(),
        )
        .unwrap();
        assert!(balance.evidence.accepted, "{:?}", balance.evidence);

        let beam = &summary.structural_beam_recovery[0];
        let axial_stiffness = beam.material.youngs_modulus_pa.canonical_value()
            * beam.section.area_m2.canonical_value()
            / beam.length_m.canonical_value();
        let expected_force = axial_stiffness * displacement_m;
        let expected_work = expected_force * displacement_m;
        let tip_row = summary
            .structural_dof_layout
            .index(1, StructuralDofKind::Ux)
            .unwrap();
        let root_row = summary
            .structural_dof_layout
            .index(0, StructuralDofKind::Ux)
            .unwrap();
        assert!((solve.solution[tip_row] - displacement_m).abs() <= 1.0e-14);
        assert!((balance.reaction[tip_row] - expected_force).abs() <= expected_force * 1.0e-12);
        assert!((balance.reaction[root_row] + expected_force).abs() <= expected_force * 1.0e-12);
        assert!((balance.evidence.constraint_work_joules - expected_work).abs() <= 1.0e-10);
        assert_eq!(balance.evidence.applied_load_work_joules, 0.0);
        assert!((balance.evidence.internal_work_joules - expected_work).abs() <= 1.0e-10);

        for thread_count in [2, 3, 7] {
            let plan =
                LocalExecutionPlan::fixed(LocalThreadCount::new(thread_count).unwrap()).unwrap();
            let scoped =
                recover_structural_static_balance(&summary, &solve.solution, plan).unwrap();
            assert_eq!(scoped.reaction, balance.reaction);
            assert!(scoped.evidence.accepted);
            assert!((scoped.evidence.internal_work_joules - expected_work).abs() <= 1.0e-10);
            assert!((scoped.evidence.constraint_work_joules - expected_work).abs() <= 1.0e-10);
            assert!(scoped.evidence.equilibrium_relative_error <= RELATIVE_TOLERANCE);
        }

        let residual =
            crate::algebra::spmv(summary.operator.operators().stiffness(), &solve.solution)
                .unwrap();
        let nested_reaction = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            recover_structural_reaction_scoped(&summary, &residual, LocalExecutionPlan::serial())
        });
        assert!(matches!(
            nested_reaction,
            Err(ScopedLocalExecutionError::Work {
                source: StructuralReactionError::Scoped(ScopedLocalExecutionError::NestedExecution),
                ..
            })
        ));

        {
            let context = crate::progress::FeaProgressContext::default();
            context.cancellation.cancel();
            let _guard = crate::progress::replace_fea_progress_context(context);
            let cancelled = recover_structural_reaction_scoped(
                &summary,
                &residual,
                LocalExecutionPlan::serial(),
            )
            .unwrap_err();
            assert!(matches!(
                cancelled,
                StructuralReactionError::Scoped(ScopedLocalExecutionError::Work {
                    source: ReactionWorkError::Cancelled,
                    ..
                })
            ));
        }

        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            recover_structural_static_balance(
                &summary,
                &solve.solution,
                LocalExecutionPlan::serial(),
            )
        });
        assert!(matches!(
            nested,
            Err(ScopedLocalExecutionError::Work {
                source: StructuralStaticBalanceError::Scoped(ScopedSpmvError::Execution(
                    ScopedLocalExecutionError::NestedExecution
                )),
                ..
            })
        ));

        {
            let context = crate::progress::FeaProgressContext::default();
            context.cancellation.cancel();
            let _guard = crate::progress::replace_fea_progress_context(context);
            let cancelled = recover_structural_static_balance(
                &summary,
                &solve.solution,
                LocalExecutionPlan::serial(),
            )
            .unwrap_err();
            assert!(cancelled.is_cancelled());
        }

        let plan = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
        let result = CanonicalStructuralConstraintResult::recover_for_model_from_solver_state(
            &plan,
            &model,
            None,
            &solve.solution,
            &balance.reaction,
        )
        .unwrap();
        assert!((result.constraint_work_joules - expected_work).abs() <= 1.0e-10);
        assert!(result.maximum_displacement_error_meters <= 1.0e-14);
        result.validate_against_model(&plan, &model, None).unwrap();

        let run = crate::run_linear_static_with_options(
            &model,
            crate::ComputeBackend::Cpu,
            crate::LinearStaticSolveOptions {
                ..crate::LinearStaticSolveOptions::default()
            },
        )
        .unwrap();
        let reactions = run
            .field(crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert!((reactions[0] + expected_force).abs() <= expected_force * 1.0e-12);
        assert!((reactions[3] - expected_force).abs() <= expected_force * 1.0e-12);
        let published = run.structural_static_balance.as_ref().unwrap();
        assert!(published.accepted);
        assert!((published.constraint_work_joules - expected_work).abs() <= 1.0e-10);
        let scoped_run = crate::run_linear_static_with_options(
            &model,
            crate::ComputeBackend::Cpu,
            crate::LinearStaticSolveOptions {
                local_execution: LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap())
                    .unwrap(),
                ..crate::LinearStaticSolveOptions::default()
            },
        )
        .unwrap();
        let scoped_reactions = scoped_run
            .field(crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE)
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(scoped_reactions, reactions);
        assert!(
            scoped_run
                .structural_static_balance
                .as_ref()
                .unwrap()
                .accepted
        );
        assert!(
            (scoped_run
                .structural_static_balance
                .as_ref()
                .unwrap()
                .constraint_work_joules
                - expected_work)
                .abs()
                <= 1.0e-10
        );
        let published_constraints = run.structural_constraint_result.as_ref().unwrap();
        assert!((published_constraints.constraint_work_joules - expected_work).abs() <= 1.0e-10);
        assert!(published_constraints.maximum_displacement_error_meters <= 1.0e-14);
        assert!(published_constraints.maximum_rotation_error_radians <= 1.0e-14);
        published_constraints
            .validate_against_model(&plan, &model, None)
            .unwrap();
        assert!(run.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
                && diagnostic.message.contains("balance_accepted=true")
                && diagnostic
                    .message
                    .contains(&format!("constraint_work_joules={expected_work}"))
        }));

        let serialized = serde_json::to_value(&run).unwrap();
        let mut tampered_balance = serialized.clone();
        tampered_balance["structural_static_balance"]["constraint_work_joules"] =
            serde_json::json!(expected_work * 2.0);
        assert!(serde_json::from_value::<crate::FeaRunResult>(tampered_balance).is_err());
        let mut tampered_constraints = serialized;
        tampered_constraints["structural_constraint_result"]["constraint_work_joules"] =
            serde_json::json!(expected_work * 2.0);
        assert!(serde_json::from_value::<crate::FeaRunResult>(tampered_constraints).is_err());
    }
}
