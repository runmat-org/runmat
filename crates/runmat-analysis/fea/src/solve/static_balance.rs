//! Physical equilibrium and work balance for a completed structural static solve.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    algebra::apply_k,
    assembly::AssemblySummary,
    solve::reaction::{recover_structural_reaction, StructuralReactionError},
};

const RELATIVE_TOLERANCE: f64 = 1.0e-8;

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

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum StructuralStaticBalanceError {
    #[error("structural static balance state has the wrong dimension or a nonfinite value")]
    InvalidState,
    #[error("structural static reaction recovery failed: {0}")]
    Reaction(#[from] StructuralReactionError),
    #[error("structural static balance evidence is inconsistent")]
    InvalidEvidence,
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
            .max(self.constraint_work_joules.abs())
            .max(1.0);
        let accepted = self.work_balance_relative_error <= self.relative_tolerance
            && self.equilibrium_relative_error <= self.relative_tolerance;
        if values.iter().any(|value| !value.is_finite())
            || self.work_balance_relative_error < 0.0
            || self.equilibrium_residual_norm_newtons < 0.0
            || self.equilibrium_scale_newtons < 1.0
            || self.equilibrium_relative_error < 0.0
            || self.relative_tolerance <= 0.0
            || (work_residual - self.work_balance_residual_joules).abs()
                > work_scale * f64::EPSILON * 16.0
            || (work_residual.abs() / work_scale - self.work_balance_relative_error).abs()
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
) -> Result<StructuralStaticBalance, StructuralStaticBalanceError> {
    if displacement.len() != summary.dof_count
        || summary.physical_load.len() != summary.dof_count
        || displacement.iter().any(|value| !value.is_finite())
        || summary.physical_load.iter().any(|value| !value.is_finite())
    {
        return Err(StructuralStaticBalanceError::InvalidState);
    }
    let internal_force = apply_k(&summary.operator, displacement);
    let equilibrium_residual = internal_force
        .iter()
        .zip(&summary.physical_load)
        .map(|(internal, applied)| internal - applied)
        .collect::<Vec<_>>();
    let reaction = recover_structural_reaction(summary, &equilibrium_residual)?;
    let free_residual = equilibrium_residual
        .iter()
        .zip(&reaction)
        .map(|(residual, reaction)| residual - reaction)
        .collect::<Vec<_>>();
    let equilibrium_residual_norm_newtons = norm(&free_residual);
    let equilibrium_scale_newtons = norm(&internal_force)
        .max(norm(&summary.physical_load))
        .max(norm(&reaction))
        .max(1.0);
    let equilibrium_relative_error = equilibrium_residual_norm_newtons / equilibrium_scale_newtons;
    let internal_work_joules = dot(displacement, &internal_force);
    let applied_load_work_joules = dot(displacement, &summary.physical_load);
    let constraint_work_joules = dot(displacement, &reaction);
    let work_balance_residual_joules =
        internal_work_joules - applied_load_work_joules - constraint_work_joules;
    let work_scale = internal_work_joules
        .abs()
        .max(applied_load_work_joules.abs())
        .max(constraint_work_joules.abs())
        .max(1.0);
    let work_balance_relative_error = work_balance_residual_joules.abs() / work_scale;
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

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
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
        fixtures::{fixture_model, replace_fixture_excitations, FixtureId},
        solve::{
            backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
            linear::solve_linear_system,
            preconditioner::SpdPreconditionerKind,
        },
    };

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
        );
        assert!(solve.converged, "{:?}", solve.diagnostics);
        let balance = recover_structural_static_balance(&summary, &solve.solution).unwrap();
        assert!(balance.evidence.accepted, "{:?}", balance.evidence);

        let beam = &summary.structural_beam_recovery[0];
        let axial_stiffness =
            beam.material.youngs_modulus_pa * beam.section.area_m2 / beam.length_m;
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
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
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
