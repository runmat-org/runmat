use runmat_analysis_core::{
    AnalysisModel, AnalysisProcedure, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep,
    SolutionSpace, StructuralConstraintKind,
};

use crate::{
    admission::{
        admit_formulation, require_available_formulation, structural_exact_realization,
        thermomechanical_exact_realization, AnalysisRunIdentity, FormulationAdmissionRequest,
        StructuralRealizationInput, ThermomechanicalRealizationInput,
    },
    assembly::AssemblySummary,
    contracts::FeaRunError,
};

pub(super) fn require_structural(
    model: &AnalysisModel,
    procedure: AnalysisProcedure,
) -> Result<&ProcedureStep, FeaRunError> {
    let mut matching_steps = model.sequence.steps.iter().filter(|step| {
        step.procedure == procedure
            && matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
    });
    let step = matching_steps.next().ok_or_else(|| {
        FeaRunError::InvalidModel(format!(
            "structural {procedure:?} execution requires exactly one canonical Structural or Thermomechanical procedure step, but none was found"
        ))
    })?;
    if matching_steps.next().is_some() {
        return Err(FeaRunError::InvalidModel(format!(
            "structural {procedure:?} execution requires exactly one canonical Structural or Thermomechanical procedure step, but multiple steps were found"
        )));
    }
    require_structural_step(model, step)?;
    Ok(step)
}

/// Apply the same structural admission to an explicitly selected, authored procedure.
pub(super) fn require_structural_step(
    model: &AnalysisModel,
    step: &ProcedureStep,
) -> Result<(), FeaRunError> {
    let procedure = step.procedure;
    require_available_formulation(step.family, step.procedure, step.solution_space)
        .map_err(FeaRunError::from)?;
    if !matches!(
        procedure,
        AnalysisProcedure::ModalExtraction | AnalysisProcedure::TransientResponse
    ) && model.constraints.iter().any(|constraint| {
        constraint.structural().is_some_and(|constraint| {
            matches!(
                constraint.kind,
                StructuralConstraintKind::PrescribedMotion(_)
            )
        })
    }) {
        return Err(FeaRunError::InvalidModel(format!(
            "prescribed structural motion is admitted as a homogeneous support for modal extraction and as physical kinematics for transient response, not {procedure:?}"
        )));
    }
    // Couplings, ties, periodic, symmetry and equation constraints reduce every structural
    // operator through the same affine constraint reduction (static, modal, buckling, direct
    // harmonic and transient solves); procedures that need homogeneous rows check them there.
    Ok(())
}

pub(super) fn require_nonstructural(
    family: PhysicsFamily,
    procedure: AnalysisProcedure,
) -> Result<(), FeaRunError> {
    require_available_formulation(family, procedure, SolutionSpace::Direct).map_err(Into::into)
}

pub(super) fn admit_structural(
    summary: &AssemblySummary,
    step: &ProcedureStep,
) -> Result<AnalysisRunIdentity, FeaRunError> {
    if summary.structural_formulations.is_empty() {
        return Err(FeaRunError::Assembly(
            "structural assembly did not report its exact formulation realization".to_string(),
        ));
    }
    let assemblies = summary.structural_formulations.clone();
    if summary
        .electro_thermal
        .as_ref()
        .is_some_and(|value| value.enabled)
    {
        return Err(FeaRunError::InvalidModel(
            "the structural pipelines do not implement an electrothermal formulation".to_string(),
        ));
    }
    let solved_thermal = summary
        .structural_solid_recovery
        .iter()
        .any(|element| element.solved_thermal_quadrature.is_some());
    let thermomechanical = solved_thermal
        || summary
            .thermo_mechanical
            .as_ref()
            .is_some_and(|value| value.enabled);
    let realized_family = if thermomechanical {
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    } else {
        PhysicsFamily::Structural
    };
    if step.family != realized_family {
        return Err(FeaRunError::InvalidModel(format!(
            "canonical procedure step '{}' requests {:?}, but structural assembly realizes {:?}; coupling must be represented by both the typed step and the assembled formulation",
            step.step_id.as_str(),
            step.family,
            realized_family,
        )));
    }
    let coupling = if solved_thermal {
        runmat_analysis_core::CouplingImplementation::OneWaySolvedThermoelasticState
    } else if thermomechanical {
        runmat_analysis_core::CouplingImplementation::OnePassPrescribedThermomechanicalProfile
    } else {
        runmat_analysis_core::CouplingImplementation::Uncoupled
    };
    let input = StructuralRealizationInput {
        assemblies,
        procedure: step.procedure,
        kinematics: step.structural_kinematics(),
        coupling,
        sources: summary.structural_sources.clone(),
        boundary_transfers: summary.structural_boundary_transfers.clone(),
    };
    let realization = if thermomechanical {
        thermomechanical_exact_realization(ThermomechanicalRealizationInput {
            structural: input,
            profile_sources: if solved_thermal {
                vec![runmat_analysis_core::SourceImplementation::SolvedThermalStateNodalField]
            } else {
                summary
                    .thermo_mechanical
                    .as_ref()
                    .map(|value| value.profile_sources.clone())
                    .unwrap_or_default()
            },
        })
    } else {
        structural_exact_realization(input)
    }
    .map_err(|reason| {
        FeaRunError::Assembly(format!(
            "structural assembly produced an invalid formulation realization: {reason:?}"
        ))
    })?;
    admit_formulation(FormulationAdmissionRequest {
        family: step.family,
        procedure: step.procedure,
        solution_space: step.solution_space,
        realization,
    })
    .map(|receipt| receipt.into_identity())
    .map_err(Into::into)
}
