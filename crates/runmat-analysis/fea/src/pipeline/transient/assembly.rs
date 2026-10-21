//! One load-sampling owner for full-run and portable-block transient execution.

use runmat_analysis_core::{AnalysisModel, ProcedureStep, SignalSampleApproach};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    assembly::{
        assemble_structural_system_borrowed, AssemblySummary, PreparedStructuralTransientBasis,
        StructuralFormulationRequirement,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    FeaRunError,
};

pub(super) struct TransientAssemblySampler<'a> {
    model: &'a AnalysisModel,
    step: &'a ProcedureStep,
    mesh: Option<&'a SolverMeshArtifact>,
    initial: &'a AssemblySummary,
    prepared: Option<PreparedStructuralTransientBasis<'a>>,
}

impl<'a> TransientAssemblySampler<'a> {
    pub(super) fn new(
        model: &'a AnalysisModel,
        step: &'a ProcedureStep,
        mesh: Option<&'a SolverMeshArtifact>,
        initial: &'a AssemblySummary,
    ) -> Result<Self, FeaRunError> {
        let prepared = PreparedStructuralTransientBasis::from_admitted(model, step, mesh, initial)
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
        Ok(Self {
            model,
            step,
            mesh,
            initial,
            prepared,
        })
    }

    pub(super) fn sample(
        &self,
        time_s: f64,
        approach: SignalSampleApproach,
    ) -> Result<AssemblySummary, FeaRunError> {
        let point = ExcitationSamplePoint::TimeSeconds {
            value: time_s,
            approach,
        };
        let assembly = if let Some(prepared) = &self.prepared {
            prepared
                .sample(point)
                .map_err(|error| FeaRunError::Assembly(error.to_string()))?
        } else {
            let excitations = realize_step_excitations(self.model, self.step, point)
                .map_err(|error| FeaRunError::InvalidModel(error.to_string()))?;
            assemble_structural_system_borrowed(
                self.model,
                &excitations,
                self.mesh,
                StructuralFormulationRequirement::SyntheticOperator,
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?
        };
        if assembly.dof_count != self.initial.dof_count {
            return Err(FeaRunError::Assembly(format!(
                "transient reassembly changed structural DOF count from {} to {}",
                self.initial.dof_count, assembly.dof_count,
            )));
        }
        Ok(assembly)
    }
}
