//! Portable static scalar FEM contracts. Numerical execution only: no scheduler allocation,
//! Native retained-field/state contracts do not imply restart or independent accuracy promotion.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::admission::AnalysisRunIdentity;
use runmat_analysis_core::{
    AnalysisModel, ElectromagneticDomain, ProcedureStepId, StaticElectricalEquation,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

mod admitted;
mod result;
pub use result::ElectricalStaticSolverResult;
pub(in crate::execution) mod fields;
pub use fields::{
    electrical_native_field_definition_inventory, electrical_native_field_definitions,
};
pub(in crate::execution) use fields::{
    native_electrical_entity_counts, native_electrical_field_definition,
    native_electrical_field_plan,
};
mod publication;
mod receipt;
pub use receipt::{
    ElectricalNumericalReceipt, ElectricalNumericalReceiptArtifact,
    ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES,
};
#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-electrical-static-workload/v1\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStaticWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub relative_residual_target: f64,
    /// Solved element temperatures for steady conduction at an electrothermal operating
    /// point (canonical element order); conductivity follows the material temperature law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element_temperature_k: Option<Vec<f64>>,
    /// Time at which time-dependent terminal sources are sampled (the operating point of a
    /// transient electrothermal analysis); `None` for a static source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operating_time_s: Option<f64>,
}

impl ElectricalStaticWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        crate::physics::electrical::validate_canonical_input(
            &self.model,
            self.step_id.as_str(),
            self.relative_residual_target,
        )
        .map_err(invalid)?;
        if let Some(temperatures) = &self.element_temperature_k {
            if temperatures.is_empty()
                || temperatures.iter().any(|t| !t.is_finite() || *t <= 0.0)
                || !matches!(
                    self.model.electromagnetic,
                    Some(ElectromagneticDomain::Static {
                        equation: StaticElectricalEquation::SteadyConduction
                    })
                )
            {
                return Err(invalid(
                    "an element temperature field applies to steady conduction with finite absolute temperatures",
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn sample_point(&self) -> crate::excitation::ExcitationSamplePoint<'static> {
        match self.operating_time_s {
            Some(time) => crate::excitation::ExcitationSamplePoint::time(time),
            None => crate::excitation::ExcitationSamplePoint::Static,
        }
    }
    /// Element conductivities at the workload's temperature field, or `None` at reference.
    pub(crate) fn element_conductivity(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Option<Vec<f64>>> {
        let Some(temperatures) = &self.element_temperature_k else {
            return Ok(None);
        };
        let projection = crate::topology::AnalysisProjection::derive(mesh).map_err(invalid)?;
        if temperatures.len() != projection.elements.len() {
            return Err(invalid("element temperature field does not match the mesh"));
        }
        let properties =
            crate::properties::compile_solid_property_map(&self.model, mesh).map_err(invalid)?;
        projection
            .elements
            .iter()
            .zip(temperatures)
            .map(|(element, temperature)| {
                let law = properties
                    .solid_material(element.stable_identity)
                    .and_then(|m| m.material.electrical.as_ref())
                    .ok_or_else(|| invalid("element without an electrical law"))?;
                super::conductivity_at(law, *temperature)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()
            .map(Some)
    }
    pub fn equation(&self) -> AnalysisExecutionResult<StaticElectricalEquation> {
        self.validate()?;
        match &self.model.electromagnetic {
            Some(ElectromagneticDomain::Static { equation }) => Ok(*equation),
            _ => Err(invalid("static electrical equation required")),
        }
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "electrical-static-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "electrical-static-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "electrical-static-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        self.validate()?;
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            &self.model.sequence.steps[0],
        )
    }
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        crate::admission::admit_canonical_electrical_static(
            self.equation()?,
            crate::physics::electrical::sources::kind(&self.model),
            mesh,
        )
        .map_err(invalid)
    }
    /// Executes the existing canonical CPU numerical kernel. This is not an admitted
    /// resource plan or semantic study completion. Result import never reruns this solve.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<ElectricalStaticSolverResult> {
        let formulation = self.formulation(mesh)?;
        let conductivity = self.element_conductivity(mesh)?;
        let response = crate::physics::electrical::solve_canonical_static_with_admission(
            &self.model,
            mesh,
            self.step_id.as_str(),
            self.relative_residual_target,
            None,
            conductivity.as_deref(),
            self.sample_point(),
        )
        .map_err(electrical_failure)?;
        ElectricalStaticSolverResult::new(self, mesh, formulation, response)
    }
}

fn electrical_failure(
    error: crate::physics::electrical::ElectricalStaticError,
) -> AnalysisExecutionError {
    match error {
        crate::physics::electrical::ElectricalStaticError::Accounting(cause) => {
            AnalysisExecutionError::Usage(cause)
        }
        other => invalid(other),
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
