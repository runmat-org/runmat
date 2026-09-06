use runmat_analysis_core::{
    AnalysisFieldValues, AnalysisModel, AnalysisProcedure, CoupledPhysicsFamily,
    FormulationRequirement, PhysicsFamily, SolutionSpace,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::{
    ComputeBackend, FeaElectroThermalContext, FeaRunResult, FeaThermoMechanicalContext,
};
use crate::solve::backend::kind::LinearAlgebraBackendKind;
use crate::solve::preconditioner::SpdPreconditionerKind;

use super::{AnalysisExecutionError, AnalysisExecutionResult};

const SOLVE_PREFIX: &[u8] = b"runmat-analysis-solve/v1\0";
const SOLVE_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Solver-owned, path-free input for one structural linear workload. The mesh is a separate
/// numerical object so it can be cached, transferred, and shared independently of authored state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub preconditioner: SpdPreconditionerKind,
    pub algebra_backend: LinearAlgebraBackendKind,
    pub require_solver_mesh_for_solid: bool,
    pub thermo_mechanical_context: Option<FeaThermoMechanicalContext>,
    pub electro_thermal_context: Option<FeaElectroThermalContext>,
}

impl StructuralLinearWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 {
            return Err(AnalysisExecutionError::Invalid(
                "unsupported structural linear workload schema".into(),
            ));
        }
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear workload requires exactly one procedure step".into(),
            ));
        };
        if step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
            || !matches!(
                step.family,
                PhysicsFamily::Structural
                    | PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear workload requires a structural or thermomechanical static direct procedure".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-workload",
            bytes,
            SOLVE_LIMITS,
        )?;
        workload.validate()?;
        Ok(workload)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLinearWorkloadResult {
    pub schema_version: u16,
    pub run: FeaRunResult,
}

impl StructuralLinearWorkloadResult {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.run.fields_are_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "structural linear result schema or field set is invalid".into(),
            ));
        }
        let mut field_ids = std::collections::BTreeSet::new();
        for field in &self.run.fields {
            if field.field_id.trim().is_empty() || !field_ids.insert(field.field_id.as_str()) {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural result field identities must be non-empty and unique"
                        .into(),
                ));
            }
            let AnalysisFieldValues::HostF64(values) = &field.values else {
                return Err(AnalysisExecutionError::Invalid(
                    "portable structural results must contain host-backed fields".into(),
                ));
            };
            let Some(expected_values) = field
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            else {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` shape overflows the host address space",
                    field.field_id
                )));
            };
            if values.len() != expected_values || values.iter().any(|value| !value.is_finite()) {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "portable structural field `{}` has invalid shape or non-finite values",
                    field.field_id
                )));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            self,
            SOLVE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result: Self = decode_contract(
            SOLVE_PREFIX,
            "structural-linear-result",
            bytes,
            SOLVE_LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
}
