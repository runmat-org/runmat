//! Binding selection and borrowed nodal values; no numerical field fabrication.
use runmat_analysis_core::{
    AnalysisModel, InitialConditionValue, InitialValue, ProcedureStep, Temperature,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

use crate::{execution::InitialFieldBinding, FeaRunError};

pub(crate) fn validate_selected(
    model: &AnalysisModel,
    step: &ProcedureStep,
    mesh: &SolverMeshArtifact,
    fields: &[InitialFieldBinding],
) -> Result<(), FeaRunError> {
    crate::execution::validate_initial_field_selection(
        model,
        step,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        mesh.topology.nodes.len(),
        fields,
    )
    .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
}

pub(crate) fn values<'a>(
    model: &AnalysisModel,
    condition: &InitialConditionValue<Temperature>,
    topology: Digest,
    node_count: usize,
    fields: &'a [InitialFieldBinding],
) -> Result<Option<&'a [f64]>, FeaRunError> {
    let InitialValue::FieldArtifact(reference) = &condition.value else {
        return Ok(None);
    };
    let binding = fields
        .iter()
        .find(|binding| binding.reference() == reference)
        .ok_or_else(|| {
            FeaRunError::InvalidModel(format!(
                "thermal initial condition {} has no field binding",
                condition.initial_condition_id
            ))
        })?;
    let selected = model
        .initial_conditions
        .iter()
        .find(|candidate| candidate.initial_condition_id() == condition.initial_condition_id)
        .ok_or_else(|| {
            FeaRunError::InvalidModel("thermal initial condition is not model-owned".into())
        })?;
    binding
        .admit(selected, topology, node_count)
        .map(|field| Some(field.values()))
        .map_err(|error| FeaRunError::InvalidModel(error.to_string()))
}
