//! Binding selection and borrowed nodal values; no numerical field fabrication.
use runmat_analysis_core::{
    AnalysisModel, InitialCondition, InitialConditionValue, InitialValue, ProcedureStep,
    Temperature,
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
    let invalid =
        |message: &str| FeaRunError::InvalidModel(format!("thermal initial fields: {message}"));
    for (index, binding) in fields.iter().enumerate() {
        if fields[..index]
            .iter()
            .any(|other| other.reference() == binding.reference())
        {
            return Err(invalid("duplicate binding"));
        }
        if !model.initial_conditions.iter().any(|condition| {
            condition.applies_to_step(step)
                && matches!(condition,
                InitialCondition::Temperature(value) if matches!(&value.value,
                    InitialValue::FieldArtifact(reference) if reference == binding.reference()))
        }) {
            return Err(invalid(
                "binding is not selected by the consuming temperature step",
            ));
        }
    }
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    for condition in model
        .initial_conditions
        .iter()
        .filter(|condition| condition.applies_to_step(step))
    {
        if let InitialCondition::Temperature(value) = condition {
            values(model, value, topology, mesh.topology.nodes.len(), fields)?;
        }
    }
    Ok(())
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
