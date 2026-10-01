//! Canonical thermal inputs and shared bridge-derived output reservations.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::execution::{
    AnalysisTransportProfiler, ThermalWorkload, THERMAL_RECEIPT_MAX_ENCODED_BYTES,
};
use runmat_analysis_fea::planning::resource::{
    build_thermal_resource_plan, FeaResourcePlan, PreparedThermalResourceFacts,
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod source;
pub use source::validate_thermal_task_source;
#[cfg(test)]
mod tests;

pub struct PreparedThermalInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

/// Recompute the worker plan using validated imported inputs, without cloning their buffers.
pub fn rederive_thermal_resource_plan(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let field_source = source::decode(workload, mesh, inputs, true)?;
    let (result_output, receipt_output, state_profiles_output) =
        projected_thermal_outputs(workload, mesh, policy)?;
    let facts = PreparedThermalResourceFacts {
        field_source: source::facts(field_source.as_ref(), mesh, inputs)?,
        inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
        result_output,
        receipt_output,
        state_profiles_output,
        output_set: projected_output_set_resource_facts(),
    };
    Ok(build_thermal_resource_plan(
        workload,
        mesh,
        policy.chunk_bytes(),
        facts,
    )?)
}

/// Prepare governed numerical inputs, not an executable worker task. Native execution must
/// rederive this plan and compare it with the admitted plan before invoking the numerical solve.
pub fn prepare_thermal_inputs(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedThermalInputs> {
    prepare_thermal_inputs_with_source(workload, mesh, policy, None)
}

pub(crate) fn prepare_thermal_inputs_with_source(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    field_source: Option<&runmat_analysis_fea::execution::RetainedJouleHeatSource>,
) -> AnalysisBridgeResult<PreparedThermalInputs> {
    if field_source.is_none()
        && runmat_analysis_fea::execution::RetainedJouleHeatSource::requires_binding(workload)
    {
        return Err(AnalysisBridgeError::Invalid(
            "thermal retained field requires its original source binding".into(),
        ));
    }
    let (result_output, receipt_output, state_profiles_output) =
        projected_thermal_outputs(workload, mesh, policy)?;
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ThermalWorkload,
        workload.identity()?.digest(),
        &workload.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    let mesh_bytes = mesh
        .canonical_encode()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh_bytes,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?;
    let mut inputs = vec![workload_objects, mesh_objects];
    if let Some(source) = field_source {
        let payload = source.canonical_bytes()?;
        inputs.push(prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldSourceBinding,
            source.identity()?,
            &payload,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?);
    }
    let facts = PreparedThermalResourceFacts {
        field_source: source::facts(field_source, mesh, &inputs)?,
        inputs: prepared_input_resource_facts(&inputs)?,
        result_output,
        receipt_output,
        state_profiles_output,
        output_set: projected_output_set_resource_facts(),
    };
    let resource_plan = build_thermal_resource_plan(workload, mesh, policy.chunk_bytes(), facts)?;
    Ok(PreparedThermalInputs {
        inputs,
        resource_plan,
    })
}

fn projected_thermal_outputs(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<(
    runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts,
    runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts,
    runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts,
)> {
    let numerical = workload.prepare(mesh)?;
    let result = numerical.transport_profile(policy.chunk_bytes())?;
    let mut receipt = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    receipt.push(THERMAL_RECEIPT_MAX_ENCODED_BYTES)?;
    let receipt = receipt.finish(AnalysisNumericalObjectKind::NumericalReceipt)?;
    let profile_metadata = workload.state_publication_profiles_encoding_profile(
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )?;
    let mut profiles = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiles.push(profile_metadata.encoded_bytes_upper_bound)?;
    let profiles = profiles.finish(AnalysisNumericalObjectKind::StatePublicationProfiles)?;
    crate::thermal_execution::require_inventory(&result, policy.inventory_limits())?;
    crate::thermal_execution::require_inventory(&receipt, policy.inventory_limits())?;
    crate::thermal_execution::require_inventory(&profiles, policy.inventory_limits())?;
    Ok((
        projected_transport_output_resource_facts(&result)?,
        projected_transport_output_resource_facts(&receipt)?,
        projected_transport_output_resource_facts(&profiles)?,
    ))
}
