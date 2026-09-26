//! Canonical thermal inputs and shared bridge-derived output reservations.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts, projected_output_set_resource_facts,
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

pub struct PreparedThermalInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

/// Prepare governed numerical inputs, not an executable worker task. Native execution must
/// rederive this plan and compare it with the admitted plan before invoking the numerical solve.
pub fn prepare_thermal_inputs(
    workload: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<PreparedThermalInputs> {
    let numerical = workload.prepare(mesh)?;
    let result_profile = numerical.transport_profile(policy.chunk_bytes())?;
    crate::thermal_execution::require_inventory(&result_profile, policy.inventory_limits())?;
    let mut receipt_profile = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    receipt_profile.push(THERMAL_RECEIPT_MAX_ENCODED_BYTES)?;
    let receipt_profile = receipt_profile.finish(AnalysisNumericalObjectKind::NumericalReceipt)?;
    crate::thermal_execution::require_inventory(&receipt_profile, policy.inventory_limits())?;
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
    let inputs = vec![workload_objects, mesh_objects];
    let facts = PreparedThermalResourceFacts {
        inputs: prepared_input_resource_facts(&inputs)?,
        result_output: projected_transport_output_resource_facts(&result_profile)?,
        receipt_output: projected_transport_output_resource_facts(&receipt_profile)?,
        output_set: projected_output_set_resource_facts(),
    };
    // Drop the preparation before the source-bound plan rederives it; two compiled property
    // maps must not coexist as an accidental preparation duplicate.
    drop(numerical);
    let resource_plan = build_thermal_resource_plan(workload, mesh, policy.chunk_bytes(), facts)?;
    Ok(PreparedThermalInputs {
        inputs,
        resource_plan,
    })
}
