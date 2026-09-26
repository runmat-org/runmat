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
    let count = |kind| {
        inputs
            .iter()
            .filter(|input| input.manifest.kind == kind)
            .count()
    };
    if inputs.len() != 3
        || count(AnalysisNumericalObjectKind::ThermalWorkload) != 1
        || count(AnalysisNumericalObjectKind::Mesh) != 1
        || count(AnalysisNumericalObjectKind::ResourcePlan) != 1
    {
        return Err(AnalysisBridgeError::Invalid(
            "thermal execution requires exactly workload, mesh and resource-plan inputs".into(),
        ));
    }
    let workload_identity = workload.identity()?.digest();
    if !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload
            && input.manifest.logical_identity == workload_identity
    }) || !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::Mesh
            && input.manifest.logical_identity == Digest::from_bytes(*mesh.canonical_digest.bytes())
    }) {
        return Err(AnalysisBridgeError::Identity(
            "thermal resource inputs differ from decoded workload or mesh",
        ));
    }
    let (result_output, receipt_output) = projected_thermal_outputs(workload, mesh, policy)?;
    let facts = PreparedThermalResourceFacts {
        inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
        result_output,
        receipt_output,
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
    let (result_output, receipt_output) = projected_thermal_outputs(workload, mesh, policy)?;
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
        result_output,
        receipt_output,
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
)> {
    let numerical = workload.prepare(mesh)?;
    let result = numerical.transport_profile(policy.chunk_bytes())?;
    let mut receipt = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    receipt.push(THERMAL_RECEIPT_MAX_ENCODED_BYTES)?;
    let receipt = receipt.finish(AnalysisNumericalObjectKind::NumericalReceipt)?;
    crate::thermal_execution::require_inventory(&result, policy.inventory_limits())?;
    crate::thermal_execution::require_inventory(&receipt, policy.inventory_limits())?;
    Ok((
        projected_transport_output_resource_facts(&result)?,
        projected_transport_output_resource_facts(&receipt)?,
    ))
}
