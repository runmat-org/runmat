//! Canonical acoustic_harmonic inputs and shared bridge-derived output reservations.
use crate::{
    prepare_analysis_objects, prepared_input_resource_facts,
    prepared_input_resource_facts_without_resource_plan, projected_output_set_resource_facts,
    projected_transport_output_resource_facts, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisOutputPolicy, PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::AnalysisNumericalObjectKind;
use runmat_analysis_fea::execution::{AcousticHarmonicWorkload, AnalysisTransportProfiler};
use runmat_analysis_fea::planning::resource::{
    build_acoustic_resource_plan, FeaResourcePlan, LocalExecutionPlan,
    PreparedAcousticResourceFacts,
};
use runmat_execution::Digest;
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct PreparedAcousticHarmonicInputs {
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

/// Recompute the worker plan using validated imported inputs, without cloning their buffers.
pub fn rederive_acoustic_harmonic_resource_plan(
    workload: &AcousticHarmonicWorkload,
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
        || count(AnalysisNumericalObjectKind::AcousticHarmonicWorkload) != 1
        || count(AnalysisNumericalObjectKind::Mesh) != 1
        || count(AnalysisNumericalObjectKind::ResourcePlan) != 1
    {
        return Err(AnalysisBridgeError::Invalid(
            "acoustic_harmonic execution requires exactly workload, mesh and resource-plan inputs"
                .into(),
        ));
    }
    let workload_identity = workload.identity()?.digest();
    if !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::AcousticHarmonicWorkload
            && input.manifest.logical_identity == workload_identity
    }) || !inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::Mesh
            && input.manifest.logical_identity == Digest::from_bytes(*mesh.canonical_digest.bytes())
    }) {
        return Err(AnalysisBridgeError::Identity(
            "acoustic_harmonic resource inputs differ from decoded workload or mesh",
        ));
    }
    let result_output = projected_acoustic_harmonic_output(workload, mesh, policy)?;
    let state_profiles_output = projected_acoustic_state_profiles_output(workload, mesh, policy)?;
    let declaration = inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| AnalysisBridgeError::Invalid("acoustic resource plan missing".into()))?;
    let declared = FeaResourcePlan::from_canonical_bytes(&declaration.payload())?;
    if declaration.manifest.logical_identity != declared.identity.digest() {
        return Err(AnalysisBridgeError::Identity(
            "acoustic CPU allocation differs from its resource identity",
        ));
    }
    let facts = PreparedAcousticResourceFacts {
        local_execution: LocalExecutionPlan::from_cpu_millicores(declared.request.cpu_millicores)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
        result_output,
        state_profiles_output,
        receipt_output: projected_acoustic_receipt_output(policy)?,
        output_set: projected_output_set_resource_facts(),
    };
    Ok(build_acoustic_resource_plan(workload, mesh, facts)?)
}

/// Prepare governed numerical inputs, not an executable worker task. Native execution must
/// rederive this plan and compare it with the admitted plan before invoking the numerical solve.
pub fn prepare_acoustic_harmonic_inputs(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
    local_execution: LocalExecutionPlan,
) -> AnalysisBridgeResult<PreparedAcousticHarmonicInputs> {
    let result_output = projected_acoustic_harmonic_output(workload, mesh, policy)?;
    let state_profiles_output = projected_acoustic_state_profiles_output(workload, mesh, policy)?;
    let workload_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::AcousticHarmonicWorkload,
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
    let facts = PreparedAcousticResourceFacts {
        local_execution,
        inputs: prepared_input_resource_facts(&inputs)?,
        result_output,
        state_profiles_output,
        receipt_output: projected_acoustic_receipt_output(policy)?,
        output_set: projected_output_set_resource_facts(),
    };
    let resource_plan = build_acoustic_resource_plan(workload, mesh, facts)?;
    Ok(PreparedAcousticHarmonicInputs {
        inputs,
        resource_plan,
    })
}

fn projected_acoustic_receipt_output(
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(runmat_analysis_fea::execution::ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES)?;
    let output = profiler.finish(AnalysisNumericalObjectKind::NumericalReceipt)?;
    crate::thermal_execution::require_inventory(&output, policy.inventory_limits())?;
    projected_transport_output_resource_facts(&output)
}

fn projected_acoustic_harmonic_output(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let bytes = runmat_analysis_fea::planning::resource::acoustic_harmonic_result_payload_bytes_upper_bound(workload, mesh)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(bytes)?;
    let result = profiler.finish(AnalysisNumericalObjectKind::SolverResult)?;
    crate::thermal_execution::require_inventory(&result, policy.inventory_limits())?;
    projected_transport_output_resource_facts(&result)
}

fn projected_acoustic_state_profiles_output(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    policy: AnalysisOutputPolicy,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let profile = workload.state_publication_profiles_encoding_profile(Digest::from_bytes(
        *mesh.canonical_digest.bytes(),
    ))?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    profiler.push(profile.encoded_bytes_upper_bound)?;
    let output = profiler.finish(AnalysisNumericalObjectKind::StatePublicationProfiles)?;
    crate::thermal_execution::require_inventory(&output, policy.inventory_limits())?;
    projected_transport_output_resource_facts(&output)
}
