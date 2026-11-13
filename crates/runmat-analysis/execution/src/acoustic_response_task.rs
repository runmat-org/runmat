//! Portable, explicitly planned static acoustic response work; no driver-side numerical solve.
use crate::{
    prepare_acoustic_response_inputs, prepare_analysis_objects, prepare_analysis_program,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::{
    AcousticResponseWorkload, AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity,
    NumericalPlanIdentity, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeSet;

pub struct AcousticResponseTaskPreparation {
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: AcousticResponseWorkload,
    pub mesh: SolverMeshArtifact,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedAcousticResponseTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}

pub fn prepare_acoustic_response_task(
    preparation: AcousticResponseTaskPreparation,
) -> AnalysisBridgeResult<PreparedAcousticResponseTask> {
    let prepared = prepare_acoustic_response_inputs(
        &preparation.workload,
        &preparation.mesh,
        preparation.output_policy,
        preparation.local_execution,
    )?;
    let resource_plan = prepared.resource_plan;
    let mut inputs = prepared.inputs;
    inputs.push(prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        preparation.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        preparation.output_policy.inventory_limits(),
    )?);
    let study = StudyIdentity::from_authored_digest(
        preparation.authored_study_digest,
        preparation.frozen_project_digest,
    )?;
    let step = preparation.workload.producer_identity()?;
    let numerical = NumericalPlanIdentity::from_acoustic_response_workload(
        preparation.workload.identity()?,
        Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
    )?;
    let execution = ExecutionPlanIdentity::new(numerical, resource_plan.identity.digest())?;
    let mut capabilities = preparation.required_capabilities;
    capabilities.insert(AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    });
    let request = AnalysisWorkloadRequest::new(
        study,
        step,
        execution,
        AnalysisWorkKind::SolveAcousticResponse,
        inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect(),
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
            AnalysisNumericalObjectKind::StatePublicationProfiles,
        ],
        capabilities,
    )?;
    let host = AnalysisHostWorkload::new(
        request,
        preparation.artifact_access,
        preparation.output_policy,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedAcousticResponseTask {
        host,
        program,
        inputs,
        resource_plan,
        output_policy: preparation.output_policy,
    })
}
