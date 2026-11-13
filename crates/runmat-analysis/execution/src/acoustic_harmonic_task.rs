//! Portable task construction for standalone host acoustic_harmonic solves.
use crate::{
    prepare_acoustic_harmonic_inputs, prepare_analysis_objects, prepare_analysis_program,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::{
    AcousticHarmonicWorkload, AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity,
    NumericalPlanIdentity, SequenceIdentity, StepIdentity, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeSet;

pub struct AcousticHarmonicTaskPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: AcousticHarmonicWorkload,
    pub mesh: SolverMeshArtifact,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

pub struct PreparedAcousticHarmonicTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

pub fn prepare_acoustic_harmonic_task(
    preparation: AcousticHarmonicTaskPreparation,
) -> AnalysisBridgeResult<PreparedAcousticHarmonicTask> {
    let prepared = prepare_acoustic_harmonic_inputs(
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
    let sequence = SequenceIdentity::from_solver_contract(&preparation.workload.model.sequence)?;
    let step = preparation
        .workload
        .model
        .sequence
        .steps
        .iter()
        .find(|s| s.step_id == preparation.workload.step_id)
        .ok_or_else(|| {
            crate::AnalysisBridgeError::Invalid("acoustic_harmonic step absent".into())
        })?;
    let step = StepIdentity::from_solver_contract(sequence, step)?;
    let numerical = NumericalPlanIdentity::from_acoustic_workload(
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
        AnalysisWorkKind::SolveAcousticHarmonic,
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
    Ok(PreparedAcousticHarmonicTask {
        host,
        program,
        inputs,
        resource_plan,
    })
}
