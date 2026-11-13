//! Portable, explicitly planned magnetostatic work; no driver-side numerical solve.
use crate::{
    prepare_analysis_objects, prepare_analysis_program, prepare_magnetostatic_inputs,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisOutputPolicy, AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity, MagnetostaticWorkload,
    NumericalPlanIdentity, StudyIdentity,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeSet;

pub struct MagnetostaticTaskPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: MagnetostaticWorkload,
    pub mesh: SolverMeshArtifact,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

pub struct PreparedMagnetostaticTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}

pub fn prepare_magnetostatic_task(
    preparation: MagnetostaticTaskPreparation,
) -> AnalysisBridgeResult<PreparedMagnetostaticTask> {
    let prepared = prepare_magnetostatic_inputs(
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
    let numerical = NumericalPlanIdentity::from_magnetostatic_workload(
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
        AnalysisWorkKind::SolveMagnetostatic,
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
    Ok(PreparedMagnetostaticTask {
        host,
        program,
        inputs,
        resource_plan,
        output_policy: preparation.output_policy,
    })
}
