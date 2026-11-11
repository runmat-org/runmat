//! Portable task construction for standalone host thermal solves.
use crate::{
    prepare_analysis_objects, prepare_analysis_program, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputPolicy,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity, NumericalPlanIdentity,
    SequenceIdentity, StepIdentity, StudyIdentity, ThermalWorkload,
};
use runmat_analysis_fea::planning::resource::FeaResourcePlan;
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeSet;

pub struct ThermalTaskPreparation {
    /// Placement budget; the authored model and numerical intent retain their original identity.
    pub maximum_resident_bytes: Option<u64>,
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub workload: ThermalWorkload,
    pub field_source: Option<runmat_analysis_fea::execution::RetainedJouleHeatSource>,
    pub mesh: SolverMeshArtifact,
    pub output_policy: AnalysisOutputPolicy,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

pub struct PreparedThermalTask {
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
}

pub fn prepare_thermal_task(
    preparation: ThermalTaskPreparation,
) -> AnalysisBridgeResult<PreparedThermalTask> {
    if let Some(source) = &preparation.field_source {
        source.validate_for(&preparation.workload, &preparation.mesh)?;
    }
    let prepared = crate::thermal_plan::prepare_thermal_inputs_with_source(
        &preparation.workload,
        &preparation.mesh,
        preparation.output_policy,
        preparation.field_source.as_ref(),
        preparation.local_execution,
        preparation.maximum_resident_bytes.unwrap_or(u64::MAX),
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
        .ok_or_else(|| crate::AnalysisBridgeError::Invalid("thermal step absent".into()))?;
    let step = StepIdentity::from_solver_contract(sequence, step)?;
    let numerical = NumericalPlanIdentity::from_thermal_intent(
        preparation.workload.numerical_intent_identity()?,
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
        AnalysisWorkKind::SolveThermal,
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
    Ok(PreparedThermalTask {
        host,
        program,
        inputs,
        resource_plan,
        output_policy: preparation.output_policy,
    })
}
