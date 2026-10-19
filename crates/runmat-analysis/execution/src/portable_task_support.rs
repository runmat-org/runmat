use std::collections::BTreeSet;

use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity, StepIdentity,
        StudyIdentity,
    },
    planning::resource::FeaResourcePlan,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace};

use crate::{
    prepare_analysis_objects, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, ANALYSIS_HOST_ABI,
};

pub(crate) struct PortableTaskHostPreparation<'a> {
    pub study: StudyIdentity,
    pub step: StepIdentity,
    pub execution_plan: ExecutionPlanIdentity,
    pub work_kind: AnalysisWorkKind,
    pub inputs: &'a [PreparedAnalysisObjects],
    pub outputs: Vec<AnalysisNumericalObjectKind>,
    pub artifact_access: ArtifactAccess,
    pub output_policy: crate::AnalysisOutputPolicy,
}

pub(crate) fn task_host(
    preparation: PortableTaskHostPreparation<'_>,
) -> AnalysisBridgeResult<AnalysisHostWorkload> {
    let PortableTaskHostPreparation {
        study,
        step,
        execution_plan,
        work_kind,
        inputs,
        outputs,
        artifact_access,
        output_policy,
    } = preparation;
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::SemanticState
                    | AnalysisNumericalObjectKind::StructuralModalResponseSpectrumDirectionResult
                    | AnalysisNumericalObjectKind::StructuralRandomResponseIntervalResult
            ) {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect();
    let request = AnalysisWorkloadRequest::new(
        study,
        step,
        execution_plan,
        work_kind,
        workload_inputs,
        outputs,
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    AnalysisHostWorkload::new(request, artifact_access, output_policy)
}

pub(crate) fn resource_plan_objects(
    plan: &FeaResourcePlan,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        chunk_bytes,
        ObjectNamespace::InputValue,
        limits,
    )
}
