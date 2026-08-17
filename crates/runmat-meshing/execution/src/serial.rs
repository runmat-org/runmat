use crate::{
    import_exact_geometry_input, import_faceted_geometry_input, import_result_publication,
    prepare_result_publication, prepare_stage_objects, MeshingExecutionError, MeshingHostWorkload,
    PreparedExactGeometryInput, PreparedFacetedGeometryInput, PreparedMeshingResultPublication,
};
use runmat_execution::value::ValuePayload;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_meshing_core::{
    build_chunked_stage_payload, build_closed_stage_manifest, CanonicalMeshingContract,
    MeshingCancellationSignal, MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStream,
    MeshingFailure, MeshingFailureCategory, MeshingInputKind, MeshingManifestDisposition,
    MeshingPartitionIdentity, MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind,
    MeshingWorkloadResult, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
};

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedMeshingInput {
    ExactGeometry(Box<PreparedExactGeometryInput>),
    FacetedGeometry(Box<PreparedFacetedGeometryInput>),
    StageArtifact(Box<PreparedMeshingResultPublication>),
}

impl PreparedMeshingInput {
    pub const fn exact_geometry(&self) -> Option<&PreparedExactGeometryInput> {
        match self {
            Self::ExactGeometry(input) => Some(input),
            Self::FacetedGeometry(_) | Self::StageArtifact(_) => None,
        }
    }

    pub const fn faceted_geometry(&self) -> Option<&PreparedFacetedGeometryInput> {
        match self {
            Self::FacetedGeometry(input) => Some(input),
            Self::ExactGeometry(_) | Self::StageArtifact(_) => None,
        }
    }

    pub const fn stage_artifact(&self) -> Option<&PreparedMeshingResultPublication> {
        match self {
            Self::ExactGeometry(_) | Self::FacetedGeometry(_) => None,
            Self::StageArtifact(input) => Some(input),
        }
    }

    fn objects(&self) -> &[runmat_execution_artifact::LogicalObject] {
        match self {
            Self::ExactGeometry(input) => &input.geometry_objects().objects,
            Self::FacetedGeometry(input) => &input.geometry_objects().objects,
            Self::StageArtifact(input) => &input.stage_objects().objects,
        }
    }
}

use super::budget::{failure, MeshingProgressSink, MeshingStageCheckpoint, MeshingStageControl};

pub struct MeshingStageInvocation<'a, 'control> {
    pub host: &'a MeshingHostWorkload,
    pub inputs: &'a [PreparedMeshingInput],
    pub control: &'a mut MeshingStageControl<'control>,
}

/// Meshing-owned semantic implementation for one stage invocation.
///
/// The execution bridge supplies only validated contracts and artifact closures. Concrete kernels
/// remain responsible for constructive geometry work and independent stage validation.
pub trait MeshingStageKernel: Send + Sync {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedMeshingStageOutput {
    pub invariant_summary_digest: StableDigest,
    pub streams: Vec<MeshingChunkStream>,
    pub final_checkpoint: MeshingStageCheckpoint,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedMeshingStage {
    workload_result: MeshingWorkloadResult,
    publication: PreparedMeshingResultPublication,
}

impl CompletedMeshingStage {
    pub const fn workload_result(&self) -> &MeshingWorkloadResult {
        &self.workload_result
    }

    pub const fn publication(&self) -> &PreparedMeshingResultPublication {
        &self.publication
    }

    pub fn attempt_success(&self) -> runmat_execution_runner::AttemptSuccess {
        self.publication.attempt_success()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MeshingSerialExecutionError {
    #[error("meshing execution bridge rejected serial stage: {0}")]
    Bridge(#[from] MeshingExecutionError),
    #[error("meshing stage failed: {0}")]
    Stage(#[from] Box<MeshingFailure>),
}

impl From<runmat_meshing_core::MeshingContractError> for MeshingSerialExecutionError {
    fn from(error: runmat_meshing_core::MeshingContractError) -> Self {
        Self::Bridge(MeshingExecutionError::from(error))
    }
}

impl MeshingSerialExecutionError {
    pub fn workload_result(&self) -> Option<MeshingWorkloadResult> {
        match self {
            Self::Stage(failure) => Some(MeshingWorkloadResult::Failed {
                failure: failure.as_ref().clone(),
            }),
            Self::Bridge(_) => None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute_serial_stage<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    cancellation: &dyn MeshingCancellationSignal,
    progress: &mut dyn MeshingProgressSink,
    chunk_policy: MeshingChunkPolicy,
    inventory_limits: ObjectInventoryLimits,
) -> Result<CompletedMeshingStage, MeshingSerialExecutionError>
where
    S: CacheImport + CacheExport,
    K: MeshingStageKernel + ?Sized,
{
    let host = MeshingHostWorkload::from_program_request(request)?;
    let roots = input_roots(request)?;
    let limits = effective_limits(&host, inventory_limits)?;
    let inputs = host
        .workload
        .inputs
        .iter()
        .zip(&roots)
        .map(|(input, root)| match input.kind {
            MeshingInputKind::ExactGeometry => import_exact_geometry_input(
                store,
                host.geometry_document.clone().ok_or_else(|| {
                    MeshingExecutionError::Invalid(
                        "exact geometry input is missing its host document".into(),
                    )
                })?,
                root,
                host.artifact_access.clone(),
                limits,
            )
            .map(Box::new)
            .map(PreparedMeshingInput::ExactGeometry),
            MeshingInputKind::FacetedGeometry => import_faceted_geometry_input(
                store,
                host.geometry_document.clone().ok_or_else(|| {
                    MeshingExecutionError::Invalid(
                        "faceted geometry input is missing its host document".into(),
                    )
                })?,
                root,
                host.artifact_access.clone(),
                limits,
            )
            .map(Box::new)
            .map(PreparedMeshingInput::FacetedGeometry),
            MeshingInputKind::StageArtifact => {
                import_result_publication(store, root, host.artifact_access.clone(), limits)
                    .map(Box::new)
                    .map(PreparedMeshingInput::StageArtifact)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_combined_input_inventory(&inputs, limits)?;

    let mut control = MeshingStageControl::new(
        host.workload.stage,
        host.workload.partition.partition_index,
        &host.resolved_request,
        cancellation,
        progress,
    )?;
    control.checkpoint(MeshingStageCheckpoint::default())?;
    let output = kernel.execute(MeshingStageInvocation {
        host: &host,
        inputs: &inputs,
        control: &mut control,
    })?;
    control.checkpoint(output.final_checkpoint.clone())?;
    control.guard()?;
    validate_stage_streams(host.workload.stage, &output.streams)?;
    output
        .invariant_summary_digest
        .validate_nonzero("stage invariant summary")
        .map_err(MeshingExecutionError::from)?;

    let policy = effective_chunk_policy(&host, chunk_policy)?;
    let payload = build_chunked_stage_payload(&output.streams, policy).map_err(|error| {
        if error.field == "meshing chunks" && error.reason.contains("hard total byte limit") {
            MeshingSerialExecutionError::Stage(artifact_budget_failure(&host))
        } else {
            MeshingSerialExecutionError::Bridge(MeshingExecutionError::from(error))
        }
    })?;
    let partition_identity = MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: host.workload.stage_identity_digest,
        partition: host.workload.partition.clone(),
    };
    let result_kind = match host.workload.partition.kind {
        MeshingPartitionKind::WholeStage => MeshingStageResultKind::WholeStage,
        MeshingPartitionKind::CanonicalEntityBatch
        | MeshingPartitionKind::DisconnectedComponent => MeshingStageResultKind::Partition,
    };
    let (result_identity, manifest) = build_closed_stage_manifest(
        host.workload.stage,
        result_kind,
        partition_identity.canonical_digest()?,
        output.invariant_summary_digest,
        host.workload
            .inputs
            .iter()
            .map(|input| input.digest)
            .collect(),
        MeshingManifestDisposition::ValidatedDependency,
        &payload,
    )?;
    let stage_objects = prepare_stage_objects(result_identity, manifest, payload.chunks, limits)
        .map_err(|error| map_artifact_limit(&host, error))?;
    let publication =
        prepare_result_publication(stage_objects, host.artifact_access.clone(), limits)
            .map_err(|error| map_artifact_limit(&host, error))?;
    for object in &publication.stage_objects().objects {
        store
            .write_verified(object)
            .map_err(MeshingExecutionError::from)?;
    }
    let manifest_digest =
        StableDigest::from_bytes(*publication.stage_objects().root.digest.bytes());
    let workload_result = MeshingWorkloadResult::Validated {
        stage_manifest_digest: manifest_digest,
    };
    workload_result.validate_against(&host.workload)?;
    Ok(CompletedMeshingStage {
        workload_result,
        publication,
    })
}

fn validate_stage_streams(
    stage: MeshingStageKind,
    streams: &[MeshingChunkStream],
) -> Result<(), MeshingSerialExecutionError> {
    let legal = |media| match stage {
        MeshingStageKind::GeometryAdmission | MeshingStageKind::Healing => {
            matches!(
                media,
                MeshingChunkMediaType::ExactGeometry | MeshingChunkMediaType::FacetedGeometry
            )
        }
        MeshingStageKind::Sizing => matches!(media, MeshingChunkMediaType::MetricField),
        MeshingStageKind::CurveMesh => matches!(media, MeshingChunkMediaType::CurvePartitions),
        MeshingStageKind::SurfaceMesh => {
            matches!(media, MeshingChunkMediaType::SurfacePartitions)
        }
        MeshingStageKind::ProtectedBoundaryComplex => {
            matches!(media, MeshingChunkMediaType::ProtectedBoundaryComplex)
        }
        MeshingStageKind::Tetrahedralization
        | MeshingStageKind::ConstraintRecovery
        | MeshingStageKind::Refinement
        | MeshingStageKind::Optimization => matches!(
            media,
            MeshingChunkMediaType::VolumeTopology
                | MeshingChunkMediaType::MeshNodes
                | MeshingChunkMediaType::MeshElements
                | MeshingChunkMediaType::MeshClassification
        ),
        MeshingStageKind::OrderElevation => matches!(
            media,
            MeshingChunkMediaType::MeshNodes
                | MeshingChunkMediaType::MeshElements
                | MeshingChunkMediaType::MeshClassification
        ),
        MeshingStageKind::Validation => {
            matches!(media, MeshingChunkMediaType::ValidationEvidence)
        }
        MeshingStageKind::Serialization | MeshingStageKind::Publication => matches!(
            media,
            MeshingChunkMediaType::AnalysisMeshArtifact | MeshingChunkMediaType::MeshingEvidence
        ),
    };
    if streams.is_empty() || streams.iter().any(|stream| !legal(stream.media_type)) {
        return Err(failure(
            stage,
            MeshingFailureCategory::InternalInvariantViolation,
            "emit only the canonical artifact media owned by this meshing stage",
            None,
        )
        .into());
    }
    Ok(())
}

fn input_roots(
    request: &ProgramExecutionRequest,
) -> Result<Vec<runmat_execution::value::ValueRef>, MeshingExecutionError> {
    request
        .arguments
        .iter()
        .map(|value| match value {
            ValuePayload::Object(root) => Ok((**root).clone()),
            ValuePayload::Inline(_) => Err(MeshingExecutionError::Invalid(
                "serial meshing inputs must remain externalized artifact roots".into(),
            )),
        })
        .collect()
}

fn effective_limits(
    host: &MeshingHostWorkload,
    mut limits: ObjectInventoryLimits,
) -> Result<ObjectInventoryLimits, MeshingExecutionError> {
    if limits.max_objects == 0 || limits.max_object_bytes == 0 || limits.max_total_bytes == 0 {
        return Err(MeshingExecutionError::Invalid(
            "serial meshing object inventory limits must be non-zero".into(),
        ));
    }
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(host.resolved_request.resources.maximum_artifact_bytes);
    limits.max_object_bytes = limits.max_object_bytes.min(limits.max_total_bytes);
    Ok(limits)
}

fn effective_chunk_policy(
    host: &MeshingHostWorkload,
    mut policy: MeshingChunkPolicy,
) -> Result<MeshingChunkPolicy, MeshingSerialExecutionError> {
    policy.maximum_total_encoded_bytes = policy
        .maximum_total_encoded_bytes
        .min(host.resolved_request.resources.maximum_artifact_bytes);
    if policy.maximum_chunk_bytes > policy.maximum_total_encoded_bytes {
        policy.maximum_chunk_bytes = policy.maximum_total_encoded_bytes;
    }
    policy
        .validate()
        .map_err(|_| MeshingSerialExecutionError::Stage(artifact_budget_failure(host)))?;
    Ok(policy)
}

fn validate_combined_input_inventory(
    inputs: &[PreparedMeshingInput],
    limits: ObjectInventoryLimits,
) -> Result<(), MeshingExecutionError> {
    let mut objects = std::collections::BTreeMap::new();
    let mut total = 0_u64;
    for input in inputs {
        for object in input.objects() {
            match objects.insert(object.descriptor.digest, object.descriptor.encoded_length) {
                Some(length) if length != object.descriptor.encoded_length => {
                    return Err(MeshingExecutionError::Identity(
                        "input object digest was reused with a different length",
                    ));
                }
                Some(_) => continue,
                None => {}
            }
            total = total
                .checked_add(object.descriptor.encoded_length)
                .ok_or_else(|| {
                    MeshingExecutionError::Invalid("input artifact byte total overflow".into())
                })?;
        }
    }
    if objects.len() > limits.max_objects || total > limits.max_total_bytes {
        return Err(MeshingExecutionError::Invalid(
            "combined meshing input artifact inventory exceeds its hard bound".into(),
        ));
    }
    Ok(())
}

fn map_artifact_limit(
    host: &MeshingHostWorkload,
    error: MeshingExecutionError,
) -> MeshingSerialExecutionError {
    if matches!(
        error,
        MeshingExecutionError::Artifact(runmat_execution_artifact::ArtifactError::Limit(_))
    ) {
        MeshingSerialExecutionError::Stage(artifact_budget_failure(host))
    } else {
        MeshingSerialExecutionError::Bridge(error)
    }
}

fn artifact_budget_failure(host: &MeshingHostWorkload) -> Box<MeshingFailure> {
    failure(
        host.workload.stage,
        MeshingFailureCategory::ArtifactBudgetExceeded,
        "increase the artifact budget or reduce the stage output size",
        None,
    )
}
