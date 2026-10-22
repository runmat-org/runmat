use crate::{MeshingExecutionError, MeshingHostWorkload, PreparedMeshingResultPublication};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionRequest;
use runmat_meshing_core::{
    build_chunked_stage_payload_with_cancellation, profile_chunked_stage_payload,
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingChunkMediaType, MeshingChunkPolicy,
    MeshingChunkStream, MeshingFailure, MeshingFailureCategory, MeshingManifestDisposition,
    MeshingPartitionIdentity, MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind,
    MeshingWorkloadResult, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
};

mod admission;
mod input;
mod input_inventory;
mod publication;
use publication::publish_stage;

pub use input::PreparedMeshingInput;

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
    stage_evidence: runmat_meshing_core::MeshingStageEvidence,
    stage_evidence_reference: runmat_execution::value::ValueRef,
}

impl CompletedMeshingStage {
    pub const fn workload_result(&self) -> &MeshingWorkloadResult {
        &self.workload_result
    }

    pub const fn publication(&self) -> &PreparedMeshingResultPublication {
        &self.publication
    }

    pub const fn stage_evidence(&self) -> &runmat_meshing_core::MeshingStageEvidence {
        &self.stage_evidence
    }

    pub const fn stage_evidence_reference(&self) -> &runmat_execution::value::ValueRef {
        &self.stage_evidence_reference
    }

    pub fn attempt_success(&self) -> runmat_execution_runner::AttemptSuccess {
        let mut success = self.publication.attempt_success();
        success
            .push_result_object(self.stage_evidence_reference.clone())
            .expect("meshing publications are value results");
        success
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
    let host = MeshingHostWorkload::from_program_request(request, store)?;
    let roots = MeshingHostWorkload::semantic_input_roots(request)?;
    let limits = effective_limits(&host, inventory_limits)?;
    let mut control = MeshingStageControl::new(
        host.workload.stage,
        host.workload.partition.partition_index,
        &host.resolved_request,
        host.ceilings,
        cancellation,
        progress,
    )?;
    let input_control = control.geometry_evaluation_control();
    let input_scope = runmat_geometry_core::GeometryAllocationScope::new(&input_control);
    let input_reader = admission::InputReader::new(&*store, &input_scope);
    let inputs = admission::import_inputs(&host, &roots, &input_reader, limits)?;
    drop(input_reader);
    input_inventory::validate(&inputs, limits, &input_control, host.workload.stage)?;

    control.checkpoint(MeshingStageCheckpoint::default())?;
    control.begin_kernel_allocations()?;
    let produced = kernel.execute(MeshingStageInvocation {
        host: &host,
        inputs: &inputs,
        control: &mut control,
    })?;
    let record_storage = control.finish_kernel_allocations(&produced.streams)?;
    // Keep the returned record value after its guard in drop order.
    let output = produced;
    drop(inputs);
    drop(input_scope);
    let ValidatedMeshingStageOutput {
        invariant_summary_digest,
        streams,
        final_checkpoint,
    } = output;
    control.checkpoint(final_checkpoint)?;
    control.guard()?;
    validate_stage_streams(host.workload.stage, &streams)?;
    invariant_summary_digest
        .validate_nonzero("stage invariant summary")
        .map_err(MeshingExecutionError::from)?;

    let policy = effective_chunk_policy(&host, chunk_policy)?;
    let publication_control = control.geometry_evaluation_control();
    let publication_codec = crate::codec_control::CodecControl::new(&publication_control);
    let chunk_profile = profile_chunked_stage_payload(&streams, policy, &publication_codec)
        .map_err(|error| chunk_failure(&host, error, &publication_codec))?;
    // Records and chunks coexist during encoding. Keep the additional charge
    // through publication; never allocate the payload before this admission.
    let _chunk_storage = runmat_geometry_core::GeometryAllocationReservation::new(
        &publication_control,
        chunk_profile.requested_storage_bytes,
    )
    .map_err(|error| {
        MeshingSerialExecutionError::Stage(failure(
            host.workload.stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        ))
    })?;
    let payload =
        build_chunked_stage_payload_with_cancellation(&streams, policy, &publication_codec)
            .map_err(|error| chunk_failure(&host, error, &publication_codec))?;
    // Chunk ownership has moved into the payload; release the original record buffers.
    drop(streams);
    drop(record_storage);
    publish_stage(
        &host,
        store,
        &mut control,
        invariant_summary_digest,
        payload,
        limits,
    )
}

fn chunk_failure(
    host: &MeshingHostWorkload,
    error: runmat_meshing_core::MeshingContractError,
    codec: &crate::codec_control::CodecControl<'_>,
) -> MeshingSerialExecutionError {
    if let Some(original) = codec.take_failure() {
        return MeshingSerialExecutionError::Stage(failure(
            host.workload.stage,
            crate::geometry_failure::category(original.kind),
            &original.reason,
            None,
        ));
    }
    if error.field == "meshing chunks" && error.reason.contains("hard total byte limit") {
        let mut failure = artifact_budget_failure(host);
        failure
            .achieved_values
            .push(runmat_meshing_core::MeshingDiagnosticEntry {
                name: "chunk_artifact_failure".into(),
                value: runmat_meshing_core::MeshingDiagnosticValue::Text(
                    crate::diagnostic::bounded_diagnostic_text(
                        &error.to_string(),
                        "chunk artifact limit exceeded",
                    ),
                ),
                unit: None,
            });
        MeshingSerialExecutionError::Stage(failure)
    } else {
        MeshingSerialExecutionError::Bridge(MeshingExecutionError::from(error))
    }
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
        MeshingStageKind::CurveMesh => matches!(
            media,
            MeshingChunkMediaType::CurvePartitions | MeshingChunkMediaType::CurveMesh
        ),
        MeshingStageKind::SurfaceMesh => {
            matches!(
                media,
                MeshingChunkMediaType::SurfacePartitions | MeshingChunkMediaType::SurfaceMesh
            )
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
            MeshingChunkMediaType::SolverMeshProjection
                | MeshingChunkMediaType::MeshNodes
                | MeshingChunkMediaType::MeshElements
                | MeshingChunkMediaType::MeshClassification
        ),
        MeshingStageKind::Validation => {
            matches!(media, MeshingChunkMediaType::ValidationEvidence)
        }
        MeshingStageKind::Serialization | MeshingStageKind::Publication => matches!(
            media,
            MeshingChunkMediaType::SolverMeshArtifact | MeshingChunkMediaType::MeshingEvidence
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
        .min(host.resources().maximum_artifact_bytes);
    limits.max_object_bytes = limits.max_object_bytes.min(limits.max_total_bytes);
    Ok(limits)
}

fn effective_chunk_policy(
    host: &MeshingHostWorkload,
    mut policy: MeshingChunkPolicy,
) -> Result<MeshingChunkPolicy, MeshingSerialExecutionError> {
    policy.maximum_total_encoded_bytes = policy
        .maximum_total_encoded_bytes
        .min(host.resources().maximum_artifact_bytes);
    if policy.maximum_chunk_bytes > policy.maximum_total_encoded_bytes {
        policy.maximum_chunk_bytes = policy.maximum_total_encoded_bytes;
    }
    policy
        .validate()
        .map_err(|_| MeshingSerialExecutionError::Stage(artifact_budget_failure(host)))?;
    Ok(policy)
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
