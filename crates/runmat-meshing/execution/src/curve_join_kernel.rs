use runmat_execution::Digest;
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::{
    decode_shared_curve_batch, encode_shared_curve_mesh, join_shared_curve_batches,
    validate_shared_curve_geometry, SharedCurveBatch, SHARED_CURVE_BATCH_SCHEMA_VERSION,
    SHARED_CURVE_MESH_SCHEMA_VERSION,
};

use crate::curve_kernel::{curve_failure, curve_options, map_curve_error, resolved_curve_metric};
use crate::{
    ExactCurveEvaluatorProvider, MeshingStageCheckpoint, MeshingStageInvocation,
    MeshingStageKernel, PortableCurveEvaluatorProvider, PreparedExactGeometryObjects,
    PreparedMeshingInput, PreparedMeshingResultPublication, ValidatedMeshingStageOutput,
};

#[derive(Clone, Debug)]
pub struct ExactCurveJoinKernel<P = PortableCurveEvaluatorProvider> {
    evaluator_provider: P,
}

impl Default for ExactCurveJoinKernel<PortableCurveEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableCurveEvaluatorProvider,
        }
    }
}

impl<P> ExactCurveJoinKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
    }
}

impl<P: ExactCurveEvaluatorProvider> MeshingStageKernel for ExactCurveJoinKernel<P> {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::CurveMesh
            || invocation.host.workload.partition.kind
                != runmat_meshing_core::MeshingPartitionKind::DeterministicJoin
        {
            return Err(curve_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "submit the curve join with its deterministic-join descriptor",
                "curve join stage shape",
            ));
        }
        let (geometry, partitions) = inputs(invocation.inputs)?;
        let geometry_root_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
        let mut encoded_input_bytes = 0_u64;
        let mut batches = Vec::<SharedCurveBatch>::with_capacity(partitions.len());
        for partition in partitions {
            let streams = partition
                .stage_objects()
                .decoded_streams()
                .map_err(|error| invalid_partition(&error.to_string()))?;
            let [stream] = streams.as_slice() else {
                return Err(invalid_partition(
                    "curve partition closure must contain exactly one logical stream",
                ));
            };
            if partition.stage_objects().result_identity.stage != MeshingStageKind::CurveMesh
                || partition.stage_objects().result_identity.result_kind
                    != MeshingStageResultKind::Partition
                || stream.media_type != MeshingChunkMediaType::CurvePartitions
                || stream.schema_version != SHARED_CURVE_BATCH_SCHEMA_VERSION
                || stream.records.len() != 1
                || partition
                    .stage_objects()
                    .manifest
                    .prerequisite_manifest_digests
                    != [geometry_root_digest]
            {
                return Err(invalid_partition(
                    "curve partition identity, source geometry, media, schema, or record count is invalid",
                ));
            }
            encoded_input_bytes = encoded_input_bytes
                .checked_add(stream.records[0].len() as u64)
                .ok_or_else(|| invalid_partition("curve partition byte count overflowed"))?;
            batches.push(
                decode_shared_curve_batch(&stream.records[0], &geometry.topology)
                    .map_err(map_curve_error)?,
            );
        }

        let mesh =
            join_shared_curve_batches(&geometry.topology, batches).map_err(map_curve_error)?;
        let evaluator = self
            .evaluator_provider
            .evaluator(geometry)
            .map_err(|error| invalid_partition(&error.to_string()))?;
        let control = invocation.control.geometry_evaluation_control();
        let metric = resolved_curve_metric(
            geometry,
            evaluator.as_ref(),
            &control,
            &invocation.host.resolved_request,
        )?;
        let options = curve_options(invocation.host);
        let report = validate_shared_curve_geometry(
            &mesh,
            &geometry.topology,
            evaluator.as_ref(),
            evaluator.as_ref(),
            &metric,
            &control,
            options,
        )
        .map_err(map_curve_error)?;
        let encoded =
            encode_shared_curve_mesh(&mesh, &geometry.topology).map_err(map_curve_error)?;
        let usage = control.usage();
        drop(control);

        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("curve_edges".into(), report.edge_count);
        entity_counts.insert("curve_nodes".into(), report.node_count);
        entity_counts.insert(
            "curve_metric_evaluations".into(),
            report.metric_evaluation_count,
        );
        let checkpoint = MeshingStageCheckpoint {
            completed_work: report.edge_count,
            estimated_work: report.edge_count,
            node_count: report.node_count,
            peak_memory_bytes: usage
                .allocation_bytes
                .saturating_add(encoded_input_bytes)
                .saturating_add(encoded.len() as u64),
            search_work: usage.search_work,
            iterations: usage.iterations,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation_digest(&encoded),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::CurveMesh,
                schema_version: SHARED_CURVE_MESH_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

fn inputs(
    inputs: &[PreparedMeshingInput],
) -> Result<
    (
        &PreparedExactGeometryObjects,
        Vec<&PreparedMeshingResultPublication>,
    ),
    Box<MeshingFailure>,
> {
    let mut geometry = None;
    let mut partitions = Vec::new();
    for input in inputs {
        match input {
            PreparedMeshingInput::ExactGeometry(input) if geometry.is_none() => {
                geometry = Some(input.geometry_objects());
            }
            PreparedMeshingInput::StageArtifact(input) => partitions.push(input.as_ref()),
            PreparedMeshingInput::ExactGeometry(_) | PreparedMeshingInput::FacetedGeometry(_) => {
                return Err(invalid_partition("curve join input kinds are invalid"));
            }
        }
    }
    let geometry = geometry.ok_or_else(|| invalid_partition("curve join lacks exact geometry"))?;
    if partitions.is_empty() || partitions.len() > 63 {
        return Err(invalid_partition(
            "curve join requires one through 63 partition artifacts",
        ));
    }
    Ok((geometry, partitions))
}

fn invalid_partition(detail: &str) -> Box<MeshingFailure> {
    curve_failure(
        MeshingFailureCategory::InvalidGeometry,
        None,
        "regenerate every curve partition from the same admitted exact revision",
        detail,
    )
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-exact-global-curve-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}
