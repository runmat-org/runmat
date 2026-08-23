use runmat_execution::Digest;
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::{
    apply_shared_curve_splits, decode_shared_curve_mesh, encode_shared_curve_mesh,
    SharedCurveEvaluationContext, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    decode_exact_face_partition_result, decode_exact_surface_pass_result, ExactSurfacePassOutcome,
    EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};

use crate::curve_kernel::{curve_failure, curve_options, map_curve_error, resolved_curve_metric};
use crate::surface_join_kernel::partition_record;
use crate::surface_kernel::{curve_record, exact_surface_join_options};
use crate::{
    ExactMeshingEvaluatorProvider, MeshingStageCheckpoint, MeshingStageInvocation,
    MeshingStageKernel, PortableMeshingEvaluatorProvider, PreparedExactGeometryObjects,
    PreparedMeshingInput, PreparedMeshingResultPublication, ValidatedMeshingStageOutput,
};

#[derive(Clone, Debug)]
pub struct ExactCurveRefinementKernel<P = PortableMeshingEvaluatorProvider> {
    evaluator_provider: P,
}

impl Default for ExactCurveRefinementKernel<PortableMeshingEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableMeshingEvaluatorProvider,
        }
    }
}

impl<P> ExactCurveRefinementKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
    }
}

impl<P: ExactMeshingEvaluatorProvider> MeshingStageKernel for ExactCurveRefinementKernel<P> {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::CurveMesh
            || invocation.host.workload.partition.kind != MeshingPartitionKind::WholeStage
        {
            return Err(curve_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "submit curve refinement as one whole-stage barrier",
                "curve refinement stage shape",
            ));
        }
        let (geometry, curve_publication, pass_publication, partition_publications) =
            inputs(invocation.inputs)?;
        let curve_bytes = curve_record(geometry, curve_publication)?;
        let curves =
            decode_shared_curve_mesh(&curve_bytes, &geometry.topology).map_err(map_curve_error)?;
        let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
        let curve_digest =
            StableDigest::from_bytes(*curve_publication.stage_objects().root.digest.bytes());
        let mut encoded_input_bytes = checked_count(
            std::iter::once(curve_bytes.len()),
            "shared curve byte count overflowed",
        )?;
        let mut partitions = Vec::with_capacity(partition_publications.len());
        let mut pass_prerequisites = vec![geometry_digest, curve_digest];
        for publication in partition_publications {
            let bytes = partition_record(publication, geometry_digest, curve_digest)?;
            encoded_input_bytes = add_bytes(encoded_input_bytes, bytes.len())?;
            partitions.push(
                decode_exact_face_partition_result(&bytes, &geometry.topology, &curves)
                    .map_err(|error| invalid_input(&error.to_string()))?,
            );
            pass_prerequisites.push(StableDigest::from_bytes(
                *publication.stage_objects().root.digest.bytes(),
            ));
            invocation
                .control
                .checkpoint(MeshingStageCheckpoint::default())?;
        }
        pass_prerequisites.sort();
        let pass_bytes = pass_record(pass_publication, &pass_prerequisites)?;
        encoded_input_bytes = add_bytes(encoded_input_bytes, pass_bytes.len())?;
        let pass = decode_exact_surface_pass_result(
            &pass_bytes,
            &geometry.topology,
            &curves,
            &partitions,
            exact_surface_join_options(invocation.host),
        )
        .map_err(|error| invalid_input(&error.to_string()))?;
        let ExactSurfacePassOutcome::RequiresCurveSplits { splits } = pass.outcome else {
            return Err(invalid_input(
                "a converged surface pass cannot enter curve refinement",
            ));
        };

        let evaluator = self
            .evaluator_provider
            .evaluator(geometry)
            .map_err(|error| invalid_input(&error.to_string()))?;
        let control = invocation.control.geometry_evaluation_control();
        let metric = resolved_curve_metric(
            geometry,
            evaluator.as_ref(),
            &control,
            &invocation.host.resolved_request,
        )?;
        let refined = apply_shared_curve_splits(
            &curves,
            SharedCurveEvaluationContext::new(
                &geometry.topology,
                evaluator.as_ref(),
                evaluator.as_ref(),
                &metric,
                &control,
            ),
            curve_options(invocation.host),
            &splits,
        )
        .map_err(map_curve_error)?;
        let encoded =
            encode_shared_curve_mesh(&refined, &geometry.topology).map_err(map_curve_error)?;
        let usage = control.usage();

        let edge_count = checked_count(
            std::iter::once(refined.edges.len()),
            "refined curve edge count overflowed",
        )?;
        let node_count = checked_count(
            refined.edges.iter().map(|edge| edge.nodes.len()),
            "refined curve node count overflowed",
        )?;
        let split_count = checked_count(
            std::iter::once(splits.len()),
            "applied surface split count overflowed",
        )?;
        let encoded_bytes = checked_count(
            std::iter::once(encoded.len()),
            "refined curve byte count overflowed",
        )?;
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("curve_edges".into(), edge_count);
        entity_counts.insert("curve_nodes".into(), node_count);
        entity_counts.insert("applied_surface_splits".into(), split_count);
        let checkpoint = MeshingStageCheckpoint {
            completed_work: edge_count,
            estimated_work: edge_count,
            node_count,
            peak_memory_bytes: usage
                .allocation_bytes
                .saturating_add(encoded_input_bytes)
                .saturating_add(encoded_bytes),
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
        &PreparedMeshingResultPublication,
        &PreparedMeshingResultPublication,
        Vec<&PreparedMeshingResultPublication>,
    ),
    Box<MeshingFailure>,
> {
    let mut geometry = None;
    let mut curves = None;
    let mut pass = None;
    let mut partitions = Vec::new();
    for input in inputs {
        match input {
            PreparedMeshingInput::ExactGeometry(input) if geometry.is_none() => {
                geometry = Some(input.geometry_objects());
            }
            PreparedMeshingInput::StageArtifact(input)
                if input.stage_objects().result_identity.stage == MeshingStageKind::CurveMesh
                    && curves.is_none() =>
            {
                curves = Some(input.as_ref());
            }
            PreparedMeshingInput::StageArtifact(input)
                if input.stage_objects().result_identity.stage == MeshingStageKind::SurfaceMesh
                    && input.stage_objects().result_identity.result_kind
                        == MeshingStageResultKind::DeterministicJoin
                    && pass.is_none() =>
            {
                pass = Some(input.as_ref());
            }
            PreparedMeshingInput::StageArtifact(input) => partitions.push(input.as_ref()),
            PreparedMeshingInput::ExactGeometry(_)
            | PreparedMeshingInput::FacetedGeometry(_)
            | PreparedMeshingInput::DomainModel(_)
            | PreparedMeshingInput::Evidence(_) => {
                return Err(invalid_input("curve refinement input kinds are invalid"));
            }
        }
    }
    if partitions.is_empty() || partitions.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid_input(
            "curve refinement requires the complete bounded face partition set",
        ));
    }
    Ok((
        geometry.ok_or_else(|| invalid_input("curve refinement lacks exact geometry"))?,
        curves.ok_or_else(|| invalid_input("curve refinement lacks current shared curves"))?,
        pass.ok_or_else(|| invalid_input("curve refinement lacks a surface pass"))?,
        partitions,
    ))
}

fn pass_record(
    publication: &PreparedMeshingResultPublication,
    prerequisites: &[StableDigest],
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let streams = publication
        .stage_objects()
        .decoded_streams()
        .map_err(|error| invalid_input(&error.to_string()))?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_input(
            "surface pass must contain one logical stream",
        ));
    };
    if publication.stage_objects().result_identity.stage != MeshingStageKind::SurfaceMesh
        || publication.stage_objects().result_identity.result_kind
            != MeshingStageResultKind::DeterministicJoin
        || publication
            .stage_objects()
            .manifest
            .prerequisite_manifest_digests
            != prerequisites
        || stream.media_type != MeshingChunkMediaType::SurfacePartitions
        || stream.schema_version != EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION
        || stream.records.len() != 1
    {
        return Err(invalid_input(
            "surface pass identity, prerequisites, media, schema, or record count is invalid",
        ));
    }
    Ok(stream.records[0].clone())
}

fn add_bytes(total: u64, length: usize) -> Result<u64, Box<MeshingFailure>> {
    let length = checked_count(
        std::iter::once(length),
        "curve refinement byte count overflowed",
    )?;
    total
        .checked_add(length)
        .ok_or_else(|| invalid_input("curve refinement byte count overflowed"))
}

fn checked_count(
    lengths: impl Iterator<Item = usize>,
    detail: &str,
) -> Result<u64, Box<MeshingFailure>> {
    crate::accounting::checked_sum_lengths(lengths).ok_or_else(|| {
        curve_failure(
            MeshingFailureCategory::InternalInvariantViolation,
            None,
            "reduce the admitted curve refinement size",
            detail,
        )
    })
}

fn invalid_input(detail: &str) -> Box<MeshingFailure> {
    curve_failure(
        MeshingFailureCategory::InvalidGeometry,
        None,
        "regenerate curve refinement from one admitted surface restart pass",
        detail,
    )
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-refined-global-curve-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}
