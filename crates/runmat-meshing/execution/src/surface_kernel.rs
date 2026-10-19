use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::{
    decode_shared_curve_mesh_with_cancellation, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    visit_exact_face_partition, visit_exact_face_partition_with_reuse, ExactFaceDelaunayOptions,
    ExactFaceJoinOptions, ExactFacePartitionContext, ExactFacePartitionOptions,
    ExactSurfaceJoinOptions, EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION,
};

mod options;
mod records;
#[cfg(test)]
mod tests;

pub(crate) use options::exact_surface_join_options;
use options::surface_options;
pub(crate) mod error;

use error::{invalid_input, map_surface_error, surface_failure};

use crate::{
    ExactMeshingEvaluatorProvider, MeshingStageCheckpoint, MeshingStageInvocation,
    MeshingStageKernel, PortableMeshingEvaluatorProvider, PreparedExactGeometryObjects,
    PreparedMeshingInput, PreparedMeshingResultPublication, ValidatedMeshingStageOutput,
};

#[derive(Clone, Debug)]
pub struct ExactSurfacePartitionKernel<P = PortableMeshingEvaluatorProvider> {
    evaluator_provider: P,
    subtask_cache: Option<std::sync::Arc<dyn crate::MeshingSubtaskCacheStore>>,
}

impl Default for ExactSurfacePartitionKernel<PortableMeshingEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableMeshingEvaluatorProvider,
            subtask_cache: None,
        }
    }
}

impl<P> ExactSurfacePartitionKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self {
            evaluator_provider,
            subtask_cache: None,
        }
    }

    pub fn with_subtask_cache(
        mut self,
        cache: Option<std::sync::Arc<dyn crate::MeshingSubtaskCacheStore>>,
    ) -> Self {
        self.subtask_cache = cache;
        self
    }
}

impl<P: ExactMeshingEvaluatorProvider> MeshingStageKernel for ExactSurfacePartitionKernel<P> {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::SurfaceMesh
            || invocation.host.workload.partition.kind != MeshingPartitionKind::CanonicalEntityBatch
        {
            return Err(surface_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "submit exact surface work only as canonical face batches",
                "surface partition stage shape",
            ));
        }
        let options = surface_options(invocation.host)?;
        let (geometry, curve_publication) = inputs(invocation.inputs)?;
        let control = invocation.control.geometry_evaluation_control();
        let reconstruction = runmat_geometry_core::GeometryAllocationScope::new(&control);
        let curve_bytes = curve_record(
            geometry,
            curve_publication,
            Some(&control),
            Some(&reconstruction),
        )?;
        let curves =
            decode_shared_curve_mesh_with_cancellation(&curve_bytes, &geometry.topology, &control)
                .map_err(crate::curve_kernel::map_curve_error)?;
        let curve_encoded_bytes = curve_bytes.len();
        drop(curve_bytes);
        drop(reconstruction);
        let evaluator = self
            .evaluator_provider
            .evaluator(geometry, &control)
            .map_err(|error| invalid_input(&error.to_string()))?;
        let metric = self
            .evaluator_provider
            .geometry_metric(
                geometry,
                evaluator.as_ref(),
                &control,
                &invocation.host.resolved_request,
            )
            .map_err(crate::curve_kernel::map_curve_error)?;
        let numerical_phase = tracing::info_span!("meshing.surface_numerical").entered();
        let context = ExactFacePartitionContext {
            topology: &geometry.topology,
            curves: &curves,
            metric_request: &metric,
            quality: invocation.host.resolved_request.quality.surface,
            evaluator: evaluator.as_ref(),
            geometry_control: &control,
            cancellation: &control,
        };
        let mut face_reuse = self
            .subtask_cache
            .as_ref()
            .map(|store| {
                crate::subtask_cache::SurfaceFaceReuse::new(
                    std::sync::Arc::clone(store),
                    geometry,
                    &invocation.host.resolved_request,
                    &metric,
                    &curves,
                    &control,
                )
            })
            .transpose()
            .map_err(map_surface_error)?
            .flatten();
        let partition = invocation.host.workload.partition.clone();
        let expected = partition
            .entity_range
            .as_ref()
            .map_or(0, |range| range.entity_count);
        let mut encoded = records::Records::new(&control, expected).map_err(map_surface_error)?;
        let completion = match face_reuse.as_mut() {
            Some(reuse) => visit_exact_face_partition_with_reuse(
                partition,
                context,
                options,
                reuse,
                &mut |face| encoded.accept(face),
            ),
            None => visit_exact_face_partition(partition, context, options, &mut |face| {
                encoded.accept(face)
            }),
        }
        .map_err(map_surface_error)?;
        let encoded = encoded.finish(completion).map_err(map_surface_error)?;
        drop(numerical_phase);
        let usage = control.usage();
        tracing::info!(
            event_code = "meshing.surface_storage.inventory",
            admitted_peak_bytes = usage.allocation_bytes,
            admitted_live_bytes = usage.live_allocation_bytes,
            encoded_input_bytes = curve_encoded_bytes,
            encoded_output_bytes = encoded.encoded_bytes,
            "surface storage inventory before terminal resource admission"
        );
        let (face_count, node_count, element_count, split_count, completed_work) = (
            encoded.faces,
            encoded.nodes,
            encoded.triangles,
            encoded.splits,
            encoded.completed,
        );
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("surface_faces".into(), face_count);
        entity_counts.insert("surface_nodes".into(), node_count);
        entity_counts.insert("surface_triangles".into(), element_count);
        entity_counts.insert("curve_split_demands".into(), split_count);
        entity_counts.insert(
            "reused_surface_faces".into(),
            face_reuse.as_ref().map_or(0, |reuse| reuse.hits()),
        );
        entity_counts.insert(
            "retained_surface_faces".into(),
            face_reuse.as_ref().map_or(0, |reuse| reuse.retained()),
        );
        let checkpoint = MeshingStageCheckpoint {
            completed_work,
            estimated_work: completed_work,
            node_count,
            element_count,
            peak_memory_bytes: usage
                .allocation_bytes
                .saturating_add(curve_encoded_bytes as u64),
            search_work: usage.search_work,
            iterations: usage.iterations,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        let streams = vec![MeshingChunkStream {
            media_type: MeshingChunkMediaType::SurfacePartitions,
            schema_version: EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION,
            records: encoded.records,
        }];
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: crate::publication_digest::streams(
                b"runmat-exact-face-partition-validation/v2\0",
                &streams,
                invocation.control,
                &checkpoint,
            )?,
            streams,
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
    ),
    Box<MeshingFailure>,
> {
    let mut geometry = None;
    let mut curves = None;
    for input in inputs {
        match input {
            PreparedMeshingInput::ExactGeometry(input) if geometry.is_none() => {
                geometry = Some(input.geometry_objects());
            }
            PreparedMeshingInput::StageArtifact(input) if curves.is_none() => {
                curves = Some(input.as_ref());
            }
            PreparedMeshingInput::ExactGeometry(_)
            | PreparedMeshingInput::FacetedGeometry(_)
            | PreparedMeshingInput::Evidence(_)
            | PreparedMeshingInput::StageArtifact(_) => {
                return Err(invalid_input("surface partition input kinds are invalid"));
            }
        }
    }
    Ok((
        geometry.ok_or_else(|| invalid_input("surface partition lacks exact geometry"))?,
        curves.ok_or_else(|| invalid_input("surface partition lacks a global shared curve"))?,
    ))
}

pub(crate) fn curve_record(
    geometry: &PreparedExactGeometryObjects,
    publication: &PreparedMeshingResultPublication,
    cancellation: Option<&dyn runmat_meshing_core::MeshingCancellationSignal>,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let cancellation = cancellation.unwrap_or(&runmat_meshing_core::NeverCancelled);
    let streams = match allocations {
        Some(allocations) => crate::controlled_streams::decode_admitted(
            publication.stage_objects(),
            cancellation,
            allocations,
        ),
        None => crate::controlled_streams::decode(publication.stage_objects(), cancellation),
    }
    .map_err(|error| {
        surface_failure(
            error.category,
            None,
            "regenerate the shared curve publication",
            &error.detail,
        )
    })?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_input(
            "surface curve closure must contain one logical stream",
        ));
    };
    let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
    if publication.stage_objects().result_identity.stage != MeshingStageKind::CurveMesh
        || !matches!(
            publication.stage_objects().result_identity.result_kind,
            MeshingStageResultKind::DeterministicJoin | MeshingStageResultKind::WholeStage
        )
        || publication
            .stage_objects()
            .manifest
            .prerequisite_manifest_digests
            .binary_search(&geometry_digest)
            .is_err()
        || stream.media_type != MeshingChunkMediaType::CurveMesh
        || stream.schema_version != SHARED_CURVE_MESH_SCHEMA_VERSION
        || stream.records.len() != 1
    {
        return Err(invalid_input(
            "surface curve identity, geometry, media, schema, or record count is invalid",
        ));
    }
    Ok(crate::controlled_streams::take_single_record(streams, 0).expect("one admitted record"))
}

pub(crate) fn checked_count(
    lengths: impl Iterator<Item = usize>,
    detail: &str,
) -> Result<u64, Box<MeshingFailure>> {
    crate::accounting::checked_sum_lengths(lengths).ok_or_else(|| {
        surface_failure(
            MeshingFailureCategory::InternalInvariantViolation,
            None,
            "reduce the admitted surface partition size",
            detail,
        )
    })
}
