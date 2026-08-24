use runmat_execution::Digest;
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::{decode_shared_curve_mesh, SHARED_CURVE_MESH_SCHEMA_VERSION};
use runmat_meshing_surface::{
    encode_exact_face_partition_result, mesh_exact_face_partition, ExactFaceJoinOptions,
    ExactFacePartitionContext, ExactFacePartitionOptions, ExactFacePartitionOutcome,
    ExactSurfaceJoinOptions, EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
};

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
}

impl Default for ExactSurfacePartitionKernel<PortableMeshingEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableMeshingEvaluatorProvider,
        }
    }
}

impl<P> ExactSurfacePartitionKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
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
        let (geometry, curve_publication) = inputs(invocation.inputs)?;
        let curve_bytes = curve_record(geometry, curve_publication)?;
        let curves = decode_shared_curve_mesh(&curve_bytes, &geometry.topology)
            .map_err(crate::curve_kernel::map_curve_error)?;
        let evaluator = self
            .evaluator_provider
            .evaluator(geometry)
            .map_err(|error| invalid_input(&error.to_string()))?;
        let control = invocation.control.geometry_evaluation_control();
        let options = surface_options(invocation.host);
        let result = mesh_exact_face_partition(
            invocation.host.workload.partition.clone(),
            ExactFacePartitionContext {
                topology: &geometry.topology,
                curves: &curves,
                metric_request: &invocation.host.resolved_request.metric,
                quality: invocation.host.resolved_request.quality.surface,
                evaluator: evaluator.as_ref(),
                geometry_control: &control,
                cancellation: &control,
            },
            options,
        )
        .map_err(map_surface_error)?;
        let encoded = encode_exact_face_partition_result(&result, &geometry.topology, &curves)
            .map_err(|error| {
                surface_failure(
                    MeshingFailureCategory::InvalidGeometry,
                    error.source_face_id.as_deref().cloned(),
                    "regenerate the exact face partition from its admitted prerequisites",
                    &error.to_string(),
                )
            })?;
        let usage = control.usage();

        let (face_count, node_count, element_count, split_count) = match &result.outcome {
            ExactFacePartitionOutcome::Converged { faces } => (
                checked_count(
                    std::iter::once(faces.len()),
                    "surface face count overflowed",
                )?,
                checked_count(
                    faces.iter().map(|face| face.nodes.len()),
                    "surface node count overflowed",
                )?,
                checked_count(
                    faces.iter().map(|face| face.triangles.len()),
                    "surface triangle count overflowed",
                )?,
                0,
            ),
            ExactFacePartitionOutcome::RequiresCurveSplits { splits } => (
                0,
                0,
                0,
                checked_count(
                    std::iter::once(splits.len()),
                    "curve split demand count overflowed",
                )?,
            ),
        };
        let completed_work = result
            .partition
            .entity_range
            .as_ref()
            .map_or(0, |range| range.entity_count);
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("surface_faces".into(), face_count);
        entity_counts.insert("surface_nodes".into(), node_count);
        entity_counts.insert("surface_triangles".into(), element_count);
        entity_counts.insert("curve_split_demands".into(), split_count);
        let checkpoint = MeshingStageCheckpoint {
            completed_work,
            estimated_work: completed_work,
            node_count,
            element_count,
            peak_memory_bytes: usage
                .allocation_bytes
                .saturating_add(curve_bytes.len() as u64)
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
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
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
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let streams = publication
        .stage_objects()
        .decoded_streams()
        .map_err(|error| invalid_input(&error.to_string()))?;
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
    Ok(stream.records[0].clone())
}

fn surface_options(host: &crate::MeshingHostWorkload) -> ExactFacePartitionOptions {
    let resources = &host.resolved_request.resources;
    let maximum_elements = usize::try_from(resources.maximum_elements).unwrap_or(usize::MAX);
    let maximum_iterations = resources.maximum_iterations.min(u64::from(u32::MAX)) as u32;
    let maximum_depth = resources.maximum_recursion_depth.min(16) as u8;
    ExactFacePartitionOptions {
        charts: runmat_meshing_surface::ExactFaceChartOptions {
            maximum_charts_per_face: resources.maximum_recursion_depth.min(u32::from(u16::MAX))
                as u16,
            ..runmat_meshing_surface::ExactFaceChartOptions::default()
        },
        delaunay: runmat_meshing_surface::ExactFaceDelaunayOptions {
            maximum_triangles: maximum_elements,
            maximum_predicate_evaluations: resources.maximum_search_work,
            maximum_edge_flips: resources.maximum_search_work,
            maximum_cavity_retriangulations: resources.maximum_iterations,
            cancellation_check_interval: host
                .resolved_request
                .cancellation
                .maximum_work_units_between_checks,
        },
        refinement: runmat_meshing_surface::ExactFaceRefinementOptions {
            maximum_interior_insertions: maximum_iterations,
        },
        chart_refinement: runmat_meshing_surface::ExactFaceChartRefinementOptions {
            maximum_chart_cut_splits: maximum_iterations,
        },
        acceptance: runmat_meshing_surface::ExactFaceAcceptanceOptions {
            minimum_subdivision_depth: 1_u8.min(maximum_depth),
            maximum_subdivision_depth: maximum_depth,
            refinement_margin_ratio: 0.5,
            maximum_samples: resources.maximum_search_work,
        },
        face_join: surface_join_options(host),
    }
}

pub(crate) fn surface_join_options(host: &crate::MeshingHostWorkload) -> ExactFaceJoinOptions {
    let resources = &host.resolved_request.resources;
    ExactFaceJoinOptions {
        coordinate_tolerance_m: host.resolved_request.tolerance.absolute_floor_m,
        maximum_nodes: resources.maximum_nodes,
        maximum_triangles: resources.maximum_elements,
        maximum_boundary_segments: resources.maximum_elements,
    }
}

pub(crate) fn exact_surface_join_options(
    host: &crate::MeshingHostWorkload,
) -> ExactSurfaceJoinOptions {
    let resources = &host.resolved_request.resources;
    ExactSurfaceJoinOptions {
        coordinate_tolerance_m: host.resolved_request.tolerance.absolute_floor_m,
        maximum_nodes: resources.maximum_nodes,
        maximum_triangles: resources.maximum_elements,
        maximum_boundary_segments: resources.maximum_elements,
    }
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

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-exact-face-partition-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}
