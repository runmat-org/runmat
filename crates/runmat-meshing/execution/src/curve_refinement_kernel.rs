use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::{
    apply_shared_curve_splits, decode_shared_curve_mesh_with_cancellation,
    encode_shared_curve_mesh_admitted, SharedCurveEvaluationContext,
    SHARED_CURVE_MESH_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    decode_exact_face_partition_records_admitted,
    decode_exact_surface_pass_result_with_cancellation, ExactSurfacePassOutcome,
    EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};

use crate::curve_kernel::{curve_failure, curve_options, map_curve_error, resolved_curve_metric};
use crate::surface_join_kernel::partition_records;
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
        let reconstruction_control = invocation.control.geometry_evaluation_control();
        let curve_reconstruction =
            runmat_geometry_core::GeometryAllocationScope::new(&reconstruction_control);
        let (curve_bytes, curves) = {
            let control = invocation.control.geometry_evaluation_control();
            let bytes = curve_record(
                geometry,
                curve_publication,
                Some(&control),
                Some(&curve_reconstruction),
            )?;
            let curves =
                decode_shared_curve_mesh_with_cancellation(&bytes, &geometry.topology, &control)
                    .map_err(map_curve_error)?;
            (bytes, curves)
        };
        let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
        let curve_digest =
            StableDigest::from_bytes(*curve_publication.stage_objects().root.digest.bytes());
        let mut encoded_input_bytes = checked_count(
            std::iter::once(curve_bytes.len()),
            "shared curve byte count overflowed",
        )?;
        drop(curve_bytes);
        drop(curve_reconstruction);
        let partitions_owner =
            runmat_geometry_core::GeometryAllocationScope::new(&reconstruction_control);
        let mut partitions = Vec::with_capacity(partition_publications.len());
        let mut pass_prerequisites = vec![geometry_digest, curve_digest];
        for publication in partition_publications {
            let partition_reconstruction =
                runmat_geometry_core::GeometryAllocationScope::new(&partitions_owner);
            let bytes = {
                let control = invocation.control.geometry_evaluation_control();
                partition_records(
                    publication,
                    geometry_digest,
                    curve_digest,
                    &control,
                    &partition_reconstruction,
                )?
            };
            encoded_input_bytes = add_bytes(encoded_input_bytes, bytes.iter().map(Vec::len).sum())?;
            let partition = {
                let codec = crate::codec_control::CodecControl::new(&partition_reconstruction);
                decode_exact_face_partition_records_admitted(
                    &bytes,
                    &geometry.topology,
                    &curves,
                    &codec,
                    &codec,
                )
                .map_err(|error| {
                    if let Some(original) = codec.take_failure() {
                        return curve_failure(
                            crate::geometry_failure::category(original.kind),
                            None,
                            "honor the original face reconstruction admission",
                            &original.to_string(),
                        );
                    }

                    if error.kind == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled {
                        curve_failure(
                            MeshingFailureCategory::Cancelled,
                            None,
                            "cancelled surface partition read",
                            &error.to_string(),
                        )
                    } else {
                        invalid_input(&error.to_string())
                    }
                })?
            };
            pass_prerequisites.push(StableDigest::from_bytes(
                *publication.stage_objects().root.digest.bytes(),
            ));
            let retained = partition
                .retained_storage_bytes(&partition_reconstruction)
                .map_err(|error| {
                    curve_failure(
                        crate::geometry_failure::category(error.kind),
                        None,
                        "admit the complete retained face partition",
                        &error.to_string(),
                    )
                })?;
            partition_reconstruction.retain_bytes(retained);
            partitions.push(partition);
            drop(bytes);
            drop(partition_reconstruction);
            invocation
                .control
                .checkpoint(MeshingStageCheckpoint::default())?;
        }
        pass_prerequisites.sort();
        let pass_reconstruction =
            runmat_geometry_core::GeometryAllocationScope::new(&reconstruction_control);
        let pass_bytes = {
            let control = invocation.control.geometry_evaluation_control();
            pass_record(
                pass_publication,
                &pass_prerequisites,
                &control,
                &pass_reconstruction,
            )?
        };
        encoded_input_bytes = add_bytes(encoded_input_bytes, pass_bytes.len())?;
        let pass = {
            let control = invocation.control.geometry_evaluation_control();
            decode_exact_surface_pass_result_with_cancellation(
                &pass_bytes,
                &geometry.topology,
                &curves,
                &partitions,
                exact_surface_join_options(invocation.host),
                &control,
            )
            .map_err(map_surface_error)?
        };
        drop(pass_bytes);
        drop(pass_reconstruction);
        let ExactSurfacePassOutcome::RequiresCurveSplits { splits } = pass.outcome else {
            return Err(invalid_input(
                "a converged surface pass cannot enter curve refinement",
            ));
        };

        let control = invocation.control.geometry_evaluation_control();
        let evaluator = self
            .evaluator_provider
            .evaluator(geometry, &control)
            .map_err(|error| invalid_input(&error.to_string()))?;
        let metric = resolved_curve_metric(
            &self.evaluator_provider,
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
            encode_shared_curve_mesh_admitted(&refined, &geometry.topology, &control, &control)
                .map_err(map_curve_error)?;
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
            invariant_summary_digest: crate::publication_digest::record(
                b"runmat-refined-global-curve-validation/v1\0",
                &encoded,
                invocation.control,
                &checkpoint,
            )?,
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
    cancellation: &dyn runmat_meshing_core::MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let streams = crate::controlled_streams::decode_admitted(
        publication.stage_objects(),
        cancellation,
        allocations,
    )
    .map_err(|error| {
        crate::surface_kernel::error::surface_failure(
            error.category,
            None,
            "regenerate the exact surface pass publication",
            &error.detail,
        )
    })?;
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
    Ok(crate::controlled_streams::take_single_record(streams, 0).expect("one admitted record"))
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

fn map_surface_error(error: runmat_meshing_surface::ExactSurfaceMeshError) -> Box<MeshingFailure> {
    if error.kind == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled {
        crate::surface_kernel::error::surface_failure(
            MeshingFailureCategory::Cancelled,
            None,
            "retry the cancelled surface pass admission",
            &error.to_string(),
        )
    } else {
        invalid_input(&error.to_string())
    }
}
