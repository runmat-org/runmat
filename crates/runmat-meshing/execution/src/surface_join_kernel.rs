use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingChunkMediaType, MeshingChunkStream, MeshingFailure,
    MeshingFailureCategory, MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind,
    StableDigest,
};
use runmat_meshing_curve::decode_shared_curve_mesh_with_cancellation;
use runmat_meshing_surface::{
    decide_exact_surface_pass_owned_admitted, decode_exact_face_partition_records_admitted,
    ExactSurfacePassOutcome, EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
    MAX_EXACT_FACE_PARTITIONS,
};

use crate::surface_kernel::{
    checked_count, curve_record,
    error::{invalid_input, map_convergence_error, surface_failure},
    exact_surface_join_options,
};
use crate::{
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedExactGeometryObjects, PreparedMeshingInput, PreparedMeshingResultPublication,
    ValidatedMeshingStageOutput,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactSurfaceJoinKernel;

impl MeshingStageKernel for ExactSurfaceJoinKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::SurfaceMesh
            || invocation.host.workload.partition.kind != MeshingPartitionKind::DeterministicJoin
        {
            return Err(surface_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "submit the surface barrier with its deterministic-join descriptor",
                "surface join stage shape",
            ));
        }
        let (geometry, curve_publication, partition_publications) = inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let reconstruction_control = invocation.control.geometry_evaluation_control();
        let curve_reconstruction =
            runmat_geometry_core::GeometryAllocationScope::new(&reconstruction_control);
        let (curve_bytes, curves) = {
            let control = invocation.control.geometry_evaluation_control();
            let curve_bytes = curve_record(
                geometry,
                curve_publication,
                Some(&control),
                Some(&curve_reconstruction),
            )?;
            let curves = decode_shared_curve_mesh_with_cancellation(
                &curve_bytes,
                &geometry.topology,
                &control,
            )
            .map_err(crate::curve_kernel::map_curve_error)?;
            (curve_bytes, curves)
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
            let byte_count = checked_count(
                bytes.iter().map(Vec::len),
                "surface partition byte count overflowed",
            )?;
            encoded_input_bytes = encoded_input_bytes
                .checked_add(byte_count)
                .ok_or_else(|| invalid_input("surface partition byte count overflowed"))?;
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
                        return surface_failure(
                            crate::geometry_failure::category(original.kind),
                            None,
                            "honor the original face reconstruction admission",
                            &original.to_string(),
                        );
                    }

                    surface_failure(
                        if error.kind
                            == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled
                        {
                            MeshingFailureCategory::Cancelled
                        } else {
                            MeshingFailureCategory::InvalidGeometry
                        },
                        error.source_face_id.as_deref().cloned(),
                        "regenerate every face partition from the same current shared curve",
                        &error.to_string(),
                    )
                })?
            };
            let retained = partition
                .retained_storage_bytes(&partition_reconstruction)
                .map_err(|error| {
                    surface_failure(
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
        let options = exact_surface_join_options(invocation.host);
        let accepted = {
            let control = invocation.control.geometry_evaluation_control();
            let codec = crate::codec_control::CodecControl::new(&partitions_owner);
            decide_exact_surface_pass_owned_admitted(
                &curves,
                partitions,
                &geometry.topology,
                options,
                &control,
                Some(&codec),
            )
            .map_err(|error| match codec.take_failure() {
                Some(original) => surface_failure(
                    crate::geometry_failure::category(original.kind),
                    None,
                    "honor original surface join storage/work admission",
                    &original.to_string(),
                ),
                None => map_convergence_error(error),
            })?
        };
        // Input roots and merge scratch are gone. Transfer only the live final
        // surface before releasing obsolete partition charges. Restart demands
        // keep their original scope until their encoding completes.
        let surface_storage = match &accepted.result().outcome {
            ExactSurfacePassOutcome::Converged { surface } => {
                let retained =
                    surface
                        .retained_storage_bytes(&partitions_owner)
                        .map_err(|error| {
                            surface_failure(
                                crate::geometry_failure::category(error.kind),
                                None,
                                "inventory the original joined surface",
                                &error.to_string(),
                            )
                        })?;
                Some(
                    partitions_owner
                        .transfer_reservation(retained)
                        .map_err(|error| {
                            surface_failure(
                                crate::geometry_failure::category(error.kind),
                                None,
                                "retain the original joined surface",
                                &error.to_string(),
                            )
                        })?,
                )
            }
            ExactSurfacePassOutcome::RequiresCurveSplits { .. } => None,
        };
        let restart_storage = if surface_storage.is_some() {
            drop(partitions_owner);
            None
        } else {
            Some(partitions_owner)
        };
        // Move the accepted value after its storage owners so all error paths
        // drop the value before releasing its retained allocation reservation.
        let result = accepted;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let (encoded, encoded_surface) = {
            let control = invocation.control.geometry_evaluation_control();
            let codec = crate::codec_control::CodecControl::new(&control);
            let map_error =
                |error: runmat_meshing_surface::ExactSurfaceMeshError| match codec.take_failure() {
                    Some(error) => surface_failure(
                        crate::geometry_failure::category(error.kind),
                        None,
                        "honor admitted global surface encoding",
                        &error.to_string(),
                    ),
                    None => map_surface_error(error),
                };
            let preparation_bytes = result
                .encoding_preparation_storage_bytes(&codec)
                .map_err(map_error)?;
            let _preparation =
                runmat_geometry_core::GeometryAllocationReservation::new(&codec, preparation_bytes)
                    .map_err(|error| {
                        surface_failure(
                            crate::geometry_failure::category(error.kind),
                            None,
                            "admit original global surface dictionaries",
                            &error.to_string(),
                        )
                    })?;
            let prepared = result.prepare_encoding(&codec).map_err(map_error)?;
            let profile = prepared.encoding_profile(&codec).map_err(map_error)?;
            let mut storage = runmat_geometry_core::GeometryAllocationReservation::new(
                &codec,
                profile.resident_workspace_bytes_estimate,
            )
            .map_err(|error| {
                surface_failure(
                    crate::geometry_failure::category(error.kind),
                    None,
                    "admit original global surface encoder workspace",
                    &error.to_string(),
                )
            })?;
            let encoded = prepared
                .encode_profiled(&profile, &codec)
                .map_err(map_error)?;
            if encoded.len() as u64 > profile.encoded_bytes_upper_bound
                || encoded.capacity() as u64 > profile.resident_workspace_bytes_estimate
            {
                return Err(invalid_input(
                    "global surface output exceeds admitted profile",
                ));
            }
            storage.retain_bytes(encoded.capacity() as u64);
            match &result.result().outcome {
                ExactSurfacePassOutcome::Converged { .. } => (None, Some(encoded)),
                ExactSurfacePassOutcome::RequiresCurveSplits { .. } => (Some(encoded), None),
            }
        };

        let face_count = checked_count(
            std::iter::once(geometry.topology.faces.len()),
            "surface face count overflowed",
        )?;
        let (completed_work, node_count, element_count, split_count) =
            match &result.result().outcome {
                ExactSurfacePassOutcome::Converged { surface } => (
                    checked_count(
                        std::iter::once(surface.face_ids.len()),
                        "joined surface face count overflowed",
                    )?,
                    checked_count(
                        std::iter::once(surface.nodes.len()),
                        "joined surface node count overflowed",
                    )?,
                    checked_count(
                        std::iter::once(surface.triangles.len()),
                        "joined surface triangle count overflowed",
                    )?,
                    0,
                ),
                ExactSurfacePassOutcome::RequiresCurveSplits { splits } => (
                    face_count,
                    0,
                    0,
                    checked_count(
                        std::iter::once(splits.len()),
                        "curve split demand count overflowed",
                    )?,
                ),
            };
        drop(result);
        drop(surface_storage);
        drop(restart_storage);
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("surface_faces".into(), completed_work);
        entity_counts.insert("surface_nodes".into(), node_count);
        entity_counts.insert("surface_triangles".into(), element_count);
        entity_counts.insert("curve_split_demands".into(), split_count);
        let encoded_output_bytes = checked_count(
            encoded.iter().chain(encoded_surface.iter()).map(Vec::len),
            "surface publication byte count overflowed",
        )?;
        let checkpoint = MeshingStageCheckpoint {
            completed_work,
            estimated_work: face_count,
            node_count,
            element_count,
            peak_memory_bytes: encoded_input_bytes.saturating_add(encoded_output_bytes),
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        let mut streams = Vec::with_capacity(1);
        if let Some(pass) = encoded {
            streams.push(MeshingChunkStream {
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
                records: vec![pass],
            });
        }
        if let Some(surface) = encoded_surface {
            streams.push(MeshingChunkStream {
                media_type: MeshingChunkMediaType::SurfaceMesh,
                schema_version: EXACT_SURFACE_MESH_SCHEMA_VERSION,
                records: vec![surface],
            });
        }
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: crate::publication_digest::streams(
                b"runmat-exact-surface-publication-validation/v1\0",
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
        Vec<&PreparedMeshingResultPublication>,
    ),
    Box<MeshingFailure>,
> {
    let mut geometry = None;
    let mut curves = None;
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
            PreparedMeshingInput::StageArtifact(input) => partitions.push(input.as_ref()),
            PreparedMeshingInput::ExactGeometry(_)
            | PreparedMeshingInput::FacetedGeometry(_)
            | PreparedMeshingInput::Evidence(_) => {
                return Err(invalid_input("surface join input kinds are invalid"));
            }
        }
    }
    if partitions.is_empty() || partitions.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid_input(
            "surface join requires one through 62 face partition artifacts",
        ));
    }
    Ok((
        geometry.ok_or_else(|| invalid_input("surface join lacks exact geometry"))?,
        curves.ok_or_else(|| invalid_input("surface join lacks a current shared curve"))?,
        partitions,
    ))
}

pub(crate) fn partition_records(
    publication: &PreparedMeshingResultPublication,
    geometry_digest: StableDigest,
    curve_digest: StableDigest,
    cancellation: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<Vec<Vec<u8>>, Box<MeshingFailure>> {
    let streams = crate::controlled_streams::decode_admitted(
        publication.stage_objects(),
        cancellation,
        allocations,
    )
    .map_err(|error| {
        surface_failure(
            error.category,
            None,
            "regenerate the exact surface partition publication",
            &error.detail,
        )
    })?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_input(
            "surface partition closure must contain one logical stream",
        ));
    };
    let prerequisites = &publication
        .stage_objects()
        .manifest
        .prerequisite_manifest_digests;
    if publication.stage_objects().result_identity.stage != MeshingStageKind::SurfaceMesh
        || publication.stage_objects().result_identity.result_kind
            != MeshingStageResultKind::Partition
        || prerequisites.binary_search(&geometry_digest).is_err()
        || prerequisites.binary_search(&curve_digest).is_err()
        || stream.media_type != MeshingChunkMediaType::SurfacePartitions
        || ![
            EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
            runmat_meshing_surface::EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION,
        ]
        .contains(&stream.schema_version)
        || stream.records.is_empty()
    {
        return Err(invalid_input(
            "surface partition identity, prerequisites, media, schema, or record count is invalid",
        ));
    }
    Ok(streams
        .into_iter()
        .next()
        .expect("one admitted stream")
        .records)
}

fn map_surface_error(error: runmat_meshing_surface::ExactSurfaceMeshError) -> Box<MeshingFailure> {
    surface_failure(
        if error.kind == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled {
            MeshingFailureCategory::Cancelled
        } else {
            MeshingFailureCategory::InvalidGeometry
        },
        None,
        "regenerate the exact surface publication from its current prerequisites",
        &error.to_string(),
    )
}
