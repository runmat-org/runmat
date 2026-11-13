use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingChunkMediaType, MeshingChunkStream, MeshingFailure,
    MeshingFailureCategory, MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind,
    StableDigest,
};
use runmat_meshing_surface::{
    decode_published_exact_surface_mesh_admitted, ExactSurfaceJoinOptions,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
};
use runmat_meshing_tetrahedron::cdt::{
    construct_delaunay_volume_mesh_with_quality_observer, DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};

mod diagnostic;
mod error;
mod options;
#[cfg(test)]
mod tests;

use error::{invalid_input, map_codec_error, map_volume_error, volume_failure};
pub(crate) use options::volume_options;

use crate::{
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedExactGeometryObjects, PreparedMeshingInput, PreparedMeshingResultPublication,
    ValidatedMeshingStageOutput,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactVolumeKernel;

impl MeshingStageKernel for ExactVolumeKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        self.execute_with_provider(invocation, &crate::PortableMeshingEvaluatorProvider)
    }
}

impl ExactVolumeKernel {
    pub(crate) fn execute_with_provider(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
        provider: &impl crate::ExactMeshingEvaluatorProvider,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        self.execute_with_diagnostics(invocation, provider, None)
    }

    pub(crate) fn execute_with_diagnostics(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
        provider: &impl crate::ExactMeshingEvaluatorProvider,
        cache: Option<&dyn crate::MeshingSubtaskCacheStore>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::Tetrahedralization
            || invocation.host.workload.partition.kind != MeshingPartitionKind::WholeStage
        {
            return Err(volume_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                "submit general CDT construction as one whole connected volume stage",
                "tetrahedralization stage shape",
            ));
        }
        let (geometry, surface_publication) = inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let cancellation = invocation.control.geometry_evaluation_control();
        let reconstruction = runmat_geometry_core::GeometryAllocationScope::new(&cancellation);
        let surface_bytes = surface_record(
            geometry,
            surface_publication,
            &cancellation,
            &reconstruction,
        )?;
        let surface_options = ExactSurfaceJoinOptions {
            coordinate_tolerance_m: runmat_types::physical::Length::new(
                invocation
                    .host
                    .resolved_request
                    .tolerance
                    .absolute_floor_m
                    .canonical_value(),
            ),
            maximum_nodes: invocation.host.resolved_request.limits.maximum_nodes,
            maximum_triangles: invocation.host.resolved_request.limits.maximum_elements,
            maximum_boundary_segments: invocation.host.resolved_request.limits.maximum_elements,
        };
        let surface_codec = crate::codec_control::CodecControl::new(&reconstruction);
        let surface = decode_published_exact_surface_mesh_admitted(
            &surface_bytes,
            &geometry.topology,
            surface_options,
            &surface_codec,
            &surface_codec,
        )
        .map_err(|error| {
            if let Some(original) = surface_codec.take_failure() {
                return volume_failure(
                    crate::geometry_failure::category(original.kind),
                    "honor original surface reconstruction admission",
                    &original.to_string(),
                );
            }

            volume_failure(
                if error.kind == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled {
                    MeshingFailureCategory::Cancelled
                } else {
                    MeshingFailureCategory::InvalidGeometry
                },
                "regenerate the final exact surface",
                &error.to_string(),
            )
        })?;
        let retained_surface =
            surface
                .retained_storage_bytes(&reconstruction)
                .map_err(|error| {
                    volume_failure(
                        crate::geometry_failure::category(error.kind),
                        "retain the original exact surface",
                        &error.to_string(),
                    )
                })?;
        let surface_storage = reconstruction
            .transfer_reservation(retained_surface)
            .map_err(|error| {
                volume_failure(
                    crate::geometry_failure::category(error.kind),
                    "transfer the original surface storage owner",
                    &error.to_string(),
                )
            })?;
        drop(surface_codec);
        let surface_encoded_bytes = surface_bytes.len() as u64;
        drop(surface_bytes);
        drop(reconstruction);
        let evaluator = provider
            .evaluator(geometry, &cancellation)
            .map_err(|error| invalid_input(&error.to_string()))?;
        let metric = provider
            .geometry_metric(
                geometry,
                evaluator.as_ref(),
                &cancellation,
                &invocation.host.resolved_request,
            )
            .map_err(crate::curve_kernel::map_curve_error)?;
        drop(evaluator);
        let options = volume_options(&invocation.host.resolved_request);
        tracing::info!(
            surface_encoded_bytes,
            "volume construction started; encoded input released"
        );
        let construction_started = runmat_time::Instant::now();
        let observer = diagnostic::Observer {
            store: cache,
            binding: runmat_meshing_tetrahedron::cdt::VolumeQualityBinding {
                request: invocation.host.stage_identity.resolved_request_digest,
                geometry: StableDigest::from_bytes(*geometry.root.digest.bytes()),
                implementation: StableDigest::ZERO,
            },
            control: &cancellation,
            failure: std::sync::Mutex::new(None),
        };
        let mesh = construct_delaunay_volume_mesh_with_quality_observer(
            &geometry.topology,
            &surface,
            &metric,
            options,
            &cancellation,
            &observer,
        )
        .map_err(|error| {
            match observer
                .failure
                .lock()
                .expect("diagnostic failure lock")
                .take()
            {
                Some(failure) => error::map_diagnostic_error(failure),
                None => map_volume_error(error),
            }
        })?;
        drop(surface);
        drop(surface_storage);
        tracing::info!(
            elapsed_seconds = construction_started.elapsed().as_secs_f64(),
            node_count = mesh.topology.nodes.len(),
            element_count = mesh.topology.tetrahedra.len(),
            "volume construction independently validated; publication encoding started"
        );
        let encoding_started = runmat_time::Instant::now();
        let codec = crate::codec_control::CodecControl::new(&cancellation);
        let codec_failure = |error| match codec.take_failure() {
            Some(failure) => {
                error::map_diagnostic_error(crate::MeshingSubtaskCacheError::Control(failure))
            }
            None => map_codec_error(error),
        };
        let preparation_bytes = mesh
            .encoding_preparation_storage_bytes()
            .map_err(codec_failure)?;
        let preparation_storage = runmat_geometry_core::GeometryAllocationReservation::new(
            &cancellation,
            preparation_bytes,
        )
        .map_err(|error| {
            error::map_diagnostic_error(crate::MeshingSubtaskCacheError::Control(error))
        })?;
        let prepared_encoding = mesh.prepare_encoding(&codec).map_err(codec_failure)?;
        let profile = prepared_encoding
            .encoding_profile(&codec)
            .map_err(codec_failure)?;
        let mut encoding_storage = runmat_geometry_core::GeometryAllocationReservation::new(
            &cancellation,
            profile.resident_workspace_bytes_estimate,
        )
        .map_err(|error| {
            error::map_diagnostic_error(crate::MeshingSubtaskCacheError::Control(error))
        })?;
        let encoded = prepared_encoding.encode(&codec).map_err(codec_failure)?;
        drop(prepared_encoding);
        drop(preparation_storage);

        let node_count = mesh.topology.nodes.len() as u64;
        let element_count = mesh.topology.tetrahedra.len() as u64;
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("volume_nodes".into(), node_count);
        entity_counts.insert("volume_tetrahedra".into(), element_count);
        entity_counts.insert(
            "volume_boundary_facets".into(),
            mesh.provenance.facets.len() as u64,
        );
        let mutation_count = mesh.quality_mutations.len() as u64;
        entity_counts.insert("volume_mutations".into(), mutation_count);
        entity_counts.insert("encoded_surface_bytes".into(), surface_encoded_bytes);
        entity_counts.insert("encoded_volume_bytes".into(), encoded.len() as u64);
        entity_counts.insert("construction_memory_measurement_unavailable".into(), 1);
        entity_counts.insert(
            "encoding_preparation_requested_bytes".into(),
            preparation_bytes,
        );
        entity_counts.insert(
            "encoding_workspace_requested_bytes".into(),
            profile.resident_workspace_bytes_estimate,
        );
        entity_counts.insert(
            "admitted_geometry_storage_peak_bytes".into(),
            cancellation.usage().allocation_bytes,
        );
        // Numeric construction and independent encoding validation have finished.
        // Publication hashing/chunking retain only encoded output and counters.
        drop(mesh);
        // Retain exactly the owned wire bytes through publication, releasing
        // temporary encoding workspace before hashing and chunk transport.
        encoding_storage.retain_bytes(encoded.capacity() as u64);
        drop(encoding_storage);
        tracing::info!(
            elapsed_seconds = encoding_started.elapsed().as_secs_f64(),
            encoded_volume_bytes = encoded.len(),
            "volume encoding finished; numeric inventories released before publication"
        );
        let checkpoint = MeshingStageCheckpoint {
            completed_work: element_count,
            estimated_work: element_count,
            node_count,
            element_count,
            // Encoded lengths are not a construction resident-memory high water
            // mark. The native process guard reports physical memory separately.
            peak_memory_bytes: 0,
            iterations: mutation_count,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: crate::publication_digest::record(
                b"runmat-general-cdt-volume-publication-validation/v1\0",
                &encoded,
                invocation.control,
                &checkpoint,
            )?,
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::VolumeTopology,
                schema_version: DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
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
    match inputs {
        [PreparedMeshingInput::ExactGeometry(geometry), PreparedMeshingInput::StageArtifact(surface)]
        | [PreparedMeshingInput::StageArtifact(surface), PreparedMeshingInput::ExactGeometry(geometry)] => {
            Ok((geometry.geometry_objects(), surface))
        }
        _ => Err(invalid_input(
            "tetrahedralization requires exactly one exact geometry and one final surface artifact",
        )),
    }
}

pub(crate) fn surface_record(
    geometry: &PreparedExactGeometryObjects,
    publication: &PreparedMeshingResultPublication,
    cancellation: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let stage = publication.stage_objects();
    let streams = crate::controlled_streams::decode_admitted(
        publication.stage_objects(),
        cancellation,
        allocations,
    )
    .map_err(|error| {
        volume_failure(
            error.category,
            "regenerate the exact surface publication",
            &error.detail,
        )
    })?;
    let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
    if stage.result_identity.stage != MeshingStageKind::SurfaceMesh
        || stage.result_identity.result_kind != MeshingStageResultKind::DeterministicJoin
        || stage
            .manifest
            .prerequisite_manifest_digests
            .binary_search(&geometry_digest)
            .is_err()
    {
        return Err(invalid_input(
            "final surface identity or exact-geometry prerequisite is invalid",
        ));
    }
    let (index, stream) = match streams.as_slice() {
        [surface] => (0, surface),
        // Legacy publications included the complete pass as well as the surface.
        [pass, surface]
            if pass.media_type == MeshingChunkMediaType::SurfacePartitions
                && pass.schema_version == EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION
                && pass.records.len() == 1 =>
        {
            (1, surface)
        }
        _ => {
            return Err(invalid_input(
                "final surface publication has an invalid stream shape",
            ))
        }
    };
    if stream.media_type != MeshingChunkMediaType::SurfaceMesh
        || stream.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
        || stream.records.len() != 1
    {
        return Err(invalid_input(
            "final surface publication stream media, schema, or record count is invalid",
        ));
    }
    Ok(crate::controlled_streams::take_single_record(streams, index).expect("one admitted record"))
}
