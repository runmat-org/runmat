use std::collections::BTreeMap;
mod error;
mod storage;
use error::{failure, invalid_inputs, map_solver_error};

use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingChunkMediaType, MeshingChunkStream, MeshingDiagnosticEntry,
    MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory, MeshingPartitionKind,
    MeshingStageKind, MeshingStageResultKind, StableDigest, MESHING_FAILURE_SCHEMA_VERSION,
    SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    decode_published_exact_surface_mesh_admitted, ExactSurfaceJoinOptions, ExactSurfaceMesh,
};
use runmat_meshing_tetrahedron::cdt::{
    decode_delaunay_solver_topology, DelaunayExactEvaluation, DelaunaySolverProjectionError,
    DelaunaySolverProjectionInput, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};

use crate::volume_kernel::volume_options;
use crate::{
    ExactMeshingEvaluatorProvider, MeshingStageCheckpoint, MeshingStageControl,
    MeshingStageInvocation, MeshingStageKernel, PortableMeshingEvaluatorProvider,
    PreparedExactGeometryObjects, PreparedMeshingInput, PreparedMeshingResultPublication,
    ValidatedMeshingStageOutput,
};

#[derive(Clone, Debug)]
pub struct ExactSolverProjectionKernel<P = PortableMeshingEvaluatorProvider> {
    evaluator_provider: P,
}

impl Default for ExactSolverProjectionKernel<PortableMeshingEvaluatorProvider> {
    fn default() -> Self {
        Self {
            evaluator_provider: PortableMeshingEvaluatorProvider,
        }
    }
}

impl<P> ExactSolverProjectionKernel<P> {
    pub const fn new(evaluator_provider: P) -> Self {
        Self { evaluator_provider }
    }
}

impl<P: ExactMeshingEvaluatorProvider> MeshingStageKernel for ExactSolverProjectionKernel<P> {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::OrderElevation
            || invocation.host.workload.partition.kind != MeshingPartitionKind::WholeStage
        {
            return Err(failure(
                MeshingFailureCategory::InternalInvariantViolation,
                "submit solver projection as one whole-stage order projection",
            ));
        }
        let sources = SolverProjectionSources::admit(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let control = invocation.control.geometry_evaluation_control();
        let (surface, surface_storage) = sources.surface(&invocation, &control)?;
        log_phase(&control, "exact surface decoded");
        phase_checkpoint(
            invocation.control,
            "exact surface artifact decoding and validation",
        )?;

        let request = &invocation.host.resolved_request;
        if request.dimension.is_solid() != sources.volume.is_some() {
            return Err(invalid_inputs());
        }
        let retained = runmat_geometry_core::GeometryAllocationScope::new(&control);
        let codec = crate::codec_control::CodecControl::new(&retained);
        let topology = if sources.volume.is_none() {
            // Shell and beam meshes publish the exact surface (or its edges) directly.
            let topology = runmat_meshing_surface::build_surface_solver_topology(
                &surface,
                &sources.geometry.topology,
                request,
                &codec,
            )
            .map_err(|error| {
                failure(MeshingFailureCategory::InvalidGeometry, &error.to_string())
            })?;
            drop(surface);
            drop(surface_storage);
            topology
        } else {
            let evaluator = self
                .evaluator_provider
                .evaluator(sources.geometry, &control)
                .map_err(|error| {
                    failure(MeshingFailureCategory::InvalidGeometry, &error.to_string())
                })?;
            let metric = self
                .evaluator_provider
                .geometry_metric(
                    sources.geometry,
                    evaluator.as_ref(),
                    &control,
                    &invocation.host.resolved_request,
                )
                .map_err(crate::curve_kernel::map_curve_error)?;
            log_phase(&control, "geometry metric prepared");
            let reconstruction = runmat_geometry_core::GeometryAllocationScope::new(&codec);
            let volume_bytes = sources
                .volume_record(&codec, &reconstruction)
                .map_err(|failure| error::map_record_error(failure, &codec))?;
            log_phase(&control, "volume record admitted");
            let topology = decode_delaunay_solver_topology(
                volume_bytes,
                DelaunaySolverProjectionInput {
                    exact_topology: &sources.geometry.topology,
                    exact_surface: &surface,
                    volume_metric: &metric,
                    volume_options: volume_options(&invocation.host.resolved_request),
                    request: &invocation.host.resolved_request,
                    exact_evaluation: Some(DelaunayExactEvaluation {
                        evaluator: evaluator.as_ref(),
                        control: &codec,
                    }),
                },
                solver_options(&invocation.host.resolved_request),
                &codec,
            )
            .map_err(|error| match error {
                DelaunaySolverProjectionError::Control(error) => {
                    failure(crate::geometry_failure::category(error.kind), &error.reason)
                }
                DelaunaySolverProjectionError::Decode(error) => {
                    error::map_volume_codec_error(error, &codec)
                }
                DelaunaySolverProjectionError::Projection(error) => {
                    if let Some(original) = codec.take_failure() {
                        failure(
                            crate::geometry_failure::category(original.kind),
                            &original.reason,
                        )
                    } else {
                        map_solver_error(error)
                    }
                }
            })?;
            drop(reconstruction);
            drop(metric);
            drop(evaluator);
            drop(surface);
            drop(surface_storage);
            topology
        };
        log_phase(&control, "solver topology constructed");
        phase_checkpoint(
            invocation.control,
            "solver topology construction and validation",
        )?;
        let elevation = topology.elevation_report();
        let projection = topology
            .into_projection(invocation.host.stage_identity.geometry.clone())
            .map_err(|error| {
                failure(
                    MeshingFailureCategory::InternalInvariantViolation,
                    &error.to_string(),
                )
            })?;
        if sources.volume.is_none() {
            // The surface topology was built from owned vectors, not through the geometry
            // allocator; charge its retained size to this scope before transferring it.
            let bytes = projection
                .retained_storage_bytes(u64::MAX, &mut || Ok(()))
                .map_err(|error| {
                    failure(
                        MeshingFailureCategory::MemoryBudgetExceeded,
                        &error.to_string(),
                    )
                })?;
            runmat_geometry_core::GeometryEvaluationControl::consume_allocation_bytes(
                &retained, bytes,
            )
            .map_err(|error| {
                failure(crate::geometry_failure::category(error.kind), &error.reason)
            })?;
        }
        let projection = storage::admit(projection, &retained, &codec)?;
        log_phase(&control, "projection retained");
        drop(codec);
        drop(retained);
        let codec = crate::codec_control::CodecControl::new(&control);
        let retained_bytes = projection.retained_bytes();
        let encoding_profile = projection.encoding_profile(&codec).map_err(|error| {
            failure(
                MeshingFailureCategory::InternalInvariantViolation,
                &error.to_string(),
            )
        })?;
        let mut encoding_reservation = runmat_geometry_core::GeometryAllocationReservation::new(
            &control,
            encoding_profile.resident_workspace_bytes_estimate,
        )
        .map_err(|error| failure(crate::geometry_failure::category(error.kind), &error.reason))?;
        log_phase(&control, "encoding workspace reserved");
        let encoded = projection.encode(&codec).map_err(|error| {
            if let Some(original) = codec.take_failure() {
                failure(
                    crate::geometry_failure::category(original.kind),
                    &original.reason,
                )
            } else {
                failure(
                    MeshingFailureCategory::InternalInvariantViolation,
                    &error.to_string(),
                )
            }
        })?;
        phase_checkpoint(invocation.control, "solver projection canonical encoding")?;
        let node_count = projection.topology.nodes.len() as u64;
        let element_count = (match projection.resolved_request.dimension {
            runmat_meshing_core::MeshDimension::Solid => projection.topology.volume_elements.len(),
            runmat_meshing_core::MeshDimension::Surface => projection.topology.boundary_faces.len(),
            runmat_meshing_core::MeshDimension::Curve => projection.topology.boundary_edges.len(),
        }) as u64;
        let completed_work = node_count.checked_add(element_count).ok_or_else(|| {
            failure(
                MeshingFailureCategory::ElementBudgetExceeded,
                "solver projection work inventory overflowed",
            )
        })?;
        let checkpoint = MeshingStageCheckpoint {
            completed_work,
            estimated_work: completed_work,
            node_count,
            element_count,
            // Requested workspace and wire bytes are not measured whole-process residency.
            peak_memory_bytes: 0,
            entity_counts: BTreeMap::from([
                ("construction_memory_measurement_unavailable".into(), 1),
                ("projection_retained_requested_bytes".into(), retained_bytes),
                ("encoded_projection_bytes".into(), encoded.len() as u64),
                (
                    "encoding_workspace_requested_bytes".into(),
                    encoding_profile.resident_workspace_bytes_estimate,
                ),
                (
                    "admitted_geometry_storage_peak_bytes".into(),
                    control.usage().allocation_bytes,
                ),
                ("solver_nodes".into(), node_count),
                (
                    "tet10_straightened_elements".into(),
                    elevation.straightened_elements,
                ),
                (
                    "tet10_straightened_edges".into(),
                    elevation.straightened_edges,
                ),
                ("solver_volume_elements".into(), element_count),
                (
                    "solver_boundary_faces".into(),
                    projection.topology.boundary_faces.len() as u64,
                ),
                (
                    "solver_boundary_edges".into(),
                    projection.topology.boundary_edges.len() as u64,
                ),
            ]),
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;
        drop(projection);
        encoding_reservation.retain_bytes(encoded.capacity() as u64);
        drop(encoding_reservation);
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: crate::publication_digest::record(
                b"runmat-solver-projection-validation/v1\0",
                &encoded,
                invocation.control,
                &checkpoint,
            )?,
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SolverMeshProjection,
                schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

/// Live and peak admitted geometry storage at each order-elevation phase.
fn log_phase(control: &crate::MeshingGeometryEvaluationControl<'_>, phase: &'static str) {
    let usage = control.usage();
    tracing::info!(
        event_code = "meshing.order_elevation.phase",
        phase,
        live_bytes = usage.live_allocation_bytes,
        peak_bytes = usage.allocation_bytes,
        "order elevation phase"
    );
}

fn phase_checkpoint(
    control: &mut MeshingStageControl<'_>,
    phase: &'static str,
) -> Result<(), Box<MeshingFailure>> {
    control
        .checkpoint(MeshingStageCheckpoint::default())
        .map_err(|mut failure| {
            failure.remediation = format!("{}; phase: {phase}", failure.remediation);
            failure
        })
}

struct SolverProjectionSources<'a> {
    geometry: &'a PreparedExactGeometryObjects,
    surface: &'a PreparedMeshingResultPublication,
    /// Absent for surface and curve requests, which project the exact surface directly.
    volume: Option<&'a PreparedMeshingResultPublication>,
}

impl<'a> SolverProjectionSources<'a> {
    fn admit(inputs: &'a [PreparedMeshingInput]) -> Result<Self, Box<MeshingFailure>> {
        let mut geometry = None;
        let mut surface = None;
        let mut volume = None;
        for input in inputs {
            match input {
                PreparedMeshingInput::ExactGeometry(input) if geometry.is_none() => {
                    geometry = Some(input.geometry_objects());
                }
                PreparedMeshingInput::StageArtifact(input)
                    if input.stage_objects().result_identity.stage
                        == MeshingStageKind::SurfaceMesh
                        && surface.is_none() =>
                {
                    surface = Some(input.as_ref());
                }
                PreparedMeshingInput::StageArtifact(input)
                    if input.stage_objects().result_identity.stage
                        == MeshingStageKind::Tetrahedralization
                        && volume.is_none() =>
                {
                    volume = Some(input.as_ref());
                }
                _ => return Err(invalid_inputs()),
            }
        }
        Ok(Self {
            geometry: geometry.ok_or_else(invalid_inputs)?,
            surface: surface.ok_or_else(invalid_inputs)?,
            volume,
        })
        .and_then(Self::validate_source_closure)
    }

    fn validate_source_closure(self) -> Result<Self, Box<MeshingFailure>> {
        let geometry_digest = StableDigest::from_bytes(*self.geometry.root.digest.bytes());
        let surface_digest =
            StableDigest::from_bytes(*self.surface.stage_objects().root.digest.bytes());
        let surface_prerequisites = &self
            .surface
            .stage_objects()
            .manifest
            .prerequisite_manifest_digests;
        let volume_closes = self.volume.is_none_or(|volume| {
            let volume_prerequisites = &volume
                .stage_objects()
                .manifest
                .prerequisite_manifest_digests;
            volume_prerequisites.binary_search(&geometry_digest).is_ok()
                && volume_prerequisites.binary_search(&surface_digest).is_ok()
        });
        if surface_prerequisites
            .binary_search(&geometry_digest)
            .is_err()
            || !volume_closes
        {
            return Err(invalid_inputs());
        }
        Ok(self)
    }

    fn surface<'storage>(
        &self,
        invocation: &MeshingStageInvocation<'_, '_>,
        allocations: &'storage dyn runmat_geometry_core::GeometryEvaluationControl,
    ) -> Result<
        (
            ExactSurfaceMesh,
            runmat_geometry_core::GeometryAllocationReservation<'storage>,
        ),
        Box<MeshingFailure>,
    > {
        let reconstruction = runmat_geometry_core::GeometryAllocationScope::new(allocations);
        let codec = crate::codec_control::CodecControl::new(&reconstruction);
        let bytes = crate::volume_kernel::surface_record(
            self.geometry,
            self.surface,
            &codec,
            &reconstruction,
        )
        .map_err(|failure| error::map_record_error(failure, &codec))?;
        let surface = decode_published_exact_surface_mesh_admitted(
            &bytes,
            &self.geometry.topology,
            ExactSurfaceJoinOptions {
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
            },
            &codec,
            &codec,
        )
        .map_err(|error| {
            if let Some(original) = codec.take_failure() {
                return failure(
                    crate::geometry_failure::category(original.kind),
                    &original.reason,
                );
            }
            failure(
                if error.kind == runmat_meshing_surface::ExactSurfaceMeshErrorKind::Cancelled {
                    MeshingFailureCategory::Cancelled
                } else {
                    MeshingFailureCategory::InvalidGeometry
                },
                &error.to_string(),
            )
        })?;
        let retained = surface
            .retained_storage_bytes(&reconstruction)
            .map_err(|error| {
                failure(
                    crate::geometry_failure::category(error.kind),
                    &error.to_string(),
                )
            })?;
        let storage = reconstruction
            .transfer_reservation(retained)
            .map_err(|error| {
                failure(
                    crate::geometry_failure::category(error.kind),
                    &error.to_string(),
                )
            })?;
        Ok((surface, storage))
    }

    fn volume_record(
        &self,
        control: &dyn MeshingCancellationSignal,
        allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
    ) -> Result<Vec<u8>, Box<MeshingFailure>> {
        let stage = self.volume.ok_or_else(invalid_inputs)?.stage_objects();
        if stage.result_identity.stage != MeshingStageKind::Tetrahedralization
            || stage.result_identity.result_kind != MeshingStageResultKind::WholeStage
        {
            return Err(invalid_inputs());
        }
        let streams = crate::controlled_streams::decode_admitted(stage, control, allocations)
            .map_err(|error| failure(error.category, &error.detail))?;
        let [stream] = streams.as_slice() else {
            return Err(invalid_inputs());
        };
        if stream.media_type != MeshingChunkMediaType::VolumeTopology
            || !matches!(
                stream.schema_version,
                3 | DELAUNAY_VOLUME_MESH_SCHEMA_VERSION
            )
        {
            return Err(invalid_inputs());
        }
        let [_record] = stream.records.as_slice() else {
            return Err(invalid_inputs());
        };
        Ok(crate::controlled_streams::take_single_record(streams, 0).expect("one admitted record"))
    }
}

/// Algorithm bounds for curved-node optimization. They are properties of the algorithm, not of
/// the machine: the same request produces the same mesh under any execution ceiling.
const CURVED_OPTIMIZATION_CANDIDATES: u64 = 1_000_000_000;
const CURVED_OPTIMIZATION_ROUNDS: u32 = 1_000_000_000;

fn solver_options(request: &runmat_meshing_core::MeshingRequest) -> DelaunaySolverTopologyOptions {
    let limits = request.limits;
    DelaunaySolverTopologyOptions {
        maximum_boundary_faces: limits.maximum_elements.saturating_mul(4),
        maximum_boundary_edges: limits.maximum_elements.saturating_mul(6),
        maximum_curved_optimization_candidates: CURVED_OPTIMIZATION_CANDIDATES,
        maximum_curved_optimization_rounds: CURVED_OPTIMIZATION_ROUNDS,
        maximum_curved_optimization_sweeps: DelaunaySolverTopologyOptions::default()
            .maximum_curved_optimization_sweeps,
        trim_boundary_tolerance_uv: 1.0e-10,
        cancellation_check_interval: request
            .cancellation
            .maximum_work_units_between_checks
            .max(1)
            .min(DelaunaySolverTopologyOptions::default().cancellation_check_interval),
    }
}

#[cfg(test)]
mod tests;
