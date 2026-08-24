use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, MeshingChunkMediaType, MeshingChunkStream, MeshingDiagnosticEntry,
    MeshingDiagnosticValue, MeshingFailure, MeshingFailureCategory, MeshingPartitionKind,
    MeshingStageKind, MeshingStageResultKind, SolverMeshProjection, StableDigest,
    MESHING_FAILURE_SCHEMA_VERSION, SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
};
use runmat_meshing_surface::{
    decode_published_exact_surface_mesh, ExactSurfaceJoinOptions, ExactSurfaceMesh,
};
use runmat_meshing_tetrahedron::cdt::{
    build_delaunay_solver_topology, decode_delaunay_volume_mesh, DelaunayExactEvaluation,
    DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, DelaunaySolverTopologyInput,
    DelaunaySolverTopologyOptions, DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};

use crate::volume_kernel::volume_options;
use crate::{
    ExactMeshingEvaluatorProvider, MeshingStageCheckpoint, MeshingStageInvocation,
    MeshingStageKernel, PortableMeshingEvaluatorProvider, PreparedExactGeometryObjects,
    PreparedMeshingInput, PreparedMeshingResultPublication, ValidatedMeshingStageOutput,
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
        let surface = sources.surface(&invocation)?;
        let control = invocation.control.geometry_evaluation_control();
        let volume = decode_delaunay_volume_mesh(
            &sources.volume_record()?,
            &sources.geometry.topology,
            &surface,
            &invocation.host.resolved_request.metric,
            volume_options(&invocation.host.resolved_request),
            &control,
        )
        .map_err(|error| failure(MeshingFailureCategory::InvalidGeometry, &error.to_string()))?;
        let evaluator = self
            .evaluator_provider
            .evaluator(sources.geometry)
            .map_err(|error| {
                failure(MeshingFailureCategory::InvalidGeometry, &error.to_string())
            })?;
        let topology = build_delaunay_solver_topology(
            DelaunaySolverTopologyInput {
                exact_topology: &sources.geometry.topology,
                exact_surface: &surface,
                volume_mesh: &volume,
                volume_options: volume_options(&invocation.host.resolved_request),
                request: &invocation.host.resolved_request,
                exact_evaluation: Some(DelaunayExactEvaluation {
                    evaluator: evaluator.as_ref(),
                    control: &control,
                }),
            },
            solver_options(&invocation.host.resolved_request),
            &control,
        )
        .map_err(map_solver_error)?;
        let projection = SolverMeshProjection {
            schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
            geometry: invocation.host.stage_identity.geometry.clone(),
            resolved_request: invocation.host.resolved_request.clone(),
            topology,
        };
        let encoded = projection.canonical_encode().map_err(|error| {
            failure(
                MeshingFailureCategory::InternalInvariantViolation,
                &error.to_string(),
            )
        })?;
        let node_count = projection.topology.nodes.len() as u64;
        let element_count = projection.topology.volume_elements.len() as u64;
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
            peak_memory_bytes: encoded.len() as u64,
            entity_counts: BTreeMap::from([
                ("solver_nodes".into(), node_count),
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
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation_digest(&encoded),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SolverMeshProjection,
                schema_version: SOLVER_MESH_PROJECTION_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

struct SolverProjectionSources<'a> {
    geometry: &'a PreparedExactGeometryObjects,
    surface: &'a PreparedMeshingResultPublication,
    volume: &'a PreparedMeshingResultPublication,
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
            volume: volume.ok_or_else(invalid_inputs)?,
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
        let volume_prerequisites = &self
            .volume
            .stage_objects()
            .manifest
            .prerequisite_manifest_digests;
        if surface_prerequisites
            .binary_search(&geometry_digest)
            .is_err()
            || volume_prerequisites
                .binary_search(&geometry_digest)
                .is_err()
            || volume_prerequisites.binary_search(&surface_digest).is_err()
        {
            return Err(invalid_inputs());
        }
        Ok(self)
    }

    fn surface(
        &self,
        invocation: &MeshingStageInvocation<'_, '_>,
    ) -> Result<ExactSurfaceMesh, Box<MeshingFailure>> {
        let bytes = crate::volume_kernel::surface_record(self.geometry, self.surface)?;
        decode_published_exact_surface_mesh(
            &bytes,
            &self.geometry.topology,
            ExactSurfaceJoinOptions {
                coordinate_tolerance_m: invocation.host.resolved_request.tolerance.absolute_floor_m,
                maximum_nodes: invocation.host.resolved_request.resources.maximum_nodes,
                maximum_triangles: invocation.host.resolved_request.resources.maximum_elements,
                maximum_boundary_segments: invocation
                    .host
                    .resolved_request
                    .resources
                    .maximum_elements,
            },
        )
        .map_err(|error| failure(MeshingFailureCategory::InvalidGeometry, &error.to_string()))
    }

    fn volume_record(&self) -> Result<Vec<u8>, Box<MeshingFailure>> {
        let stage = self.volume.stage_objects();
        if stage.result_identity.stage != MeshingStageKind::Tetrahedralization
            || stage.result_identity.result_kind != MeshingStageResultKind::WholeStage
        {
            return Err(invalid_inputs());
        }
        let streams = stage.decoded_streams().map_err(|error| {
            failure(MeshingFailureCategory::InvalidGeometry, &error.to_string())
        })?;
        let [stream] = streams.as_slice() else {
            return Err(invalid_inputs());
        };
        if stream.media_type != MeshingChunkMediaType::VolumeTopology
            || stream.schema_version != DELAUNAY_VOLUME_MESH_SCHEMA_VERSION
        {
            return Err(invalid_inputs());
        }
        let [record] = stream.records.as_slice() else {
            return Err(invalid_inputs());
        };
        Ok(record.clone())
    }
}

fn solver_options(request: &runmat_meshing_core::MeshingRequest) -> DelaunaySolverTopologyOptions {
    let resources = request.resources;
    DelaunaySolverTopologyOptions {
        maximum_boundary_faces: resources.maximum_elements.saturating_mul(4),
        maximum_boundary_edges: resources.maximum_elements.saturating_mul(6),
        maximum_curved_optimization_candidates: resources.maximum_search_work,
        maximum_curved_optimization_rounds: resources.maximum_iterations.min(u64::from(u32::MAX))
            as u32,
        trim_boundary_tolerance_uv: 1.0e-10,
        cancellation_check_interval: request
            .cancellation
            .maximum_work_units_between_checks
            .max(1),
    }
}

fn map_solver_error(error: DelaunaySolverTopologyError) -> Box<MeshingFailure> {
    let category = match error.kind {
        DelaunaySolverTopologyErrorKind::InvalidOptions => {
            MeshingFailureCategory::InternalInvariantViolation
        }
        DelaunaySolverTopologyErrorKind::InvalidGeometry => MeshingFailureCategory::InvalidGeometry,
        DelaunaySolverTopologyErrorKind::InvalidMesh => {
            MeshingFailureCategory::QualityTargetUnreachable
        }
        DelaunaySolverTopologyErrorKind::ResourceLimit => resource_category(&error.reason),
        DelaunaySolverTopologyErrorKind::Cancelled => MeshingFailureCategory::Cancelled,
    };
    failure(category, &error.to_string())
}

fn resource_category(reason: &str) -> MeshingFailureCategory {
    if reason.contains("IterationBudgetExceeded") || reason.contains("round limit") {
        MeshingFailureCategory::IterationBudgetExceeded
    } else if reason.contains("TimeBudgetExceeded") {
        MeshingFailureCategory::TimeBudgetExceeded
    } else if reason.contains("AllocationBudgetExceeded") {
        MeshingFailureCategory::MemoryBudgetExceeded
    } else if reason.contains("node inventory") {
        MeshingFailureCategory::NodeBudgetExceeded
    } else if reason.contains("boundary face") || reason.contains("boundary edge") {
        MeshingFailureCategory::ElementBudgetExceeded
    } else {
        MeshingFailureCategory::SearchWorkBudgetExceeded
    }
}

fn invalid_inputs() -> Box<MeshingFailure> {
    failure(
        MeshingFailureCategory::InvalidGeometry,
        "solver projection requires one exact geometry, final surface, and final volume",
    )
}

fn failure(category: MeshingFailureCategory, detail: &str) -> Box<MeshingFailure> {
    let detail = crate::diagnostic::bounded_diagnostic_text(detail, "solver projection failed");
    Box::new(MeshingFailure {
        schema_version: MESHING_FAILURE_SCHEMA_VERSION,
        category,
        stage: MeshingStageKind::OrderElevation,
        operation: MeshingStageKind::OrderElevation.operation(),
        entity_ids: Vec::new(),
        witnesses: Vec::new(),
        request_values: Vec::new(),
        achieved_values: vec![MeshingDiagnosticEntry {
            name: "solver_projection_failure".into(),
            value: MeshingDiagnosticValue::Text(detail),
            unit: None,
        }],
        remediation:
            "regenerate terminal inputs, increase the exhausted budget, or repair exact geometry"
                .into(),
    })
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-solver-projection-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_failures_keep_their_specific_budget_category() {
        assert_eq!(
            resource_category("geometry evaluation IterationBudgetExceeded"),
            MeshingFailureCategory::IterationBudgetExceeded
        );
        assert_eq!(
            resource_category("geometry evaluation AllocationBudgetExceeded"),
            MeshingFailureCategory::MemoryBudgetExceeded
        );
        assert_eq!(
            resource_category("boundary face inventory exceeds its hard limit"),
            MeshingFailureCategory::ElementBudgetExceeded
        );
        assert_eq!(
            resource_category("curved-node candidate work exceeds its hard limit"),
            MeshingFailureCategory::SearchWorkBudgetExceeded
        );
    }
}
