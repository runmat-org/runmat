//! Own the cold validation boundary and release volume-only evidence before projection.
use super::*;
use crate::cdt::{DelaunayVolumeMeshCodecError, DelaunayVolumeProvenance, DelaunayVolumeTopology};
mod storage;

/// Original CAD, surface, metric and request used to independently admit a cold volume.
pub struct DelaunaySolverProjectionInput<'a> {
    pub exact_topology: &'a ExactBRepTopology,
    pub exact_surface: &'a ExactSurfaceMesh,
    pub volume_options: DelaunayVolumeMeshOptions,
    pub volume_metric: &'a MetricFieldRequest,
    pub request: &'a MeshingRequest,
    pub exact_evaluation: Option<DelaunayExactEvaluation<'a>>,
}

pub(super) struct VolumeView<'a> {
    pub topology: &'a DelaunayVolumeTopology,
    pub provenance: &'a DelaunayVolumeProvenance,
}
pub(super) struct ProjectionInput<'a> {
    pub exact_topology: &'a ExactBRepTopology,
    pub exact_surface: &'a ExactSurfaceMesh,
    pub volume_mesh: VolumeView<'a>,
    pub request: &'a MeshingRequest,
    pub exact_evaluation: Option<DelaunayExactEvaluation<'a>>,
}
impl<'a> From<DelaunaySolverTopologyInput<'a>> for ProjectionInput<'a> {
    fn from(input: DelaunaySolverTopologyInput<'a>) -> Self {
        Self {
            exact_topology: input.exact_topology,
            exact_surface: input.exact_surface,
            volume_mesh: VolumeView {
                topology: &input.volume_mesh.topology,
                provenance: &input.volume_mesh.provenance,
            },
            request: input.request,
            exact_evaluation: input.exact_evaluation,
        }
    }
}

/// Validate a cold volume once against its original immutable context, then
/// project only its topology/provenance. Neither serialized claims nor callers
/// can bypass the volume check. Mutation and quality history die before solver
/// construction, and the compact source dies before this function returns.
pub fn decode_delaunay_solver_topology(
    bytes: Vec<u8>,
    input: DelaunaySolverProjectionInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ValidatedSolverMeshTopology, DelaunaySolverProjectionError> {
    validate_options(options).map_err(DelaunaySolverProjectionError::Projection)?;
    input
        .request
        .validate()
        .map_err(error::request)
        .map_err(DelaunaySolverProjectionError::Projection)?;
    let allocations = input
        .exact_evaluation
        .map(|evaluation| runmat_geometry_core::GeometryAllocationScope::new(evaluation.control));
    let mesh = crate::cdt::volume_mesh::decode_delaunay_volume_mesh_admitted(
        &bytes,
        input.exact_topology,
        input.exact_surface,
        input.volume_metric,
        input.volume_options,
        cancellation,
        allocations
            .as_ref()
            .map(|scope| scope as &dyn GeometryEvaluationControl),
    )
    .map_err(DelaunaySolverProjectionError::Decode)?;
    drop(bytes);
    let mut source = compact_source(mesh);
    if let Some(scope) = &allocations {
        let retained = storage::retained_bytes(&source.topology, &source.provenance, scope)
            .map_err(DelaunaySolverProjectionError::Control)?;
        source.storage = Some(
            scope
                .transfer_reservation(retained)
                .map_err(DelaunaySolverProjectionError::Control)?,
        );
    }
    // The compact source keeps its original charge; release dropped quality,
    // mutation and decoding workspace before allocating the solver topology.
    drop(allocations);
    project(
        ProjectionInput {
            exact_topology: input.exact_topology,
            exact_surface: input.exact_surface,
            volume_mesh: VolumeView {
                topology: &source.topology,
                provenance: &source.provenance,
            },
            request: input.request,
            exact_evaluation: input.exact_evaluation,
        },
        options,
        cancellation,
    )
    .map_err(DelaunaySolverProjectionError::Projection)
}

struct CompactSource<'a> {
    topology: DelaunayVolumeTopology,
    provenance: DelaunayVolumeProvenance,
    // Rust drops fields in declaration order: release data before its reservation.
    storage: Option<runmat_geometry_core::GeometryAllocationReservation<'a>>,
}
fn compact_source<'a>(mesh: DelaunayVolumeMesh) -> CompactSource<'a> {
    // The unmoved fields are dropped on return, before project() starts.
    CompactSource {
        topology: mesh.topology,
        provenance: mesh.provenance,
        storage: None,
    }
}

/// Preserve the decoding/semantic boundary so execution retains original failure categories.
#[derive(Debug)]
pub enum DelaunaySolverProjectionError {
    Decode(DelaunayVolumeMeshCodecError),
    Projection(DelaunaySolverTopologyError),
    Control(runmat_geometry_core::GeometryEvaluationError),
}
impl std::fmt::Display for DelaunaySolverProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(error) => std::fmt::Display::fmt(error, f),
            Self::Projection(error) => std::fmt::Display::fmt(error, f),
            Self::Control(error) => std::fmt::Display::fmt(error, f),
        }
    }
}
impl std::error::Error for DelaunaySolverProjectionError {}
