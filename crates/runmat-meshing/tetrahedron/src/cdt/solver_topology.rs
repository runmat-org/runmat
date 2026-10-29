use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    ExactTrimClassifier, GeometryEvaluationControl,
};
use runmat_meshing_core::{
    MeshingCancellationSignal, MeshingRequest, MetricFieldRequest, SolverMeshTopology,
    ValidatedSolverMeshTopology,
};
use runmat_meshing_surface::ExactSurfaceMesh;

use super::{
    validate_delaunay_volume_mesh, DelaunayVolumeMesh, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshOptions,
};

mod boundaries;
mod classification;
mod construction;
mod node_index;
use node_index::NodeIndex;
mod error;
mod inventories;
mod order_elevation;
mod parameters;
mod source;
mod storage;
use source::ProjectionInput;
pub use source::{
    decode_delaunay_solver_topology, DelaunaySolverProjectionError, DelaunaySolverProjectionInput,
};

pub use error::{DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind};

pub struct DelaunaySolverTopologyInput<'a> {
    pub exact_topology: &'a ExactBRepTopology,
    pub exact_surface: &'a ExactSurfaceMesh,
    pub volume_mesh: &'a DelaunayVolumeMesh,
    pub volume_options: DelaunayVolumeMeshOptions,
    /// Complete metric authority used to construct and validate the original volume.
    /// The authored request remains unchanged for solver topology and resource admission.
    pub volume_metric: &'a MetricFieldRequest,
    pub request: &'a MeshingRequest,
    pub exact_evaluation: Option<DelaunayExactEvaluation<'a>>,
}

pub trait DelaunayExactEvaluator:
    ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ExactTrimClassifier
{
}

impl<T> DelaunayExactEvaluator for T where
    T: ExactCurveEvaluator + ExactPcurveEvaluator + ExactSurfaceEvaluator + ExactTrimClassifier
{
}

#[derive(Clone, Copy)]
pub struct DelaunayExactEvaluation<'a> {
    pub evaluator: &'a dyn DelaunayExactEvaluator,
    pub control: &'a dyn GeometryEvaluationControl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaunaySolverTopologyOptions {
    pub maximum_boundary_faces: u64,
    pub maximum_boundary_edges: u64,
    pub maximum_curved_optimization_candidates: u64,
    pub maximum_curved_optimization_rounds: u32,
    pub maximum_curved_optimization_sweeps: u32,
    pub trim_boundary_tolerance_uv: f64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunaySolverTopologyOptions {
    fn default() -> Self {
        Self {
            maximum_boundary_faces: 2_000_000_000,
            maximum_boundary_edges: 3_000_000_000,
            maximum_curved_optimization_candidates: 12_000_000_000,
            maximum_curved_optimization_rounds: 4_096,
            maximum_curved_optimization_sweeps: 64,
            trim_boundary_tolerance_uv: 1.0e-10,
            cancellation_check_interval: 1_024,
        }
    }
}

/// Projects a validated linear CDT into the requested canonical solver-topology order.
pub fn build_delaunay_solver_topology(
    input: DelaunaySolverTopologyInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<SolverMeshTopology, DelaunaySolverTopologyError> {
    validate_options(options)?;
    input.request.validate().map_err(error::request)?;
    validate_delaunay_volume_mesh(
        input.exact_topology,
        input.exact_surface,
        input.volume_metric,
        input.volume_mesh,
        input.volume_options,
        cancellation,
    )
    .map_err(|failure| error::volume(failure.kind, failure.to_string()))?;
    project(ProjectionInput::from(input), options, cancellation)
        .map(ValidatedSolverMeshTopology::into_topology)
}

fn project(
    input: ProjectionInput<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ValidatedSolverMeshTopology, DelaunaySolverTopologyError> {
    let storage = storage::Storage::for_input(&input);
    storage.slots::<runmat_meshing_core::SolverMeshProjection>(1)?;
    let linear = construction::construct(&input, options, cancellation)?;
    let (result, elevation) = match input.request.element_order {
        runmat_meshing_core::ElementOrder::Tet4 => (
            linear,
            runmat_meshing_core::SolverElevationReport::default(),
        ),
        runmat_meshing_core::ElementOrder::Tet10 => {
            order_elevation::elevate(&input, linear, options, cancellation)?
        }
    };
    let validation_bytes = result
        .validation_scratch_bytes(input.request.element_order)
        .map_err(error::solver)?;
    let _validation_storage = input
        .exact_evaluation
        .map(|evaluation| {
            runmat_geometry_core::GeometryAllocationReservation::new(
                evaluation.control,
                validation_bytes,
            )
        })
        .transpose()
        .map_err(error::geometry)?;
    let request_bytes = input
        .request
        .retained_storage_bytes(u64::MAX, &mut || {
            if cancellation.is_cancelled() {
                Err(runmat_meshing_core::MeshingContractError::invalid(
                    "solver request clone",
                    "cancelled",
                ))
            } else {
                Ok(())
            }
        })
        .map_err(error::solver)?;
    storage.charge(request_bytes)?;
    ValidatedSolverMeshTopology::new(result, input.request.clone(), cancellation)
        .map(|topology| topology.with_elevation_report(elevation))
        .map_err(error::solver)
}

fn validate_options(
    options: DelaunaySolverTopologyOptions,
) -> Result<(), DelaunaySolverTopologyError> {
    if options.maximum_boundary_faces == 0
        || options.maximum_boundary_edges == 0
        || options.maximum_curved_optimization_candidates == 0
        || options.maximum_curved_optimization_rounds == 0
        || options.maximum_curved_optimization_sweeps == 0
        || !options.trim_boundary_tolerance_uv.is_finite()
        || options.trim_boundary_tolerance_uv < 0.0
        || options.cancellation_check_interval == 0
    {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::InvalidOptions,
            "solver topology limits and cancellation interval must be nonzero",
        ));
    }
    Ok(())
}

pub(super) fn checkpoint(
    work: u64,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    if work.is_multiple_of(options.cancellation_check_interval) && cancellation.is_cancelled() {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::Cancelled,
            "cancelled",
        ));
    }
    Ok(())
}

pub(super) fn require_capacity(
    field: &str,
    current: usize,
    maximum: u64,
) -> Result<(), DelaunaySolverTopologyError> {
    if current as u64 >= maximum {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::ResourceLimit,
            format!("{field} exceeds its hard limit of {maximum}"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "solver_topology/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "solver_topology/test_evaluator.rs"]
mod test_evaluator;
