use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    ExactTrimClassifier, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{
    validate_solver_mesh_topology, MeshingCancellationSignal, MeshingRequest, SolverMeshTopology,
};
use runmat_meshing_surface::ExactSurfaceMesh;

use super::{
    validate_delaunay_volume_mesh, DelaunayVolumeMesh, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshOptions,
};

mod boundaries;
mod classification;
mod construction;
mod error;
mod inventories;
mod order_elevation;
mod parameters;

pub use error::{DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayRegionMaterial {
    pub region_id: PersistentEntityId,
    pub material_id: String,
}

pub struct DelaunaySolverTopologyInput<'a> {
    pub exact_topology: &'a ExactBRepTopology,
    pub exact_surface: &'a ExactSurfaceMesh,
    pub volume_mesh: &'a DelaunayVolumeMesh,
    pub volume_options: DelaunayVolumeMeshOptions,
    pub request: &'a MeshingRequest,
    pub region_materials: &'a [DelaunayRegionMaterial],
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
    pub trim_boundary_tolerance_uv: f64,
    pub cancellation_check_interval: u64,
}

impl Default for DelaunaySolverTopologyOptions {
    fn default() -> Self {
        Self {
            maximum_boundary_faces: 2_000_000_000,
            maximum_boundary_edges: 3_000_000_000,
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
        &input.request.metric,
        input.volume_mesh,
        input.volume_options,
        cancellation,
    )
    .map_err(|failure| error::volume(failure.kind, failure.to_string()))?;
    let linear = construction::construct(&input, options, cancellation)?;
    let result = match input.request.element_order {
        runmat_meshing_core::ElementOrder::Tet4 => linear,
        runmat_meshing_core::ElementOrder::Tet10 => {
            order_elevation::elevate(&input, linear, options, cancellation)?
        }
    };
    validate_solver_mesh_topology(&result, input.request).map_err(error::solver)?;
    Ok(result)
}

fn validate_options(
    options: DelaunaySolverTopologyOptions,
) -> Result<(), DelaunaySolverTopologyError> {
    if options.maximum_boundary_faces == 0
        || options.maximum_boundary_edges == 0
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
