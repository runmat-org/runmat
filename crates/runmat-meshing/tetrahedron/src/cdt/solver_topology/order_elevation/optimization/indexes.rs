//! Immutable connectivity/lineage is prepared once for all curved sweeps. Node
//! coordinates may change, but these source-bound indexes do not.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};

pub(in crate::cdt::solver_topology::order_elevation) struct OptimizationTopology<'a> {
    pub(super) incidence: incidence::Incidence<'a>,
    pub(super) surfaces: geometry::BoundarySurfaceUses<'a>,
    pub(super) edges: geometry::BoundaryExactEdges<'a>,
    boundary_nodes: Vec<bool>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> OptimizationTopology<'a> {
    pub fn prepare(
        input: &ProjectionInput<'a>,
        topology: &SolverMeshTopology,
        control: &'a dyn GeometryEvaluationControl,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let storage = GeometryAllocationReservation::new(control, topology.nodes.len() as u64)
            .map_err(error::geometry)?;
        let mut boundary_nodes = vec![false; topology.nodes.len()];
        for face in &topology.boundary_faces {
            control.checkpoint().map_err(error::geometry)?;
            for &id in face.node_ids.iter().take(3) {
                boundary_nodes[geometry::node_slot(&topology.nodes, id)?] = true;
            }
        }
        Ok(Self {
            incidence: incidence::Incidence::prepare(topology, control)?,
            surfaces: geometry::boundary_sources::surface_uses(
                input,
                topology,
                control,
                cancellation,
            )?,
            edges: geometry::boundary_sources::exact_edges(input, topology, control, cancellation)?,
            boundary_nodes,
            _storage: storage,
        })
    }
    /// The elements that reference a node.
    pub fn incident_elements(&self, id: u64) -> Result<&[usize], DelaunaySolverTopologyError> {
        self.incidence.elements(id)
    }
    pub fn is_boundary(&self, slot: usize) -> bool {
        self.boundary_nodes[slot]
    }
}
