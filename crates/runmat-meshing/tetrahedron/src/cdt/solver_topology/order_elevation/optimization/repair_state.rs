//! What node repairs remember between sweeps.
//!
//! A repair is a pure function of its node's star: the coordinates of every node of the node's
//! incident elements, and the exact parameters of the node and its edge endpoints. A repair that
//! declined therefore declines again until something in that star changes, and is skipped until
//! then. An element's margin is a pure function of its ten node coordinates, so it is reused
//! while they are unchanged.
use super::*;
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
use std::collections::HashMap;

#[derive(PartialEq)]
pub(super) struct StarSignature {
    coordinates: Vec<u64>,
    parameters: [Vec<SolverNodeExactParameter>; 3],
}

impl StarSignature {
    pub fn of(
        topology: &SolverMeshTopology,
        movable: &MovableNode,
        nodes: [&runmat_meshing_core::SolverMeshNode; 3],
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let mut coordinates = Vec::with_capacity(movable.incident_elements.len() * 30);
        for &element in movable.incident_elements {
            for point in jacobian::element_coordinates(topology, element)? {
                coordinates.extend(point.map(f64::to_bits));
            }
        }
        Ok(Self {
            coordinates,
            parameters: nodes.map(|node| node.exact_parameters.clone()),
        })
    }

    fn bytes(&self) -> u64 {
        let parameters = self
            .parameters
            .iter()
            .map(|parameters| {
                parameters.len() * std::mem::size_of::<SolverNodeExactParameter>()
                    + parameters
                        .iter()
                        .map(|parameter| match parameter {
                            SolverNodeExactParameter::Curve { source_edge_id, .. } => {
                                source_edge_id.source_topology_id.len()
                            }
                            SolverNodeExactParameter::Surface { source_face_id, .. } => {
                                source_face_id.source_topology_id.len()
                            }
                        })
                        .sum::<usize>()
            })
            .sum::<usize>();
        (std::mem::size_of::<Self>() + self.coordinates.len() * 8 + parameters) as u64
    }
}

pub(in crate::cdt::solver_topology::order_elevation) struct RepairState<'a> {
    scope: GeometryAllocationScope<'a>,
    declined: HashMap<u64, StarSignature>,
    margins: HashMap<usize, ([u64; 30], u64)>,
}

const MARGIN_ENTRY_BYTES: u64 =
    (std::mem::size_of::<usize>() + std::mem::size_of::<([u64; 30], u64)>()) as u64;

impl<'a> RepairState<'a> {
    pub fn new(control: &'a dyn GeometryEvaluationControl) -> Self {
        Self {
            scope: GeometryAllocationScope::new(control),
            declined: HashMap::new(),
            margins: HashMap::new(),
        }
    }

    /// Whether the node's last repair declined with this same star.
    pub(super) fn declined_unchanged(&self, node_id: u64, signature: &StarSignature) -> bool {
        self.declined
            .get(&node_id)
            .is_some_and(|declined| declined == signature)
    }

    pub(super) fn decline(
        &mut self,
        node_id: u64,
        signature: StarSignature,
    ) -> Result<(), DelaunaySolverTopologyError> {
        self.scope
            .consume_allocation_bytes(signature.bytes())
            .map_err(error::geometry)?;
        if let Some(previous) = self.declined.insert(node_id, signature) {
            self.scope.release_allocation_bytes(previous.bytes());
        }
        Ok(())
    }

    pub(super) fn moved(&mut self, node_id: u64) {
        if let Some(previous) = self.declined.remove(&node_id) {
            self.scope.release_allocation_bytes(previous.bytes());
        }
    }

    /// The sorted margin profile of the given elements, computing only margins whose element
    /// coordinates changed since they were last computed.
    pub(super) fn profile(
        &mut self,
        topology: &SolverMeshTopology,
        elements: &[usize],
        cancellation: &dyn MeshingCancellationSignal,
        counters: &super::super::counters::ElevationCounters,
    ) -> Result<Vec<u64>, DelaunaySolverTopologyError> {
        use super::super::counters::ElevationCounters as C;
        if elements.is_empty() {
            return Err(invalid("Tet10 optimization margin profile is empty"));
        }
        C::add(&counters.profiles, 1);
        let mut profile = Vec::with_capacity(elements.len());
        for &element in elements {
            candidate_checkpoint(cancellation)?;
            let coordinates = jacobian::element_coordinates(topology, element)?;
            let mut bits = [0_u64; 30];
            for (point, slot) in coordinates.iter().zip(bits.chunks_exact_mut(3)) {
                slot.copy_from_slice(&point.map(f64::to_bits));
            }
            let key = match self.margins.get(&element) {
                Some((cached, key)) if *cached == bits => {
                    C::add(&counters.profile_margins_reused, 1);
                    *key
                }
                _ => {
                    C::add(&counters.profile_margins, 1);
                    let key = jacobian::margin_key(coordinates, cancellation)?;
                    if self.margins.insert(element, (bits, key)).is_none() {
                        self.scope
                            .consume_allocation_bytes(MARGIN_ENTRY_BYTES)
                            .map_err(error::geometry)?;
                    }
                    key
                }
            };
            profile.push(key);
        }
        profile.sort_unstable();
        Ok(profile)
    }
}

#[cfg(test)]
mod tests;
