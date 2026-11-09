//! The invalid-element inventory across curved-node optimization. An element's validity depends
//! only on its ten node coordinates, so after nodes move only the elements that reference a moved
//! node are certified again. The inventory equals a full certification of every element.
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use runmat_meshing_core::{MeshingCancellationSignal, SolverMeshTopology};

use super::{counters::ElevationCounters, error, jacobian};
use crate::cdt::solver_topology::{
    storage::{ReservedBuffer, Storage},
    DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

/// Certification limits shared by every inventory.
#[derive(Clone, Copy)]
pub(super) struct CertificationLimits {
    pub maximum_search_work: u64,
    pub maximum_recursion_depth: u32,
    pub cancellation_check_interval: u64,
}

pub(super) struct ElementValidity<'a> {
    /// Node coordinate bits at the last certification, by node slot.
    positions: Vec<[u64; 3]>,
    /// Whether each element failed its last certification, by element index.
    invalid: Vec<bool>,
    limits: CertificationLimits,
    _storage: GeometryAllocationReservation<'a>,
}

fn position_bits(topology: &SolverMeshTopology, slot: usize) -> [u64; 3] {
    topology.nodes[slot]
        .coordinates_m
        .canonical_value()
        .map(f64::to_bits)
}

impl<'a> ElementValidity<'a> {
    /// Certifies every element.
    pub fn certify(
        topology: &SolverMeshTopology,
        limits: CertificationLimits,
        control: &'a dyn GeometryEvaluationControl,
        cancellation: &dyn MeshingCancellationSignal,
        jacobian_work: &mut u64,
        counters: &ElevationCounters,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let bytes = topology
            .nodes
            .len()
            .checked_mul(std::mem::size_of::<[u64; 3]>())
            .and_then(|bytes| bytes.checked_add(topology.volume_elements.len()))
            .ok_or_else(|| resource("Tet10 validity inventory storage overflow"))?;
        let storage =
            GeometryAllocationReservation::new(control, bytes as u64).map_err(error::geometry)?;
        let positions = (0..topology.nodes.len())
            .map(|slot| position_bits(topology, slot))
            .collect();
        let indices = (0..topology.volume_elements.len()).collect::<Vec<_>>();
        let invalid = certify(
            topology,
            &indices,
            limits,
            control,
            cancellation,
            jacobian_work,
        )?;
        ElevationCounters::add(&counters.inventories, 1);
        ElevationCounters::add(&counters.inventory_certified, indices.len() as u64);
        Ok(Self {
            positions,
            invalid,
            limits,
            _storage: storage,
        })
    }

    /// Certifies again exactly the elements that reference a node whose coordinates changed
    /// since the last certification.
    pub fn refresh<'i>(
        &mut self,
        topology: &SolverMeshTopology,
        incident_elements: impl Fn(u64) -> Result<&'i [usize], DelaunaySolverTopologyError>,
        control: &dyn GeometryEvaluationControl,
        cancellation: &dyn MeshingCancellationSignal,
        jacobian_work: &mut u64,
        counters: &ElevationCounters,
    ) -> Result<(), DelaunaySolverTopologyError> {
        if self.positions.len() != topology.nodes.len()
            || self.invalid.len() != topology.volume_elements.len()
        {
            return Err(invalid_mesh(
                "Tet10 validity inventory no longer matches the mesh",
            ));
        }
        let mut touched = Vec::new();
        for slot in 0..topology.nodes.len() {
            let bits = position_bits(topology, slot);
            if bits != self.positions[slot] {
                self.positions[slot] = bits;
                touched.extend_from_slice(incident_elements(topology.nodes[slot].node_id)?);
            }
        }
        touched.sort_unstable();
        touched.dedup();
        let invalid = certify(
            topology,
            &touched,
            self.limits,
            control,
            cancellation,
            jacobian_work,
        )?;
        for (&element, invalid) in touched.iter().zip(invalid) {
            self.invalid[element] = invalid;
        }
        ElevationCounters::add(&counters.inventories, 1);
        ElevationCounters::add(&counters.inventory_certified, touched.len() as u64);
        ElevationCounters::add(
            &counters.inventory_reused,
            (topology.volume_elements.len() - touched.len()) as u64,
        );
        Ok(())
    }

    /// The invalid element indices in ascending order, in a buffer with room for every element
    /// so that elements with folded boundary faces can be merged in.
    pub fn indices<'storage>(
        &self,
        storage: Storage<'storage>,
    ) -> Result<ReservedBuffer<'storage, usize>, DelaunaySolverTopologyError> {
        let mut indices = ReservedBuffer::new(self.invalid.len(), storage)?;
        for (element, invalid) in self.invalid.iter().enumerate() {
            if *invalid {
                indices.push(element)?;
            }
        }
        Ok(indices)
    }
}

fn certify(
    topology: &SolverMeshTopology,
    indices: &[usize],
    limits: CertificationLimits,
    control: &dyn GeometryEvaluationControl,
    cancellation: &dyn MeshingCancellationSignal,
    jacobian_work: &mut u64,
) -> Result<Vec<bool>, DelaunaySolverTopologyError> {
    jacobian::certify_indices(
        topology,
        indices,
        limits.maximum_search_work,
        limits.maximum_recursion_depth,
        limits.cancellation_check_interval,
        cancellation,
        jacobian::ValidationWork {
            counter: jacobian_work,
            storage: Storage(Some(control)),
        },
    )
}

fn invalid_mesh(reason: &str) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn resource(reason: &str) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}

#[cfg(test)]
mod tests;
