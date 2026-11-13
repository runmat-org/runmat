//! Local mutation storage for the tetrahedralization hot path.
//!
//! Canonical topology is an artifact contract: nodes and tetrahedra are densely ordered and all
//! incidence is independently reconstructed. Meshing mutations have different needs. They must
//! replace a bounded cavity without copying and reindexing the rest of the mesh. This module owns
//! that distinction and is the only bridge between mutable working topology and canonical output.

mod boundary;
mod closed_cavity;
mod edge_changes;
pub(in crate::cdt) use edge_changes::{EdgeChangeObserver, ObservedCellChanges};
mod freeze;
mod interface;
mod physical_adjacencies;
mod point_update;
mod relocation;
pub(in crate::cdt) use relocation::CavityRelocation;
mod replace;
mod spatial;
mod spatial_cells;
mod storage;
mod trace;
mod trial;

#[cfg(test)]
mod tests;

pub(in crate::cdt) use storage::{
    TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
pub(in crate::cdt) use trace::{TopologyEdit, TopologyEdits};
pub(in crate::cdt) use trial::CavityTrial;
