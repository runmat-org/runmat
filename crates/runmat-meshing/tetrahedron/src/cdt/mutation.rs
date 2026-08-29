//! Local mutation storage for the tetrahedralization hot path.
//!
//! Canonical topology is an artifact contract: nodes and tetrahedra are densely ordered and all
//! incidence is independently reconstructed. Meshing mutations have different needs. They must
//! replace a bounded cavity without copying and reindexing the rest of the mesh. This module owns
//! that distinction and is the only bridge between mutable working topology and canonical output.

mod freeze;
mod replace;
mod storage;

#[cfg(test)]
mod tests;

pub(in crate::cdt) use storage::{
    TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
