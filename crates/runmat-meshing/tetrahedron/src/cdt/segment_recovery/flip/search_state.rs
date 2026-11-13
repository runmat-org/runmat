//! Cycle hints retain fixed-size fingerprints, never one full mesh per visited state.
//! Nodes and regions are fixed within this flip transaction. This is not geometry
//! identity or an admission certificate; every accepted topology still needs validation.
use crate::cdt::mutation::TopologyMutation;

pub(super) fn topology_state<E>(
    topology: &TopologyMutation,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<[u8; 32], E> {
    topology.connectivity_search_fingerprint(checkpoint)
}

#[cfg(test)]
mod tests;
