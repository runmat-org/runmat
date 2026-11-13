//! Construction-only canonical topology authority. No mutable borrow or wire decode.
//! Consuming it to obtain raw topology discards authority; cold or edited topology
//! must pass independent validation before authority can be restored.
use super::*;

pub(in crate::cdt) struct CanonicalVolumeTopology(DelaunayVolumeTopology);

impl CanonicalVolumeTopology {
    pub(super) fn from_construction(topology: DelaunayVolumeTopology) -> Self {
        Self(topology)
    }

    #[cfg(test)]
    pub(in crate::cdt) fn validate(
        topology: DelaunayVolumeTopology,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayTopologyError> {
        validate_canonical_volume_topology(&topology, options, cancellation)?;
        Ok(Self(topology))
    }

    pub(in crate::cdt) fn into_inner(self) -> DelaunayVolumeTopology {
        self.0
    }
}

impl std::ops::Deref for CanonicalVolumeTopology {
    type Target = DelaunayVolumeTopology;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests;
