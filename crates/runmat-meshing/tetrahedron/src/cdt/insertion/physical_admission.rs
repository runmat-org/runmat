//! Construction-only authority for maintained physical neighbor admission.
//! Canonical publication and terminal whole-topology checks remain independent.
use super::*;

pub(in crate::cdt) struct AdmittedPhysicalTopology {
    topology: DelaunayVolumeTopology,
}

impl AdmittedPhysicalTopology {
    pub(in crate::cdt) fn verify_global(
        topology: DelaunayVolumeTopology,
        options: DelaunayInsertionOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayInsertionError> {
        validate_delaunay_volume_topology(&topology, options, cancellation)?;
        Self::retain(topology)
    }
    pub(in crate::cdt) fn verify(
        topology: DelaunayVolumeTopology,
        options: DelaunayInsertionOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Self, DelaunayInsertionError> {
        validate_locally_delaunay_volume_topology(&topology, options, cancellation)?;
        Self::retain(topology)
    }

    fn retain(topology: DelaunayVolumeTopology) -> Result<Self, DelaunayInsertionError> {
        if !topology.incidence.regions.is_empty() {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "unconstrained physical admission must precede region assignment",
            ));
        }
        Ok(Self { topology })
    }

    pub(in crate::cdt) fn topology(&self) -> &DelaunayVolumeTopology {
        &self.topology
    }

    pub(in crate::cdt) fn into_topology(self) -> DelaunayVolumeTopology {
        self.topology
    }
}

pub(in crate::cdt) fn restore_admitted_mutation(
    topology: &mut TopologyMutation,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayInsertionError> {
    let mut work = InsertionWork::new(options, cancellation);
    legalization::mutable_repair::restore_admitted(topology, &mut work).map_err(|mut failure| {
        failure.reason = format!(
            "{}; maintained physical repair nodes={}, dirty_faces={:?}, work={:?}",
            failure.reason,
            topology.nodes.len(),
            topology.physical_dirty_faces().map(|faces| faces.len()),
            work.inventory()
        );
        failure
    })
}
