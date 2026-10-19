use runmat_meshing_core::MeshingCancellationSignal;

use super::storage::{
    error, execution_error, TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
use crate::cdt::{
    incidence::build_volume_incidence, topology::CanonicalVolumeTopology, DelaunayTopologyError,
    DelaunayTopologyErrorKind, DelaunayTopologyOptions, DelaunayVolumeTetrahedron,
    DelaunayVolumeTopology,
};

mod canonical;
use canonical::{admit, prepare};

impl TopologyMutation {
    pub(in crate::cdt) fn freeze(
        self,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, TopologyMutationError> {
        self.freeze_canonical(options, cancellation)
            .map(CanonicalVolumeTopology::into_inner)
    }

    pub(in crate::cdt) fn freeze_canonical(
        mut self,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<CanonicalVolumeTopology, TopologyMutationError> {
        let nodes = std::mem::take(&mut self.nodes);
        let tetrahedra = std::mem::take(&mut self.tetrahedra);
        // All spatial, incidence, journal and search indexes are obsolete here.
        drop(self);
        let slot_count = tetrahedra.len();
        let parts = prepare(
            &nodes,
            tetrahedra.into_iter(),
            slot_count,
            options,
            cancellation,
        )?;
        drop(nodes);
        admit(parts, options, cancellation)
    }

    /// Perform the same independent full admission as consuming `freeze`, while
    /// retaining the working owner for a checked canonical handoff.
    pub(in crate::cdt) fn validated_snapshot(
        &self,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, TopologyMutationError> {
        let parts = prepare(
            &self.nodes,
            self.tetrahedra.iter().cloned(),
            self.tetrahedra.len(),
            options,
            cancellation,
        )?;
        admit(parts, options, cancellation).map(CanonicalVolumeTopology::into_inner)
    }

    pub(in crate::cdt) fn snapshot(
        &self,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, TopologyMutationError> {
        snapshot_parts(&self.nodes, &self.tetrahedra, options, cancellation)
    }
}

/// Canonicalizes working topology whose mutations have already passed the
/// checked cavity-replacement boundary. Orientation follows the checked vertex
/// permutation; compact adjacency construction checks shared-face ownership and
/// overlap without retaining a second face tree.
/// `freeze` and the public artifact validator retain the independent full check.
fn snapshot_parts(
    source_nodes: &[crate::cdt::DelaunayVolumeNode],
    source_tetrahedra: &[Option<super::storage::MutableTetrahedron>],
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, TopologyMutationError> {
    if options.maximum_nodes == 0
        || options.maximum_tetrahedra == 0
        || options.cancellation_check_interval == 0
    {
        return Err(error("canonical snapshot options must be nonzero"));
    }
    let active_tetrahedron_count = source_tetrahedra.iter().flatten().count();
    if source_nodes.is_empty()
        || source_nodes.len() as u64 > options.maximum_nodes
        || active_tetrahedron_count == 0
        || active_tetrahedron_count as u64 > options.maximum_tetrahedra
    {
        return Err(execution_error(
            TopologyMutationErrorKind::ResourceLimit,
            "canonical snapshot inventory is empty or exceeds its hard limit",
        ));
    }

    let mut node_order = (0..source_nodes.len()).collect::<Vec<_>>();
    node_order.sort_by_key(|index| source_nodes[*index].identity);
    if node_order
        .windows(2)
        .any(|pair| source_nodes[pair[0]].identity >= source_nodes[pair[1]].identity)
    {
        return Err(error("mutable topology contains duplicate node identities"));
    }
    let mut node_remap = vec![0_u32; source_nodes.len()];
    let nodes = node_order
        .into_iter()
        .enumerate()
        .map(|(canonical, mutable)| {
            node_remap[mutable] = canonical as u32;
            source_nodes[mutable]
        })
        .collect::<Vec<_>>();

    let mut ordered = source_tetrahedra
        .iter()
        .filter_map(|tetrahedron| {
            tetrahedron.as_ref().map(|tetrahedron| {
                let remapped = tetrahedron
                    .vertex_indices
                    .map(|vertex| node_remap[vertex as usize]);
                let mut key = remapped;
                key.sort_unstable();
                // Mutable tetrahedra are positively oriented. Sorting their
                // remapped indices reverses orientation exactly when the
                // permutation is odd, so canonical orientation can be
                // recovered without repeating a spatial predicate.
                let mut vertex_indices = key;
                if permutation_is_odd(remapped) {
                    vertex_indices.swap(0, 1);
                }
                (key, vertex_indices, tetrahedron.region_id.clone())
            })
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(key, _, _)| *key);
    for (canonical, _) in ordered.iter().enumerate() {
        if (canonical as u64).is_multiple_of(options.cancellation_check_interval) {
            checkpoint(cancellation)?;
        }
    }
    let mut tetrahedra = ordered
        .into_iter()
        .map(|(_, vertex_indices, region_id)| DelaunayVolumeTetrahedron {
            vertex_indices,
            neighbors: [None; 4],
            region_id,
        })
        .collect::<Vec<_>>();
    // A retained transaction needs one canonical view, not a second tree of
    // every face and its heap-owned use list. The topology owner provides the
    // same compact, independently checked adjacency construction as freeze.
    crate::cdt::topology::connect_volume_neighbors(&nodes, &mut tetrahedra, options, cancellation)
        .map_err(topology_error)?;
    let incidence = build_volume_incidence(&nodes, &tetrahedra, cancellation, options)
        .map_err(topology_error)?;
    Ok(DelaunayVolumeTopology {
        nodes,
        tetrahedra,
        incidence,
    })
}

fn permutation_is_odd(values: [u32; 4]) -> bool {
    let inversions = (0..4)
        .flat_map(|left| ((left + 1)..4).map(move |right| (left, right)))
        .filter(|(left, right)| values[*left] > values[*right])
        .count();
    inversions % 2 == 1
}

fn checkpoint(cancellation: &dyn MeshingCancellationSignal) -> Result<(), TopologyMutationError> {
    if cancellation.is_cancelled() {
        return Err(execution_error(
            TopologyMutationErrorKind::Cancelled,
            "cancelled while freezing mutable topology",
        ));
    }
    Ok(())
}

fn topology_error(failure: DelaunayTopologyError) -> TopologyMutationError {
    let kind = match failure.kind {
        DelaunayTopologyErrorKind::ResourceLimit => TopologyMutationErrorKind::ResourceLimit,
        DelaunayTopologyErrorKind::Cancelled => TopologyMutationErrorKind::Cancelled,
        _ => TopologyMutationErrorKind::InvalidTopology,
    };
    execution_error(kind, failure.to_string())
}
