//! Consuming canonical materialization releases working indexes before full admission.
use super::*;
use crate::cdt::topology::build_canonical_volume_topology;

pub(super) struct CanonicalParts {
    nodes: Vec<crate::cdt::DelaunayVolumeNode>,
    tetrahedra: Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
}

pub(super) fn prepare(
    source_nodes: &[crate::cdt::DelaunayVolumeNode],
    source_tetrahedra: impl Iterator<Item = Option<super::super::storage::MutableTetrahedron>>,
    slot_count: usize,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<CanonicalParts, TopologyMutationError> {
    checkpoint(cancellation)?;
    let mut node_order = (0..source_nodes.len()).collect::<Vec<_>>();
    node_order.sort_by_key(|index| source_nodes[*index].identity);
    checkpoint(cancellation)?;
    if node_order
        .windows(2)
        .any(|pair| source_nodes[pair[0]].identity >= source_nodes[pair[1]].identity)
    {
        return Err(error("mutable topology contains duplicate node identities"));
    }
    let mut remap = vec![0_u32; source_nodes.len()];
    let nodes = node_order
        .into_iter()
        .enumerate()
        .map(|(canonical, mutable)| {
            remap[mutable] = canonical as u32;
            source_nodes[mutable]
        })
        .collect::<Vec<_>>();
    checkpoint(cancellation)?;
    let mut tetrahedra = Vec::with_capacity(slot_count);
    for (index, tetrahedron) in source_tetrahedra.enumerate() {
        if (index as u64).is_multiple_of(options.cancellation_check_interval) {
            checkpoint(cancellation)?;
        }
        let Some(tetrahedron) = tetrahedron else {
            continue;
        };
        tetrahedra.push((
            tetrahedron
                .vertex_indices
                .map(|vertex| remap[vertex as usize]),
            tetrahedron.region_id,
        ));
    }
    checkpoint(cancellation)?;
    Ok(CanonicalParts { nodes, tetrahedra })
}

pub(super) fn admit(
    parts: CanonicalParts,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<CanonicalVolumeTopology, TopologyMutationError> {
    let CanonicalParts { nodes, tetrahedra } = parts;
    build_canonical_volume_topology(nodes, tetrahedra, options, cancellation).map_err(|failure| {
        let kind = match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => TopologyMutationErrorKind::ResourceLimit,
            DelaunayTopologyErrorKind::Cancelled => TopologyMutationErrorKind::Cancelled,
            _ => TopologyMutationErrorKind::InvalidTopology,
        };
        execution_error(
            kind,
            format!("canonical freeze rejected mutable topology: {failure}"),
        )
    })
}
