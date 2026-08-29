use runmat_meshing_core::MeshingCancellationSignal;

use super::storage::{
    error, execution_error, TopologyMutation, TopologyMutationError, TopologyMutationErrorKind,
};
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyErrorKind,
    DelaunayTopologyOptions, DelaunayVolumeTopology,
};

impl TopologyMutation {
    pub(in crate::cdt) fn freeze(
        self,
        options: DelaunayTopologyOptions,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<DelaunayVolumeTopology, TopologyMutationError> {
        let mut node_order = (0..self.nodes.len()).collect::<Vec<_>>();
        node_order.sort_by_key(|index| self.nodes[*index].identity);
        if node_order
            .windows(2)
            .any(|pair| self.nodes[pair[0]].identity >= self.nodes[pair[1]].identity)
        {
            return Err(error("mutable topology contains duplicate node identities"));
        }
        let mut remap = vec![0_u32; self.nodes.len()];
        let nodes = node_order
            .into_iter()
            .enumerate()
            .map(|(canonical, mutable)| {
                remap[mutable] = canonical as u32;
                self.nodes[mutable]
            })
            .collect::<Vec<_>>();
        let tetrahedra = self
            .tetrahedra
            .into_iter()
            .flatten()
            .map(|tetrahedron| {
                (
                    tetrahedron
                        .vertex_indices
                        .map(|vertex| remap[vertex as usize]),
                    tetrahedron.region_id,
                )
            })
            .collect();
        build_delaunay_volume_topology_with_regions(nodes, tetrahedra, options, cancellation)
            .map_err(|failure| {
                let kind = match failure.kind {
                    DelaunayTopologyErrorKind::ResourceLimit => {
                        TopologyMutationErrorKind::ResourceLimit
                    }
                    DelaunayTopologyErrorKind::Cancelled => TopologyMutationErrorKind::Cancelled,
                    _ => TopologyMutationErrorKind::InvalidTopology,
                };
                execution_error(
                    kind,
                    format!("canonical freeze rejected mutable topology: {failure}"),
                )
            })
    }
}
