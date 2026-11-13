//! Consume completed components. Only multi-region meshes need a new incidence graph.
use std::collections::BTreeMap;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
use runmat_meshing_size::metric::MetricFieldRequest;

use crate::cdt::{
    evaluate_delaunay_volume_quality, topology::build_delaunay_volume_topology_with_regions,
    DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshOptions, DelaunayVolumeMeshStage, DelaunayVolumeNode,
    DelaunayVolumeProvenance,
};

pub(in crate::cdt) fn assemble(
    mut meshes: Vec<DelaunayVolumeMesh>,
    metric: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshError> {
    let interval = options
        .refinement
        .step
        .insertion
        .topology
        .cancellation_check_interval;
    if interval == 0 {
        return Err(super::super::error(
            DelaunayVolumeMeshStage::Refinement,
            DelaunayVolumeMeshErrorKind::InvalidOptions,
            "component cancellation interval must be nonzero",
        ));
    }
    check_cancelled(cancellation)?;
    if meshes.len() == 1 {
        let mut mesh = meshes.pop().expect("one completed component");
        sort_insertions(&mut mesh);
        return Ok(mesh);
    }
    let mut nodes = BTreeMap::<StableDigest, DelaunayVolumeNode>::new();
    let mut cells = Vec::new();
    let mut provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let mut facet_recovery_insertions = Vec::new();
    let mut quality_mutations = Vec::new();
    for mesh in meshes {
        let DelaunayVolumeMesh {
            topology,
            quality,
            provenance: component_provenance,
            facet_recovery_insertions: component_facets,
            quality_mutations: component_quality,
        } = mesh;
        // Neither the old quality nor incidence belongs to the aggregate graph.
        drop(quality);
        drop(topology.incidence);
        for (index, cell) in topology.tetrahedra.into_iter().enumerate() {
            if (index as u64).is_multiple_of(interval) {
                check_cancelled(cancellation)?;
            }
            cells.push((
                cell.vertex_indices
                    .map(|index| topology.nodes[index as usize].identity),
                cell.region_id,
            ));
        }
        for (index, node) in topology.nodes.into_iter().enumerate() {
            if (index as u64).is_multiple_of(interval) {
                check_cancelled(cancellation)?;
            }
            if nodes
                .insert(node.identity, node)
                .is_some_and(|previous| previous != node)
            {
                return Err(super::super::error(
                    DelaunayVolumeMeshStage::Refinement,
                    DelaunayVolumeMeshErrorKind::InvalidTopology,
                    "independent regions produced conflicting coordinates for one node identity",
                ));
            }
        }
        provenance.nodes.extend(component_provenance.nodes);
        provenance.segments.extend(component_provenance.segments);
        provenance.facets.extend(component_provenance.facets);
        facet_recovery_insertions.extend(component_facets);
        // Components share no mutable identities; each keeps its own applied order.
        quality_mutations.extend(component_quality);
    }
    let nodes = nodes.into_values().collect::<Vec<_>>();
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    let cells = cells
        .into_iter()
        .map(|(identities, region)| (identities.map(|identity| node_indices[&identity]), region))
        .collect();
    drop(node_indices);
    let topology = build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        options.refinement.step.insertion.topology,
        cancellation,
    )
    .map_err(|failure| {
        super::super::error(
            DelaunayVolumeMeshStage::Refinement,
            match failure.kind {
                crate::cdt::DelaunayTopologyErrorKind::Cancelled => {
                    DelaunayVolumeMeshErrorKind::Cancelled
                }
                crate::cdt::DelaunayTopologyErrorKind::ResourceLimit => {
                    DelaunayVolumeMeshErrorKind::ResourceLimit
                }
                _ => DelaunayVolumeMeshErrorKind::InvalidTopology,
            },
            format!("independent region assembly failed: {failure}"),
        )
    })?;
    let provenance = super::provenance::normalize(provenance);
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        metric,
        &provenance,
        options.quality,
        cancellation,
    )
    .map_err(super::super::quality_error)?;
    let mut mesh = DelaunayVolumeMesh {
        topology,
        quality,
        provenance,
        facet_recovery_insertions,
        quality_mutations,
    };
    sort_insertions(&mut mesh);
    Ok(mesh)
}

fn sort_insertions(mesh: &mut DelaunayVolumeMesh) {
    mesh.facet_recovery_insertions.sort_by_key(|insertion| {
        (
            insertion.constraint_index,
            insertion.support_node_identities,
            insertion.insertion_round,
            insertion.candidate_rank,
            insertion.node_identity,
        )
    });
}

fn check_cancelled(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeMeshError> {
    if cancellation.is_cancelled() {
        Err(super::super::error(
            DelaunayVolumeMeshStage::Refinement,
            DelaunayVolumeMeshErrorKind::Cancelled,
            "component assembly cancelled",
        ))
    } else {
        Ok(())
    }
}
