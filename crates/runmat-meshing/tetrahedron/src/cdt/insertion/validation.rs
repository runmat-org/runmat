use runmat_meshing_core::{
    quality::predicate::{insphere3d, PredicateSign},
    MeshingCancellationSignal,
};

use super::{
    error, predicate_error, topology_error, validate_options, DelaunayInsertionError,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunayVolumeNode,
    DelaunayVolumeTopology, InsertionWork,
};
use crate::cdt::build_delaunay_volume_topology;

pub fn validate_delaunay_volume_topology(
    topology: &DelaunayVolumeTopology,
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayInsertionError> {
    validate_options(options)?;
    let rebuilt = build_delaunay_volume_topology(
        topology.nodes.clone(),
        topology
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.vertex_indices)
            .collect(),
        options.topology,
        cancellation,
    )
    .map_err(topology_error)?;
    if rebuilt != *topology {
        return Err(error(
            DelaunayInsertionErrorKind::InvalidTopology,
            "tetrahedra or neighbor links are not in canonical checked form",
        ));
    }

    let mut work = InsertionWork::new(options, cancellation);
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for neighbor_index in tetrahedron.neighbors.iter().flatten().copied() {
            if neighbor_index as usize <= tetrahedron_index {
                continue;
            }
            work.checkpoint()?;
            let neighbor = &topology.tetrahedra[neighbor_index as usize];
            let opposite = neighbor
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !tetrahedron.vertex_indices.contains(vertex))
                .ok_or_else(|| {
                    error(
                        DelaunayInsertionErrorKind::InvalidTopology,
                        "neighbors do not share exactly one triangular face",
                    )
                })?;
            if in_circumsphere_exact(
                topology,
                tetrahedron.vertex_indices,
                topology.nodes[opposite as usize],
                &mut work,
            )? {
                return Err(error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    format!(
                        "tetrahedron {tetrahedron_index} has neighbor {neighbor_index} strictly inside its circumsphere"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn in_circumsphere_exact(
    topology: &DelaunayVolumeTopology,
    vertices: [u32; 4],
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    work.predicate()?;
    let points = vertices.map(|vertex| topology.nodes[vertex as usize].coordinates_m);
    insphere3d([
        points[0],
        points[1],
        points[2],
        points[3],
        node.coordinates_m,
    ])
    .map(|sign| sign == PredicateSign::Positive)
    .map_err(predicate_error)
}
