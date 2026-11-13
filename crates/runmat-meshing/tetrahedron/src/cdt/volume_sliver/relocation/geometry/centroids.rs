use super::*;
pub(in crate::cdt::volume_sliver) fn neighbor_centroid(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
) -> Result<[f64; 3], DelaunayVolumeSliverError> {
    let mut neighbors = topology
        .ordered_vertex_star((source_index) as u32)
        .iter()
        .flat_map(|tetrahedron| topology.cell(*tetrahedron).vertex_indices)
        .filter(|vertex| *vertex as usize != source_index)
        .map(|vertex| topology.node(vertex).identity)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|identity| {
            let index = topology.node_index(identity).ok_or_else(|| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    "vertex-star neighbor is absent from topology",
                )
            })?;
            Ok(topology.node(index).coordinates_m.canonical_value())
        })
        .collect::<Result<Vec<_>, DelaunayVolumeSliverError>>()?;
    if neighbors.len() < 4 {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "interior relocation requires at least four distinct star neighbors",
        ));
    }
    let count = neighbors.len() as f64;
    let sum = neighbors.drain(..).fold([0.0; 3], |mut sum, point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        sum
    });
    Ok(sum.map(|value| value / count))
}

/// Returns the volume-weighted centroid of the complete tetrahedral cavity around a vertex.
///
/// This is the conservative three-dimensional Laplacian move used by production tetrahedral
/// optimizers: weighting tetrahedron centroids by physical volume avoids bias toward a densely
/// subdivided side of the star while preserving the cavity's center of mass.
pub(in crate::cdt::volume_sliver) fn volume_weighted_star_centroid(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source_index: usize,
) -> Result<[f64; 3], DelaunayVolumeSliverError> {
    let mut weighted_sum = [0.0; 3];
    let mut total_volume = 0.0;
    for tetrahedron_index in &topology.ordered_vertex_star((source_index) as u32) {
        let points = topology
            .cell(*tetrahedron_index)
            .vertex_indices
            .map(|vertex| topology.node(vertex).coordinates_m.canonical_value());
        let volume = tetrahedron_volume(points);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior relocation cavity contains a non-positive tetrahedron",
            ));
        }
        for axis in 0..3 {
            let centroid = points.iter().map(|point| point[axis]).sum::<f64>() / 4.0;
            weighted_sum[axis] += centroid * volume;
        }
        total_volume += volume;
    }
    if !total_volume.is_finite() || total_volume <= 0.0 {
        return Err(error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "interior relocation cavity has no positive finite volume",
        ));
    }
    Ok(weighted_sum.map(|value| value / total_volume))
}
