use super::*;
pub(in crate::cdt::volume_sliver) fn point_is_in_vertex_star_kernel(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    source_index: usize,
    point: [f64; 3],
) -> Result<bool, DelaunayVolumeSliverError> {
    let source = topology.node(source_index as u32);
    let probe = DelaunayVolumeNode::explicit(replacement_identity(source.identity), point);
    for tetrahedron in star {
        let face = topology
            .cell(*tetrahedron)
            .vertex_indices
            .into_iter()
            .filter(|vertex| *vertex as usize != source_index)
            .map(|vertex| topology.node(vertex))
            .collect::<Vec<_>>();
        let face: [DelaunayVolumeNode; 3] = face
            .try_into()
            .expect("vertex-star tetrahedron has one source vertex");
        let expected =
            orient_volume_nodes([face[0], face[1], face[2], source]).map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("vertex-star smoothing predicate failed: {failure:?}"),
                )
            })?;
        let observed =
            orient_volume_nodes([face[0], face[1], face[2], probe]).map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("vertex-star candidate predicate failed: {failure:?}"),
                )
            })?;
        if observed != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
fn relocated_star_minimum_jacobian_with_metrics(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    source_index: usize,
    point: [f64; 3],
    metrics: &dyn Fn(u32) -> runmat_meshing_size::metric::MetricTensor3,
) -> f64 {
    star.iter()
        .filter_map(|tetrahedron| {
            let points = topology.cell(*tetrahedron).vertex_indices.map(|vertex| {
                if vertex as usize == source_index {
                    point
                } else {
                    topology.node(vertex).coordinates_m.canonical_value()
                }
            });
            crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(points, metrics(*tetrahedron))
                .unwrap()
        })
        .min_by(f64::total_cmp)
        .unwrap_or(0.0)
}

pub(in crate::cdt::volume_sliver) fn sliver_height_target(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    source: &DelaunayTetrahedronQuality,
    source_identity: StableDigest,
    minimum_scaled_jacobian: f64,
) -> Result<Option<[f64; 3]>, DelaunayVolumeSliverError> {
    let source_index = topology.node_index(source_identity).ok_or_else(|| {
        error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver source node is absent from canonical topology",
        )
    })?;
    let opposite = source
        .node_identities
        .iter()
        .copied()
        .filter(|identity| *identity != source_identity)
        .map(|identity| {
            topology
                .node_index(identity)
                .map(|index| topology.node(index).coordinates_m.canonical_value())
                .ok_or_else(|| {
                    error(
                        DelaunayVolumeSliverErrorKind::InvalidTopology,
                        "sliver opposite face is absent from canonical topology",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [first, second, third]: [[f64; 3]; 3] = opposite.try_into().map_err(|_| {
        error(
            DelaunayVolumeSliverErrorKind::InvalidTopology,
            "sliver source identity is absent from its tetrahedron",
        )
    })?;
    let subtract = |left: [f64; 3], right: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|axis| left[axis] - right[axis])
    };
    let first_edge = subtract(second, first);
    let second_edge = subtract(third, first);
    let normal = [
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    ];
    let normal_length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !normal_length.is_finite() || normal_length == 0.0 {
        return Ok(None);
    }
    let unit_normal = normal.map(|value| value / normal_length);
    let coordinates = topology.node(source_index).coordinates_m.canonical_value();
    let signed_height = subtract(coordinates, first)
        .into_iter()
        .zip(unit_normal)
        .map(|(value, normal)| value * normal)
        .sum::<f64>();
    if !signed_height.is_finite() || signed_height == 0.0 {
        return Ok(None);
    }
    let face_scale = [first_edge, second_edge, subtract(third, second)]
        .into_iter()
        .map(|edge| edge.iter().map(|value| value * value).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max);
    let desired_height = (face_scale * minimum_scaled_jacobian).max(signed_height.abs() * 2.0);
    Ok(Some(std::array::from_fn(|axis| {
        coordinates[axis]
            + unit_normal[axis] * signed_height.signum() * (desired_height - signed_height.abs())
    })))
}

#[cfg(test)]
pub(in crate::cdt::volume_sliver) fn relocated_star_minimum_jacobian(
    topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
    star: &[u32],
    source_index: usize,
    point: [f64; 3],
    metric: runmat_meshing_size::metric::MetricTensor3,
) -> f64 {
    relocated_star_minimum_jacobian_with_metrics(topology, star, source_index, point, &|_| metric)
}
