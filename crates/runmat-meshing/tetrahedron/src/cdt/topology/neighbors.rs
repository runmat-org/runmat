//! Compact, deterministic face incidence for canonical positive tetrahedra.
//! Sorting one contiguous inventory avoids one tree node and heap vector per face.
use super::*;

pub(in crate::cdt) fn connect(
    nodes: &[DelaunayVolumeNode],
    cells: &mut [DelaunayVolumeTetrahedron],
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayTopologyError> {
    let capacity = cells.len().checked_mul(4).ok_or_else(|| {
        error(
            DelaunayTopologyErrorKind::ResourceLimit,
            "face inventory overflow",
        )
    })?;
    let mut faces = Vec::with_capacity(capacity);
    for (index, cell) in cells.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        for opposite in 0..4 {
            let mut face = [0; 3];
            let mut cursor = 0;
            for (slot, vertex) in cell.vertex_indices.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            faces.push((face, index as u32, opposite as u8));
        }
    }
    faces.sort_unstable();
    checkpoint(0, options, cancellation)?;
    let mut start = 0;
    let mut groups = 0;
    while start < faces.len() {
        checkpoint(groups, options, cancellation)?;
        groups += 1;
        let end = start + faces[start..].partition_point(|entry| entry.0 == faces[start].0);
        match &faces[start..end] {
            [_] => {}
            [(face, left, left_opposite), (_, right, right_opposite)] => {
                let left = *left as usize;
                let right = *right as usize;
                let left_opposite = *left_opposite as usize;
                let right_opposite = *right_opposite as usize;
                let points = face.map(|vertex| nodes[vertex as usize]);
                let sign = |cell: usize, opposite: usize| {
                    orient_materialized_volume_nodes([
                        points[0],
                        points[1],
                        points[2],
                        nodes[cells[cell].vertex_indices[opposite] as usize],
                    ])
                    .map_err(predicate_error)
                };
                if sign(left, left_opposite)? == sign(right, right_opposite)? {
                    return Err(error(
                        DelaunayTopologyErrorKind::InvalidTetrahedron,
                        format!(
                            "tetrahedra {left} and {right} overlap across shared face {face:?}"
                        ),
                    ));
                }
                cells[left].neighbors[left_opposite] = Some(right as u32);
                cells[right].neighbors[right_opposite] = Some(left as u32);
            }
            _ => {
                return Err(error(
                    DelaunayTopologyErrorKind::NonManifoldFace,
                    format!("more than two tetrahedra share face {:?}", faces[start].0),
                ))
            }
        }
        start = end;
    }
    Ok(())
}
