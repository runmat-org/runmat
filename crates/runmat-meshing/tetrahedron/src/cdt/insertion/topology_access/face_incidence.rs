//! Shared exact face-owner queries. Mutable topology already maintains this
//! inventory across cavity commit/rollback; immutable topology uses its smallest
//! vertex star. Neither path changes geometric or empty-sphere admission.
use super::{TetrahedralTopology, TopologyMutation};

pub(super) fn scan<E>(
    topology: &(impl TetrahedralTopology + ?Sized),
    face: [u32; 3],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Vec<u32>, E> {
    checkpoint()?;
    let start = face
        .into_iter()
        .min_by_key(|vertex| topology.vertex_star_len(*vertex))
        .expect("three face vertices");
    let mut owners = Vec::new();
    for index in topology.vertex_star(start) {
        checkpoint()?;
        if topology.tetrahedron(index).is_some_and(|cell| {
            face.iter()
                .all(|vertex| cell.vertex_indices.contains(vertex))
        }) {
            owners.push(index);
        }
    }
    Ok(owners)
}

pub(super) fn maintained<E>(
    topology: &TopologyMutation,
    face: [u32; 3],
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Vec<u32>, E> {
    checkpoint()?;
    let mut owners = Vec::new();
    for (index, _) in topology.face_incidence(face) {
        checkpoint()?;
        owners.push(*index);
    }
    owners.sort_unstable();
    Ok(owners)
}

#[cfg(test)]
mod tests;
