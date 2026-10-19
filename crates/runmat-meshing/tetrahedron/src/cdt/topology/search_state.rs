//! Bounded cycle detection inside a transaction with fixed nodes and region inventory.
//! This fingerprint is not a geometry identity or a topology-admission certificate.
#[cfg(test)]
use super::DelaunayVolumeTopology;
use sha2::{Digest, Sha256};

/// Commutative cycle hint for fixed nodes/regions, not a cryptographic identity
/// or admission proof. Cell hashes permit constant-size add/remove maintenance.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::cdt) struct ConnectivitySearchState {
    lanes: [u64; 4],
    count: u64,
}
impl ConnectivitySearchState {
    pub(in crate::cdt) fn insert(&mut self, vertices: [u32; 4]) {
        let value = cell_hash(vertices);
        for (lane, value) in self.lanes.iter_mut().zip(value) {
            *lane = lane.wrapping_add(value);
        }
        self.count = self.count.checked_add(1).expect("bounded cell inventory");
    }
    pub(in crate::cdt) fn remove(&mut self, vertices: [u32; 4]) {
        let value = cell_hash(vertices);
        for (lane, value) in self.lanes.iter_mut().zip(value) {
            *lane = lane.wrapping_sub(value);
        }
        self.count = self.count.checked_sub(1).expect("retained cycle inventory");
    }
    pub(in crate::cdt) fn fingerprint(self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"runmat-meshing-connectivity-cycle/v2\0");
        digest.update(self.count.to_be_bytes());
        for lane in self.lanes {
            digest.update(lane.to_be_bytes());
        }
        digest.finalize().into()
    }
}
fn cell_hash(mut vertices: [u32; 4]) -> [u64; 4] {
    vertices.sort_unstable();
    let mut digest = Sha256::new();
    digest.update(b"runmat-meshing-connectivity-cycle-cell/v2\0");
    for vertex in vertices {
        digest.update(vertex.to_be_bytes());
    }
    let bytes = digest.finalize();
    std::array::from_fn(|lane| {
        u64::from_be_bytes(
            bytes[lane * 8..lane * 8 + 8]
                .try_into()
                .expect("digest word"),
        )
    })
}

#[cfg(test)]
pub(in crate::cdt) fn connectivity_search_fingerprint(
    topology: &DelaunayVolumeTopology,
) -> [u8; 32] {
    let tetrahedra = topology
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.vertex_indices)
        .collect::<Vec<_>>();
    connectivity_search_fingerprint_from_keys(tetrahedra)
}

#[cfg(test)]
pub(in crate::cdt) fn connectivity_search_fingerprint_from_keys(
    mut tetrahedra: Vec<[u32; 4]>,
) -> [u8; 32] {
    for vertices in &mut tetrahedra {
        vertices.sort_unstable();
    }
    tetrahedra.sort_unstable();
    connectivity_search_fingerprint_from_ordered_keys(tetrahedra)
}

/// Stream canonical corner keys in lexicographic order. The caller owns that
/// ordering invariant; this cycle hint does not validate or admit topology.
#[cfg(test)]
pub(in crate::cdt) fn connectivity_search_fingerprint_from_ordered_keys(
    tetrahedra: impl IntoIterator<Item = [u32; 4]>,
) -> [u8; 32] {
    connectivity_search_fingerprint_with_checkpoint(tetrahedra, &mut || {
        Ok::<_, std::convert::Infallible>(())
    })
    .unwrap_or_else(|never| match never {})
}

/// The inventory traversal itself is governed work. Rejection returns no partial
/// fingerprint and does not change the caller's connectivity index.
#[cfg(test)]
pub(in crate::cdt) fn connectivity_search_fingerprint_with_checkpoint<E>(
    tetrahedra: impl IntoIterator<Item = [u32; 4]>,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<[u8; 32], E> {
    let mut state = ConnectivitySearchState::default();
    for vertices in tetrahedra {
        checkpoint()?;
        state.insert(vertices);
    }
    Ok(state.fingerprint())
}
