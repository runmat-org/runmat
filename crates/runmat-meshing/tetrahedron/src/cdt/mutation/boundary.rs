//! Incremental boundary index, maintained with the shared face-incidence index.
use super::TopologyMutation;
use std::collections::BTreeSet;

impl TopologyMutation {
    pub(in crate::cdt) fn face_incidence(&self, mut face: [u32; 3]) -> &[(u32, usize)] {
        face.sort_unstable();
        self.face_uses.get(&face).map_or(&[], Vec::as_slice)
    }

    pub(in crate::cdt) fn face_adjacency(&self, face: [u32; 3]) -> Option<[(u32, usize); 2]> {
        let uses = self.face_uses.get(&face)?;
        let [left, right] = uses.as_slice() else {
            return None;
        };
        Some([*left, *right])
    }

    pub(in crate::cdt) fn boundary_face_owner(&self, face: [u32; 3]) -> Option<(u32, usize)> {
        self.boundary_faces.get(&face).copied()
    }
    pub(super) fn sync_boundary_face(&mut self, face: [u32; 3]) {
        if let Some(uses) = self.face_uses.get(&face).filter(|uses| uses.len() == 1) {
            self.boundary_faces.insert(face, uses[0]);
        } else {
            self.boundary_faces.remove(&face);
        }
    }

    pub(in crate::cdt) fn active_tetrahedron_count(&self) -> usize {
        self.tetrahedron_indices.len()
    }

    /// Boundary triangles sharing an edge, found through local vertex stars.
    pub(in crate::cdt) fn boundary_neighbors(&self, face: [u32; 3]) -> BTreeSet<[u32; 3]> {
        self.boundary_neighbors_with_checkpoint(face, &mut || Ok::<_, std::convert::Infallible>(()))
            .expect("infallible checkpoint")
    }

    pub(in crate::cdt) fn boundary_neighbors_with_checkpoint<E>(
        &self,
        face: [u32; 3],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<BTreeSet<[u32; 3]>, E> {
        let mut result = BTreeSet::new();
        for edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            let (start, other) = if self.vertex_stars[edge[0] as usize].len()
                <= self.vertex_stars[edge[1] as usize].len()
            {
                (edge[0], edge[1])
            } else {
                (edge[1], edge[0])
            };
            for index in self.vertex_stars[start as usize].iter().copied() {
                checkpoint()?;
                let tet = self
                    .tetrahedron(index)
                    .expect("vertex star references active tetrahedron");
                if !tet.vertex_indices.contains(&other) {
                    continue;
                }
                for opposite in 0..4 {
                    if tet.neighbors[opposite].is_some() {
                        continue;
                    }
                    let candidate = Self::face(tet.vertex_indices, opposite);
                    if candidate != face && edge.iter().all(|vertex| candidate.contains(vertex)) {
                        result.insert(candidate);
                    }
                }
            }
        }
        Ok(result)
    }
}
