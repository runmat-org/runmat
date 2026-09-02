use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;

use super::storage::{error, MutableTetrahedron, TopologyMutation, TopologyMutationError};

type ExternalBoundary = BTreeMap<[u32; 3], Option<(u32, usize)>>;
type ReplacementNeighbors = (Vec<[Option<u32>; 4]>, Vec<(u32, usize, u32)>);

struct ReplacementPlan {
    removed: BTreeSet<u32>,
    created: Vec<u32>,
    tetrahedra: Vec<(u32, MutableTetrahedron)>,
    external_neighbors: Vec<(u32, usize, u32)>,
    final_len: usize,
}

impl TopologyMutation {
    pub(in crate::cdt) fn replace_cavity(
        &mut self,
        removed: &BTreeSet<u32>,
        created: Vec<([u32; 4], Option<PersistentEntityId>)>,
    ) -> Result<Vec<u32>, TopologyMutationError> {
        let plan = self.prepare_cavity_replacement(removed, created)?;
        let created = plan.created.clone();
        self.commit_cavity_replacement(plan);
        Ok(created)
    }

    fn prepare_cavity_replacement(
        &self,
        removed: &BTreeSet<u32>,
        created: Vec<([u32; 4], Option<PersistentEntityId>)>,
    ) -> Result<ReplacementPlan, TopologyMutationError> {
        if removed.is_empty() || created.is_empty() {
            return Err(error(
                "cavity replacement requires removed and created tetrahedra",
            ));
        }
        for tetrahedron in removed {
            if self.tetrahedron(*tetrahedron).is_none() {
                return Err(error("cavity replacement names an inactive tetrahedron"));
            }
        }

        let external = self.external_boundary(removed)?;
        let expected_interface = external
            .iter()
            .filter_map(|(face, outside)| outside.is_some().then_some(*face))
            .collect::<BTreeSet<_>>();
        let created_len = created.len();
        let oriented = created
            .into_iter()
            .map(|(vertices, region)| {
                self.oriented_vertices(vertices)
                    .map(|vertices| (vertices, region))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let created_boundary = boundary_faces(oriented.iter().map(|(vertices, _)| *vertices))?;
        if !expected_interface.is_subset(&created_boundary) {
            return Err(error(
                "cavity replacement does not reproduce its exact retained interface",
            ));
        }
        self.reject_duplicate_tetrahedra(removed, &oriented)?;

        let mut available = self.free_tetrahedra.clone();
        available.extend(removed.iter().copied());
        let appended = created_len.saturating_sub(available.len());
        let final_len = self
            .tetrahedra
            .len()
            .checked_add(appended)
            .ok_or_else(|| error("mutable topology exhausted its tetrahedron index space"))?;
        if final_len > u32::MAX as usize {
            return Err(error(
                "mutable topology exhausted its tetrahedron index space",
            ));
        }
        let mut next_append = self.tetrahedra.len() as u32;
        let assigned = (0..oriented.len())
            .map(|_| {
                available.pop_first().unwrap_or_else(|| {
                    let index = next_append;
                    next_append += 1;
                    index
                })
            })
            .collect::<Vec<_>>();
        let (neighbors, external_neighbors) =
            replacement_neighbors(&oriented, &assigned, &external)?;
        let tetrahedra = oriented
            .into_iter()
            .zip(assigned.iter().copied())
            .zip(neighbors)
            .map(|(((vertices, region_id), index), neighbors)| {
                (
                    index,
                    MutableTetrahedron {
                        vertex_indices: vertices,
                        neighbors,
                        region_id,
                    },
                )
            })
            .collect();
        Ok(ReplacementPlan {
            removed: removed.clone(),
            created: assigned,
            tetrahedra,
            external_neighbors,
            final_len,
        })
    }

    fn reject_duplicate_tetrahedra(
        &self,
        removed: &BTreeSet<u32>,
        created: &[([u32; 4], Option<PersistentEntityId>)],
    ) -> Result<(), TopologyMutationError> {
        let mut created_keys = BTreeSet::new();
        for (vertices, _) in created {
            let key = sorted_tetrahedron(*vertices);
            let duplicates_retained = self
                .tetrahedron_indices
                .get(&key)
                .is_some_and(|index| !removed.contains(index));
            if duplicates_retained || !created_keys.insert(key) {
                return Err(error(
                    "cavity replacement would create a duplicate tetrahedron",
                ));
            }
        }
        Ok(())
    }

    fn external_boundary(
        &self,
        removed: &BTreeSet<u32>,
    ) -> Result<ExternalBoundary, TopologyMutationError> {
        let mut boundary = BTreeMap::new();
        for tetrahedron_index in removed {
            let tetrahedron = self
                .tetrahedron(*tetrahedron_index)
                .ok_or_else(|| error("cavity boundary references an inactive tetrahedron"))?;
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| removed.contains(&neighbor))
                {
                    continue;
                }
                let face = Self::face(tetrahedron.vertex_indices, opposite);
                let outside = tetrahedron.neighbors[opposite]
                    .map(|neighbor| {
                        let neighbor_tetrahedron = self
                            .tetrahedron(neighbor)
                            .ok_or_else(|| error("cavity boundary neighbor is inactive"))?;
                        let neighbor_opposite = neighbor_tetrahedron
                            .neighbors
                            .iter()
                            .position(|candidate| *candidate == Some(*tetrahedron_index))
                            .ok_or_else(|| error("cavity boundary neighbor is not reciprocal"))?;
                        Ok((neighbor, neighbor_opposite))
                    })
                    .transpose()?;
                if boundary.insert(face, outside).is_some() {
                    return Err(error("removed cavity has a nonmanifold boundary face"));
                }
            }
        }
        Ok(boundary)
    }

    fn commit_cavity_replacement(&mut self, plan: ReplacementPlan) {
        for tetrahedron_index in &plan.removed {
            let tetrahedron = self.tetrahedra[*tetrahedron_index as usize]
                .take()
                .expect("prepared cavity tetrahedron must remain active");
            self.tetrahedron_indices
                .remove(&sorted_tetrahedron(tetrahedron.vertex_indices));
            self.remove_face_uses(*tetrahedron_index, tetrahedron.vertex_indices);
            for vertex in tetrahedron.vertex_indices {
                self.vertex_stars[vertex as usize].remove(tetrahedron_index);
            }
            self.free_tetrahedra.insert(*tetrahedron_index);
        }
        self.tetrahedra.resize(plan.final_len, None);
        for (index, tetrahedron) in plan.tetrahedra {
            self.tetrahedron_indices
                .insert(sorted_tetrahedron(tetrahedron.vertex_indices), index);
            for vertex in tetrahedron.vertex_indices {
                self.vertex_stars[vertex as usize].insert(index);
            }
            self.insert_face_uses(index, tetrahedron.vertex_indices);
            self.tetrahedra[index as usize] = Some(tetrahedron);
            self.free_tetrahedra.remove(&index);
        }
        for (neighbor, opposite, replacement) in plan.external_neighbors {
            self.tetrahedra[neighbor as usize]
                .as_mut()
                .expect("prepared external neighbor must remain active")
                .neighbors[opposite] = Some(replacement);
        }
    }
}

fn replacement_neighbors(
    tetrahedra: &[([u32; 4], Option<PersistentEntityId>)],
    assigned: &[u32],
    external: &ExternalBoundary,
) -> Result<ReplacementNeighbors, TopologyMutationError> {
    let mut uses = BTreeMap::<[u32; 3], Vec<(usize, usize)>>::new();
    for (created, (vertices, _)) in tetrahedra.iter().enumerate() {
        for opposite in 0..4 {
            uses.entry(TopologyMutation::face(*vertices, opposite))
                .or_default()
                .push((created, opposite));
        }
    }
    let mut neighbors = vec![[None; 4]; tetrahedra.len()];
    let mut external_neighbors = Vec::new();
    for (face, face_uses) in uses {
        match face_uses.as_slice() {
            [(created, opposite)] => {
                if let Some(Some((neighbor, neighbor_opposite))) = external.get(&face) {
                    neighbors[*created][*opposite] = Some(*neighbor);
                    external_neighbors.push((*neighbor, *neighbor_opposite, assigned[*created]));
                }
            }
            [(left, left_opposite), (right, right_opposite)] => {
                neighbors[*left][*left_opposite] = Some(assigned[*right]);
                neighbors[*right][*right_opposite] = Some(assigned[*left]);
            }
            _ => return Err(error("new cavity contains a nonmanifold face")),
        }
    }
    Ok((neighbors, external_neighbors))
}

fn sorted_tetrahedron(mut vertices: [u32; 4]) -> [u32; 4] {
    vertices.sort_unstable();
    vertices
}

fn boundary_faces(
    tetrahedra: impl Iterator<Item = [u32; 4]>,
) -> Result<BTreeSet<[u32; 3]>, TopologyMutationError> {
    let mut counts = BTreeMap::<[u32; 3], usize>::new();
    for tetrahedron in tetrahedra {
        for opposite in 0..4 {
            let face = TopologyMutation::face(tetrahedron, opposite);
            let count = counts.entry(face).or_default();
            *count += 1;
            if *count > 2 {
                return Err(error("new cavity contains a nonmanifold face"));
            }
        }
    }
    Ok(counts
        .into_iter()
        .filter_map(|(face, count)| (count == 1).then_some(face))
        .collect())
}
