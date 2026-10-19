//! Pure boundary incidence for edge-star proposals. This is not geometry admission.
//! Paired boundary faces identify ring edges directly; no all-node-pair search.
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct BoundaryEdgeStarCandidates<'a> {
    boundary: &'a BTreeSet<[u32; 3]>,
    rings: BTreeMap<[u32; 2], Vec<[u32; 2]>>,
}
impl<'a> BoundaryEdgeStarCandidates<'a> {
    pub(crate) fn new<E>(
        boundary: &'a BTreeSet<[u32; 3]>,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let mut apices = BTreeMap::<[u32; 2], BTreeSet<u32>>::new();
        for face in boundary {
            checkpoint()?;
            for slot in 0..3 {
                let mut edge = [face[(slot + 1) % 3], face[(slot + 2) % 3]];
                edge.sort_unstable();
                apices.entry(edge).or_default().insert(face[slot]);
            }
        }
        let mut rings = BTreeMap::<[u32; 2], Vec<[u32; 2]>>::new();
        for (edge, owners) in apices {
            for (index, left) in owners.iter().enumerate() {
                for right in owners.iter().skip(index + 1) {
                    checkpoint()?;
                    if edge.contains(left) || edge.contains(right) {
                        continue;
                    }
                    rings.entry([*left, *right]).or_default().push(edge);
                }
            }
        }
        Ok(Self { boundary, rings })
    }
    /// Every replacement with at least two cells appears in lexical target order.
    /// Geometry, cavity coverage, barriers and retained cells remain caller checks.
    pub(crate) fn targets(&self) -> impl Iterator<Item = [u32; 2]> + '_ {
        self.rings
            .iter()
            .filter_map(|(target, edges)| (edges.len() >= 2).then_some(*target))
    }
    pub(crate) fn tetrahedra<E>(
        &self,
        target: [u32; 2],
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<Vec<[u32; 4]>, E> {
        let mut key = target;
        key.sort_unstable();
        let Some(ring) = self.rings.get(&key) else {
            return Ok(Vec::new());
        };
        let mut created = Vec::with_capacity(ring.len());
        for edge in ring {
            checkpoint()?;
            created.push([target[0], target[1], edge[0], edge[1]]);
        }
        Ok(created)
    }
    pub(crate) fn boundary(&self) -> &BTreeSet<[u32; 3]> {
        self.boundary
    }
}

#[cfg(test)]
mod tests;
