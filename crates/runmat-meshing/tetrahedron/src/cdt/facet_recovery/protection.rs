//! Maintained protection membership. Preserve original oriented triangle order
//! for lineage, and materialize sorted face arrays only at canonical providers.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct FacetProtection {
    pub(super) triangles: Vec<DelaunayRecoveredFacetTriangle>,
    oriented: BTreeMap<[StableDigest; 3], usize>,
    pub(super) faces: BTreeSet<[StableDigest; 3]>,
}
impl FacetProtection {
    pub(super) fn insert(&mut self, triangle: DelaunayRecoveredFacetTriangle) {
        if self.insert_key(triangle.node_identities) {
            self.triangles.push(triangle);
        }
    }
    fn insert_key(&mut self, oriented: [StableDigest; 3]) -> bool {
        if self.oriented.contains_key(&oriented) {
            return false;
        }
        self.oriented.insert(oriented, 1);
        let mut face = oriented;
        face.sort_unstable();
        self.faces.insert(face);
        true
    }
    /// Removes a protected triangle (the inverse of [`Self::insert`]). Returns whether it was
    /// protected. The current facet's triangles are the newest, so the search runs from the end.
    pub(super) fn remove(&mut self, triangle: DelaunayRecoveredFacetTriangle) -> bool {
        let oriented = triangle.node_identities;
        let Some(count) = self.oriented.get_mut(&oriented) else {
            return false;
        };
        *count -= 1;
        if *count == 0 {
            self.oriented.remove(&oriented);
        }
        let mut face = oriented;
        face.sort_unstable();
        let [a, b, c] = face;
        if ![
            [a, b, c],
            [a, c, b],
            [b, a, c],
            [b, c, a],
            [c, a, b],
            [c, b, a],
        ]
        .iter()
        .any(|order| self.oriented.contains_key(order))
        {
            self.faces.remove(&face);
        }
        if let Some(index) = self.triangles.iter().rposition(|entry| *entry == triangle) {
            self.triangles.remove(index);
        }
        true
    }
    pub(super) fn ordered_faces(&self) -> Vec<[StableDigest; 3]> {
        self.faces.iter().copied().collect()
    }
    pub(super) fn apply_splits(
        &mut self,
        facets: &mut [DelaunayRecoveredFacet],
        splits: Vec<conforming_split::ProtectedFacetSplit>,
    ) {
        for split in &splits {
            if let Some(count) = self.oriented.get_mut(&split.source.node_identities) {
                *count -= 1;
                if *count == 0 {
                    self.oriented.remove(&split.source.node_identities);
                }
                let mut face = split.source.node_identities;
                face.sort_unstable();
                let [a, b, c] = face;
                if ![
                    [a, b, c],
                    [a, c, b],
                    [b, a, c],
                    [b, c, a],
                    [c, a, b],
                    [c, b, a],
                ]
                .iter()
                .any(|order| self.oriented.contains_key(order))
                {
                    self.faces.remove(&face);
                }
                for child in &split.children {
                    *self.oriented.entry(child.node_identities).or_default() += 1;
                    let mut face = child.node_identities;
                    face.sort_unstable();
                    self.faces.insert(face);
                }
            }
        }
        conforming_split::apply_protected_facet_splits(facets, &mut self.triangles, splits);
    }
}

#[cfg(test)]
mod tests;
