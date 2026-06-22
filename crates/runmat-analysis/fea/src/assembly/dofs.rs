use runmat_analysis_core::{AnalysisModel, LoadKind};
use serde::{Deserialize, Serialize};

use crate::contracts::FeaRunError;

const MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE: &str =
    "moment loads require rotational-DOF structural elements";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralDofKind {
    Ux,
    Uy,
    Uz,
    Rx,
    Ry,
    Rz,
}

impl StructuralDofKind {
    pub const ORDER: [StructuralDofKind; 6] = [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
        StructuralDofKind::Rx,
        StructuralDofKind::Ry,
        StructuralDofKind::Rz,
    ];

    pub fn is_translational(self) -> bool {
        matches!(
            self,
            StructuralDofKind::Ux | StructuralDofKind::Uy | StructuralDofKind::Uz
        )
    }

    pub fn is_rotational(self) -> bool {
        matches!(
            self,
            StructuralDofKind::Rx | StructuralDofKind::Ry | StructuralDofKind::Rz
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralNodeDofSet {
    active: Vec<StructuralDofKind>,
}

impl StructuralNodeDofSet {
    pub fn translational() -> Self {
        Self::from_kinds([
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ])
    }

    pub fn translational_rotational() -> Self {
        Self::from_kinds(StructuralDofKind::ORDER)
    }

    pub fn from_kinds<const N: usize>(kinds: [StructuralDofKind; N]) -> Self {
        let mut active = Vec::with_capacity(N);
        for kind in StructuralDofKind::ORDER {
            if kinds.contains(&kind) {
                active.push(kind);
            }
        }
        Self { active }
    }

    pub fn contains(&self, kind: StructuralDofKind) -> bool {
        self.active.contains(&kind)
    }

    pub fn active(&self) -> &[StructuralDofKind] {
        &self.active
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralDofAddress {
    pub node_index: usize,
    pub kind: StructuralDofKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralDofLayout {
    node_sets: Vec<StructuralNodeDofSet>,
    rows: Vec<StructuralDofAddress>,
}

impl StructuralDofLayout {
    pub fn from_node_sets(node_sets: Vec<StructuralNodeDofSet>) -> Self {
        let rows = node_sets
            .iter()
            .enumerate()
            .flat_map(|(node_index, set)| {
                set.active()
                    .iter()
                    .copied()
                    .map(move |kind| StructuralDofAddress { node_index, kind })
            })
            .collect();
        Self { node_sets, rows }
    }

    pub fn legacy_translational_rows(dof_count: usize) -> Self {
        if dof_count == 0 {
            return Self::from_node_sets(Vec::new());
        }
        let node_count = dof_count.div_ceil(3);
        let mut kinds_by_node = vec![Vec::new(); node_count];
        for row in 0..dof_count {
            let kind = match row % 3 {
                0 => StructuralDofKind::Ux,
                1 => StructuralDofKind::Uy,
                _ => StructuralDofKind::Uz,
            };
            kinds_by_node[row / 3].push(kind);
        }
        let node_sets = kinds_by_node
            .into_iter()
            .map(|kinds| {
                let mut active = Vec::new();
                for kind in StructuralDofKind::ORDER {
                    if kinds.contains(&kind) {
                        active.push(kind);
                    }
                }
                StructuralNodeDofSet { active }
            })
            .collect();
        Self::from_node_sets(node_sets)
    }

    pub fn total_dof_count(&self) -> usize {
        self.rows.len()
    }

    pub fn node_count(&self) -> usize {
        self.node_sets.len()
    }

    pub fn rotation_node_count(&self) -> usize {
        self.node_sets
            .iter()
            .filter(|set| set.active().iter().any(|kind| kind.is_rotational()))
            .count()
    }

    pub fn translational_dof_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|address| address.kind.is_translational())
            .count()
    }

    pub fn rotational_dof_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|address| address.kind.is_rotational())
            .count()
    }

    pub fn index(&self, node_index: usize, kind: StructuralDofKind) -> Option<usize> {
        self.rows
            .iter()
            .position(|address| address.node_index == node_index && address.kind == kind)
    }

    pub fn address(&self, row: usize) -> Option<StructuralDofAddress> {
        self.rows.get(row).copied()
    }

    pub fn has_rotational_dofs(&self) -> bool {
        self.rotational_dof_count() > 0
    }
}

pub(crate) fn validate_moment_loads_against_layout(
    model: &AnalysisModel,
    layout: &StructuralDofLayout,
) -> Result<(), FeaRunError> {
    if layout.has_rotational_dofs() {
        return Ok(());
    }
    if let Some(load) = model
        .loads
        .iter()
        .find(|load| matches!(load.kind, LoadKind::Moment { .. }))
    {
        return Err(FeaRunError::InvalidModel(format!(
            "{}; load_id={} region_id={}",
            MOMENT_REQUIRES_ROTATIONAL_DOF_MESSAGE, load.load_id, load.region_id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_dof_layout_indexes_full_rotational_nodes_deterministically() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
            StructuralNodeDofSet::translational(),
        ]);

        assert_eq!(layout.total_dof_count(), 9);
        assert_eq!(layout.node_count(), 2);
        assert_eq!(layout.translational_dof_count(), 6);
        assert_eq!(layout.rotational_dof_count(), 3);
        assert_eq!(layout.rotation_node_count(), 1);
        assert_eq!(layout.index(0, StructuralDofKind::Ux), Some(0));
        assert_eq!(layout.index(0, StructuralDofKind::Rz), Some(5));
        assert_eq!(layout.index(1, StructuralDofKind::Ux), Some(6));
        assert_eq!(layout.index(1, StructuralDofKind::Rz), None);
        assert_eq!(
            layout.address(4),
            Some(StructuralDofAddress {
                node_index: 0,
                kind: StructuralDofKind::Ry,
            })
        );
    }

    #[test]
    fn legacy_translational_layout_preserves_existing_row_count() {
        let layout = StructuralDofLayout::legacy_translational_rows(5);

        assert_eq!(layout.total_dof_count(), 5);
        assert_eq!(layout.node_count(), 2);
        assert_eq!(layout.translational_dof_count(), 5);
        assert_eq!(layout.rotational_dof_count(), 0);
        assert_eq!(layout.index(1, StructuralDofKind::Ux), Some(3));
        assert_eq!(layout.index(1, StructuralDofKind::Uz), None);
    }
}
