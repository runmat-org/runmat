//! Counted candidate storage and union-find; no numerical basis allocation or row cloning.
use super::{invalid, reserved, CanonicalStructuralConstraintRow, ConstraintPlanError, DofKey};

#[derive(Clone, Copy, Default)]
pub(super) struct Component {
    pub dofs: usize,
    pub rows: usize,
}

pub(super) struct Components {
    pub components: Vec<Component>,
    pub row_components: Vec<usize>,
    pub candidate_count: usize,
    pub dof_count: usize,
}

impl Components {
    pub(super) fn derive(
        rows: &[CanonicalStructuralConstraintRow],
    ) -> Result<Self, ConstraintPlanError> {
        // Count-only reservation is available to callers before graph construction itself;
        // check the aggregate slot arithmetic here as well, before the first allocation.
        super::constraint_rank_graph_workspace_bytes(rows)?;
        let candidate_count = rows.iter().try_fold(0usize, |total, row| {
            if row.terms.is_empty() {
                return Err(invalid(
                    "constraint rank classification requires nonempty rows",
                ));
            }
            if !row.right_hand_side.is_finite()
                || row.terms.iter().any(|term| !term.coefficient.is_finite())
            {
                return Err(invalid(
                    "constraint rank classification requires finite equations",
                ));
            }
            total
                .checked_add(row.terms.len())
                .ok_or_else(|| invalid("constraint rank candidate count overflow"))
        })?;
        let mut candidates = reserved::<DofKey>(candidate_count)?;
        for row in rows {
            candidates.extend(row.terms.iter().map(|term| (term.owner, term.dof)));
        }
        // Unstable sorting has no auxiliary heap buffer. Keep the counted candidate capacity
        // after deduplication; the profile includes this full reservation.
        candidates.sort_unstable();
        candidates.dedup();
        let dof_count = candidates.len();
        let mut parents = reserved(dof_count)?;
        parents.extend(0..dof_count);
        for row in rows {
            let first = key_index(&candidates, (row.terms[0].owner, row.terms[0].dof))?;
            for term in &row.terms[1..] {
                let index = key_index(&candidates, (term.owner, term.dof))?;
                let left = find(&mut parents, first);
                let right = find(&mut parents, index);
                if left != right {
                    parents[left.max(right)] = left.min(right);
                }
            }
        }
        let mut root_components = reserved(dof_count)?;
        root_components.resize(dof_count, usize::MAX);
        let mut components = reserved::<Component>(dof_count)?;
        for index in 0..dof_count {
            let root = find(&mut parents, index);
            if root_components[root] == usize::MAX {
                root_components[root] = components.len();
                components.push(Component::default());
            }
            components[root_components[root]].dofs += 1;
        }
        let mut row_components = reserved(rows.len())?;
        for row in rows {
            let first = key_index(&candidates, (row.terms[0].owner, row.terms[0].dof))?;
            let component = root_components[find(&mut parents, first)];
            components[component].rows += 1;
            row_components.push(component);
        }
        Ok(Self {
            components,
            row_components,
            candidate_count,
            dof_count,
        })
    }
}

fn key_index(keys: &[DofKey], key: DofKey) -> Result<usize, ConstraintPlanError> {
    keys.binary_search(&key)
        .map_err(|_| invalid("constraint rank graph omits a row DOF"))
}

fn find(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}
