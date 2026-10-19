use super::{
    dofs::{StructuralDofKind, StructuralDofLayout},
    StructuralAssemblyError,
};

const RELATIVE_VALUE_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone)]
struct DirectConstraintAssignment {
    constraint_id: String,
    value: f64,
}

/// Canonical owner for direct structural DOF assignments.
///
/// Solid and explicit topology both lower into this state before it mutates the assembled right
/// hand side. That makes one solver row one authority even when two authored selections reach the
/// same attached node.
pub(super) struct DirectConstraintAssignments<'a> {
    layout: &'a StructuralDofLayout,
    assignments: Vec<Option<DirectConstraintAssignment>>,
}

impl<'a> DirectConstraintAssignments<'a> {
    pub fn new(layout: &'a StructuralDofLayout) -> Self {
        Self {
            layout,
            assignments: vec![None; layout.total_dof_count()],
        }
    }

    pub fn assign(
        &mut self,
        constraint_id: &str,
        node_index: usize,
        kind: StructuralDofKind,
        value: f64,
    ) -> Result<(), StructuralAssemblyError> {
        let Some(row) = self.layout.index(node_index, kind) else {
            return Ok(());
        };
        if !value.is_finite() {
            return Err(StructuralAssemblyError::InvalidDirectConstraintValue {
                constraint_id: constraint_id.into(),
            });
        }
        if let Some(existing) = &self.assignments[row] {
            let scale = existing.value.abs().max(value.abs()).max(1.0);
            if (existing.value - value).abs() > scale * RELATIVE_VALUE_TOLERANCE {
                return Err(StructuralAssemblyError::ConflictingDirectConstraints {
                    first_constraint_id: existing.constraint_id.clone(),
                    second_constraint_id: constraint_id.into(),
                    node_index,
                    dof: kind,
                });
            }
            return Ok(());
        }
        self.assignments[row] = Some(DirectConstraintAssignment {
            constraint_id: constraint_id.into(),
            value,
        });
        Ok(())
    }

    pub fn apply(&self, rhs: &mut [f64]) -> Result<Vec<bool>, StructuralAssemblyError> {
        if rhs.len() != self.assignments.len() {
            return Err(StructuralAssemblyError::InvalidDirectConstraintState {
                expected: self.assignments.len(),
                actual: rhs.len(),
            });
        }
        let mut constrained = vec![false; rhs.len()];
        for (row, assignment) in self.assignments.iter().enumerate() {
            if let Some(assignment) = assignment {
                constrained[row] = true;
                rhs[row] = assignment.value;
            }
        }
        Ok(constrained)
    }

    pub fn constrained_dof_count(&self) -> usize {
        self.assignments.iter().flatten().count()
    }

    pub fn rotational_constraint_count(&self) -> usize {
        self.assignments
            .iter()
            .enumerate()
            .filter(|(_, assignment)| assignment.is_some())
            .filter(|(row, _)| {
                self.layout
                    .address(*row)
                    .is_some_and(|address| address.kind.is_rotational())
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::dofs::StructuralNodeDofSet;

    #[test]
    fn repeated_equal_assignment_is_redundant_but_conflicting_value_fails() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
        ]);
        let mut assignments = DirectConstraintAssignments::new(&layout);
        assignments
            .assign("solid", 0, StructuralDofKind::Ux, 0.25)
            .unwrap();
        assignments
            .assign("explicit_equal", 0, StructuralDofKind::Ux, 0.25)
            .unwrap();
        assert!(matches!(
            assignments.assign("explicit_conflict", 0, StructuralDofKind::Ux, 0.5),
            Err(StructuralAssemblyError::ConflictingDirectConstraints {
                first_constraint_id,
                second_constraint_id,
                node_index: 0,
                dof: StructuralDofKind::Ux,
            }) if first_constraint_id == "solid" && second_constraint_id == "explicit_conflict"
        ));
    }
}
