use thiserror::Error;

use crate::assembly::dofs::{StructuralDofKind, StructuralDofLayout};

const VECTOR_COMPONENT_COUNT: usize = 3;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructuralNodalSnapshot {
    pub translations: Vec<f64>,
    pub rotations: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum StructuralNodalProjectionError {
    #[error("structural result state length differs from its canonical DOF layout")]
    StateLengthMismatch,
}

pub(crate) fn project_structural_nodal_snapshot(
    layout: &StructuralDofLayout,
    state: &[f64],
) -> Result<StructuralNodalSnapshot, StructuralNodalProjectionError> {
    if state.len() != layout.total_dof_count() {
        return Err(StructuralNodalProjectionError::StateLengthMismatch);
    }
    let value_count = layout
        .node_count()
        .checked_mul(VECTOR_COMPONENT_COUNT)
        .ok_or(StructuralNodalProjectionError::StateLengthMismatch)?;
    let mut translations = vec![0.0; value_count];
    let mut rotations = vec![0.0; value_count];
    for (row, &value) in state.iter().take(layout.node_dof_count()).enumerate() {
        let address = layout
            .address(row)
            .ok_or(StructuralNodalProjectionError::StateLengthMismatch)?;
        let target = address
            .node_index
            .checked_mul(VECTOR_COMPONENT_COUNT)
            .and_then(|offset| offset.checked_add(component_index(address.kind)))
            .ok_or(StructuralNodalProjectionError::StateLengthMismatch)?;
        let values = if address.kind.is_translational() {
            &mut translations
        } else {
            &mut rotations
        };
        let slot = values
            .get_mut(target)
            .ok_or(StructuralNodalProjectionError::StateLengthMismatch)?;
        *slot = value;
    }
    Ok(StructuralNodalSnapshot {
        translations,
        rotations,
    })
}

pub(crate) fn project_structural_nodal_snapshots(
    layout: &StructuralDofLayout,
    states: &[Vec<f64>],
) -> Result<Vec<StructuralNodalSnapshot>, StructuralNodalProjectionError> {
    states
        .iter()
        .map(|state| project_structural_nodal_snapshot(layout, state))
        .collect()
}

pub(crate) fn nodal_vector_shape(layout: &StructuralDofLayout) -> Vec<usize> {
    vec![layout.node_count(), VECTOR_COMPONENT_COUNT]
}

const fn component_index(kind: StructuralDofKind) -> usize {
    match kind {
        StructuralDofKind::Ux | StructuralDofKind::Rx => 0,
        StructuralDofKind::Uy | StructuralDofKind::Ry => 1,
        StructuralDofKind::Uz | StructuralDofKind::Rz => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::dofs::StructuralNodeDofSet;

    #[test]
    fn projection_separates_interleaved_translation_rotation_and_reference_point_rows() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational(),
            StructuralNodeDofSet::translational_rotational(),
        ])
        .with_reference_points(1);
        let state = (1..=layout.total_dof_count())
            .map(|value| value as f64)
            .collect::<Vec<_>>();

        let projected = project_structural_nodal_snapshot(&layout, &state).unwrap();

        assert_eq!(projected.translations, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(projected.rotations, vec![0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);
        assert_eq!(nodal_vector_shape(&layout), vec![2, 3]);
    }

    #[test]
    fn projection_rejects_truncated_or_padded_solver_state() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
        ]);
        assert_eq!(
            project_structural_nodal_snapshot(&layout, &[0.0; 5]),
            Err(StructuralNodalProjectionError::StateLengthMismatch)
        );
        assert_eq!(
            project_structural_nodal_snapshot(&layout, &[0.0; 7]),
            Err(StructuralNodalProjectionError::StateLengthMismatch)
        );
    }
}
