//! A strict subset of homogeneous constraints, never an approximation to an affine map.
use super::algebra::{AffineConstraintReduction, ConstraintAlgebraError};

pub(crate) struct HomogeneousTranslationSelection {
    axis: usize,
    rows: Vec<Option<usize>>,
}

impl HomogeneousTranslationSelection {
    pub fn from_reduction(
        reduction: &AffineConstraintReduction,
        node_count: usize,
    ) -> Result<Option<Self>, ConstraintAlgebraError> {
        reduction.validate()?;
        if reduction.full_dof_count != node_count.checked_mul(3).unwrap_or(0)
            || reduction.offset.iter().any(|&value| value != 0.0)
        {
            return Ok(None);
        }
        let mut selected = Vec::with_capacity(reduction.full_dof_count);
        for full in 0..reduction.full_dof_count {
            let range =
                reduction.recovery_row_offsets[full]..reduction.recovery_row_offsets[full + 1];
            match range.len() {
                0 => selected.push(None),
                1 if reduction.recovery_coefficients[range.start] == 1.0 => {
                    selected.push(Some(reduction.recovery_column_indices[range.start]))
                }
                _ => return Ok(None),
            }
        }
        Ok(Self::from_rows(selected))
    }

    pub fn from_fixed_rows(fixed: &[bool], node_count: usize) -> Option<Self> {
        if fixed.len() != node_count.checked_mul(3)? {
            return None;
        }
        let mut row = 0;
        Self::from_rows(
            fixed
                .iter()
                .map(|&fixed| {
                    if fixed {
                        None
                    } else {
                        let index = row;
                        row += 1;
                        Some(index)
                    }
                })
                .collect(),
        )
    }

    fn from_rows(selected: Vec<Option<usize>>) -> Option<Self> {
        let axis = selected.iter().position(Option::is_some)? % 3;
        if selected
            .iter()
            .enumerate()
            .any(|(full, row)| row.is_some() && full % 3 != axis)
        {
            return None;
        }
        // Reject repeated columns and holes, including a malformed decoded reduction.
        let count = selected.iter().flatten().count();
        let mut seen = vec![false; count];
        for &row in selected.iter().flatten() {
            let entry = seen.get_mut(row)?;
            if *entry {
                return None;
            }
            *entry = true;
        }
        Some(Self {
            axis,
            rows: selected.into_iter().skip(axis).step_by(3).collect(),
        })
    }

    pub fn axis(&self) -> usize {
        self.axis
    }
    pub fn into_rows(self) -> Vec<Option<usize>> {
        self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_translation_axis_preserves_the_original_free_order() {
        for axis in 0..3 {
            let mut fixed = vec![true; 9];
            fixed[3 + axis] = false;
            fixed[6 + axis] = false;
            let selected = HomogeneousTranslationSelection::from_fixed_rows(&fixed, 3).unwrap();
            assert_eq!(selected.axis(), axis);
            assert_eq!(selected.into_rows(), vec![None, Some(0), Some(1)]);
            fixed[3 + (axis + 1) % 3] = false;
            assert!(HomogeneousTranslationSelection::from_fixed_rows(&fixed, 3).is_none());
        }
        assert!(HomogeneousTranslationSelection::from_fixed_rows(&[true; 6], 2).is_none());
        assert!(HomogeneousTranslationSelection::from_fixed_rows(&[false; 5], 2).is_none());
    }

    #[test]
    fn homogeneous_selection_does_not_approximate_ties_or_prescribed_offsets() {
        let mut reduction = AffineConstraintReduction {
            full_dof_count: 6,
            reduced_dof_count: 1,
            free_dof_indices: vec![3],
            pivot_dof_indices: vec![0, 1, 2, 4, 5],
            recovery_row_offsets: vec![0, 0, 0, 0, 1, 1, 1],
            recovery_column_indices: vec![0],
            recovery_coefficients: vec![1.],
            offset: vec![0.; 6],
        };
        let selected = HomogeneousTranslationSelection::from_reduction(&reduction, 2)
            .unwrap()
            .unwrap();
        assert_eq!(selected.into_rows(), vec![None, Some(0)]);
        reduction.offset[0] = 1.;
        assert!(
            HomogeneousTranslationSelection::from_reduction(&reduction, 2)
                .unwrap()
                .is_none()
        );
        reduction.offset[0] = 0.;
        reduction.recovery_row_offsets = vec![0, 1, 1, 1, 2, 2, 2];
        reduction.recovery_column_indices = vec![0, 0];
        reduction.recovery_coefficients = vec![0.5, 1.];
        assert!(
            HomogeneousTranslationSelection::from_reduction(&reduction, 2)
                .unwrap()
                .is_none()
        );
    }
}
