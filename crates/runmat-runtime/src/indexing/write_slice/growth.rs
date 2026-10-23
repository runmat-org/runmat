//! Borrowed column-major relocation geometry; payload construction lives once
//! at the canonical final numeric output owner, never an intermediate COW.
use super::{checked_assignment_element_count, mex, RuntimeError};
use crate::context::RuntimeContext;
use crate::indexing::plan::effective_index_shape_in_context;
use runmat_value::SharedShape;

pub(super) enum Relocation<'a> {
    Linear,
    Rebased {
        old_shape: SharedShape,
        new_shape: &'a [usize],
    },
}
impl<'a> Relocation<'a> {
    pub(super) fn new(
        context: Option<&RuntimeContext>,
        old_shape: &[usize],
        old_len: usize,
        new_shape: &'a [usize],
    ) -> Result<Self, RuntimeError> {
        if checked_assignment_element_count(new_shape)? < old_len {
            return Err(mex(
                "InvalidAssignmentPlan",
                "slice assignment plan cannot shrink storage",
            ));
        }
        let old_shape = effective_index_shape_in_context(context, old_shape, new_shape.len())?;
        if old_shape.iter().zip(new_shape).any(|(old, new)| new < old) {
            return Err(mex(
                "InvalidAssignmentPlan",
                "slice assignment plan cannot shrink an addressed dimension",
            ));
        }
        let rebase = old_shape
            .iter()
            .zip(new_shape)
            .take(old_shape.len().saturating_sub(1))
            .any(|(old, new)| old != new);
        Ok(if rebase {
            Self::Rebased {
                old_shape,
                new_shape,
            }
        } else {
            Self::Linear
        })
    }
    pub(super) fn position(&self, mut old_index: usize) -> usize {
        let Self::Rebased {
            old_shape,
            new_shape,
        } = self
        else {
            return old_index;
        };
        let mut new_index = 0;
        let mut new_stride = 1;
        for (&old_extent, &new_extent) in old_shape.iter().zip(*new_shape) {
            let coordinate = old_index % old_extent;
            old_index /= old_extent;
            new_index += coordinate * new_stride;
            new_stride *= new_extent;
        }
        new_index
    }
}
