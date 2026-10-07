//! Bounded admission of owned dense numeric array storage.
//!
//! The tensor cursors own candidate storage and shape; borrowed storage
//! validators retain their existing owners. Both routes use the same bounded
//! shape arithmetic. Completed owned parts are sealed: callers can consume
//! them into the corresponding array, but cannot manufacture unchecked parts.

mod complex;
mod numeric;
mod shape;

pub use complex::{ComplexTensorAdmission, ValidatedComplexTensorParts};
pub use numeric::{NumericTensorAdmission, ValidatedNumericTensorParts};

pub(crate) use shape::BorrowedShapeAdmission;

use shape::ShapeProgress;
use std::num::NonZeroUsize;

struct ShapeAdmission {
    shape: Vec<usize>,
    progress: ShapeProgress,
}

impl ShapeAdmission {
    fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            progress: ShapeProgress::new(),
        }
    }

    fn advance(
        &mut self,
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        self.progress.advance(&self.shape, maximum_work, overflow)
    }

    fn is_complete(&self) -> bool {
        self.progress.is_complete()
    }

    fn elements(&self) -> usize {
        self.progress.elements
    }

    fn dimensions(&self) -> (usize, usize) {
        ShapeProgress::dimensions(&self.shape)
    }
}

fn synchronous_quantum() -> NonZeroUsize {
    NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero")
}
