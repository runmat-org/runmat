//! Bounded admission of owned dense numeric array storage.
//!
//! These cursors own the candidate storage and shape while checking shape
//! arithmetic a bounded number of dimensions at a time. Completed parts are
//! sealed: callers can consume them into the corresponding array, but cannot
//! manufacture unchecked parts.

mod complex;
mod numeric;

pub use complex::{ComplexTensorAdmission, ValidatedComplexTensorParts};
pub use numeric::{NumericTensorAdmission, ValidatedNumericTensorParts};

use std::num::NonZeroUsize;

struct ShapeAdmission {
    shape: Vec<usize>,
    cursor: usize,
    elements: usize,
    complete: bool,
}

impl ShapeAdmission {
    fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            cursor: 0,
            elements: 1,
            complete: false,
        }
    }

    fn advance(
        &mut self,
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        let mut completed = 0;
        while self.cursor < self.shape.len() && completed < maximum_work.get() {
            self.elements = self
                .elements
                .checked_mul(self.shape[self.cursor])
                .ok_or_else(|| overflow(&self.shape))?;
            self.cursor += 1;
            completed += 1;
        }
        if self.cursor == self.shape.len() && !self.complete && completed < maximum_work.get() {
            self.complete = true;
            completed += 1;
        }
        Ok(completed)
    }

    fn is_complete(&self) -> bool {
        self.complete
    }

    fn dimensions(&self) -> (usize, usize) {
        if self.shape.len() >= 2 {
            (self.shape[0], self.shape[1])
        } else if self.shape.len() == 1 {
            (1, self.shape[0])
        } else {
            (0, 0)
        }
    }
}

fn synchronous_quantum() -> NonZeroUsize {
    NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero")
}
