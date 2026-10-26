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

pub(crate) use shape::ShapeProgress;
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

    fn finish_synchronously(
        mut self,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<Self, String> {
        while !self.is_complete() {
            self.advance(synchronous_quantum(), &overflow)?;
        }
        Ok(self)
    }
}

pub(in crate::array) struct ValidatedShape {
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    elements: usize,
}

impl ValidatedShape {
    pub(in crate::array) fn elements(&self) -> usize {
        self.elements
    }

    pub(in crate::array) fn into_components(self) -> (Vec<usize>, usize, usize) {
        (self.shape, self.rows, self.cols)
    }
}

pub(in crate::array) fn admit_shape_for_resize(
    shape: Vec<usize>,
    overflow: impl Fn(&[usize]) -> String,
) -> Result<ValidatedShape, String> {
    let shape = ShapeAdmission::new(shape).finish_synchronously(overflow)?;
    let (rows, cols) = shape.dimensions();
    Ok(ValidatedShape {
        elements: shape.elements(),
        shape: shape.shape,
        rows,
        cols,
    })
}

pub(in crate::array) fn admit_shape(
    shape: Vec<usize>,
    storage_len: usize,
    overflow: impl Fn(&[usize]) -> String,
    mismatch: impl Fn(&[usize], usize) -> String,
) -> Result<ValidatedShape, String> {
    let shape = admit_shape_for_resize(shape, overflow)?;
    if shape.elements != storage_len {
        return Err(mismatch(&shape.shape, shape.elements));
    }
    Ok(shape)
}

pub(in crate::array) fn shape_dimensions(shape: &[usize]) -> (usize, usize) {
    ShapeProgress::dimensions(shape)
}

fn synchronous_quantum() -> NonZeroUsize {
    NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero")
}
