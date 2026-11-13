//! Bounded validation of already-exact character backing and shape.

use super::*;
use crate::array::admission::ShapeProgress;
use std::num::NonZeroUsize;

pub struct CharArrayAdmission {
    data: ExactArrayStorage<Utf16CodeUnit>,
    shape: SharedShape,
    progress: ShapeProgress,
    complete: bool,
}

impl CharArrayAdmission {
    pub fn new(data: ExactArrayStorage<Utf16CodeUnit>, shape: SharedShape) -> Result<Self, String> {
        if !data.is_exact() || !shape.is_admitted() {
            return Err("Character payload and shape require exact admission".into());
        }
        Ok(Self {
            data,
            shape,
            progress: ShapeProgress::new(),
            complete: false,
        })
    }

    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<usize, String> {
        if self.complete {
            return Ok(0);
        }
        let completed = self
            .progress
            .advance_zero_aware(&self.shape, maximum_work, |_| {
                "Char data shape exceeds platform limits".into()
            })?;
        if self.progress.is_complete() {
            let expected = if self.shape.is_empty() {
                0
            } else {
                self.progress.elements
            };
            if expected != self.data.len() {
                return Err("Character payload length does not match admitted shape".into());
            }
            self.complete = true;
        }
        Ok(completed)
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    pub fn into_array(self) -> Result<CharArray, Self> {
        if !self.complete {
            return Err(self);
        }
        let (rows, cols) = shape_rows_cols(&self.shape);
        Ok(CharArray {
            data: self.data,
            shape: self.shape,
            rows,
            cols,
        })
    }
}

pub(super) fn character_len(shape: &[usize]) -> Option<usize> {
    let mut progress = ShapeProgress::new();
    while !progress.is_complete() {
        progress
            .advance_zero_aware(
                shape,
                NonZeroUsize::new(usize::MAX).expect("nonzero quantum"),
                |_| "Char data shape exceeds platform limits".into(),
            )
            .ok()?;
    }
    Some(if shape.is_empty() {
        0
    } else {
        progress.elements
    })
}
