use std::num::NonZeroUsize;

pub(crate) struct ShapeProgress {
    cursor: usize,
    pub(crate) elements: usize,
    complete: bool,
    overflowed: bool,
}

impl ShapeProgress {
    pub(crate) fn new() -> Self {
        Self {
            cursor: 0,
            elements: 1,
            complete: false,
            overflowed: false,
        }
    }

    pub(crate) fn advance(
        &mut self,
        shape: &[usize],
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        self.advance_with_zero_policy(shape, maximum_work, overflow, false)
    }

    /// Character geometry permits a later zero extent to make a virtual
    /// shape empty. Keep numeric admission's existing overflow policy intact.
    pub(crate) fn advance_zero_aware(
        &mut self,
        shape: &[usize],
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        self.advance_with_zero_policy(shape, maximum_work, overflow, true)
    }

    fn advance_with_zero_policy(
        &mut self,
        shape: &[usize],
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
        zero_aware: bool,
    ) -> Result<usize, String> {
        let mut completed = 0;
        while self.cursor < shape.len() && completed < maximum_work.get() {
            let extent = shape[self.cursor];
            if zero_aware && extent == 0 {
                self.elements = 0;
                self.overflowed = false;
            } else if !self.overflowed {
                match self.elements.checked_mul(extent) {
                    Some(elements) => self.elements = elements,
                    None if zero_aware => self.overflowed = true,
                    None => return Err(overflow(shape)),
                }
            }
            self.cursor += 1;
            completed += 1;
        }
        if self.cursor == shape.len() && !self.complete && completed < maximum_work.get() {
            if self.overflowed {
                return Err(overflow(shape));
            }
            self.complete = true;
            completed += 1;
        }
        Ok(completed)
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.complete
    }

    pub(super) fn dimensions(shape: &[usize]) -> (usize, usize) {
        match shape {
            [] => (0, 0),
            [length] => (1, *length),
            [rows, columns, ..] => (*rows, *columns),
        }
    }
}

/// Borrowed adapter for callers that need canonical shape admission without
/// copying either the shape or its associated payload storage.
pub(crate) struct BorrowedShapeAdmission<'a> {
    shape: &'a [usize],
    progress: ShapeProgress,
}

impl<'a> BorrowedShapeAdmission<'a> {
    pub(crate) fn new(shape: &'a [usize]) -> Self {
        Self {
            shape,
            progress: ShapeProgress::new(),
        }
    }

    pub(crate) fn advance(
        &mut self,
        maximum_work: NonZeroUsize,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        self.progress.advance(self.shape, maximum_work, overflow)
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.progress.is_complete()
    }

    pub(crate) fn elements(&self) -> usize {
        self.progress.elements
    }

    pub(crate) fn finish_synchronously(
        mut self,
        overflow: impl Fn(&[usize]) -> String,
    ) -> Result<usize, String> {
        let quantum = NonZeroUsize::new(usize::MAX).expect("usize::MAX is nonzero");
        while !self.is_complete() {
            self.advance(quantum, &overflow)?;
        }
        Ok(self.elements())
    }

    #[cfg(test)]
    fn shape(&self) -> &'a [usize] {
        self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::super::ShapeAdmission;
    use super::*;

    #[test]
    fn borrowed_cursor_is_incremental_and_keeps_the_original_shape() {
        let shape = vec![2, 3, 4];
        let original = shape.as_ptr();
        let mut admission = BorrowedShapeAdmission::new(&shape);
        for expected in [2, 6, 24] {
            assert_eq!(
                admission
                    .advance(NonZeroUsize::new(1).unwrap(), |_| "overflow".into())
                    .unwrap(),
                1
            );
            assert_eq!(admission.elements(), expected);
            assert_eq!(admission.shape().as_ptr(), original);
        }
        assert!(!admission.is_complete());
        admission
            .advance(NonZeroUsize::new(1).unwrap(), |_| "overflow".into())
            .unwrap();
        assert!(admission.is_complete());
    }

    #[test]
    fn empty_zero_extent_and_overflow_follow_one_cursor() {
        assert_eq!(
            BorrowedShapeAdmission::new(&[])
                .finish_synchronously(|_| "overflow".into())
                .unwrap(),
            1
        );
        assert_eq!(
            BorrowedShapeAdmission::new(&[7, 0, usize::MAX])
                .finish_synchronously(|_| "overflow".into())
                .unwrap(),
            0
        );
        assert_eq!(
            BorrowedShapeAdmission::new(&[usize::MAX, 2])
                .finish_synchronously(|_| "exact overflow".into())
                .unwrap_err(),
            "exact overflow"
        );
    }

    #[test]
    fn owned_and_borrowed_admissions_share_progress_and_overflow_order() {
        let source = vec![2, 3, 4];
        let mut owned = ShapeAdmission::new(source.clone());
        let mut borrowed = BorrowedShapeAdmission::new(&source);
        let quantum = NonZeroUsize::new(1).unwrap();
        while !owned.is_complete() {
            assert_eq!(
                owned.advance(quantum, |_| "owned overflow".into()).unwrap(),
                borrowed
                    .advance(quantum, |_| "borrowed overflow".into())
                    .unwrap()
            );
            assert_eq!(owned.elements(), borrowed.elements());
        }
        assert!(borrowed.is_complete());

        let mut owned = ShapeAdmission::new(vec![usize::MAX, 2]);
        let source = [usize::MAX, 2];
        let mut borrowed = BorrowedShapeAdmission::new(&source);
        assert_eq!(
            owned.advance(quantum, |_| "overflow".into()).unwrap(),
            borrowed.advance(quantum, |_| "overflow".into()).unwrap()
        );
        assert_eq!(
            owned.advance(quantum, |_| "overflow".into()).unwrap_err(),
            borrowed
                .advance(quantum, |_| "overflow".into())
                .unwrap_err()
        );
    }
}
