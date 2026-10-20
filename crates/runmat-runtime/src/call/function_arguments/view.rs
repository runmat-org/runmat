//! Borrowed logical argument windows over the original three segments.
use runmat_value::Value;
use std::ops::{Bound, Range, RangeBounds};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgumentRangeError;
impl std::fmt::Display for ArgumentRangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("argument range is out of bounds")
    }
}
impl std::error::Error for ArgumentRangeError {}
impl From<ArgumentRangeError> for crate::RuntimeError {
    fn from(source: ArgumentRangeError) -> Self {
        crate::runtime_error::semantic_error_with_source(
            "CallableArguments",
            "argument range is out of bounds",
            source,
        )
    }
}
pub(super) fn bounds(
    range: impl RangeBounds<usize>,
    len: usize,
) -> Result<Range<usize>, ArgumentRangeError> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.checked_add(1).ok_or(ArgumentRangeError)?,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n.checked_add(1).ok_or(ArgumentRangeError)?,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    if start > end || end > len {
        return Err(ArgumentRangeError);
    }
    Ok(start..end)
}
/// Copying a view copies only three borrowed slice descriptors. It owns no
/// Values, roots or storage authority; the original rooted owner must survive.
#[derive(Clone, Copy, Debug)]
pub struct FunctionArgumentView<'a> {
    segments: [&'a [Value]; 3],
}
impl<'a> FunctionArgumentView<'a> {
    pub(super) fn new(routing: &'a [Value], captures: &'a [Value], supplied: &'a [Value]) -> Self {
        Self {
            segments: [routing, captures, supplied],
        }
    }
    pub fn from_slice(values: &'a [Value]) -> Self {
        Self::new(&[], &[], values)
    }
    pub fn len(self) -> usize {
        self.segments.iter().map(|s| s.len()).sum()
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn get(self, mut index: usize) -> Option<&'a Value> {
        for values in self.segments {
            if index < values.len() {
                return values.get(index);
            }
            index -= values.len();
        }
        None
    }
    pub fn first(self) -> Option<&'a Value> {
        self.get(0)
    }
    pub fn last(self) -> Option<&'a Value> {
        self.len().checked_sub(1).and_then(|i| self.get(i))
    }
    /// Borrow the first logical input and remainder from the same segments.
    pub fn split_first(self) -> Option<(&'a Value, Self)> {
        let first = self.first()?;
        Some((first, self.range(1..).expect("nonempty argument view")))
    }
    /// Borrow the last logical input and prefix without collecting a slice.
    pub fn split_last(self) -> Option<(&'a Value, Self)> {
        let index = self.len().checked_sub(1)?;
        Some((
            self.get(index).expect("last argument index"),
            self.range(..index).expect("argument prefix"),
        ))
    }
    pub fn range(self, range: impl RangeBounds<usize>) -> Result<Self, ArgumentRangeError> {
        let range = bounds(range, self.len())?;
        let mut cursor = 0usize;
        let segments = self.segments.map(|values| {
            let start = range.start.saturating_sub(cursor).min(values.len());
            let end = range.end.saturating_sub(cursor).min(values.len());
            cursor += values.len();
            &values[start..end]
        });
        Ok(Self { segments })
    }
    pub fn iter(self) -> FunctionArgumentIter<'a> {
        FunctionArgumentIter {
            view: self,
            front: 0,
            back: self.len(),
        }
    }
    pub fn chunks_exact(self, width: usize) -> FunctionArgumentChunksExact<'a> {
        assert!(width != 0, "argument chunk width must not be zero");
        FunctionArgumentChunksExact {
            view: self,
            width,
            front: 0,
            back: self.len() / width,
        }
    }
}
impl std::ops::Index<usize> for FunctionArgumentView<'_> {
    type Output = Value;
    fn index(&self, index: usize) -> &Value {
        self.get(index).expect("argument index is out of bounds")
    }
}
impl<'a> IntoIterator for FunctionArgumentView<'a> {
    type Item = &'a Value;
    type IntoIter = FunctionArgumentIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &FunctionArgumentView<'a> {
    type Item = &'a Value;
    type IntoIter = FunctionArgumentIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
#[derive(Clone, Debug)]
pub struct FunctionArgumentIter<'a> {
    view: FunctionArgumentView<'a>,
    front: usize,
    back: usize,
}
impl<'a> Iterator for FunctionArgumentIter<'a> {
    type Item = &'a Value;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let index = self.front;
        self.front += 1;
        self.view.get(index)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}
impl DoubleEndedIterator for FunctionArgumentIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        self.view.get(self.back)
    }
}
impl ExactSizeIterator for FunctionArgumentIter<'_> {}
impl std::iter::FusedIterator for FunctionArgumentIter<'_> {}
#[derive(Clone, Debug)]
pub struct FunctionArgumentChunksExact<'a> {
    view: FunctionArgumentView<'a>,
    width: usize,
    front: usize,
    back: usize,
}
impl<'a> FunctionArgumentChunksExact<'a> {
    pub fn remainder(&self) -> FunctionArgumentView<'a> {
        let start = (self.view.len() / self.width) * self.width;
        self.view
            .range(start..)
            .expect("chunk remainder is in bounds")
    }
}
impl<'a> Iterator for FunctionArgumentChunksExact<'a> {
    type Item = FunctionArgumentView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        let start = self.front * self.width;
        self.front += 1;
        self.view.range(start..start + self.width).ok()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.back - self.front;
        (n, Some(n))
    }
}
impl DoubleEndedIterator for FunctionArgumentChunksExact<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        let start = self.back * self.width;
        self.view.range(start..start + self.width).ok()
    }
}
impl ExactSizeIterator for FunctionArgumentChunksExact<'_> {}
impl std::iter::FusedIterator for FunctionArgumentChunksExact<'_> {}
