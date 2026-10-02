//! Resumable stable ordering of caller-owned associated values.

use std::cmp::Ordering;
use std::collections::TryReserveError;

#[derive(Debug)]
pub(crate) struct SortResourceError(pub(crate) TryReserveError);

// One uninterrupted library sort is bounded; the execution service owns
// scheduling policy and the work quantum.
pub(crate) const RUN_LEN: usize = 256;
const MAX_MERGE_ELEMENTS: u64 = 4096;

#[derive(Clone, Copy)]
pub(crate) enum SortProgress {
    Pending(u64),
    Complete(u64),
}

enum Phase {
    Runs,
    Merge,
    Complete,
}

pub(crate) struct StableSortWork<'a, T, F> {
    values: &'a mut Vec<T>,
    compare: F,
    merged: Vec<T>,
    phase: Phase,
    cursor: usize,
    width: usize,
    left: usize,
    middle: usize,
    right: usize,
    end: usize,
}

impl<'a, T: Copy, F: FnMut(&T, &T) -> Ordering> StableSortWork<'a, T, F> {
    pub(crate) fn new(values: &'a mut Vec<T>, compare: F) -> Self {
        Self {
            values,
            compare,
            merged: Vec::new(),
            phase: Phase::Runs,
            cursor: 0,
            width: RUN_LEN,
            left: 0,
            middle: 0,
            right: 0,
            end: 0,
        }
    }

    pub(crate) fn advance(&mut self) -> Result<SortProgress, SortResourceError> {
        let mut work = 0_u64;
        while work < MAX_MERGE_ELEMENTS {
            match self.phase {
                Phase::Runs => {
                    if self.cursor >= self.values.len() {
                        self.phase = if self.values.len() <= RUN_LEN {
                            Phase::Complete
                        } else {
                            Phase::Merge
                        };
                        continue;
                    }
                    let end = self.cursor.saturating_add(RUN_LEN).min(self.values.len());
                    self.values[self.cursor..end].sort_by(&mut self.compare);
                    work += (end - self.cursor) as u64;
                    self.cursor = end;
                    // A bounded library sort is one indivisible unit. Return
                    // between runs so the invocation budget can observe each
                    // completed unit before starting the next run or merge.
                    return Ok(SortProgress::Pending(work));
                }
                Phase::Merge => {
                    if self.merged.is_empty() && self.cursor >= self.values.len() {
                        self.merged
                            .try_reserve_exact(self.values.len())
                            .map_err(SortResourceError)?;
                        self.cursor = 0;
                        self.begin_segment();
                        continue;
                    }
                    if self.left == self.middle && self.right == self.end {
                        self.cursor = self.end;
                        if self.cursor == self.values.len() {
                            *self.values = std::mem::take(&mut self.merged);
                            self.width = self.width.saturating_mul(2);
                            if self.width >= self.values.len() {
                                self.phase = Phase::Complete;
                            } else {
                                self.cursor = self.values.len();
                            }
                        } else {
                            self.begin_segment();
                        }
                        continue;
                    }
                    let select_left = if self.left == self.middle {
                        false
                    } else if self.right == self.end {
                        true
                    } else {
                        (self.compare)(&self.values[self.left], &self.values[self.right])
                            != Ordering::Greater
                    };
                    if select_left {
                        self.merged.push(self.values[self.left]);
                        self.left += 1;
                    } else {
                        self.merged.push(self.values[self.right]);
                        self.right += 1;
                    }
                    work += 1;
                }
                Phase::Complete => return Ok(SortProgress::Complete(work)),
            }
        }
        Ok(SortProgress::Pending(work))
    }

    fn begin_segment(&mut self) {
        self.left = self.cursor;
        self.middle = self
            .cursor
            .saturating_add(self.width)
            .min(self.values.len());
        self.right = self.middle;
        self.end = self
            .middle
            .saturating_add(self.width)
            .min(self.values.len());
    }
}

#[cfg(any(feature = "plot-core", test))]
pub(crate) fn stable_sort_by_sync<T: Copy>(
    values: &mut Vec<T>,
    compare: impl FnMut(&T, &T) -> Ordering,
) -> Result<(), SortResourceError> {
    let mut work = StableSortWork::new(values, compare);
    loop {
        match work.advance()? {
            SortProgress::Pending(_) => {}
            SortProgress::Complete(_) => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn associated_values_keep_stable_ties_across_runs_and_merge_passes() {
        for count in [0, 1, RUN_LEN, RUN_LEN + 1, 3 * RUN_LEN + 9] {
            let mut actual = (0..count)
                .rev()
                .map(|index| (index, index % 7))
                .collect::<Vec<_>>();
            let mut expected = actual.clone();
            expected.sort_by_key(|item| item.1);
            stable_sort_by_sync(&mut actual, |left, right| left.1.cmp(&right.1)).unwrap();
            assert_eq!(actual, expected, "count {count}");
        }
    }

    #[test]
    fn caller_owned_float_order_keeps_index_association_and_nan_policy() {
        let mut eigenpairs = (0..RUN_LEN + 5)
            .map(|index| {
                let eigenvalue = match index % 4 {
                    0 => f64::NAN,
                    1 => -0.0,
                    2 => 0.0,
                    _ => 2.0,
                };
                (index, eigenvalue)
            })
            .collect::<Vec<_>>();
        let mut expected = eigenpairs.clone();
        let comparator =
            |left: &(usize, f64), right: &(usize, f64)| match (left.1.is_nan(), right.1.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal),
            };
        expected.sort_by(comparator);
        stable_sort_by_sync(&mut eigenpairs, comparator).unwrap();
        assert_eq!(
            eigenpairs.iter().map(|item| item.0).collect::<Vec<_>>(),
            expected.iter().map(|item| item.0).collect::<Vec<_>>()
        );
        assert!(eigenpairs.iter().all(|item| item.1.to_bits()
            == expected
                .iter()
                .find(|expected| expected.0 == item.0)
                .unwrap()
                .1
                .to_bits()));
    }
}
