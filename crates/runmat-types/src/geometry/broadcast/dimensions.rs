//! Borrowed concrete implicit expansion, before allocating output metadata.
use super::ConcreteBroadcastError;

#[derive(Clone, Copy, Debug)]
pub struct ConcreteBroadcastDimensions<'a> {
    left: &'a [usize],
    right: &'a [usize],
    rank: usize,
}

impl<'a> ConcreteBroadcastDimensions<'a> {
    pub fn new(left: &'a [usize], right: &'a [usize]) -> Result<Self, ConcreteBroadcastError> {
        let dimensions = Self {
            left,
            right,
            rank: left.len().max(right.len()),
        };
        for axis in 0..dimensions.rank {
            let a = dimensions.input_extent(left, axis);
            let b = dimensions.input_extent(right, axis);
            if a != b && a != 1 && b != 1 {
                return Err(ConcreteBroadcastError::IncompatibleAxis {
                    axis: axis + 1,
                    left: a,
                    right: b,
                });
            }
        }
        Ok(dimensions)
    }

    pub fn rank(self) -> usize {
        self.rank
    }

    pub fn dimension(self, axis: usize) -> Option<usize> {
        if axis >= self.rank {
            return None;
        }
        let a = self.input_extent(self.left, axis);
        let b = self.input_extent(self.right, axis);
        Some(if a == 1 { b } else { a })
    }

    pub fn element_count(self) -> Option<usize> {
        if (0..self.rank).any(|axis| self.dimension(axis) == Some(0)) {
            return Some(0);
        }
        (0..self.rank).try_fold(1usize, |count, axis| {
            count.checked_mul(
                self.dimension(axis)
                    .expect("axis belongs to validated rank"),
            )
        })
    }

    fn input_extent(self, shape: &[usize], axis: usize) -> usize {
        if self.rank >= 2 && shape.len() == 1 {
            if axis == 1 {
                shape[0]
            } else {
                1
            }
        } else {
            shape.get(axis).copied().unwrap_or(1)
        }
    }
}
