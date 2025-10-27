//! Broadcasting utilities shared across builtin implementations.
//!
//! The helpers in this module mirror MATLAB's implicit expansion rules.
//! They operate purely on shape metadata and element strides, allowing
//! callers to iterate over broadcasted indices without materialising
//! expanded tensors.

/// Compute column-major strides for a given shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim.max(1));
    }
    strides
}

/// Broadcast plan describing how two tensors can be implicitly expanded.
#[derive(Debug, Clone)]
pub struct BroadcastPlan {
    output_shape: Vec<usize>,
    len: usize,
    advance_a: Vec<usize>,
    advance_b: Vec<usize>,
}

impl BroadcastPlan {
    /// Construct a broadcast plan for two shapes, returning an error when they
    /// cannot be implicitly expanded under MATLAB rules.
    pub fn new(shape_a: &[usize], shape_b: &[usize]) -> Result<Self, String> {
        let ndims = shape_a.len().max(shape_b.len());

        let mut ext_a = Vec::with_capacity(ndims);
        ext_a.extend(std::iter::repeat(1).take(ndims.saturating_sub(shape_a.len())));
        ext_a.extend_from_slice(shape_a);

        let mut ext_b = Vec::with_capacity(ndims);
        ext_b.extend(std::iter::repeat(1).take(ndims.saturating_sub(shape_b.len())));
        ext_b.extend_from_slice(shape_b);

        let mut output_shape = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let da = ext_a[i];
            let db = ext_b[i];
            if da == db {
                output_shape.push(da);
            } else if da == 1 {
                output_shape.push(db);
            } else if db == 1 {
                output_shape.push(da);
            } else {
                return Err(format!(
                    "broadcast: non-singleton dimension mismatch (dimension {}: {} vs {})",
                    i + 1,
                    da,
                    db
                ));
            }
        }

        let len = output_shape.iter().copied().product();
        let strides_a = compute_strides(&ext_a);
        let strides_b = compute_strides(&ext_b);

        let advance_a = ext_a
            .iter()
            .enumerate()
            .map(|(dim, &size)| if size <= 1 { 0 } else { strides_a[dim] })
            .collect::<Vec<_>>();
        let advance_b = ext_b
            .iter()
            .enumerate()
            .map(|(dim, &size)| if size <= 1 { 0 } else { strides_b[dim] })
            .collect::<Vec<_>>();

        Ok(Self {
            output_shape,
            len,
            advance_a,
            advance_b,
        })
    }

    /// Total number of elements produced by the broadcast.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Output shape after broadcasting both operands.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Iterator yielding `(output_index, index_a, index_b)` triples for each element.
    pub fn iter(&self) -> BroadcastIter<'_> {
        BroadcastIter {
            plan: self,
            offset: 0,
            index_a: 0,
            index_b: 0,
            coords: vec![0usize; self.output_shape.len()],
        }
    }
}

/// Iterator over broadcast indices.
pub struct BroadcastIter<'a> {
    plan: &'a BroadcastPlan,
    offset: usize,
    index_a: usize,
    index_b: usize,
    coords: Vec<usize>,
}

impl<'a> Iterator for BroadcastIter<'a> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.plan.len {
            return None;
        }
        let current = (self.offset, self.index_a, self.index_b);
        self.offset += 1;
        if self.offset == self.plan.len {
            return Some(current);
        }
        for dim in 0..self.plan.output_shape.len() {
            if self.plan.output_shape[dim] == 0 {
                continue;
            }
            self.coords[dim] += 1;
            if self.coords[dim] < self.plan.output_shape[dim] {
                self.index_a += self.plan.advance_a[dim];
                self.index_b += self.plan.advance_b[dim];
                break;
            }
            self.coords[dim] = 0;
            let rewind = self.plan.output_shape[dim].saturating_sub(1);
            let rewind_a = self.plan.advance_a[dim] * rewind;
            let rewind_b = self.plan.advance_b[dim] * rewind;
            if rewind_a != 0 {
                self.index_a = self.index_a.saturating_sub(rewind_a);
            }
            if rewind_b != 0 {
                self.index_b = self.index_b.saturating_sub(rewind_b);
            }
        }
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_same_shape() {
        let plan = BroadcastPlan::new(&[2, 3], &[2, 3]).unwrap();
        assert_eq!(plan.output_shape(), &[2, 3]);
        assert_eq!(plan.len(), 6);
        let indices: Vec<(usize, usize, usize)> = plan.iter().collect();
        assert_eq!(
            indices,
            vec![
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                (4, 4, 4),
                (5, 5, 5)
            ]
        );
    }

    #[test]
    fn broadcast_scalar_expansion() {
        let plan = BroadcastPlan::new(&[1, 3], &[1, 1]).unwrap();
        assert_eq!(plan.output_shape(), &[1, 3]);
        assert_eq!(plan.len(), 3);
        let indices: Vec<(usize, usize, usize)> = plan.iter().collect();
        assert_eq!(indices, vec![(0, 0, 0), (1, 1, 0), (2, 2, 0)]);
    }

    #[test]
    fn broadcast_mismatch_errors() {
        let err = BroadcastPlan::new(&[2, 3], &[4, 3]).unwrap_err();
        assert!(err.contains("dimension 1"));
    }

    #[test]
    fn broadcast_zero_sized_dimension() {
        let plan = BroadcastPlan::new(&[0, 3], &[1, 3]).unwrap();
        assert_eq!(plan.output_shape(), &[0, 3]);
        assert_eq!(plan.len(), 0);
        assert_eq!(plan.iter().next(), None);
    }
}
