//! Block compressed sparse rows for vector degrees of freedom (`b × b` dense blocks per node
//! pair). SpMV streams one column index per block instead of per entry, which cuts index traffic
//! by `b²` for elasticity (`b = 3`).

use rayon::prelude::*;

use crate::csr::{CsrMatrix, CsrRef};
use crate::error::{SolverError, SolverResult};
use crate::operator::LinearOperator;
use crate::par;
use crate::scalar::Scalar;

const NODES_PER_TASK: usize = 512;
/// Largest block size `from_csr` merges (rows of one block row held in a fixed array).
const MAX_MERGE: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct BsrMatrix<T> {
    block: usize,
    node_rows: usize,
    node_cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    /// Row-major `b × b` blocks, contiguous per stored block.
    values: Vec<T>,
}

impl<T: Scalar> BsrMatrix<T> {
    /// Heap bytes of the index and value arrays.
    pub fn heap_bytes(&self) -> u64 {
        let word = std::mem::size_of::<usize>() as u64;
        (self.row_ptr.len() + self.col_idx.len()) as u64 * word
            + self.values.len() as u64 * std::mem::size_of::<T>() as u64
    }
    /// Converts a CSR matrix whose dimensions are multiples of `block`; missing entries inside a
    /// touched block are stored as zeros.
    pub fn from_csr(matrix: CsrRef<'_, T>, block: usize) -> SolverResult<Self> {
        if block == 0
            || !matrix.nrows().is_multiple_of(block)
            || !matrix.ncols().is_multiple_of(block)
        {
            return Err(SolverError::Dimension(format!(
                "block size {block} does not divide the matrix"
            )));
        }
        if block > MAX_MERGE {
            return Err(SolverError::Unsupported(format!(
                "block size {block} exceeds {MAX_MERGE}"
            )));
        }
        let node_rows = matrix.nrows() / block;
        let node_cols = matrix.ncols() / block;
        // Union of the block columns of a node's `block` sorted rows, in increasing order.
        let block_columns = |node: usize, emit: &mut dyn FnMut(usize)| {
            let mut rows: [&[usize]; MAX_MERGE] = [&[]; MAX_MERGE];
            for (r, row) in rows.iter_mut().enumerate().take(block) {
                *row = matrix.row(node * block + r).0;
            }
            loop {
                let Some(next) = rows[..block]
                    .iter()
                    .filter_map(|row| row.first())
                    .map(|&c| c / block)
                    .min()
                else {
                    break;
                };
                for row in &mut rows[..block] {
                    while row.first().is_some_and(|&c| c / block == next) {
                        *row = &row[1..];
                    }
                }
                emit(next);
            }
        };
        // Pass 1: block-column counts, then offsets.
        let counts: Vec<usize> = par::map_range(node_rows, NODES_PER_TASK, |node| {
            let mut count = 0usize;
            block_columns(node, &mut |_| count += 1);
            count
        });
        let mut row_ptr = Vec::with_capacity(node_rows + 1);
        row_ptr.push(0usize);
        for count in &counts {
            row_ptr.push(row_ptr.last().unwrap() + count);
        }
        let stored = *row_ptr.last().unwrap();
        let mut col_idx = vec![0usize; stored];
        let mut values = vec![T::zero(); stored * block * block];
        // Pass 2: each task fills the disjoint slices of its nodes.
        let mut tasks = Vec::new();
        {
            let (mut cols_rest, mut values_rest) = (&mut col_idx[..], &mut values[..]);
            let mut first = 0usize;
            while first < node_rows {
                let last = (first + NODES_PER_TASK).min(node_rows);
                let entries = row_ptr[last] - row_ptr[first];
                let (cols_here, cols_tail) = cols_rest.split_at_mut(entries);
                let (values_here, values_tail) = values_rest.split_at_mut(entries * block * block);
                tasks.push((first, last, cols_here, values_here));
                cols_rest = cols_tail;
                values_rest = values_tail;
                first = last;
            }
        }
        runmat_cpu_execution::for_each_mut(&mut tasks, |(first, last, cols, vals)| {
            let base = row_ptr[*first];
            for (node, &row_start) in row_ptr.iter().enumerate().take(*last).skip(*first) {
                let start = row_start - base;
                let mut written = start;
                block_columns(node, &mut |c| {
                    cols[written] = c;
                    written += 1;
                });
                let node_cols = &cols[start..written];
                for r in 0..block {
                    let (c, v) = matrix.row(node * block + r);
                    let mut position = 0usize;
                    for (&col, &value) in c.iter().zip(v) {
                        while node_cols[position] != col / block {
                            position += 1;
                        }
                        vals[(start + position) * block * block + r * block + col % block] = value;
                    }
                }
            }
        });
        drop(tasks);
        Ok(Self {
            block,
            node_rows,
            node_cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block
    }
    pub fn nrows(&self) -> usize {
        self.node_rows * self.block
    }
    pub fn ncols(&self) -> usize {
        self.node_cols * self.block
    }

    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols());
        assert_eq!(y.len(), self.nrows());
        match self.block {
            2 => self.spmv_fixed::<2>(x, y),
            3 => self.spmv_fixed::<3>(x, y),
            6 => self.spmv_fixed::<6>(x, y),
            _ => self.spmv_general(x, y),
        }
    }

    /// Fixed block size: the block loops unroll and each block is loaded contiguously.
    fn spmv_fixed<const B: usize>(&self, x: &[T], y: &mut [T]) {
        let kernel = |node: usize, out: &mut [T]| {
            let mut sums = [T::zero(); B];
            for entry in self.row_ptr[node]..self.row_ptr[node + 1] {
                let col = self.col_idx[entry] * B;
                let block = &self.values[entry * B * B..(entry + 1) * B * B];
                let xs: [T; B] = std::array::from_fn(|c| x[col + c]);
                // Same summation order as the general kernel: each block row, then accumulate.
                for r in 0..B {
                    let mut sum = T::zero();
                    for c in 0..B {
                        sum += block[r * B + c] * xs[c];
                    }
                    sums[r] += sum;
                }
            }
            out.copy_from_slice(&sums);
        };
        if par::threads() > 1 && self.node_rows >= 4 * NODES_PER_TASK {
            y.par_chunks_mut(B * NODES_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    for (offset, out) in chunk.chunks_mut(B).enumerate() {
                        kernel(task * NODES_PER_TASK + offset, out);
                    }
                });
        } else {
            for (node, out) in y.chunks_mut(B).enumerate() {
                kernel(node, out);
            }
        }
    }

    fn spmv_general(&self, x: &[T], y: &mut [T]) {
        let b = self.block;
        let kernel = |node: usize, out: &mut [T]| {
            out.iter_mut().for_each(|v| *v = T::zero());
            for entry in self.row_ptr[node]..self.row_ptr[node + 1] {
                let col = self.col_idx[entry] * b;
                let block = &self.values[entry * b * b..(entry + 1) * b * b];
                for r in 0..b {
                    let mut sum = T::zero();
                    for c in 0..b {
                        sum += block[r * b + c] * x[col + c];
                    }
                    out[r] += sum;
                }
            }
        };
        if par::threads() > 1 && self.node_rows >= 4 * NODES_PER_TASK {
            y.par_chunks_mut(b * NODES_PER_TASK)
                .enumerate()
                .for_each(|(task, chunk)| {
                    for (offset, out) in chunk.chunks_mut(b).enumerate() {
                        kernel(task * NODES_PER_TASK + offset, out);
                    }
                });
        } else {
            for (node, out) in y.chunks_mut(b).enumerate() {
                kernel(node, out);
            }
        }
    }

    pub fn to_csr(&self) -> CsrMatrix<T> {
        let b = self.block;
        let rows: Vec<(Vec<usize>, Vec<T>)> = (0..self.nrows())
            .map(|row| {
                let (node, r) = (row / b, row % b);
                let mut cols = Vec::new();
                let mut vals = Vec::new();
                for entry in self.row_ptr[node]..self.row_ptr[node + 1] {
                    for c in 0..b {
                        let value = self.values[entry * b * b + r * b + c];
                        if value != T::zero() {
                            cols.push(self.col_idx[entry] * b + c);
                            vals.push(value);
                        }
                    }
                }
                (cols, vals)
            })
            .collect();
        CsrMatrix::from_rows(self.nrows(), self.ncols(), rows)
    }
}

impl<T: Scalar> LinearOperator<T> for BsrMatrix<T> {
    fn nrows(&self) -> usize {
        BsrMatrix::nrows(self)
    }
    fn ncols(&self) -> usize {
        BsrMatrix::ncols(self)
    }
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        self.spmv(x, y);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::poisson3d;

    #[test]
    fn bsr_spmv_matches_csr() {
        let a = poisson3d::<f64>(6, 5, 4); // 120 rows, divisible by 3
        let bsr = BsrMatrix::from_csr(a.as_ref(), 3).unwrap();
        let x = par::seeded_vector::<f64>(a.nrows(), 3);
        let (mut y1, mut y2) = (vec![0.0; a.nrows()], vec![0.0; a.nrows()]);
        a.as_ref().spmv(&x, &mut y1);
        bsr.spmv(&x, &mut y2);
        for (p, q) in y1.iter().zip(&y2) {
            assert!((p - q).abs() < 1e-14);
        }
        assert_eq!(bsr.to_csr(), a);
    }
}

#[cfg(test)]
mod fixed_block_tests {
    use super::*;
    use crate::testutil::elasticity::elasticity3d;

    #[test]
    fn fixed_size_block_kernels_match_csr_and_are_thread_invariant() {
        let a = elasticity3d(9, 8, 7).matrix;
        let x = par::seeded_vector::<f64>(a.nrows(), 4);
        let mut reference = vec![0.0; a.nrows()];
        a.as_ref().spmv(&x, &mut reference);
        let bsr = BsrMatrix::from_csr(a.as_ref(), 3).unwrap();
        let mut serial = vec![0.0; a.nrows()];
        bsr.spmv(&x, &mut serial);
        let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(reference
            .iter()
            .zip(&serial)
            .all(|(p, q)| (p - q).abs() <= 1e-13 * scale));
        let mut general = vec![0.0; a.nrows()];
        bsr.spmv_general(&x, &mut general);
        assert!(general
            .iter()
            .zip(&serial)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
        let mut threaded = vec![0.0; a.nrows()];
        par::with_threads(4, || bsr.spmv(&x, &mut threaded));
        assert!(threaded
            .iter()
            .zip(&serial)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}
