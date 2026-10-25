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
    /// Converts a CSR matrix whose dimensions are multiples of `block`; missing entries inside a
    /// touched block are stored as zeros.
    pub fn from_csr(matrix: CsrRef<'_, T>, block: usize) -> SolverResult<Self> {
        if block == 0 || matrix.nrows() % block != 0 || matrix.ncols() % block != 0 {
            return Err(SolverError::Dimension(format!("block size {block} does not divide the matrix")));
        }
        let node_rows = matrix.nrows() / block;
        let node_cols = matrix.ncols() / block;
        let rows: Vec<(Vec<usize>, Vec<T>)> = par::map_range(node_rows, NODES_PER_TASK, |node| {
            let mut cols: Vec<usize> = Vec::new();
            for r in 0..block {
                cols.extend(matrix.row(node * block + r).0.iter().map(|&c| c / block));
            }
            cols.sort_unstable();
            cols.dedup();
            let mut values = vec![T::zero(); cols.len() * block * block];
            for r in 0..block {
                let (c, v) = matrix.row(node * block + r);
                for (&col, &value) in c.iter().zip(v) {
                    let position = cols.binary_search(&(col / block)).unwrap();
                    values[position * block * block + r * block + col % block] = value;
                }
            }
            (cols, values)
        });
        let mut row_ptr = Vec::with_capacity(node_rows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for (c, v) in rows {
            col_idx.extend(c);
            values.extend(v);
            row_ptr.push(col_idx.len());
        }
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
        assert!(reference.iter().zip(&serial).all(|(p, q)| (p - q).abs() <= 1e-13 * scale));
        let mut general = vec![0.0; a.nrows()];
        bsr.spmv_general(&x, &mut general);
        assert!(general.iter().zip(&serial).all(|(p, q)| p.to_bits() == q.to_bits()));
        let mut threaded = vec![0.0; a.nrows()];
        par::with_threads(4, || bsr.spmv(&x, &mut threaded));
        assert!(threaded.iter().zip(&serial).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}
