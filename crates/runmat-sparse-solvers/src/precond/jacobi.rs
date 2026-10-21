//! Point and block Jacobi.

use crate::csr::CsrRef;
use crate::dense::DenseLu;
use crate::error::{SolverError, SolverResult};
use crate::operator::Preconditioner;
use crate::par;
use crate::scalar::Scalar;

/// `z_i = r_i / a_ii`.
#[derive(Debug, Clone)]
pub struct Jacobi<T> {
    inverse_diagonal: Vec<T>,
}

impl<T: Scalar> Jacobi<T> {
    pub fn new(matrix: CsrRef<'_, T>) -> SolverResult<Self> {
        let diagonal = matrix.diagonal();
        Self::from_diagonal(&diagonal)
    }
    pub fn from_diagonal(diagonal: &[T]) -> SolverResult<Self> {
        let inverse_diagonal = diagonal
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if d.abs() == 0.0 || !d.is_finite() {
                    Err(SolverError::Singular { pivot: i })
                } else {
                    Ok(T::one() / d)
                }
            })
            .collect::<SolverResult<Vec<T>>>()?;
        Ok(Self { inverse_diagonal })
    }
    pub fn inverse_diagonal(&self) -> &[T] {
        &self.inverse_diagonal
    }
}

impl<T: Scalar> Preconditioner<T> for Jacobi<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        par::mul_into(&self.inverse_diagonal, r, z);
        Ok(())
    }
    fn label(&self) -> &'static str {
        "jacobi"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        self.inverse_diagonal.iter().all(|d| d.re() > 0.0 && d.im() == 0.0)
    }
}

/// Inverse `b × b` diagonal blocks (node blocks of vector problems).
#[derive(Debug, Clone)]
pub struct BlockJacobi<T> {
    block: usize,
    /// Column-major inverse blocks, one per node.
    inverses: Vec<T>,
}

impl<T: Scalar> BlockJacobi<T> {
    pub fn new(matrix: CsrRef<'_, T>, block: usize) -> SolverResult<Self> {
        let n = matrix.nrows();
        if block == 0 || n % block != 0 {
            return Err(SolverError::Dimension(format!(
                "block size {block} does not divide {n}"
            )));
        }
        let nodes = n / block;
        let blocks: Vec<SolverResult<Vec<T>>> = par::map_range(nodes, 256, |node| {
            let mut local = vec![T::zero(); block * block];
            for a in 0..block {
                let row = node * block + a;
                let (cols, vals) = matrix.row(row);
                for (&col, &value) in cols.iter().zip(vals) {
                    if col / block == node {
                        local[(col % block) * block + a] = value;
                    }
                }
            }
            let lu = DenseLu::new(local, block)?;
            let mut inverse = vec![T::zero(); block * block];
            for c in 0..block {
                let mut e = vec![T::zero(); block];
                e[c] = T::one();
                lu.solve_in_place(&mut e);
                inverse[c * block..(c + 1) * block].copy_from_slice(&e);
            }
            Ok(inverse)
        });
        let mut inverses = Vec::with_capacity(n * block);
        for (node, b) in blocks.into_iter().enumerate() {
            inverses.extend(b.map_err(|_| SolverError::Singular { pivot: node * block })?);
        }
        Ok(Self { block, inverses })
    }
    pub fn block_size(&self) -> usize {
        self.block
    }
}

impl<T: Scalar> Preconditioner<T> for BlockJacobi<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        let b = self.block;
        par::for_each_chunk_mut(z, |start, chunk| {
            // Chunks are multiples of the reduction block, which is a multiple of b for b | 4096
            // only; fall back to per-entry node lookup.
            let mut i = 0;
            while i < chunk.len() {
                let global = start + i;
                let node = global / b;
                let offset = global % b;
                let inverse = &self.inverses[node * b * b..(node + 1) * b * b];
                let base = node * b;
                let mut sum = T::zero();
                for c in 0..b {
                    sum += inverse[c * b + offset] * r[base + c];
                }
                chunk[i] = sum;
                i += 1;
            }
        });
        Ok(())
    }
    fn label(&self) -> &'static str {
        "block_jacobi"
    }
}
