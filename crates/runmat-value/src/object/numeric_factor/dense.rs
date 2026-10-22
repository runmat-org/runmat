//! Canonical dense factor storage. These are immutable buffers, not kernels.
use super::{FactorPermutation, NumericFactorCandidate};
use crate::{ComplexTensor, ExactHostSparseClass, NumericDType, Tensor};

#[derive(Clone, Debug, PartialEq)]
pub enum NativeDenseFactorMatrix {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NativeDenseFactorMatrix {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Real(t) => t.shape(),
            Self::Complex(t) => t.shape(),
        }
    }
    pub fn dimensions(&self) -> (usize, usize) {
        let shape = self.shape();
        (
            shape.first().copied().unwrap_or(1),
            shape.get(1).copied().unwrap_or(1),
        )
    }
    pub fn storage_class(&self) -> ExactHostSparseClass {
        match self {
            Self::Real(t) => ExactHostSparseClass::Numeric(t.numeric_dtype()),
            Self::Complex(t) => ExactHostSparseClass::Complex(t.numeric_dtype()),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Real(t) => t.len(),
            Self::Complex(t) => t.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn parts(&self, index: usize) -> (f64, f64) {
        match self {
            Self::Real(t) => {
                if let Some(v) = t.as_f32_slice() {
                    (f64::from(v[index]), 0.0)
                } else {
                    (t.as_f64_slice().expect("checked native class")[index], 0.0)
                }
            }
            Self::Complex(t) => {
                if let Some(v) = t.as_f32_slice() {
                    (f64::from(v[index].0), f64::from(v[index].1))
                } else {
                    let v = t.as_f64_slice().expect("checked native class")[index];
                    (v.0, v.1)
                }
            }
        }
    }
    fn check(&self) -> Result<(), &'static str> {
        if self.shape().iter().skip(2).any(|d| *d != 1) {
            return Err("dense factor requires matrix geometry");
        }
        if !matches!(
            self.storage_class(),
            ExactHostSparseClass::Numeric(NumericDType::F32 | NumericDType::F64)
                | ExactHostSparseClass::Complex(NumericDType::F32 | NumericDType::F64)
        ) {
            return Err("dense factor requires native floating storage");
        }
        let (m, n) = self.dimensions();
        if m == 0 || n == 0 {
            return Err("zero geometry requires the closed empty factor form");
        }
        if m.checked_mul(n) != Some(self.len()) {
            return Err("dense factor payload geometry mismatch");
        }
        Ok(())
    }
}

/// Faer column-pivoted compact QR: `packed` is m-by-n, with R on and
/// above the diagonal and implicit-unit Householder vectors below it.
/// `coefficients` is block_size-by-min(m,n), in faer's blocked format.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseOrthogonalFactorData {
    pub packed: NativeDenseFactorMatrix,
    pub coefficients: NativeDenseFactorMatrix,
    pub permutation: FactorPermutation,
    pub rank: usize,
}
impl DenseOrthogonalFactorData {
    fn check(&self) -> Result<(), &'static str> {
        self.packed.check()?;
        self.coefficients.check()?;
        let (m, n) = self.packed.dimensions();
        let (block, k) = self.coefficients.dimensions();
        if block == 0
            || block > m.min(n)
            || k != m.min(n)
            || self.rank > k
            || self.coefficients.storage_class() != self.packed.storage_class()
        {
            return Err("dense orthogonal coefficient geometry, class, or rank mismatch");
        }
        Ok(())
    }
}

impl NumericFactorCandidate {
    pub(super) fn dense_matrix(&self, index: usize) -> Option<&NativeDenseFactorMatrix> {
        match self {
            Self::DenseLu { packed, .. } | Self::DenseCholesky { lower: packed } => {
                (index == 0).then_some(packed)
            }
            Self::DenseQr(first) | Self::DenseCod { first, .. } => match index {
                0 => Some(&first.packed),
                1 => Some(&first.coefficients),
                2 | 3 => {
                    if let Self::DenseCod {
                        completion: Some(second),
                        ..
                    } = self
                    {
                        if index == 2 {
                            Some(&second.packed)
                        } else {
                            Some(&second.coefficients)
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }
    pub(super) fn check_dense_geometry(&self) -> Result<(), &'static str> {
        let first = self.dense_matrix(0).expect("dense factor");
        first.check()?;
        for i in 1..4 {
            if let Some(matrix) = self.dense_matrix(i) {
                matrix.check()?;
                if matrix.storage_class() != first.storage_class() {
                    return Err("dense factor blocks must have one native class and domain");
                }
            }
        }
        match self {
            Self::DenseLu { .. } | Self::DenseCholesky { .. } => {
                let (m, n) = first.dimensions();
                if m != n {
                    return Err("dense LU/Cholesky factor must be square");
                }
            }
            Self::DenseQr(qr) => qr.check()?,
            Self::DenseCod { first, completion } => {
                first.check()?;
                let (_, n) = first.packed.dimensions();
                if let Some(second) = completion {
                    second.check()?;
                    if first.rank == 0
                        || first.rank >= n
                        || second.packed.dimensions() != (n, first.rank)
                        || second.rank != first.rank
                    {
                        return Err("dense COD completion geometry mismatch");
                    }
                } else if first.rank != 0 && first.rank != n {
                    return Err("rank-deficient dense COD requires completion");
                }
            }
            _ => unreachable!("dense geometry branch"),
        }
        for i in 0..2 {
            if let Some((p, len)) = self.permutation(i) {
                if p.forward.len() != len || p.inverse.len() != len {
                    return Err("dense factor permutation extent mismatch");
                }
            }
        }
        Ok(())
    }
    pub(super) fn check_dense_scalar(
        &self,
        block: usize,
        index: usize,
    ) -> Result<(), &'static str> {
        let matrix = self.dense_matrix(block).expect("dense block");
        let (real, imag) = matrix.parts(index);
        let (m, _) = matrix.dimensions();
        let row = index % m;
        let col = index / m;
        // Faer represents the identity reflector as tau=+Inf in
        // H=I-v*v^H/tau. This is a representation sentinel, not input data.
        let identity_coefficient =
            (block == 1 || block == 3) && row == col % m && real == f64::INFINITY && imag == 0.0;
        if (block == 1 || block == 3) && row == col % m && (real <= 0.0 || imag != 0.0) {
            return Err("dense Householder coefficient diagonal must be positive real");
        }
        if (!real.is_finite() || !imag.is_finite()) && !identity_coefficient {
            return Err("dense factor contains invalid nonfinite numerical data");
        }
        if matches!(self, Self::DenseCholesky { .. })
            && (row < col && (real != 0.0 || imag != 0.0)
                || row == col && (real <= 0.0 || imag != 0.0))
        {
            return Err("dense Cholesky lower factor or diagonal is invalid");
        }
        let rank = match (self, block) {
            (Self::DenseQr(first) | Self::DenseCod { first, .. }, 0) => Some(first.rank),
            (
                Self::DenseCod {
                    completion: Some(second),
                    ..
                },
                2,
            ) => Some(second.rank),
            _ => None,
        };
        if rank.is_some_and(|rank| row == col && col < rank) && real == 0.0 && imag == 0.0 {
            return Err("dense QR leading diagonal must be nonzero");
        }
        Ok(())
    }
}
