//! Immutable numerical factor DATA, not an execution capability.
//!
//! Every payload owner is an existing native buffer owner. Validation is
//! incremental and owns no scheduling policy; Runtime chooses its turn size
//! and admits storage before constructing this data. No issuer, callback,
//! context, mutable solve workspace, or erased numerical method is retained.

use crate::{
    ExactHostSparseClass, HostIndexBuffer, HostNumericBuffer, NumericDType, SparseCscValidation,
    SparseTensor,
};
use std::num::NonZeroUsize;
mod buffer;
pub use buffer::NumericFactorBufferAdoption;
mod dense;
pub use dense::{DenseOrthogonalFactorData, NativeDenseFactorMatrix};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericFactorKind {
    Lu,
    Cholesky,
    Qr,
    Cod,
}

/// Forward and inverse zero-based maps. Keeping both allows linear,
/// allocation-free validation of bijectivity, including imported data.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorPermutation {
    pub forward: HostIndexBuffer,
    pub inverse: HostIndexBuffer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrthogonalFactorData {
    pub r: SparseTensor,
    pub reflectors: SparseTensor,
    pub permutation: FactorPermutation,
    pub rank: usize,
}

/// Unvalidated construction/import input. Only a completed
/// `NumericFactorValidation` can produce `NumericFactorData`.
#[derive(Clone, Debug, PartialEq)]
pub enum NumericFactorCandidate {
    DenseLu {
        packed: NativeDenseFactorMatrix,
        rows: FactorPermutation,
    },
    DenseCholesky {
        lower: NativeDenseFactorMatrix,
    },
    DenseQr(DenseOrthogonalFactorData),
    DenseCod {
        first: DenseOrthogonalFactorData,
        completion: Option<DenseOrthogonalFactorData>,
    },
    /// Zero physical coefficient geometry has no CSC/map allocations,
    /// including when one empty matrix axis is a virtual `usize::MAX`.
    Empty {
        kind: NumericFactorKind,
        rows: usize,
        columns: usize,
        class: ExactHostSparseClass,
    },
    Lu {
        lower: SparseTensor,
        upper: SparseTensor,
        rows: FactorPermutation,
        columns: FactorPermutation,
        row_scale: Option<HostNumericBuffer>,
    },
    Cholesky {
        lower: SparseTensor,
        permutation: FactorPermutation,
    },
    Qr(OrthogonalFactorData),
    Cod {
        first: OrthogonalFactorData,
        completion: Option<OrthogonalFactorData>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFactorData {
    parts: NumericFactorCandidate,
}

impl NumericFactorData {
    pub fn kind(&self) -> NumericFactorKind {
        match &self.parts {
            NumericFactorCandidate::Empty { kind, .. } => *kind,
            NumericFactorCandidate::Lu { .. } | NumericFactorCandidate::DenseLu { .. } => {
                NumericFactorKind::Lu
            }
            NumericFactorCandidate::Cholesky { .. }
            | NumericFactorCandidate::DenseCholesky { .. } => NumericFactorKind::Cholesky,
            NumericFactorCandidate::Qr(_) | NumericFactorCandidate::DenseQr(_) => {
                NumericFactorKind::Qr
            }
            NumericFactorCandidate::Cod { .. } | NumericFactorCandidate::DenseCod { .. } => {
                NumericFactorKind::Cod
            }
        }
    }

    pub fn parts(&self) -> &NumericFactorCandidate {
        &self.parts
    }

    pub fn dimensions(&self) -> (usize, usize) {
        if let Some(matrix) = self.parts.dense_matrix(0) {
            return matrix.dimensions();
        }
        if let NumericFactorCandidate::Empty { rows, columns, .. } = &self.parts {
            return (*rows, *columns);
        }
        let matrix = self.parts.matrix(0).expect("factor has a first block");
        (matrix.rows, matrix.cols)
    }

    pub fn storage_class(&self) -> ExactHostSparseClass {
        if let Some(matrix) = self.parts.dense_matrix(0) {
            return matrix.storage_class();
        }
        if let NumericFactorCandidate::Empty { class, .. } = &self.parts {
            return *class;
        }
        let (dtype, complex) = native_class(self.parts.matrix(0).expect("validated factor matrix"))
            .expect("validated native factor class");
        if complex {
            ExactHostSparseClass::Complex(dtype)
        } else {
            ExactHostSparseClass::Numeric(dtype)
        }
    }
}

impl NumericFactorCandidate {
    fn orthogonal_block(&self, index: usize) -> Option<(&OrthogonalFactorData, bool)> {
        match self {
            Self::Qr(first) | Self::Cod { first, .. } if index < 2 => Some((first, index == 1)),
            Self::Cod {
                completion: Some(second),
                ..
            } if index < 4 => Some((second, index == 3)),
            _ => None,
        }
    }
    fn matrix(&self, index: usize) -> Option<&SparseTensor> {
        match self {
            Self::DenseLu { .. }
            | Self::DenseCholesky { .. }
            | Self::DenseQr(_)
            | Self::DenseCod { .. } => None,
            Self::Empty { .. } => None,
            Self::Lu { lower, upper, .. } => match index {
                0 => Some(lower),
                1 => Some(upper),
                _ => None,
            },
            Self::Cholesky { lower, .. } => (index == 0).then_some(lower),
            Self::Qr(first) | Self::Cod { first, .. } => match index {
                0 => Some(&first.r),
                1 => Some(&first.reflectors),
                2 => match self {
                    Self::Cod {
                        completion: Some(second),
                        ..
                    } => Some(&second.r),
                    _ => None,
                },
                3 => match self {
                    Self::Cod {
                        completion: Some(second),
                        ..
                    } => Some(&second.reflectors),
                    _ => None,
                },
                _ => None,
            },
        }
    }

    fn permutation(&self, index: usize) -> Option<(&FactorPermutation, usize)> {
        match self {
            Self::DenseLu { packed, rows } => (index == 0).then_some((rows, packed.dimensions().0)),
            Self::DenseCholesky { .. } => None,
            Self::DenseQr(first) | Self::DenseCod { first, .. } => match index {
                0 => Some((&first.permutation, first.packed.dimensions().1)),
                1 => {
                    if let Self::DenseCod {
                        completion: Some(second),
                        ..
                    } = self
                    {
                        Some((&second.permutation, second.packed.dimensions().1))
                    } else {
                        None
                    }
                }
                _ => None,
            },
            Self::Empty { .. } => None,
            Self::Lu {
                lower,
                rows,
                columns,
                ..
            } => match index {
                0 => Some((rows, lower.rows)),
                1 => Some((columns, lower.rows)),
                _ => None,
            },
            Self::Cholesky { lower, permutation } => {
                (index == 0).then_some((permutation, lower.rows))
            }
            Self::Qr(first) | Self::Cod { first, .. } => match index {
                0 => Some((&first.permutation, first.r.cols)),
                1 => match self {
                    Self::Cod {
                        completion: Some(second),
                        ..
                    } => Some((&second.permutation, second.r.cols)),
                    _ => None,
                },
                _ => None,
            },
        }
    }

    fn check_geometry(&self) -> Result<(), &'static str> {
        if self.dense_matrix(0).is_some() {
            return self.check_dense_geometry();
        }
        if let Self::Empty {
            kind,
            rows,
            columns,
            class,
        } = self
        {
            if *rows != 0 && *columns != 0 {
                return Err("empty factor requires a zero axis");
            }
            if matches!(kind, NumericFactorKind::Lu | NumericFactorKind::Cholesky)
                && rows != columns
            {
                return Err("empty LU/Cholesky factor must be square");
            }
            return if matches!(
                class,
                ExactHostSparseClass::Numeric(NumericDType::F32 | NumericDType::F64)
                    | ExactHostSparseClass::Complex(NumericDType::F32 | NumericDType::F64)
            ) {
                Ok(())
            } else {
                Err("empty factor requires a native floating class and domain")
            };
        }
        let first = self.matrix(0).ok_or("factor has no matrix")?;
        let class = native_class(first).ok_or("factor requires native floating storage")?;
        for index in 0..4 {
            if let Some(matrix) = self.matrix(index) {
                if native_class(matrix) != Some(class) {
                    return Err("factor blocks must have one native class and domain");
                }
            }
        }
        match self {
            Self::Lu {
                lower,
                upper,
                row_scale,
                ..
            } => {
                if lower.rows != lower.cols || upper.rows != lower.rows || upper.cols != lower.cols
                {
                    return Err("LU factor blocks must be equally sized square matrices");
                }
                if let Some(scale) = row_scale {
                    if scale.len() != lower.rows || scale.numeric_dtype() != class.0 {
                        return Err("LU row scaling must match native class and row count");
                    }
                }
            }
            Self::Cholesky { lower, .. } if lower.rows != lower.cols => {
                return Err("Cholesky factor must be square")
            }
            Self::Qr(first) => check_orthogonal(first)?,
            Self::Cod { first, completion } => {
                check_orthogonal(first)?;
                if let Some(second) = completion {
                    check_orthogonal(second)?;
                    if first.rank == 0
                        || first.rank >= first.r.cols
                        || second.r.rows != first.r.cols
                        || second.r.cols != first.rank
                        || second.rank != first.rank
                    {
                        return Err("COD completion geometry does not match the retained rank");
                    }
                } else if first.rank != 0 && first.rank != first.r.cols {
                    return Err("rank-deficient COD requires its completion factor");
                }
            }
            _ => {}
        }
        for index in 0..2 {
            if let Some((permutation, len)) = self.permutation(index) {
                if permutation.forward.len() != len || permutation.inverse.len() != len {
                    return Err("factor permutation extent mismatch");
                }
            }
        }
        Ok(())
    }
}

fn check_orthogonal(data: &OrthogonalFactorData) -> Result<(), &'static str> {
    if data.rank > data.r.rows.min(data.r.cols)
        || data.reflectors.rows != data.r.rows
        || data.reflectors.cols != data.r.rows.min(data.r.cols)
    {
        return Err("orthogonal factor geometry or rank mismatch");
    }
    Ok(())
}

fn native_class(matrix: &SparseTensor) -> Option<(NumericDType, bool)> {
    if matrix.as_f32_slice().is_some() {
        Some((NumericDType::F32, false))
    } else if matrix.as_f64_slice().is_some() {
        Some((NumericDType::F64, false))
    } else if matrix.as_complex_f32_slice().is_some() {
        Some((NumericDType::F32, true))
    } else if matrix.as_complex_f64_slice().is_some() {
        Some((NumericDType::F64, true))
    } else {
        None
    }
}

fn parts(matrix: &SparseTensor, index: usize) -> (f64, f64) {
    if let Some(values) = matrix.as_f32_slice() {
        (f64::from(values[index]), 0.0)
    } else if let Some(values) = matrix.as_f64_slice() {
        (values[index], 0.0)
    } else if let Some(values) = matrix.as_complex_f32_slice() {
        (f64::from(values[index].0), f64::from(values[index].1))
    } else {
        let value = matrix.as_complex_f64_slice().expect("checked native class")[index];
        (value.0, value.1)
    }
}

/// Sealed, bounded-turn construction proof. A work item is one topology
/// validator item, scalar, column transition, or permutation relation.
pub struct NumericFactorValidation {
    data: NumericFactorCandidate,
    block: usize,
    topology: Option<SparseCscValidation>,
    topology_done: bool,
    column: usize,
    edge: usize,
    diagonal: bool,
    permutation: usize,
    offset: usize,
    scale: usize,
    complete: bool,
}

impl NumericFactorValidation {
    pub fn new(data: NumericFactorCandidate) -> Result<Self, &'static str> {
        data.check_geometry()?;
        Ok(Self {
            data,
            block: 0,
            topology: None,
            topology_done: false,
            column: 0,
            edge: 0,
            diagonal: false,
            permutation: 0,
            offset: 0,
            scale: 0,
            complete: false,
        })
    }

    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<bool, String> {
        let one = NonZeroUsize::new(1).expect("one is nonzero");
        for _ in 0..maximum_work.get() {
            if let Some(matrix) = self.data.dense_matrix(self.block) {
                if self.edge < matrix.len() {
                    self.data
                        .check_dense_scalar(self.block, self.edge)
                        .map_err(str::to_owned)?;
                    self.edge += 1;
                } else {
                    self.block += 1;
                    self.edge = 0;
                }
                continue;
            }
            if let Some(matrix) = self.data.matrix(self.block) {
                if !self.topology_done {
                    if self.topology.is_none() {
                        self.topology = Some(SparseCscValidation::new(
                            matrix.rows,
                            matrix.cols,
                            matrix.col_ptrs.clone(),
                            matrix.row_indices.clone(),
                            matrix.nnz(),
                        )?);
                    }
                    if self
                        .topology
                        .as_mut()
                        .expect("installed validator")
                        .advance(one)?
                    {
                        self.topology_done = true;
                        self.topology = None;
                    }
                    continue;
                }
                if self.column < matrix.cols {
                    let end = matrix.col_ptrs[self.column + 1];
                    if self.edge < end {
                        let row = matrix.row_indices[self.edge];
                        let (real, imag) = parts(matrix, self.edge);
                        if !real.is_finite() || !imag.is_finite() {
                            return Err("factor contains nonfinite numerical data".into());
                        }
                        if let Some((orthogonal, reflector)) =
                            self.data.orthogonal_block(self.block)
                        {
                            if reflector {
                                if self.column >= orthogonal.rank || row < self.column {
                                    return Err(
                                        "reflector support lies outside the retained factorization"
                                            .into(),
                                    );
                                }
                            } else if row < orthogonal.rank && row > self.column {
                                return Err(
                                    "retained QR leading block is not upper triangular".into()
                                );
                            }
                        }
                        let lower = matches!(
                            self.data,
                            NumericFactorCandidate::Lu { .. }
                                | NumericFactorCandidate::Cholesky { .. }
                        ) && self.block == 0;
                        let upper = matches!(self.data, NumericFactorCandidate::Lu { .. })
                            && self.block == 1;
                        if (lower && row < self.column) || (upper && row > self.column) {
                            return Err(
                                "factor triangular storage has an off-triangle entry".into()
                            );
                        }
                        if row == self.column {
                            self.diagonal = true;
                            if let Some((orthogonal, false)) =
                                self.data.orthogonal_block(self.block)
                            {
                                if self.column < orthogonal.rank && real == 0.0 && imag == 0.0 {
                                    return Err(
                                        "retained QR leading diagonal must be nonzero".into()
                                    );
                                }
                            }
                            if lower
                                && matches!(self.data, NumericFactorCandidate::Lu { .. })
                                && (real != 1.0 || imag != 0.0)
                            {
                                return Err("LU lower factor must have a unit diagonal".into());
                            }
                            if lower
                                && matches!(self.data, NumericFactorCandidate::Cholesky { .. })
                                && (real <= 0.0 || imag != 0.0)
                            {
                                return Err("Cholesky diagonal must be positive real".into());
                            }
                        }
                        self.edge += 1;
                    } else {
                        let requires_diagonal = (self.block == 0
                            && matches!(
                                self.data,
                                NumericFactorCandidate::Lu { .. }
                                    | NumericFactorCandidate::Cholesky { .. }
                            ))
                            || self
                                .data
                                .orthogonal_block(self.block)
                                .is_some_and(|(qr, _)| self.column < qr.rank);
                        if requires_diagonal && !self.diagonal {
                            return Err("factor is missing its required diagonal".into());
                        }
                        self.column += 1;
                        self.diagonal = false;
                    }
                    continue;
                }
                self.block += 1;
                self.topology_done = false;
                self.column = 0;
                self.edge = 0;
                continue;
            }
            if let Some((permutation, len)) = self.data.permutation(self.permutation) {
                if self.offset < len {
                    let index = permutation.forward[self.offset];
                    if index >= len || permutation.inverse[index] != self.offset {
                        return Err("factor maps are not inverse permutations".into());
                    }
                    self.offset += 1;
                } else {
                    self.permutation += 1;
                    self.offset = 0;
                }
                continue;
            }
            if let NumericFactorCandidate::Lu {
                row_scale: Some(scale),
                ..
            } = &self.data
            {
                if self.scale < scale.len() {
                    let value = if let Some(values) = scale.as_f32_slice() {
                        f64::from(values[self.scale])
                    } else {
                        scale.as_f64_slice().expect("checked native scale")[self.scale]
                    };
                    if !value.is_finite() || value <= 0.0 {
                        return Err("LU row scaling must be finite and positive".into());
                    }
                    self.scale += 1;
                    continue;
                }
            }
            self.complete = true;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn finish(self) -> Result<NumericFactorData, &'static str> {
        if !self.complete {
            return Err("factor validation has not completed");
        }
        Ok(NumericFactorData { parts: self.data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(class: usize, values: &[f64], shape: [usize; 2]) -> NativeDenseFactorMatrix {
        match class {
            0 => NativeDenseFactorMatrix::Real(
                crate::Tensor::from_f32(values.iter().map(|v| *v as f32).collect(), shape.to_vec())
                    .unwrap(),
            ),
            1 => NativeDenseFactorMatrix::Real(
                crate::Tensor::new(values.to_vec(), shape.to_vec()).unwrap(),
            ),
            2 => NativeDenseFactorMatrix::Complex(
                crate::ComplexTensor::from_f32(
                    values.iter().map(|v| (*v as f32, 0.0)).collect(),
                    shape.to_vec(),
                )
                .unwrap(),
            ),
            _ => NativeDenseFactorMatrix::Complex(
                crate::ComplexTensor::new(
                    values.iter().map(|v| (*v, 0.0)).collect(),
                    shape.to_vec(),
                )
                .unwrap(),
            ),
        }
    }
    #[test]
    fn dense_native_factor_validation_preserves_compact_geometry_and_identity_coefficients() {
        for class in 0..4 {
            let qr = DenseOrthogonalFactorData {
                packed: dense(class, &[1.0, 0.0, 0.0, 0.0], [2, 2]),
                coefficients: dense(class, &[f64::INFINITY, f64::INFINITY], [1, 2]),
                permutation: identity(2),
                rank: 1,
            };
            let valid = validate(NumericFactorCandidate::DenseQr(qr.clone())).unwrap();
            assert_eq!(valid.dimensions(), (2, 2));
            assert_eq!(valid.kind(), NumericFactorKind::Qr);
            let mut invalid = qr.clone();
            invalid.coefficients = dense(class, &[f64::NAN, f64::INFINITY], [1, 2]);
            assert!(validate(NumericFactorCandidate::DenseQr(invalid)).is_err());
            let mut invalid = qr.clone();
            invalid.rank = 2;
            assert!(validate(NumericFactorCandidate::DenseQr(invalid)).is_err());
            let mut invalid = qr;
            invalid.coefficients = dense(class, &[1.0, f64::INFINITY, 0.0, 1.0], [2, 2]);
            assert!(validate(NumericFactorCandidate::DenseQr(invalid)).is_err());
            assert!(validate(NumericFactorCandidate::DenseLu {
                packed: dense(class, &[1.0, 0.0, 0.0, 0.0], [2, 2]),
                rows: identity(2)
            })
            .is_ok());
            assert!(validate(NumericFactorCandidate::DenseCholesky {
                lower: dense(class, &[1.0, 0.5, 0.0, 2.0], [2, 2])
            })
            .is_ok());
            assert!(validate(NumericFactorCandidate::DenseCholesky {
                lower: dense(class, &[1.0, 0.0, 0.5, 2.0], [2, 2])
            })
            .is_err());
        }
    }

    fn identity(len: usize) -> FactorPermutation {
        FactorPermutation {
            forward: HostIndexBuffer::new((0..len).collect()),
            inverse: HostIndexBuffer::new((0..len).collect()),
        }
    }
    fn block(
        class: usize,
        cols: usize,
        ptrs: Vec<usize>,
        rows: Vec<usize>,
        values: &[f64],
    ) -> SparseTensor {
        match class {
            0 => SparseTensor::new_f32(
                2,
                cols,
                ptrs,
                rows,
                values.iter().map(|v| *v as f32).collect(),
            ),
            1 => SparseTensor::new(2, cols, ptrs, rows, values.to_vec()),
            2 => SparseTensor::new_complex_f32(
                2,
                cols,
                ptrs,
                rows,
                values.iter().map(|v| (*v as f32, 0.0)).collect(),
            ),
            _ => SparseTensor::new_complex(
                2,
                cols,
                ptrs,
                rows,
                values.iter().map(|v| (*v, 0.0)).collect(),
            ),
        }
        .unwrap()
    }
    fn validate(data: NumericFactorCandidate) -> Result<NumericFactorData, String> {
        let mut validation = NumericFactorValidation::new(data).map_err(str::to_owned)?;
        while !validation.advance(NonZeroUsize::new(1).unwrap())? {}
        validation.finish().map_err(str::to_owned)
    }
    fn first(class: usize) -> OrthogonalFactorData {
        OrthogonalFactorData {
            r: block(class, 2, vec![0, 1, 2], vec![0, 0], &[1.0, 2.0]),
            reflectors: block(class, 2, vec![0, 1, 1], vec![0], &[1.0]),
            permutation: identity(2),
            rank: 1,
        }
    }

    #[test]
    fn native_four_factor_data_distinguishes_basic_qr_from_complete_orthogonal_data() {
        for class in 0..4 {
            let qr = validate(NumericFactorCandidate::Qr(first(class))).unwrap();
            assert_eq!(qr.kind(), NumericFactorKind::Qr);
            assert_eq!(qr.dimensions(), (2, 2));
            assert!(validate(NumericFactorCandidate::Cod {
                first: first(class),
                completion: None
            })
            .is_err());
            let completion = OrthogonalFactorData {
                r: block(class, 1, vec![0, 1], vec![0], &[1.0]),
                reflectors: block(class, 1, vec![0, 1], vec![0], &[1.0]),
                permutation: identity(1),
                rank: 1,
            };
            let cod = validate(NumericFactorCandidate::Cod {
                first: first(class),
                completion: Some(completion),
            })
            .unwrap();
            assert_eq!(cod.kind(), NumericFactorKind::Cod);
            assert_eq!(cod.clone(), cod);
            let lower = block(class, 2, vec![0, 1, 2], vec![0, 1], &[1.0, 1.0]);
            // A singular upper factor is legitimate DATA. The invocation
            // solve kernel owns singular-system behavior, not this schema.
            let upper = block(class, 2, vec![0, 1, 1], vec![0], &[1.0]);
            let lu = validate(NumericFactorCandidate::Lu {
                lower,
                upper,
                rows: identity(2),
                columns: identity(2),
                row_scale: None,
            })
            .unwrap();
            assert_eq!(lu.kind(), NumericFactorKind::Lu);
            let mut invalid = first(class);
            invalid.permutation.forward = HostIndexBuffer::new(vec![0, 0]);
            assert!(validate(NumericFactorCandidate::Qr(invalid)).is_err());
            let lower = block(class, 2, vec![0, 1, 2], vec![0, 1], &[1.0, -1.0]);
            assert!(validate(NumericFactorCandidate::Cholesky {
                lower,
                permutation: identity(2)
            })
            .is_err());
        }
    }

    #[test]
    fn empty_native_factor_data_preserves_virtual_axes_without_payload_or_maps() {
        for class in [
            ExactHostSparseClass::Numeric(NumericDType::F32),
            ExactHostSparseClass::Numeric(NumericDType::F64),
            ExactHostSparseClass::Complex(NumericDType::F32),
            ExactHostSparseClass::Complex(NumericDType::F64),
        ] {
            for kind in [NumericFactorKind::Qr, NumericFactorKind::Cod] {
                for (rows, columns) in [(0, usize::MAX), (usize::MAX, 0), (0, 0)] {
                    let data = validate(NumericFactorCandidate::Empty {
                        kind,
                        rows,
                        columns,
                        class,
                    })
                    .unwrap();
                    assert_eq!(data.kind(), kind);
                    assert_eq!(data.dimensions(), (rows, columns));
                    assert_eq!(data.storage_class(), class);
                }
            }
            for kind in [NumericFactorKind::Lu, NumericFactorKind::Cholesky] {
                assert!(validate(NumericFactorCandidate::Empty {
                    kind,
                    rows: 0,
                    columns: 1,
                    class
                })
                .is_err());
                assert!(validate(NumericFactorCandidate::Empty {
                    kind,
                    rows: 0,
                    columns: 0,
                    class
                })
                .is_ok());
            }
            assert!(validate(NumericFactorCandidate::Empty {
                kind: NumericFactorKind::Qr,
                rows: 1,
                columns: 1,
                class
            })
            .is_err());
        }
    }
}
