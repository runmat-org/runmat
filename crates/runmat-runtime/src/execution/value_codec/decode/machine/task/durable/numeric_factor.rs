//! Incremental import of closed factor DATA using canonical dense/sparse decoders.
//! No factorization is rerun; the imported numerical topology is validated once.

use super::*;
use runmat_execution::value::{
    DenseValue, DurableDenseOrthogonalFactor, DurableEmptyFactorKind, DurableFactorPermutation,
    DurableNumericFactor, DurableOrthogonalFactor, ElementType, SparseValue,
};
use runmat_value::{
    DenseOrthogonalFactorData, ExactHostSparseClass, FactorPermutation, HostIndexBuffer,
    HostNumericBuffer, NativeDenseFactorMatrix, NumericDType, NumericFactorCandidate,
    NumericFactorData, NumericFactorKind, NumericFactorValidation, OrthogonalFactorData,
    SparseTensor,
};

#[path = "numeric_factor/buffers.rs"]
mod buffers;
use buffers::{IndexCopy, ScaleCopy};

pub(super) struct NumericFactorDecode<'a> {
    source: &'a DurableNumericFactor,
    runtime: Option<&'a crate::context::RuntimeContext>,
    path: ControlPath<'a>,
    block: usize,
    active_block: Option<super::super::sparse::SparseTask<'a>>,
    blocks: [Option<SparseTensor>; 4],
    active_dense: Option<super::super::dense::DenseTask<'a>>,
    dense_blocks: [Option<NativeDenseFactorMatrix>; 4],
    map: usize,
    active_map: Option<IndexCopy<'a>>,
    maps: [Option<HostIndexBuffer>; 4],
    scale: Option<ScaleCopy<'a>>,
    scale_value: Option<HostNumericBuffer>,
    validation: Option<NumericFactorValidation>,
}

impl<'a> NumericFactorDecode<'a> {
    pub(super) fn new(
        source: &'a DurableNumericFactor,
        runtime: Option<&'a crate::context::RuntimeContext>,
        path: Result<ControlPath<'a>, ValueCodecError>,
    ) -> Result<Self, ValueCodecError> {
        Ok(Self {
            source,
            runtime,
            path: path?,
            block: 0,
            active_block: None,
            blocks: std::array::from_fn(|_| None),
            active_dense: None,
            dense_blocks: std::array::from_fn(|_| None),
            map: 0,
            active_map: None,
            maps: std::array::from_fn(|_| None),
            scale: None,
            scale_value: None,
            validation: None,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
        object_path: &str,
    ) -> Result<(usize, Option<NumericFactorData>), ValueCodecError> {
        if let Some(source) = block(self.source, self.block) {
            if self.active_block.is_none() {
                self.active_block = Some(super::super::sparse::SparseTask::new(
                    source,
                    self.path.try_clone()?,
                    self.runtime,
                )?);
            }
            let (work, value) = self
                .active_block
                .as_mut()
                .expect("factor block")
                .advance(maximum_work)?;
            if let Some(value) = value {
                let Value::SparseTensor(value) = value else {
                    return Err(ValueCodecError::invalid(
                        object_path,
                        "factor block is not sparse",
                    ));
                };
                self.blocks[self.block] = Some(value);
                self.active_block = None;
                self.block += 1;
            }
            return Ok((work, None));
        }
        if let Some(source) = dense_block(self.source, self.block) {
            if self.active_dense.is_none() {
                self.active_dense = Some(super::super::dense::DenseTask::new_with_runtime(
                    source,
                    self.path.try_clone()?,
                    self.runtime,
                )?);
            }
            let active = self.active_dense.as_mut().expect("dense factor block");
            let work = active.advance(maximum_work)?;
            if active.is_complete() {
                let value = self
                    .active_dense
                    .take()
                    .expect("complete dense factor block")
                    .finish()?;
                self.dense_blocks[self.block] = Some(match value {
                    Value::Tensor(value) => NativeDenseFactorMatrix::Real(value),
                    Value::ComplexTensor(value) => NativeDenseFactorMatrix::Complex(value),
                    _ => {
                        return Err(ValueCodecError::invalid(
                            object_path,
                            "factor block is not a floating dense matrix",
                        ))
                    }
                });
                self.block += 1;
            }
            return Ok((work, None));
        }
        if let Some(source) = map(self.source, self.map) {
            if self.active_map.is_none() {
                self.active_map = Some(IndexCopy::new(source, self.runtime, object_path)?);
            }
            let (work, value) = self.active_map.as_mut().expect("factor map").advance(
                source,
                maximum_work,
                object_path,
            )?;
            if let Some(value) = value {
                self.maps[self.map] = Some(value);
                self.active_map = None;
                self.map += 1;
            }
            return Ok((work, None));
        }
        if self.scale_value.is_none() {
            if let DurableNumericFactor::Lu {
                row_scale: Some(source),
                ..
            } = self.source
            {
                if self.scale.is_none() {
                    self.scale = Some(ScaleCopy::new(source, self.runtime, object_path)?);
                }
                let (work, value) = self.scale.as_mut().expect("factor scale").advance(
                    source,
                    maximum_work,
                    object_path,
                )?;
                if let Some(value) = value {
                    self.scale_value = Some(value);
                    self.scale = None;
                }
                return Ok((work, None));
            }
        }
        if self.validation.is_none() {
            let candidate = candidate(
                self.source,
                &mut self.blocks,
                &mut self.dense_blocks,
                &mut self.maps,
                self.scale_value.take(),
                object_path,
            )?;
            self.validation = Some(
                NumericFactorValidation::new(candidate)
                    .map_err(|error| ValueCodecError::invalid(object_path, error))?,
            );
        }
        let validation = self.validation.as_mut().expect("factor validation");
        if !validation
            .advance(NonZeroUsize::new(maximum_work).expect("machine work is nonzero"))
            .map_err(|error| ValueCodecError::invalid(object_path, error))?
        {
            return Ok((maximum_work, None));
        }
        let value = self
            .validation
            .take()
            .expect("complete factor validation")
            .finish()
            .map_err(|error| ValueCodecError::invalid(object_path, error))?;
        Ok((maximum_work, Some(value)))
    }
}

fn block(source: &DurableNumericFactor, index: usize) -> Option<&SparseValue> {
    match source {
        DurableNumericFactor::Empty { .. } => None,
        DurableNumericFactor::DenseLu { .. }
        | DurableNumericFactor::DenseCholesky { .. }
        | DurableNumericFactor::DenseQr(_)
        | DurableNumericFactor::DenseCod { .. } => None,
        DurableNumericFactor::Lu { lower, upper, .. } => match index {
            0 => Some(lower),
            1 => Some(upper),
            _ => None,
        },
        DurableNumericFactor::Cholesky { lower, .. } => (index == 0).then_some(lower),
        DurableNumericFactor::Qr(first) => orthogonal_block(first, index),
        DurableNumericFactor::Cod { first, completion } => {
            if index < 2 {
                orthogonal_block(first, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|next| orthogonal_block(next, index - 2))
            }
        }
    }
}

fn dense_block(source: &DurableNumericFactor, index: usize) -> Option<&DenseValue> {
    match source {
        DurableNumericFactor::DenseLu { packed, .. } => (index == 0).then_some(packed),
        DurableNumericFactor::DenseCholesky { lower } => (index == 0).then_some(lower),
        DurableNumericFactor::DenseQr(first) => dense_orthogonal_block(first, index),
        DurableNumericFactor::DenseCod { first, completion } => {
            if index < 2 {
                dense_orthogonal_block(first, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|next| dense_orthogonal_block(next, index - 2))
            }
        }
        _ => None,
    }
}

fn dense_orthogonal_block(
    source: &DurableDenseOrthogonalFactor,
    index: usize,
) -> Option<&DenseValue> {
    match index {
        0 => Some(&source.packed),
        1 => Some(&source.coefficients),
        _ => None,
    }
}

fn orthogonal_block(source: &DurableOrthogonalFactor, index: usize) -> Option<&SparseValue> {
    match index {
        0 => Some(&source.r),
        1 => Some(&source.reflectors),
        _ => None,
    }
}

fn map(source: &DurableNumericFactor, index: usize) -> Option<&[u64]> {
    match source {
        DurableNumericFactor::Empty { .. } => None,
        DurableNumericFactor::DenseLu { rows, .. } => permutation_map(rows, index),
        DurableNumericFactor::DenseCholesky { .. } => None,
        DurableNumericFactor::DenseQr(first) => permutation_map(&first.permutation, index),
        DurableNumericFactor::DenseCod { first, completion } => {
            if index < 2 {
                permutation_map(&first.permutation, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|next| permutation_map(&next.permutation, index - 2))
            }
        }
        DurableNumericFactor::Lu { rows, columns, .. } => match index {
            0 => Some(&rows.forward),
            1 => Some(&rows.inverse),
            2 => Some(&columns.forward),
            3 => Some(&columns.inverse),
            _ => None,
        },
        DurableNumericFactor::Cholesky { permutation, .. } => permutation_map(permutation, index),
        DurableNumericFactor::Qr(first) => permutation_map(&first.permutation, index),
        DurableNumericFactor::Cod { first, completion } => {
            if index < 2 {
                permutation_map(&first.permutation, index)
            } else {
                completion
                    .as_ref()
                    .and_then(|next| permutation_map(&next.permutation, index - 2))
            }
        }
    }
}

fn permutation_map(source: &DurableFactorPermutation, index: usize) -> Option<&[u64]> {
    match index {
        0 => Some(&source.forward),
        1 => Some(&source.inverse),
        _ => None,
    }
}

fn candidate(
    source: &DurableNumericFactor,
    blocks: &mut [Option<SparseTensor>; 4],
    dense_blocks: &mut [Option<NativeDenseFactorMatrix>; 4],
    maps: &mut [Option<HostIndexBuffer>; 4],
    row_scale: Option<HostNumericBuffer>,
    path: &str,
) -> Result<NumericFactorCandidate, ValueCodecError> {
    match source {
        DurableNumericFactor::DenseLu { .. } => Ok(NumericFactorCandidate::DenseLu {
            packed: take_dense(dense_blocks, 0, path)?,
            rows: take_permutation(maps, 0, path)?,
        }),
        DurableNumericFactor::DenseCholesky { .. } => Ok(NumericFactorCandidate::DenseCholesky {
            lower: take_dense(dense_blocks, 0, path)?,
        }),
        DurableNumericFactor::DenseQr(first) => Ok(NumericFactorCandidate::DenseQr(
            take_dense_orthogonal(dense_blocks, maps, 0, 0, first.rank, path)?,
        )),
        DurableNumericFactor::DenseCod { first, completion } => {
            Ok(NumericFactorCandidate::DenseCod {
                first: take_dense_orthogonal(dense_blocks, maps, 0, 0, first.rank, path)?,
                completion: completion
                    .as_ref()
                    .map(|next| take_dense_orthogonal(dense_blocks, maps, 2, 2, next.rank, path))
                    .transpose()?,
            })
        }
        DurableNumericFactor::Empty {
            kind,
            rows,
            columns,
            class,
        } => Ok(NumericFactorCandidate::Empty {
            kind: match kind {
                DurableEmptyFactorKind::Lu => NumericFactorKind::Lu,
                DurableEmptyFactorKind::Cholesky => NumericFactorKind::Cholesky,
                DurableEmptyFactorKind::Qr => NumericFactorKind::Qr,
                DurableEmptyFactorKind::Cod => NumericFactorKind::Cod,
            },
            rows: usize::try_from(*rows)
                .map_err(|_| ValueCodecError::invalid(path, "factor rows exceed host bounds"))?,
            columns: usize::try_from(*columns)
                .map_err(|_| ValueCodecError::invalid(path, "factor columns exceed host bounds"))?,
            class: match class {
                ElementType::F32 => ExactHostSparseClass::Numeric(NumericDType::F32),
                ElementType::F64 => ExactHostSparseClass::Numeric(NumericDType::F64),
                ElementType::ComplexF32 => ExactHostSparseClass::Complex(NumericDType::F32),
                ElementType::ComplexF64 => ExactHostSparseClass::Complex(NumericDType::F64),
                _ => {
                    return Err(ValueCodecError::invalid(
                        path,
                        "factor empty class is not floating",
                    ))
                }
            },
        }),
        DurableNumericFactor::Lu { .. } => Ok(NumericFactorCandidate::Lu {
            lower: take_block(blocks, 0, path)?,
            upper: take_block(blocks, 1, path)?,
            rows: take_permutation(maps, 0, path)?,
            columns: take_permutation(maps, 2, path)?,
            row_scale,
        }),
        DurableNumericFactor::Cholesky { .. } => Ok(NumericFactorCandidate::Cholesky {
            lower: take_block(blocks, 0, path)?,
            permutation: take_permutation(maps, 0, path)?,
        }),
        DurableNumericFactor::Qr(first) => Ok(NumericFactorCandidate::Qr(take_orthogonal(
            blocks, maps, 0, 0, first.rank, path,
        )?)),
        DurableNumericFactor::Cod { first, completion } => Ok(NumericFactorCandidate::Cod {
            first: take_orthogonal(blocks, maps, 0, 0, first.rank, path)?,
            completion: completion
                .as_ref()
                .map(|next| take_orthogonal(blocks, maps, 2, 2, next.rank, path))
                .transpose()?,
        }),
    }
}

fn take_block(
    blocks: &mut [Option<SparseTensor>; 4],
    index: usize,
    path: &str,
) -> Result<SparseTensor, ValueCodecError> {
    blocks[index]
        .take()
        .ok_or_else(|| ValueCodecError::invalid(path, "factor block is missing"))
}

fn take_dense(
    blocks: &mut [Option<NativeDenseFactorMatrix>; 4],
    index: usize,
    path: &str,
) -> Result<NativeDenseFactorMatrix, ValueCodecError> {
    blocks[index]
        .take()
        .ok_or_else(|| ValueCodecError::invalid(path, "dense factor block is missing"))
}

fn take_dense_orthogonal(
    blocks: &mut [Option<NativeDenseFactorMatrix>; 4],
    maps: &mut [Option<HostIndexBuffer>; 4],
    block: usize,
    map: usize,
    rank: u64,
    path: &str,
) -> Result<DenseOrthogonalFactorData, ValueCodecError> {
    Ok(DenseOrthogonalFactorData {
        packed: take_dense(blocks, block, path)?,
        coefficients: take_dense(blocks, block + 1, path)?,
        permutation: take_permutation(maps, map, path)?,
        rank: usize::try_from(rank)
            .map_err(|_| ValueCodecError::invalid(path, "dense rank exceeds platform limits"))?,
    })
}

fn take_permutation(
    maps: &mut [Option<HostIndexBuffer>; 4],
    index: usize,
    path: &str,
) -> Result<FactorPermutation, ValueCodecError> {
    Ok(FactorPermutation {
        forward: maps[index]
            .take()
            .ok_or_else(|| ValueCodecError::invalid(path, "factor forward map is missing"))?,
        inverse: maps[index + 1]
            .take()
            .ok_or_else(|| ValueCodecError::invalid(path, "factor inverse map is missing"))?,
    })
}

fn take_orthogonal(
    blocks: &mut [Option<SparseTensor>; 4],
    maps: &mut [Option<HostIndexBuffer>; 4],
    block: usize,
    map: usize,
    rank: u64,
    path: &str,
) -> Result<OrthogonalFactorData, ValueCodecError> {
    Ok(OrthogonalFactorData {
        r: take_block(blocks, block, path)?,
        reflectors: take_block(blocks, block + 1, path)?,
        permutation: take_permutation(maps, map, path)?,
        rank: usize::try_from(rank)
            .map_err(|_| ValueCodecError::invalid(path, "factor rank exceeds platform limits"))?,
    })
}
