//! Closed numerical DATA encoding. No solver, issuer, or receiver crosses the wire.

use super::*;
use runmat_value::{HostIndexBuffer, NumericFactorData};

pub(super) fn encode_factor(
    factor: &NumericFactorData,
    path: &str,
) -> Result<DurableNumericFactor, ValueCodecError> {
    match factor.parts() {
        NumericFactorCandidate::Empty {
            kind,
            rows,
            columns,
            class,
        } => Ok(DurableNumericFactor::Empty {
            kind: match kind {
                NumericFactorKind::Lu => DurableEmptyFactorKind::Lu,
                NumericFactorKind::Cholesky => DurableEmptyFactorKind::Cholesky,
                NumericFactorKind::Qr => DurableEmptyFactorKind::Qr,
                NumericFactorKind::Cod => DurableEmptyFactorKind::Cod,
            },
            rows: u64::try_from(*rows)
                .map_err(|_| ValueCodecError::invalid(path, "factor rows exceed durable bounds"))?,
            columns: u64::try_from(*columns).map_err(|_| {
                ValueCodecError::invalid(path, "factor columns exceed durable bounds")
            })?,
            class: match class {
                ExactHostSparseClass::Numeric(NumericDType::F32) => ElementType::F32,
                ExactHostSparseClass::Numeric(NumericDType::F64) => ElementType::F64,
                ExactHostSparseClass::Complex(NumericDType::F32) => ElementType::ComplexF32,
                ExactHostSparseClass::Complex(NumericDType::F64) => ElementType::ComplexF64,
                _ => {
                    return Err(ValueCodecError::invalid(
                        path,
                        "factor empty class is not floating",
                    ))
                }
            },
        }),
        NumericFactorCandidate::Lu {
            lower,
            upper,
            rows,
            columns,
            row_scale,
        } => Ok(DurableNumericFactor::Lu {
            lower: Box::new(encode_sparse(lower, path)?),
            upper: Box::new(encode_sparse(upper, path)?),
            rows: permutation(rows, path)?,
            columns: permutation(columns, path)?,
            row_scale: row_scale
                .as_ref()
                .map(|scale| encode_scale(scale, path))
                .transpose()?,
        }),
        NumericFactorCandidate::Cholesky {
            lower,
            permutation: map,
        } => Ok(DurableNumericFactor::Cholesky {
            lower: encode_sparse(lower, path)?,
            permutation: permutation(map, path)?,
        }),
        NumericFactorCandidate::DenseLu { packed, rows } => Ok(DurableNumericFactor::DenseLu {
            packed: dense_matrix(packed, path)?,
            rows: permutation(rows, path)?,
        }),
        NumericFactorCandidate::DenseCholesky { lower } => {
            Ok(DurableNumericFactor::DenseCholesky {
                lower: dense_matrix(lower, path)?,
            })
        }
        NumericFactorCandidate::DenseQr(first) => Ok(DurableNumericFactor::DenseQr(Box::new(
            dense_orthogonal(first, path)?,
        ))),
        NumericFactorCandidate::DenseCod { first, completion } => {
            Ok(DurableNumericFactor::DenseCod {
                first: Box::new(dense_orthogonal(first, path)?),
                completion: completion
                    .as_ref()
                    .map(|part| dense_orthogonal(part, path).map(Box::new))
                    .transpose()?,
            })
        }
        NumericFactorCandidate::Qr(first) => Ok(DurableNumericFactor::Qr(orthogonal(first, path)?)),
        NumericFactorCandidate::Cod { first, completion } => Ok(DurableNumericFactor::Cod {
            first: Box::new(orthogonal(first, path)?),
            completion: completion
                .as_ref()
                .map(|part| orthogonal(part, path).map(Box::new))
                .transpose()?,
        }),
    }
}

fn dense_matrix(
    value: &NativeDenseFactorMatrix,
    path: &str,
) -> Result<DenseValue, ValueCodecError> {
    match value {
        NativeDenseFactorMatrix::Real(value) => encode_tensor(value, path),
        NativeDenseFactorMatrix::Complex(value) => {
            let (element_type, little_endian_data) =
                encode_complex_storage(value.complex_storage());
            Ok(DenseValue {
                element_type,
                shape: shape(value.shape(), path)?,
                little_endian_data,
            })
        }
    }
}

fn dense_orthogonal(
    value: &DenseOrthogonalFactorData,
    path: &str,
) -> Result<DurableDenseOrthogonalFactor, ValueCodecError> {
    Ok(DurableDenseOrthogonalFactor {
        packed: dense_matrix(&value.packed, path)?,
        coefficients: dense_matrix(&value.coefficients, path)?,
        permutation: permutation(&value.permutation, path)?,
        rank: u64::try_from(value.rank).map_err(|_| {
            ValueCodecError::invalid(path, "dense factor rank exceeds durable bounds")
        })?,
    })
}

fn orthogonal(
    factor: &OrthogonalFactorData,
    path: &str,
) -> Result<DurableOrthogonalFactor, ValueCodecError> {
    Ok(DurableOrthogonalFactor {
        r: encode_sparse(&factor.r, path)?,
        reflectors: encode_sparse(&factor.reflectors, path)?,
        permutation: permutation(&factor.permutation, path)?,
        rank: u64::try_from(factor.rank)
            .map_err(|_| ValueCodecError::invalid(path, "factor rank exceeds durable bounds"))?,
    })
}

fn permutation(
    map: &FactorPermutation,
    path: &str,
) -> Result<DurableFactorPermutation, ValueCodecError> {
    Ok(DurableFactorPermutation {
        forward: indices(&map.forward, path)?,
        inverse: indices(&map.inverse, path)?,
    })
}

fn indices(map: &HostIndexBuffer, path: &str) -> Result<Vec<u64>, ValueCodecError> {
    map.as_ref()
        .iter()
        .map(|&index| {
            u64::try_from(index)
                .map_err(|_| ValueCodecError::invalid(path, "factor index exceeds durable bounds"))
        })
        .collect()
}

fn encode_scale(scale: &HostNumericBuffer, path: &str) -> Result<DenseValue, ValueCodecError> {
    let (element_type, little_endian_data) = if let Some(values) = scale.as_f32_slice() {
        (
            ElementType::F32,
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        )
    } else if let Some(values) = scale.as_f64_slice() {
        (
            ElementType::F64,
            values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        )
    } else {
        return Err(ValueCodecError::invalid(
            path,
            "factor scale requires floating storage",
        ));
    };
    Ok(DenseValue {
        element_type,
        shape: vec![
            u64::try_from(scale.len()).map_err(|_| {
                ValueCodecError::invalid(path, "factor scale exceeds durable bounds")
            })?,
            1,
        ],
        little_endian_data,
    })
}
