//! One output owner for both ordinary and exact-admitted numeric replay.
//! Parsing and source-digest validation remain in the parent fill loop.

use super::*;
use runmat_value::ExactHostTensor2dAllocation;

pub(super) enum FillStorage {
    OrdinaryInteger(NumericStorage),
    OrdinaryFloat(Vec<f64>),
    Exact {
        storage: ExactHostTensor2dAllocation,
        dtype: NumericDType,
    },
}

impl FillStorage {
    pub(super) fn new(
        rows: usize,
        cols: usize,
        count: usize,
        integer_dtype: Option<NumericDType>,
        options: &ReadMatrixOptions,
        permit: Option<&dyn ExactHostAllocationPermit>,
    ) -> BuiltinResult<Self> {
        if let Some(permit) = permit {
            let OutputTemplate::Numeric(dtype) = &options.output_template else {
                return Err(readmatrix_error_with(
                    &READMATRIX_ERROR_OUTPUT_TYPE,
                    "readmatrix: exact numeric output requires a resolved numeric class",
                ));
            };
            return ExactHostTensor2dAllocation::new(*dtype, rows, cols, permit)
                .map(|storage| Self::Exact {
                    storage,
                    dtype: *dtype,
                })
                .map_err(|source| {
                    readmatrix_error_with_source(
                        &READMATRIX_ERROR_TENSOR_BUILD,
                        "readmatrix: unable to allocate admitted numeric output",
                        source,
                    )
                });
        }
        Ok(if let Some(dtype) = integer_dtype {
            Self::OrdinaryInteger(NumericStorage::zeros(dtype, count))
        } else {
            Self::OrdinaryFloat(vec![options.empty_value_f64(); count])
        })
    }

    pub(super) fn set(&mut self, index: usize, value: NumericScalar) -> BuiltinResult<()> {
        let result = match self {
            Self::OrdinaryInteger(storage) => storage.set_value(index, value),
            Self::OrdinaryFloat(storage) => {
                storage[index] = value.materialize_f64();
                Ok(())
            }
            Self::Exact { storage, dtype } => {
                let value = match value {
                    NumericScalar::F64(value) if *dtype == NumericDType::F32 => {
                        NumericScalar::F32(value as f32)
                    }
                    other => other,
                };
                storage.set_value(index, value)
            }
        };
        result.map_err(|source| {
            readmatrix_error_with(
                &READMATRIX_ERROR_TENSOR_BUILD,
                format!("readmatrix: {source}"),
            )
        })
    }

    pub(super) fn into_tensor(self, rows: usize, cols: usize) -> BuiltinResult<Tensor> {
        match self {
            Self::OrdinaryInteger(storage) => {
                Tensor::from_numeric_storage(storage, vec![rows, cols]).map_err(|source| {
                    readmatrix_error_with(
                        &READMATRIX_ERROR_TENSOR_BUILD,
                        format!("readmatrix: {source}"),
                    )
                })
            }
            Self::OrdinaryFloat(storage) => {
                Tensor::new(storage, vec![rows, cols]).map_err(|source| {
                    readmatrix_error_with(
                        &READMATRIX_ERROR_TENSOR_BUILD,
                        format!("readmatrix: {source}"),
                    )
                })
            }
            Self::Exact { storage, .. } => storage.into_tensor().map_err(|source| {
                readmatrix_error_with_source(
                    &READMATRIX_ERROR_TENSOR_BUILD,
                    "readmatrix: unable to publish admitted numeric output",
                    source,
                )
            }),
        }
    }
}
