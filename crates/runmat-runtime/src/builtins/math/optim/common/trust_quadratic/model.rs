//! One retained matrix or original Hinfo owner. Matrix-free products move
//! Hinfo through the rooted callback argument scope without cloning its map.
use super::*;
use crate::builtins::math::linalg::ops::linear_operator::{
    CooperativeOperatorError, LinearOperator, LinearOperatorError,
};
use crate::builtins::math::optim::common;
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use runmat_value::{HostNumericBuffer, NumericDType, NumericStorage, SparseTensor};
use runmat_value::{Tensor, Value};

pub(crate) enum Model {
    Matrix {
        source: RootedValue,
        policy: Policy,
    },
    Multiply {
        info: Option<RootedValue>,
        function: Value,
        policy: Policy,
    },
}

mod products;
pub(crate) use products::Products;

impl Model {
    pub async fn matrix(
        policy: Policy,
        source: RootedValue,
        n: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        policy.value_source()?;
        matrix_geometry(source.value(), n, policy)?;
        let already_double = matches!(source.value(), Value::Num(_))
            || matches!(source.value(),Value::Tensor(value)if value.numeric_dtype()==NumericDType::F64)
            || matches!(source.value(),Value::SparseTensor(value)if value.numeric_dtype()==Some(NumericDType::F64));
        let source = if already_double {
            source
        } else {
            let normalization_cooperation = &mut *cooperation;
            source
                .with_owned_async(|value| async move {
                    let value = matrix_value(value, n, policy, normalization_cooperation)
                        .await
                        .map_err(|error| policy.admission(error))?;
                    RootedValue::in_optional_context(value, normalization_cooperation.runtime())
                })
                .await
                .map_err(|error| policy.admission(error))?
        };
        let operator = borrow_operator(source.value(), policy, cooperation).await?;
        if operator.rows() != n
            || operator.cols() != n
            || !operator
                .is_hermitian(cooperation.runtime())
                .await
                .map_err(|source| operator_error(source, cooperation.runtime(), policy))?
        {
            return Err(
                policy.invalid("objective Hessian must be a finite real symmetric n-by-n matrix")
            );
        }
        if !matches!(operator, LinearOperator::RealF64(_)) {
            return Err(policy.invalid("Hessian normalization did not produce real double storage"));
        }
        Ok(Self::Matrix { source, policy })
    }

    pub fn multiply(info: RootedValue, function: Value, policy: Policy) -> Self {
        Self::Multiply {
            policy,
            info: Some(info),
            function,
        }
    }

    /// A borrowed product session, never stored inside its owning Model.
    /// The returned lifetime prevents model replacement/publication while
    /// its coefficient/topology slices or Hinfo slot are in use.
    pub async fn products<'a>(
        &'a mut self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Products<'a>> {
        match self {
            Self::Matrix { policy, .. } | Self::Multiply { policy, .. } => {
                policy.value_source()?;
            }
        }
        match self {
            Self::Matrix { source, policy } => {
                let policy = *policy;
                let source = source.value();
                let operator = borrow_operator(source, policy, cooperation).await?;
                let LinearOperator::RealF64(operator) = operator else {
                    return Err(policy.invalid("Hessian operator must be real double"));
                };
                Ok(Products::Matrix {
                    source,
                    operator,
                    policy,
                })
            }
            Self::Multiply {
                info,
                function,
                policy,
            } => Ok(Products::Multiply {
                info,
                function,
                policy: *policy,
            }),
        }
    }

    pub fn into_value(self) -> RootedValue {
        match self {
            Self::Matrix { source, .. } => source,
            Self::Multiply { info, .. } => {
                info.expect("completed Hinfo product restores its owner")
            }
        }
    }
}

fn dimensions(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(value) => Some(value.shape()),
        Value::LogicalArray(value) => Some(&value.shape),
        Value::GpuTensor(value) => Some(value.shape()),
        _ => None,
    }
}
fn matrix_geometry(value: &Value, n: usize, policy: Policy) -> BuiltinResult<()> {
    let valid = match value {
        Value::SparseTensor(value) => value.rows == n && value.cols == n && !value.is_complex(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => n == 1,
        value => dimensions(value).is_some_and(|shape| {
            shape.first().copied().unwrap_or(1) == n
                && shape.get(1).copied().unwrap_or(1) == n
                && shape.iter().skip(2).all(|&size| size == 1)
        }),
    };
    if valid {
        Ok(())
    } else {
        Err(policy.invalid("objective Hessian must be a real n-by-n matrix"))
    }
}
fn vector_geometry(value: &Value, n: usize, policy: Policy) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => n == 1,
        value => dimensions(value).is_some_and(|shape| {
            shape.first().copied().unwrap_or(1) == n && shape.iter().skip(1).all(|&size| size == 1)
        }),
    };
    if valid {
        Ok(())
    } else {
        Err(policy.invalid("HessianMultiplyFcn must return a real n-by-1 product"))
    }
}

async fn matrix_value(
    value: Value,
    n: usize,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    matrix_geometry(&value, n, policy)?;
    if let Value::SparseTensor(value) = value {
        if value.integer_storage().is_some() || value.is_logical() {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                cooperation.runtime(),
                policy.value_source()?.numeric,
                policy.identity.name,
            )?;
        }
        if value.numeric_dtype() == Some(NumericDType::F64) {
            return Ok(Value::SparseTensor(value));
        }
        let mut validation = runmat_value::SparseCscValidation::new(
            value.rows,
            value.cols,
            value.col_ptrs.clone(),
            value.row_indices.clone(),
            value.nnz(),
        )
        .map_err(|error| policy.invalid(error))?;
        let quantum = std::num::NonZeroUsize::new(1024).expect("nonzero sparse validation quantum");
        while !validation
            .advance(quantum)
            .map_err(|error| policy.invalid(error))?
        {
            cooperation.completed(1024).await?;
        }
        let parts = validation
            .into_validated()
            .map_err(|_| policy.invalid("Hessian CSC validation did not complete"))?;
        let mut numbers = policy.buffer(value.nnz())?;
        for first in (0..value.nnz()).step_by(1024) {
            let end = first.saturating_add(1024).min(value.nnz());
            for index in first..end {
                let number = if value.is_logical() {
                    1.0
                } else {
                    let scalar = value
                        .numeric_host_buffer()
                        .and_then(|values| values.value_at(index))
                        .ok_or_else(|| policy.invalid("invalid Hessian numeric storage"))?;
                    if let Some(integer) = scalar.into_int_value() {
                        integer.to_exact_f64().ok_or_else(|| {
                            policy
                                .invalid("integer Hessian must be exactly representable as double")
                        })?
                    } else {
                        scalar.materialize_f64()
                    }
                };
                if !number.is_finite() {
                    return Err(policy.invalid("Hessian must contain finite values"));
                }
                numbers.push(number);
            }
            cooperation.completed((end - first) as u64).await?;
        }
        return SparseTensor::from_validated_numeric(
            parts,
            HostNumericBuffer::from_numeric_storage(NumericStorage::F64(numbers)),
        )
        .map(Value::SparseTensor)
        .map_err(|error| policy.invalid(error));
    }
    let value = common::prepare_floating_value_with_cooperation(
        policy.identity.name,
        value,
        policy.value_source()?.numeric,
        policy.value_source()?.resident,
        "objective Hessian",
        cooperation,
    )
    .await?;
    matrix_geometry(&value, n, policy)?;
    if matches!(&value, Value::Num(_))
        || matches!(&value,Value::Tensor(value)if value.numeric_dtype()==NumericDType::F64)
    {
        return Ok(value);
    }
    let shape = dimensions(&value)
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| vec![1, 1]);
    let values =
        common::value_to_real_vector_with_cooperation(policy.identity.name, value, cooperation)
            .await?;
    Tensor::new(values, shape)
        .map(Value::Tensor)
        .map_err(|error| policy.invalid(error))
}

fn operator_error(
    source: CooperativeOperatorError,
    runtime: Option<&RuntimeContext>,
    policy: Policy,
) -> RuntimeError {
    policy.admission(match source {
        CooperativeOperatorError::Execution(source) => {
            source.into_runtime_error_in_context(runtime)
        }
        CooperativeOperatorError::Operator(LinearOperatorError::Admission(
            crate::context::host_allocation::HostAllocationError::Cancelled,
        )) => crate::execution::ExecutionServiceError::Cancelled
            .into_runtime_error_in_context(runtime),
        CooperativeOperatorError::Operator(LinearOperatorError::Admission(source)) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                policy.ownership_identifier,
                "trust-region operator ownership",
                source,
            )
        }
        CooperativeOperatorError::Operator(source) => {
            policy.invalid(format!("invalid trust-region matrix operator: {source:?}"))
        }
    })
}
async fn borrow_operator<'a>(
    value: &'a Value,
    policy: Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<LinearOperator<'a>> {
    let result = match cooperation.runtime() {
        Some(runtime) => LinearOperator::borrow_cooperatively(value, runtime).await,
        None => LinearOperator::borrow_unbounded(value).map_err(CooperativeOperatorError::Operator),
    };
    result.map_err(|source| operator_error(source, cooperation.runtime(), policy))
}
