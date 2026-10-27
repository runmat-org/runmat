//! Numerical callback interpretation is distinct from callback execution.
//! Only actual undefined numeric trial values may trigger a smaller trial;
//! user exceptions, control cancellation and invalid carrier contracts escape.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{build_runtime_error, BuiltinResult};
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_types::standard;
use runmat_value::{NumericScalar, Value};

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) struct Policy {
    pub owner: BuiltinCatalogIdentity,
    pub invalid: Option<&'static BuiltinErrorDescriptor>,
    pub numeric: Option<&'static BuiltinExtensionDescriptor>,
    pub single: Option<&'static BuiltinExtensionDescriptor>,
    pub resident: Option<&'static BuiltinExtensionDescriptor>,
    pub sparse: Option<&'static BuiltinExtensionDescriptor>,
}

pub(in crate::builtins::math::optim) enum Numeric {
    Real { shape: Vec<usize>, values: Vec<f64> },
    Undefined,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim) enum Geometry {
    Array,
    MatrixAny,
    Scalar,
    Vector,
    Elements(usize),
    Matrix(usize, usize),
}

impl Geometry {
    fn accepts(self, shape: &[usize]) -> bool {
        let Some(count) = shape
            .iter()
            .try_fold(1usize, |count, &axis| count.checked_mul(axis))
        else {
            return false;
        };
        match self {
            Self::Array => true,
            Self::MatrixAny => shape.len() <= 2,
            Self::Scalar => count == 1,
            Self::Vector => {
                count == 0
                    || (shape.len() <= 2
                        && (shape.first().copied().unwrap_or(1) == 1
                            || shape.get(1).copied().unwrap_or(1) == 1))
            }
            Self::Elements(expected) => count == expected,
            Self::Matrix(rows, columns) => shape == [rows, columns] || (columns == 0 && count == 0),
        }
    }
}

pub(in crate::builtins::math::optim) fn invalid(
    policy: &Policy,
    role: &str,
    detail: &str,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {role} {detail}", policy.owner.name))
        .with_builtin(policy.owner.name)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
    if let Some(identifier) = policy.invalid.and_then(|descriptor| descriptor.identifier) {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(in crate::builtins::math::optim) fn shape(value: &Value) -> Option<Vec<usize>> {
    Some(match value {
        Value::Tensor(value) => value.shape().to_vec(),
        Value::LogicalArray(value) => value.shape.to_vec(),
        Value::SparseTensor(value) => vec![value.rows, value.cols],
        Value::GpuTensor(value) => value.shape().to_vec(),
        Value::ComplexTensor(value) => value.shape().to_vec(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => vec![1, 1],
        _ => return None,
    })
}

pub(in crate::builtins::math::optim) async fn read(
    value: Value,
    role: &str,
    geometry: Geometry,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Numeric> {
    let Some(value) = prepare(value, role, geometry, policy, cooperation).await? else {
        return Ok(Numeric::Undefined);
    };
    materialize(value, role, policy, cooperation).await
}

/// Only concrete numerical carriers reach class facts or the original provider
/// port. Kept separate so a Jacobian can retain CSC storage after this boundary.
pub(in crate::builtins::math::optim) async fn prepare(
    value: Value,
    role: &str,
    geometry: Geometry,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Value>> {
    if !preflight(&value, role, geometry, policy, cooperation.runtime())? {
        return Ok(None);
    }
    let value =
        crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await?;
    if shape(&value).is_none_or(|shape| !geometry.accepts(&shape)) {
        return Err(invalid(
            policy,
            role,
            "changed its numeric extent during gather",
        ));
    }
    if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(&value,Value::SparseTensor(value) if value.is_complex())
    {
        return Ok(None);
    }
    Ok(Some(value))
}

pub(in crate::builtins::math::optim) fn preflight(
    value: &Value,
    role: &str,
    geometry: Geometry,
    policy: &Policy,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<bool> {
    // Do not derive aggregate property/capture facts for a rejected result.
    if !matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::SparseTensor(_)
            | Value::GpuTensor(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
    ) {
        return Err(invalid(policy, role, "must be a numeric result"));
    }
    if !geometry.accepts(&shape(value).expect("concrete numerical carrier")) {
        return Err(invalid(policy, role, "has invalid scalar/vector extent"));
    }
    if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value,Value::SparseTensor(value) if value.is_complex())
    {
        return Ok(false);
    }
    let integer_or_logical = common::is_typed_integer_value(value)
        || common::is_logical_value(value)
        || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_some() || t.is_logical());
    // Absent extension authority is a primary class restriction. Reject it
    // from concrete host/owned-device metadata before provider ownership
    // queries or transfers. Existing Some paths retain their original order.
    if policy.numeric.is_none() && integer_or_logical {
        return Err(invalid(
            policy,
            role,
            "integer/logical values require an explicit numeric-class extension",
        ));
    }
    if policy.single.is_none() {
        let class = match value {
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_storage(handle)
                    != runmat_accelerate_api::GpuTensorStorage::Real
                {
                    return Ok(false);
                }
                runmat_accelerate_api::handle_class_identity(handle)
                    .ok_or_else(|| invalid(policy, role, "has no owned numeric class metadata"))?
            }
            _ => crate::builtins::introspection::class_identity::class_identity_for_value(value),
        };
        if class == standard::SINGLE {
            return Err(invalid(
                policy,
                role,
                "single values require an explicit single-class extension",
            ));
        }
    }
    common::preflight_host_residency(policy.owner.name, value, policy.resident, runtime)?;
    if matches!(value, Value::SparseTensor(_)) {
        if let Some(extension) = policy.sparse {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                extension,
                policy.owner.name,
            )?;
        }
    }

    if integer_or_logical {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            policy.numeric.ok_or_else(|| {
                invalid(
                    policy,
                    role,
                    "integer/logical values require an explicit numeric-class extension",
                )
            })?,
            policy.owner.name,
        )?;
    }
    let class = crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(value)?;
    if class == standard::SINGLE {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            policy.single.ok_or_else(|| {
                invalid(
                    policy,
                    role,
                    "single values require an explicit single-class extension",
                )
            })?,
            policy.owner.name,
        )?;
    }
    Ok(true)
}

pub(in crate::builtins::math::optim) async fn materialize(
    value: Value,
    role: &str,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Numeric> {
    let value = materialize_raw(value, role, policy, cooperation).await?;
    if let Numeric::Real { values, .. } = &value {
        for chunk in values.chunks(1024) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Ok(Numeric::Undefined);
            }
            cooperation.completed(chunk.len() as u64).await?;
        }
    }
    Ok(value)
}

/// Bounds may contain infinity. Their parser checks NaN and sidedness after
/// decoding; this path never converts an aggregate or changes provider intent.
pub(in crate::builtins::math::optim) async fn materialize_raw(
    value: Value,
    role: &str,
    policy: &Policy,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Numeric> {
    let (shape, values) = match value {
        Value::Num(value) => (vec![1, 1], vec![value]),
        Value::Int(value) => (
            vec![1, 1],
            vec![value.to_exact_f64().ok_or_else(|| {
                invalid(
                    policy,
                    role,
                    "integer must be exactly representable as double",
                )
            })?],
        ),
        Value::Bool(value) => (vec![1, 1], vec![if value { 1.0 } else { 0.0 }]),
        Value::Tensor(value) => {
            if !cooperation
                .integer_boundary(&Value::Tensor(value.clone()))
                .await?
                .is_exact()
            {
                return Err(invalid(
                    policy,
                    role,
                    "integers must be exactly representable as double",
                ));
            }
            let shape = value.shape().to_vec();
            (shape, common::tensor_values(&value, cooperation).await?)
        }
        Value::LogicalArray(value) => (
            value.shape.to_vec(),
            common::logical_values(&value.data, cooperation).await?,
        ),
        Value::SparseTensor(value) => {
            if value.is_complex() {
                return Ok(Numeric::Undefined);
            }
            let count = value
                .rows
                .checked_mul(value.cols)
                .ok_or_else(|| invalid(policy, role, "extent overflow"))?;
            let mut output = runmat_value::exact_capacity_vec(count).map_err(|source| {
                build_runtime_error("callback materialization failed")
                    .with_builtin(policy.owner.name)
                    .with_source(source)
                    .build()
            })?;
            while output.len() < count {
                let end = output.len().saturating_add(1024).min(count);
                let work = end - output.len();
                output.resize(end, 0.0);
                cooperation.completed(work as u64).await?;
            }
            let mut proof = runmat_value::SparseCscValidation::new(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                value.row_indices.clone(),
                value.nnz(),
            )
            .map_err(|detail| invalid(policy, role, &detail))?;
            let quantum = std::num::NonZeroUsize::new(1024).expect("positive work quantum");
            loop {
                let done = proof
                    .advance(quantum)
                    .map_err(|detail| invalid(policy, role, &detail))?;
                cooperation.completed(1024).await?;
                if done {
                    break;
                }
            }
            for column in 0..value.cols {
                for entry in value.col_ptrs[column]..value.col_ptrs[column + 1] {
                    let number = if value.is_logical() {
                        1.0
                    } else {
                        let scalar = value
                            .numeric_host_buffer()
                            .and_then(|buffer| buffer.value_at(entry))
                            .ok_or_else(|| invalid(policy, role, "sparse backing is absent"))?;
                        exact(scalar).ok_or_else(|| {
                            invalid(
                                policy,
                                role,
                                "integers must be exactly representable as double",
                            )
                        })?
                    };
                    output[column * value.rows + value.row_indices[entry]] = number;
                    cooperation.completed(1).await?;
                }
                cooperation.completed(1).await?;
            }
            (vec![value.rows, value.cols], output)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => return Ok(Numeric::Undefined),
        _ => {
            return Err(invalid(
                policy,
                role,
                "changed its numeric carrier during gather",
            ))
        }
    };
    Ok(Numeric::Real { shape, values })
}

pub(in crate::builtins::math::optim) fn exact(value: NumericScalar) -> Option<f64> {
    if let Some(integer) = value.into_int_value() {
        integer.to_exact_f64()
    } else {
        Some(value.materialize_f64())
    }
}
