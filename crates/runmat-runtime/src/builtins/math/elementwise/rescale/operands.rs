use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{IntValue, NumericDType, Tensor, Value};

use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin};
use crate::BuiltinResult;

use super::{defaults, error, provider, BUILTIN_NAME};

pub(super) struct RescaleInput {
    pub tensor: Tensor,
    pub output_dtype: NumericDType,
    pub resident_source: Option<GpuTensorHandle>,
}

pub(super) struct BoundOperand {
    pub tensor: Tensor,
    pub resident_source: Option<GpuTensorHandle>,
}

impl BoundOperand {
    pub fn host(tensor: Tensor) -> Self {
        Self {
            tensor,
            resident_source: None,
        }
    }

    pub fn was_resident(&self) -> bool {
        self.resident_source.is_some()
    }
}

pub(super) async fn input(value: Value) -> BuiltinResult<RescaleInput> {
    let (value, resident_source) = gather(value).await?;
    let (tensor, output_dtype) = match value {
        Value::Tensor(tensor) => {
            let dtype = if tensor.numeric_dtype() == NumericDType::F32 {
                NumericDType::F32
            } else {
                NumericDType::F64
            };
            (tensor, dtype)
        }
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .map(|&flag| if flag != 0 { 1.0 } else { 0.0 })
                .collect();
            (
                Tensor::new(data, logical.shape).map_err(error::internal)?,
                NumericDType::F64,
            )
        }
        Value::Num(number) => (defaults::scalar_tensor(number), NumericDType::F64),
        Value::Int(integer) => (
            defaults::scalar_tensor(integer_to_f64(&integer)),
            NumericDType::F64,
        ),
        Value::Bool(flag) => (
            defaults::scalar_tensor(if flag { 1.0 } else { 0.0 }),
            NumericDType::F64,
        ),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(error::invalid_input(
                "complex inputs are not supported by rescale",
            ));
        }
        other => {
            return Err(error::invalid_input(format!(
                "expected real numeric or logical input, got {other:?}"
            )));
        }
    };
    Ok(RescaleInput {
        tensor,
        output_dtype,
        resident_source,
    })
}

pub(super) async fn bound(value: Value, label: &str) -> BuiltinResult<BoundOperand> {
    let (value, resident_source) = gather(value).await?;
    let tensor = match value {
        Value::Tensor(tensor) => tensor,
        Value::LogicalArray(logical) => {
            let data = logical
                .data
                .iter()
                .map(|&flag| if flag != 0 { 1.0 } else { 0.0 })
                .collect();
            Tensor::new(data, logical.shape).map_err(error::internal)?
        }
        Value::Num(number) => defaults::scalar_tensor(number),
        Value::Int(integer) => defaults::scalar_tensor(integer_to_f64(&integer)),
        Value::Bool(flag) => defaults::scalar_tensor(if flag { 1.0 } else { 0.0 }),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            return Err(error::invalid_input(format!("{label} must be real")));
        }
        other => {
            return Err(error::invalid_input(format!(
                "{label} must be numeric or logical, got {other:?}"
            )));
        }
    };
    Ok(BoundOperand {
        tensor,
        resident_source,
    })
}

pub(super) async fn ensure_integer_boundary(value: &Value, role: &str) -> BuiltinResult<()> {
    if crate::builtins::common::validation::value_has_native_integer_class(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(error::invalid_input(format!(
            "integer {role} values must be exactly representable as double"
        )));
    }
    Ok(())
}

async fn gather(value: Value) -> BuiltinResult<(Value, Option<GpuTensorHandle>)> {
    match value {
        Value::GpuTensor(handle) => {
            provider::for_handle(&handle)?;
            let tensor = gpu_helpers::gather_tensor_async(&handle)
                .await
                .map_err(|flow| map_control_flow_with_builtin(flow, BUILTIN_NAME))?;
            Ok((Value::Tensor(tensor), Some(handle)))
        }
        other => Ok((other, None)),
    }
}

fn integer_to_f64(value: &IntValue) -> f64 {
    match value {
        IntValue::I8(value) => f64::from(*value),
        IntValue::I16(value) => f64::from(*value),
        IntValue::I32(value) => f64::from(*value),
        IntValue::I64(value) => *value as f64,
        IntValue::U8(value) => f64::from(*value),
        IntValue::U16(value) => f64::from(*value),
        IntValue::U32(value) => f64::from(*value),
        IntValue::U64(value) => *value as f64,
    }
}
