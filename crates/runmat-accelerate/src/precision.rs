use once_cell::sync::{Lazy, OnceCell};
use runmat_accelerate_api::{AccelProvider, ProviderPrecision};
use runmat_builtins::{NumericDType, Tensor, Value};
use std::env;

/// Return the logical numeric dtype associated with the provided value, if any.
pub fn value_numeric_dtype(value: &Value) -> Option<NumericDType> {
    match value {
        Value::Tensor(t) => Some(t.dtype),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(NumericDType::F64),
        Value::LogicalArray(_) | Value::CharArray(_) => Some(NumericDType::F64),
        Value::GpuTensor(_) => None, // already resident; assume provider handled dtype
        _ => None,
    }
}

/// Return the logical dtype represented by a tensor.
pub fn tensor_numeric_dtype(tensor: &Tensor) -> NumericDType {
    tensor.dtype
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

static ALLOW_DOWNCAST: Lazy<bool> = Lazy::new(|| {
    env::var("RUNMAT_ALLOW_PRECISION_DOWNCAST")
        .ok()
        .and_then(|value| parse_bool(&value))
        .unwrap_or(false)
});

static DOWNCAST_WARNING: OnceCell<()> = OnceCell::new();

/// True if the provider can execute kernels with the requested logical dtype.
pub fn provider_supports_dtype(provider: &dyn AccelProvider, dtype: NumericDType) -> bool {
    match dtype {
        NumericDType::F32 => true,
        NumericDType::F64 => provider.precision() == ProviderPrecision::F64,
    }
}

fn downcast_permitted_for(dtype: NumericDType) -> bool {
    matches!(dtype, NumericDType::F64) && *ALLOW_DOWNCAST
}

/// Returns an error message if the provider cannot execute the requested dtype.
pub fn ensure_provider_supports_dtype(
    provider: &dyn AccelProvider,
    dtype: NumericDType,
) -> Result<(), String> {
    if provider_supports_dtype(provider, dtype) {
        Ok(())
    } else if downcast_permitted_for(dtype) {
        DOWNCAST_WARNING.get_or_init(|| {
            log::warn!(
                "RUNMAT_ALLOW_PRECISION_DOWNCAST enabled: implicitly converting double inputs to the provider's native precision"
            );
        });
        Ok(())
    } else {
        Err(match dtype {
            NumericDType::F64 => {
                "active provider does not advertise f64 kernels; refusing implicit downcast"
                    .to_string()
            }
            NumericDType::F32 => "active provider does not support f32 kernels".to_string(),
        })
    }
}

pub fn downcast_permitted(dtype: NumericDType) -> bool {
    downcast_permitted_for(dtype)
}
