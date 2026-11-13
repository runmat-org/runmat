//! Pricing result precision is an input contract, not a provider preference.

use super::*;
use runmat_accelerate_api::ProviderPrecision;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PricingClass {
    Single,
    Double,
}

impl PricingClass {
    pub(super) fn provider_precision(self) -> ProviderPrecision {
        match self {
            Self::Single => ProviderPrecision::F32,
            Self::Double => ProviderPrecision::F64,
        }
    }
}

pub(super) fn select(values: &[Value]) -> PricingClass {
    let mut saw_single = false;
    for value in values {
        match value {
            Value::Tensor(tensor) => match tensor.numeric_dtype() {
                NumericDType::F32 => saw_single = true,
                NumericDType::F64 if tensor.len() == 1 => {}
                _ => return PricingClass::Double,
            },
            Value::GpuTensor(handle) => {
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_is_logical(handle)
                {
                    return PricingClass::Double;
                }
                match runmat_accelerate_api::handle_precision(handle) {
                    Some(ProviderPrecision::F32) => saw_single = true,
                    Some(ProviderPrecision::F64) if handle.shape().iter().all(|&dim| dim == 1) => {}
                    _ => return PricingClass::Double,
                }
            }
            Value::Num(_) => {}
            Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => return PricingClass::Double,
            _ => return PricingClass::Double,
        }
    }
    if saw_single {
        PricingClass::Single
    } else {
        PricingClass::Double
    }
}

pub(super) fn single_input(value: f64) -> BuiltinResult<f64> {
    let rounded = value as f32;
    if value.is_finite() && (!rounded.is_finite() || (value != 0.0 && rounded == 0.0)) {
        return Err(error(
            BLSPRICE,
            "blsprice: a finite double scalar parameter is outside the single input range",
            &ERROR_INVALID_INPUT,
        ));
    }
    Ok(f64::from(rounded))
}

/// Reject known host controls before a resident operand can be downloaded.
pub(super) fn preflight_host_scalars(values: &[Value], class: PricingClass) -> BuiltinResult<()> {
    if class != PricingClass::Single {
        return Ok(());
    }
    for value in values {
        match value {
            Value::Num(number) => {
                single_input(*number)?;
            }
            Value::Tensor(tensor)
                if tensor.numeric_dtype() == NumericDType::F64 && tensor.len() == 1 =>
            {
                single_input(tensor_utils::tensor_value_f64(tensor, 0))?;
            }
            _ => {}
        }
    }
    Ok(())
}
