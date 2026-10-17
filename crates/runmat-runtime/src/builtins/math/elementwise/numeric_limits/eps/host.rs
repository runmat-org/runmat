use runmat_value::{spacing_f32, spacing_f64, ComplexTensor, NumericDType, Tensor, Value};

use crate::context::RuntimeContext;
use crate::BuiltinResult;

pub(super) async fn real(input: &Tensor, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
    match input.numeric_dtype() {
        NumericDType::F64 => {
            let values = input
                .as_f64_slice()
                .ok_or_else(|| super::super::errors::internal("eps", "invalid double tensor"))?;
            super::output::output_f64(input.shape(), values.len(), runtime, |start, output| {
                let end = start + output.len();
                for (out, value) in output.iter_mut().zip(&values[start..end]) {
                    *out = spacing_f64(*value);
                }
                output.len()
            })
            .await
        }
        NumericDType::F32 => {
            let values = input
                .as_f32_slice()
                .ok_or_else(|| super::super::errors::internal("eps", "invalid single tensor"))?;
            super::output::output_f32(input.shape(), values.len(), runtime, |start, output| {
                let end = start + output.len();
                for (out, value) in output.iter_mut().zip(&values[start..end]) {
                    *out = spacing_f32(*value);
                }
                output.len()
            })
            .await
        }
        _ => Err(super::super::errors::class(
            "eps",
            "eps requires a single or double input",
        )),
    }
}

pub(super) async fn complex(
    input: &ComplexTensor,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Value> {
    match input.numeric_dtype() {
        NumericDType::F64 => {
            let values = input.as_f64_slice().ok_or_else(|| {
                super::super::errors::internal("eps", "invalid complex double tensor")
            })?;
            super::output::output_f64(input.shape(), values.len(), runtime, |start, output| {
                let end = start + output.len();
                for (out, value) in output.iter_mut().zip(&values[start..end]) {
                    *out = spacing_f64(value.0.hypot(value.1));
                }
                output.len()
            })
            .await
        }
        NumericDType::F32 => {
            let values = input.as_f32_slice().ok_or_else(|| {
                super::super::errors::internal("eps", "invalid complex single tensor")
            })?;
            super::output::output_f32(input.shape(), values.len(), runtime, |start, output| {
                let end = start + output.len();
                for (out, value) in output.iter_mut().zip(&values[start..end]) {
                    *out = spacing_f32(value.0.hypot(value.1));
                }
                output.len()
            })
            .await
        }
        _ => Err(super::super::errors::class(
            "eps",
            "eps requires a single or double input",
        )),
    }
}
