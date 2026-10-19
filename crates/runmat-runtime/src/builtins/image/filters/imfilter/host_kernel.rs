//! Imfilter's output-class adaptation around the shared pure filter plan.

use super::{filter_error, ImageOutputClass};
use crate::builtins::common::tensor;
use crate::builtins::image::filters::kernel::{build_imfilter_plan, FilterAccumulation};
use crate::BuiltinResult;
use runmat_accelerate_api::ImfilterOptions;
use runmat_value::{LogicalArray, NumericDType, NumericStorage, Tensor, Value};

pub(super) fn fallback_accumulation(
    output_class: ImageOutputClass,
    explicit_gpu_array: bool,
) -> FilterAccumulation {
    if !explicit_gpu_array {
        FilterAccumulation::HostDouble
    } else {
        match output_class {
            ImageOutputClass::Integer(NumericDType::U32) | ImageOutputClass::Double => {
                FilterAccumulation::HostDouble
            }
            ImageOutputClass::Integer(_) | ImageOutputClass::Logical | ImageOutputClass::Single => {
                FilterAccumulation::GpuSingle
            }
        }
    }
}

pub(super) fn apply_imfilter_host(
    image: &Tensor,
    kernel: &Tensor,
    options: &ImfilterOptions,
    output_class: ImageOutputClass,
    accumulation: FilterAccumulation,
    builtin: &str,
) -> BuiltinResult<Value> {
    let plan = build_imfilter_plan(image.shape(), kernel, options, builtin)?;
    let image_values = tensor::tensor_values_f64_cow(image);
    let data = plan.evaluate_with_accumulation(&image_values, options, accumulation);
    let shape = plan.final_shape;

    let storage = match output_class {
        ImageOutputClass::Double => NumericStorage::F64(data),
        ImageOutputClass::Single => {
            NumericStorage::F32(data.into_iter().map(|value| value as f32).collect())
        }
        ImageOutputClass::Integer(dtype) => integer_filter_storage(data, dtype),
        ImageOutputClass::Logical => {
            let bits = data
                .into_iter()
                .map(|value| u8::from(round_and_clamp(value, 0.0, 1.0) != 0.0))
                .collect();
            let logical = LogicalArray::new(bits, shape)
                .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
            return Ok(if logical.data.len() == 1 {
                Value::Bool(logical.data[0] != 0)
            } else {
                Value::LogicalArray(logical)
            });
        }
    };
    let tensor = Tensor::from_numeric_storage(storage, shape)
        .map_err(|error| filter_error(builtin, format!("{builtin}: {error}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn integer_filter_storage(data: Vec<f64>, dtype: NumericDType) -> NumericStorage {
    macro_rules! signed {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| {
                        round_and_clamp(value, <$ty>::MIN as f64, <$ty>::MAX as f64) as $ty
                    })
                    .collect(),
            )
        };
    }
    macro_rules! unsigned {
        ($variant:ident, $ty:ty) => {
            NumericStorage::$variant(
                data.into_iter()
                    .map(|value| round_and_clamp(value, 0.0, <$ty>::MAX as f64) as $ty)
                    .collect(),
            )
        };
    }
    match dtype {
        NumericDType::I8 => signed!(I8, i8),
        NumericDType::I16 => signed!(I16, i16),
        NumericDType::I32 => signed!(I32, i32),
        NumericDType::U8 => unsigned!(U8, u8),
        NumericDType::U16 => unsigned!(U16, u16),
        NumericDType::U32 => unsigned!(U32, u32),
        NumericDType::I64 | NumericDType::U64 | NumericDType::F32 | NumericDType::F64 => {
            unreachable!("unsupported imfilter output class rejected before finalization")
        }
    }
}

fn round_and_clamp(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        0.0_f64.clamp(minimum, maximum)
    } else {
        value.round().clamp(minimum, maximum)
    }
}
