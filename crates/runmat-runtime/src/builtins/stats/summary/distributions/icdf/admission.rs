use super::*;

#[derive(Clone, Copy)]
enum OutputPrecision {
    Double,
    Single,
}

pub(super) struct IcdfAdmission {
    precision: OutputPrecision,
    gpu_source: Option<GpuTensorHandle>,
}

impl IcdfAdmission {
    pub(super) fn new(p: &Value, rest: &[Value]) -> BuiltinResult<Self> {
        let values = std::iter::once(p).chain(rest.iter());
        if values.clone().any(is_typed_integer_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTEGER_INPUT_EXTENSION,
                "icdf",
            )?;
        }
        if values.clone().any(is_logical_value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &LOGICAL_INPUT_EXTENSION,
                "icdf",
            )?;
        }
        let precision = if values.clone().any(is_single_value) {
            OutputPrecision::Single
        } else {
            OutputPrecision::Double
        };
        let gpu_source = gpu_helpers::select_resident_output_source(
            values.filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "icdf",
        )?;
        Ok(Self {
            precision,
            gpu_source,
        })
    }

    pub(super) fn finish(self, output: Value) -> BuiltinResult<Value> {
        if self.gpu_source.is_none() && matches!(self.precision, OutputPrecision::Double) {
            return Ok(output);
        }
        let tensor = match output {
            Value::Num(value) => Tensor::new(vec![value], vec![1, 1]),
            Value::Tensor(tensor) => Ok(tensor),
            other => {
                return Err(super::normal_error(
                    "icdf",
                    format!("icdf: unexpected numeric output {other:?}"),
                ))
            }
        }
        .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?;
        let tensor = match self.precision {
            OutputPrecision::Double => tensor,
            OutputPrecision::Single => Tensor::from_f32(
                tensor
                    .materialize_f64()
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                tensor.shape,
            )
            .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?,
        };
        let Some(source) = self.gpu_source else {
            return Ok(Value::Tensor(tensor));
        };
        let restored =
            gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), "icdf")
                .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?;
        if runmat_accelerate_api::handle_is_explicit(&source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(super::normal_error(
                "icdf",
                "icdf: provider cannot preserve explicit gpuArray output residency",
            ));
        }
        Ok(restored)
    }
}

pub(super) async fn gather_input(value: Value) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = &value else {
        return Ok(value);
    };
    let owner = gpu_helpers::exact_provider_for_handle(handle)
        .ok_or_else(|| super::normal_error("icdf", "icdf: resident input owner is unavailable"))?;
    gpu_helpers::download_value_preserving_residency_async(owner, handle)
        .await
        .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))
}

pub(super) fn ensure_exact_integer_boundary(value: &Value, label: &str) -> BuiltinResult<()> {
    let tensor = match value {
        Value::Int(integer) => {
            let exact = match integer {
                runmat_value::IntValue::I8(value) => i128::from(*value),
                runmat_value::IntValue::I16(value) => i128::from(*value),
                runmat_value::IntValue::I32(value) => i128::from(*value),
                runmat_value::IntValue::I64(value) => i128::from(*value),
                runmat_value::IntValue::U8(value) => i128::from(*value),
                runmat_value::IntValue::U16(value) => i128::from(*value),
                runmat_value::IntValue::U32(value) => i128::from(*value),
                runmat_value::IntValue::U64(value) => i128::from(*value),
            };
            return ensure_exact_i128(exact, label);
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor,
        _ => return Ok(()),
    };
    for index in 0..tensor.len() {
        let exact = match tensor.numeric_value_at(index) {
            Some(NumericScalar::I8(value)) => i128::from(value),
            Some(NumericScalar::I16(value)) => i128::from(value),
            Some(NumericScalar::I32(value)) => i128::from(value),
            Some(NumericScalar::I64(value)) => i128::from(value),
            Some(NumericScalar::U8(value)) => i128::from(value),
            Some(NumericScalar::U16(value)) => i128::from(value),
            Some(NumericScalar::U32(value)) => i128::from(value),
            Some(NumericScalar::U64(value)) => i128::from(value),
            _ => continue,
        };
        ensure_exact_i128(exact, label)?;
    }
    Ok(())
}

fn ensure_exact_i128(exact: i128, label: &str) -> BuiltinResult<()> {
    const MAX_EXACT: i128 = 1_i128 << 53;
    if !(-MAX_EXACT..=MAX_EXACT).contains(&exact) {
        return Err(super::normal_error(
            "icdf",
            format!("icdf: integer {label} values must be exactly representable as double"),
        ));
    }
    Ok(())
}
