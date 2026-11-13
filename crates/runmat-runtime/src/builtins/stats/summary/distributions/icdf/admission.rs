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
    pub(super) fn new(
        p: &Value,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
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
        gpu_helpers::require_resident_compute_value_iter(values.clone(), false, || {
            build_runtime_error("icdf: no acceleration provider owns a resident input")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .with_builtin("icdf")
                .build()
        })?;
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
                tensor.shape().to_vec(),
            )
            .map_err(|error| super::normal_error("icdf", format!("icdf: {error}")))?,
        };
        let Some(source) = self.gpu_source else {
            return Ok(Value::Tensor(tensor));
        };
        let restored =
            gpu_helpers::restore_class_preserving_value(&source, Value::Tensor(tensor), "icdf")
                .map_err(|error| super::with_distribution_context("icdf", error))?;
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

pub(super) async fn gather_input(
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let Value::GpuTensor(handle) = &value else {
        return Ok(value);
    };
    let readback =
        runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
            let error = build_runtime_error("icdf: resident input owner is unavailable")
                .with_identifier("RunMat:gpu:ProviderOwnershipMismatch")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_builtin("icdf")
                .build();
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    gpu_helpers::download_readback_value_in_context(runtime, &readback)
        .await
        .map_err(|error| {
            let mut error = super::with_distribution_context("icdf", error);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

pub(super) fn ensure_exact_integer_boundary(value: &Value, label: &str) -> BuiltinResult<()> {
    let tensor = match value {
        Value::Int(integer) => {
            return ensure_exact_integer(
                crate::builtins::common::validation::integer_is_exact_f64(integer),
                label,
            );
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => tensor,
        _ => return Ok(()),
    };
    for index in 0..tensor.len() {
        if let Some(exact) = tensor
            .numeric_value_at(index)
            .and_then(super::integer_numeric_scalar_is_exact_f64)
        {
            ensure_exact_integer(exact, label)?;
        }
    }
    Ok(())
}

fn ensure_exact_integer(exact: bool, label: &str) -> BuiltinResult<()> {
    if !exact {
        return Err(super::normal_error(
            "icdf",
            format!("icdf: integer {label} values must be exactly representable as double"),
        ));
    }
    Ok(())
}
