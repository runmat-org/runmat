use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
use runmat_builtins::{BuiltinCatalogEntry, BuiltinErrorDescriptor};
use runmat_value::{NumericDType, NumericScalar, Tensor, Value};

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

pub(super) struct RandomBoundary {
    entry: &'static BuiltinCatalogEntry,
    invalid: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
}

impl RandomBoundary {
    pub(super) const fn new(
        entry: &'static BuiltinCatalogEntry,
        invalid: &'static BuiltinErrorDescriptor,
        internal: &'static BuiltinErrorDescriptor,
        too_many_outputs: &'static BuiltinErrorDescriptor,
    ) -> Self {
        Self {
            entry,
            invalid,
            internal,
            too_many_outputs,
        }
    }

    pub(super) fn name(&self) -> &'static str {
        self.entry.identity.name
    }

    pub(super) fn reject_excess_outputs(&self) -> BuiltinResult<()> {
        if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
            return Err(self.error(self.too_many_outputs, "only one output is defined"));
        }
        Ok(())
    }

    pub(super) fn invalid(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.invalid, detail)
    }

    pub(super) fn internal(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.internal, detail)
    }

    fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    pub(super) async fn prepare(
        &self,
        args: &[Value],
        first_role: &str,
        second_role: &str,
    ) -> BuiltinResult<PreparedRandomArgs> {
        let output = RandomOutputPlan::inspect(self, args)?;
        let first = self.value_to_tensor(&args[0]).await?;
        let second = self.value_to_tensor(&args[1]).await?;
        self.ensure_exact_integer_boundary(&first, first_role)?;
        self.ensure_exact_integer_boundary(&second, second_role)?;
        let (first, second, parameter_shape) =
            tensor::binary_numeric_tensors(&first, &second, self.name(), self.name())
                .map_err(|error| self.invalid(error.message()))?;
        let shape = if args.len() > 2 {
            self.parse_shape_args(&args[2..]).await?
        } else {
            normalize_shape(parameter_shape.clone())
        };
        if (first.len() != 1 || second.len() != 1) && normalize_shape(parameter_shape) != shape {
            return Err(self.invalid("requested size must match each nonscalar parameter"));
        }
        Ok(PreparedRandomArgs {
            random: RandomArgs {
                first,
                second,
                shape,
            },
            output,
        })
    }

    async fn value_to_tensor(&self, value: &Value) -> BuiltinResult<Tensor> {
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|error| self.invalid(error.message()))?;
        tensor::value_into_tensor_for(self.name(), gathered).map_err(|error| self.invalid(error))
    }

    pub(super) fn ensure_exact_integer_boundary(
        &self,
        value: &Tensor,
        role: &str,
    ) -> BuiltinResult<()> {
        let Some(storage) = value.integer_storage() else {
            return Ok(());
        };
        if storage
            .exact_values()
            .iter()
            .any(|integer| !crate::builtins::common::validation::integer_is_exact_f64(integer))
        {
            return Err(self.invalid(format!(
                "integer {role} values must be exactly representable as double"
            )));
        }
        Ok(())
    }

    async fn parse_shape_args(&self, values: &[Value]) -> BuiltinResult<Vec<usize>> {
        let mut dimensions = Vec::new();
        for value in values {
            let parsed = self.parse_shape_value(value).await?;
            if values.len() > 1 && parsed.len() != 1 {
                return Err(self.invalid("separate size arguments must be scalars"));
            }
            dimensions.extend(parsed);
        }
        Ok(normalize_dims(dimensions))
    }

    async fn parse_shape_value(&self, value: &Value) -> BuiltinResult<Vec<usize>> {
        let tensor = self.value_to_tensor(value).await?;
        if tensor.len() > 1 && !(tensor.shape.len() == 1 || tensor.shape.first() == Some(&1)) {
            return Err(self.invalid("size vector must be a row vector"));
        }
        (0..tensor.len())
            .map(|index| {
                let scalar = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| self.internal("size tensor storage is inconsistent"))?;
                parse_size_scalar(scalar).ok_or_else(|| {
                    self.invalid(
                        "size values must be finite integers in the supported dimension range",
                    )
                })
            })
            .collect()
    }

    pub(super) fn checked_element_count(&self, shape: &[usize]) -> BuiltinResult<usize> {
        shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| self.invalid("requested size exceeds the supported array bounds"))
        })
    }
}

pub(super) struct PreparedRandomArgs {
    pub(super) random: RandomArgs,
    output: RandomOutputPlan,
}

impl PreparedRandomArgs {
    pub(super) fn finish(self, boundary: &RandomBoundary, data: Vec<f64>) -> BuiltinResult<Value> {
        self.output.finish(boundary, data, self.random.shape)
    }
}

pub(super) struct RandomArgs {
    pub(super) first: Vec<f64>,
    pub(super) second: Vec<f64>,
    pub(super) shape: Vec<usize>,
}

struct RandomOutputPlan {
    single: bool,
    source: Option<GpuTensorHandle>,
}

impl RandomOutputPlan {
    fn inspect(boundary: &RandomBoundary, args: &[Value]) -> BuiltinResult<Self> {
        let single = args.iter().take(2).any(|value| {
            matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
                || matches!(value, Value::GpuTensor(handle)
                    if runmat_accelerate_api::handle_integer_type(handle).is_none()
                        && !runmat_accelerate_api::handle_is_logical(handle)
                        && runmat_accelerate_api::handle_storage(handle)
                            == runmat_accelerate_api::GpuTensorStorage::Real
                        && runmat_accelerate_api::handle_precision(handle)
                            == Some(ProviderPrecision::F32))
        });
        let source = gpu_helpers::select_resident_output_source(
            args.iter().take(2).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            boundary.name(),
        )
        .map_err(|error| boundary.internal(error.message()))?;
        Ok(Self { single, source })
    }

    fn finish(
        self,
        boundary: &RandomBoundary,
        data: Vec<f64>,
        shape: Vec<usize>,
    ) -> BuiltinResult<Value> {
        let host = if self.single {
            Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
                .map(Value::Tensor)
                .map_err(|error| boundary.internal(error))?
        } else {
            Tensor::new(data, shape)
                .map(tensor::tensor_into_value)
                .map_err(|error| boundary.internal(error))?
        };
        let Some(source) = self.source else {
            return Ok(host);
        };
        let restored = gpu_helpers::restore_class_preserving_value(&source, host, boundary.name())
            .map_err(|error| boundary.internal(error.message()))?;
        if runmat_accelerate_api::handle_is_explicit(&source)
            && !matches!(restored, Value::GpuTensor(_))
        {
            return Err(
                boundary.internal("provider cannot preserve explicit gpuArray output residency")
            );
        }
        Ok(restored)
    }
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

fn normalize_dims(dimensions: Vec<usize>) -> Vec<usize> {
    if dimensions.is_empty() {
        vec![0, 0]
    } else if dimensions.len() == 1 {
        vec![dimensions[0], dimensions[0]]
    } else {
        normalize_shape(dimensions)
    }
}

fn parse_size_scalar(value: NumericScalar) -> Option<usize> {
    match value {
        NumericScalar::I8(value) => signed_size(i128::from(value)),
        NumericScalar::I16(value) => signed_size(i128::from(value)),
        NumericScalar::I32(value) => signed_size(i128::from(value)),
        NumericScalar::I64(value) => signed_size(i128::from(value)),
        NumericScalar::U8(value) => unsigned_size(u128::from(value)),
        NumericScalar::U16(value) => unsigned_size(u128::from(value)),
        NumericScalar::U32(value) => unsigned_size(u128::from(value)),
        NumericScalar::U64(value) => unsigned_size(u128::from(value)),
        NumericScalar::F32(value) => floating_size(f64::from(value)),
        NumericScalar::F64(value) => floating_size(value),
    }
}

fn signed_size(value: i128) -> Option<usize> {
    if value <= 0 {
        Some(0)
    } else {
        usize::try_from(value).ok()
    }
}

fn unsigned_size(value: u128) -> Option<usize> {
    usize::try_from(value).ok()
}

fn floating_size(value: f64) -> Option<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if value <= 0.0 {
        return Some(0);
    }
    if value >= usize::MAX as f64 {
        return None;
    }
    Some(value as usize)
}
