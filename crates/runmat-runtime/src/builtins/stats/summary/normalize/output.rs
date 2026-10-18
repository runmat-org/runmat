//! Requested normalization outputs and their all-or-nothing residency restoration.

use super::{normalize_internal, value_to_single, NormalizeEval};
use crate::builtins::common::gpu_helpers;
use crate::{BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::{NumericDType, Value};

#[derive(Clone, Copy)]
enum RequestedOutputs {
    ImplicitOne,
    Discard,
    Explicit(usize),
}

impl RequestedOutputs {
    fn current() -> Self {
        match crate::output_count::current_output_count() {
            None => Self::ImplicitOne,
            Some(0) => Self::Discard,
            Some(count) => Self::Explicit(count),
        }
    }

    fn selected_count(self) -> usize {
        match self {
            Self::ImplicitOne => 1,
            Self::Discard => 0,
            Self::Explicit(count) => count.min(3),
        }
    }

    fn publish(self, mut outputs: Vec<Value>) -> Value {
        match self {
            Self::ImplicitOne => outputs.remove(0),
            Self::Discard => Value::output_list(Vec::new()),
            Self::Explicit(count) => crate::output_count::output_list_with_padding(count, outputs),
        }
    }
}

pub(super) struct NormalizeOutputPlan {
    single: bool,
    source: Option<GpuTensorHandle>,
    requested: RequestedOutputs,
}

impl NormalizeOutputPlan {
    pub(super) fn inspect(value: &Value) -> BuiltinResult<Self> {
        let single = matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            || matches!(value, Value::ComplexTensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)
            || matches!(value, Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(handle)
                    && runmat_accelerate_api::handle_precision(handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F32));
        let source = gpu_helpers::select_resident_output_source(
            std::iter::once(value).filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle.clone()),
                _ => None,
            }),
            "normalize",
        )
        .map_err(output_error)?;
        Ok(Self {
            single,
            source,
            requested: RequestedOutputs::current(),
        })
    }

    pub(super) fn finish(self, eval: NormalizeEval) -> BuiltinResult<Value> {
        finish_selected(self.requested, eval, |value, protected| {
            self.restore(value, protected)
        })
    }

    fn restore(&self, value: Value, protected: &[&GpuTensorHandle]) -> BuiltinResult<Value> {
        let value = if self.single {
            value_to_single(value)?
        } else {
            value
        };
        match &self.source {
            Some(source) => {
                let restored = gpu_helpers::restore_class_preserving_value_protecting(
                    source,
                    value,
                    "normalize",
                    protected,
                )
                .map_err(output_error)?;
                if runmat_accelerate_api::handle_is_explicit(source)
                    && !matches!(restored, Value::GpuTensor(_))
                {
                    return Err(normalize_internal(
                        "normalize: provider cannot preserve explicit gpuArray output residency",
                    ));
                }
                Ok(restored)
            }
            None => Ok(value),
        }
    }
}

fn output_error(error: RuntimeError) -> RuntimeError {
    super::normalize_transfer_error(error)
}

fn finish_selected(
    requested: RequestedOutputs,
    eval: NormalizeEval,
    mut restore: impl FnMut(Value, &[&GpuTensorHandle]) -> BuiltinResult<Value>,
) -> BuiltinResult<Value> {
    let mut staged = RestoredOutputs::default();
    for value in [eval.n, eval.c, eval.s]
        .into_iter()
        .take(requested.selected_count())
    {
        let protected = staged.resident_handles();
        let restored = match restore(value, &protected) {
            Ok(value) => value,
            Err(error) => return Err(staged.reject(error)),
        };
        staged.push(restored);
    }
    Ok(requested.publish(staged.publish()))
}

/// Staged values become public only after every selected output was restored.
/// On failure, canonical rejection consumes each owned resident reference and
/// retains any cleanup failure alongside the original error.
#[derive(Default)]
struct RestoredOutputs(Vec<Value>);

impl RestoredOutputs {
    fn resident_handles(&self) -> Vec<&GpuTensorHandle> {
        self.0
            .iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            })
            .collect()
    }

    fn push(&mut self, value: Value) {
        self.0.push(value);
    }

    fn publish(self) -> Vec<Value> {
        self.0
    }

    fn reject(self, mut error: RuntimeError) -> RuntimeError {
        for value in self.0 {
            if let Value::GpuTensor(handle) = value {
                error = gpu_helpers::reject_gpu_output(handle, error);
            }
        }
        error
    }
}

#[cfg(test)]
#[path = "output_selection_tests.rs"]
mod selection_tests;

#[cfg(test)]
#[path = "output_provider_tests.rs"]
mod provider_tests;
