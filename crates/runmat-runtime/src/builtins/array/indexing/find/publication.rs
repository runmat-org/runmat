//! The default Runtime binding transfers selected provider handles exactly once.
//! Public deferred FindEval accessors keep their existing borrowed API.
use super::*;
#[cfg(test)]
mod tests;

fn reject_value(value: Value, error: RuntimeError) -> RuntimeError {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::reject_gpu_output(handle, error),
        _ => error,
    }
}

/// Only genuinely discarded references enter this array. Selected originals
/// have already moved to the result, never cloned into a second publication.
fn settle<T>(
    output: T,
    discarded: [Option<runmat_accelerate_api::GpuTensorHandle>; 4],
    reject: impl Fn(T, RuntimeError) -> RuntimeError,
) -> crate::BuiltinResult<T> {
    let mut result = Ok(output);
    for handle in discarded.into_iter().flatten() {
        result = match result {
            Ok(value) => match handle.release() {
                Ok(_) => Ok(value),
                Err(source) => {
                    let mut error = find_provider_output_error(
                        "find: discarded provider result could not retire",
                    );
                    error.source = Some(Box::new(source));
                    Err(reject(value, error))
                }
            },
            Err(error) => Err(gpu_helpers::reject_gpu_output(handle, error)),
        };
    }
    result
}

fn provider_publication(
    result: ProviderFindResult,
    count: Option<usize>,
) -> crate::BuiltinResult<Value> {
    let ProviderFindResult {
        linear,
        rows,
        cols,
        values,
    } = result;
    let values = match values {
        Some(values) => values,
        None => {
            return Err(reject_find_result(
                ProviderFindResult {
                    linear,
                    rows,
                    cols,
                    values: None,
                },
                find_provider_output_error("find: provider omitted values"),
            ))
        }
    };
    let Some(count) = count else {
        return settle(
            Value::GpuTensor(linear),
            [None, Some(rows), Some(cols), Some(values)],
            reject_value,
        );
    };
    let (outputs, discarded) = match count {
        0 => (
            Vec::new(),
            [Some(linear), Some(rows), Some(cols), Some(values)],
        ),
        1 => (
            vec![Value::GpuTensor(linear)],
            [None, Some(rows), Some(cols), Some(values)],
        ),
        2 => (
            vec![Value::GpuTensor(rows), Value::GpuTensor(cols)],
            [Some(linear), None, None, Some(values)],
        ),
        _ => (
            vec![
                Value::GpuTensor(rows),
                Value::GpuTensor(cols),
                Value::GpuTensor(values),
            ],
            [Some(linear), None, None, None],
        ),
    };
    let outputs = settle(outputs, discarded, retire_unpublished_find_values)?;
    Ok(crate::output_count::output_list_with_padding(
        count, outputs,
    ))
}

impl FindEval {
    pub(super) fn into_publication(self, count: Option<usize>) -> crate::BuiltinResult<Value> {
        match self {
            Self {
                inner: FindEvalInner::Gpu { result },
                ..
            } => provider_publication(result, count),
            host => match count {
                None => host.linear_value(),
                Some(0) => Ok(Value::output_list(Vec::new())),
                Some(1) => {
                    let value = host.linear_value()?;
                    Ok(crate::output_count::output_list_with_padding(
                        1,
                        vec![value],
                    ))
                }
                Some(count) => {
                    let outputs = host.output_values(count)?;
                    Ok(crate::output_count::output_list_with_padding(
                        count, outputs,
                    ))
                }
            },
        }
    }
}
