//! MATLAB-compatible `histcounts` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod edges;
mod options;
use edges::{apply_normalization, compute_edges, validate_edges};
#[cfg(test)]
use options::{numeric_vector, positive_usize, scalar_value};
use options::{parse_options, HistcountsOptions};

const BUILTIN_NAME: &str = "histcounts";
const DEFAULT_BIN_COUNT: usize = 10;
fn builtin_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with(error, error.message)
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "histcounts",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::hist::histcounts"
)]
async fn histcounts_builtin(
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 3) {
        return Err(builtin_error("histcounts: too many output arguments"));
    }
    let eval = evaluate_with_bins(data, rest.view(), requested_outputs == Some(3)).await?;
    if let Some(out_count) = requested_outputs {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_counts_value()]));
        }
        if out_count == 3 {
            let (counts, edges, bins) = eval.into_triple()?;
            return Ok(Value::output_list(vec![counts, edges, bins]));
        }
        let (counts, edges) = eval.into_pair();
        return Ok(Value::output_list(vec![counts, edges]));
    }
    Ok(eval.into_counts_value())
}

/// Evaluate `histcounts` once and surface both primary outputs.
pub async fn evaluate(
    data: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<HistcountsEvaluation> {
    evaluate_with_bins(data, rest, false).await
}

async fn evaluate_with_bins(
    data: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    include_bins: bool,
) -> BuiltinResult<HistcountsEvaluation> {
    let options = parse_options(rest)?;
    match data {
        Value::GpuTensor(handle) => histcounts_gpu(handle, &options, include_bins).await,
        other => histcounts_host(other, &options, include_bins),
    }
}

async fn histcounts_gpu(
    handle: GpuTensorHandle,
    options: &HistcountsOptions,
    include_bins: bool,
) -> BuiltinResult<HistcountsEvaluation> {
    if matches!(
        runmat_accelerate_api::handle_integer_type(&handle),
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    ) {
        return Err(build_runtime_error(
            "histcounts: 64-bit integer gpuArray inputs are not supported",
        )
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    histcounts_from_tensor(tensor, options, include_bins)
}

fn histcounts_host(
    value: Value,
    options: &HistcountsOptions,
    include_bins: bool,
) -> BuiltinResult<HistcountsEvaluation> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(builtin_error)?;
    histcounts_from_tensor(tensor, options, include_bins)
}

fn histcounts_from_tensor(
    tensor: Tensor,
    options: &HistcountsOptions,
    include_bins: bool,
) -> BuiltinResult<HistcountsEvaluation> {
    let input_shape = tensor.shape().to_vec();
    let exact_samples = tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let mut values = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;
    // Public numeric edges use double. Keep exact integer observations beside
    // that representation so bin comparisons never round their source values.
    let samples = tensor::tensor_values_f64_cow(&tensor);

    for &sample in samples.iter() {
        if sample.is_nan() {
            continue;
        }
        values.push(sample);
        if sample.is_finite() {
            min_val = Some(match min_val {
                Some(current) if sample >= current => current,
                Some(_) => sample,
                None => sample,
            });
            max_val = Some(match max_val {
                Some(current) if sample <= current => current,
                Some(_) => sample,
                None => sample,
            });
        }
    }

    let original_range_zero = match (min_val, max_val) {
        (Some(minimum), Some(maximum)) => minimum == maximum,
        _ => false,
    };

    let mut edges = compute_edges(&values, min_val, max_val, original_range_zero, options)?;
    if options.explicit_edges.is_none() && options.bin_limits.is_none() {
        if let Some(exact) = exact_samples.as_deref() {
            super::integer_bins::cover_exact_integer_range(&mut edges, exact);
        }
    }
    let mut counts = vec![0.0f64; edges.len() - 1];

    let mut bin_indices = include_bins.then(|| vec![0.0; samples.len()]);
    for (position, &value) in samples.iter().enumerate() {
        let index = if let Some(integer) = exact_samples.as_ref().and_then(|v| v.get(position)) {
            if let Some(exact_edges) = options.exact_edges.as_deref() {
                super::integer_bins::exact_edge_bin(integer, exact_edges)
            } else {
                super::integer_bins::floating_edge_bin(integer, &edges)
            }
        } else if let Some(exact_edges) = options.exact_edges.as_deref() {
            super::integer_bins::floating_observation_exact_bin(value, exact_edges)
        } else {
            histogram_bin_index(value, &edges)
        };
        if let Some(index) = index {
            counts[index] += 1.0;
            if let Some(indices) = bin_indices.as_mut() {
                indices[position] = (index + 1) as f64;
            }
        }
    }

    let normalised = apply_normalization(
        &counts,
        &edges,
        options.exact_edges.as_deref(),
        options.normalization,
    );
    let counts_tensor = Tensor::new(normalised, vec![1, counts.len()])
        .map_err(|e| builtin_error(format!("histcounts: {e}")))?;
    let edges_tensor = Tensor::new(edges.clone(), vec![1, edges.len()])
        .map_err(|e| builtin_error(format!("histcounts: {e}")))?;

    let bins_tensor = bin_indices
        .map(|indices| Tensor::new(indices, input_shape))
        .transpose()
        .map_err(|e| builtin_error(format!("histcounts: {e}")))?;

    Ok(HistcountsEvaluation::new(
        counts_tensor,
        edges_tensor,
        bins_tensor,
    ))
}

fn histogram_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value.is_nan() || value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => Some(index),
        Err(index) if index > 0 && index < edges.len() => Some(index - 1),
        _ => None,
    }
}

fn validate_bin_edges(edges: &[f64], exact: Option<&[IntValue]>) -> BuiltinResult<()> {
    if let Some(exact) = exact {
        if !super::integer_bins::validate_exact_edges(exact) {
            return Err(builtin_error(
                "histcounts: BinEdges must be strictly increasing",
            ));
        }
        Ok(())
    } else {
        validate_edges(edges)
    }
}

#[derive(Clone)]
pub struct HistcountsEvaluation {
    counts: Tensor,
    edges: Tensor,
    bins: Option<Tensor>,
}

impl HistcountsEvaluation {
    fn new(counts: Tensor, edges: Tensor, bins: Option<Tensor>) -> Self {
        Self {
            counts,
            edges,
            bins,
        }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let edges = self.edges;
        (Value::Tensor(counts), Value::Tensor(edges))
    }

    fn into_triple(self) -> BuiltinResult<(Value, Value, Value)> {
        let bins = self.bins.ok_or_else(|| {
            builtin_error("histcounts: bin indices were not prepared for the requested output")
        })?;
        Ok((
            Value::Tensor(self.counts),
            Value::Tensor(self.edges),
            Value::Tensor(bins),
        ))
    }
}

#[cfg(test)]
mod tests;
