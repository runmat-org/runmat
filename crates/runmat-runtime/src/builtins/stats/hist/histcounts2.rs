//! MATLAB-compatible `histcounts2` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, Histcounts2Option, HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
    HISTCOUNTS2_GPU_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

mod edges;
mod options;
use edges::{apply_normalization_2d, collect_axis_data, compute_edges_for_axis, find_bin_index};
use options::{is_option_key, numeric_vector, parse_options, Histcounts2Options};
#[cfg(test)]
use options::{num_bins_vector, positive_usize, scalar_value};

const NAME: &str = "histcounts2";
const DEFAULT_BIN_COUNT: usize = 10;
fn builtin_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with(error, error.message)
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runtime_builtin(
    name = "histcounts2",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::hist::histcounts2"
)]
async fn histcounts2_builtin(
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 5) {
        return Err(builtin_error("histcounts2: too many output arguments"));
    }
    parse_options(rest.view())?;
    ensure_histcounts2_public_syntax(rest.view())?;
    let eval = evaluate(x, y, rest.view()).await?;
    if let Some(out_count) = requested_outputs {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_counts_value()]));
        }
        if out_count == 2 {
            let (counts, x_edges) = eval.into_pair();
            return Ok(Value::output_list(vec![counts, x_edges]));
        }
        let (counts, x_edges, y_edges, x_bins, y_bins) = eval.into_all();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![counts, x_edges, y_edges, x_bins, y_bins],
        ));
    }
    Ok(eval.into_counts_value())
}

pub(crate) fn ensure_histcounts2_public_syntax(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let mut extension = false;
    if args.len() >= 2 && !is_option_key(&args[0]) && !is_option_key(&args[1]) {
        let first = numeric_vector(&args[0], NAME, "positional X argument")?;
        let second = numeric_vector(&args[1], NAME, "positional Y argument")?;
        extension |= first.len() == 1 && second.len() == 1;
    }
    let mut index = if args.first().is_some_and(|value| !is_option_key(value)) {
        if args.get(1).is_some_and(|value| !is_option_key(value)) {
            2
        } else {
            1
        }
    } else {
        0
    };
    while index + 1 < args.len() {
        let option =
            tensor::value_to_string(&args[index]).and_then(|name| Histcounts2Option::parse(&name));
        let value = &args[index + 1];
        extension |= option.is_some_and(Histcounts2Option::requires_runmat_compat);
        if option == Some(Histcounts2Option::BinWidth) {
            extension |= numeric_vector(value, NAME, "BinWidth")?.len() == 1;
        }
        if option == Some(Histcounts2Option::BinMethod) {
            extension |= tensor::value_to_string(value).is_some_and(|method| {
                matches!(
                    method.trim().to_ascii_lowercase().as_str(),
                    "sturges" | "sqrt"
                )
            });
        }
        index += 2;
    }
    if extension {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

/// Evaluate `histcounts2` once and surface all outputs.
pub async fn evaluate(
    x: Value,
    y: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Histcounts2Evaluation> {
    let options = parse_options(rest)?;
    preflight_observation_shapes(&x, &y)?;
    if matches!(&x, Value::GpuTensor(_)) || matches!(&y, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&HISTCOUNTS2_GPU_EXTENSION, NAME)?;
    }
    let x_tensor = match x {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    let y_tensor = match y {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    histcounts2_from_tensors(x_tensor, y_tensor, &options)
}

fn preflight_observation_shapes(x: &Value, y: &Value) -> BuiltinResult<()> {
    fn known_shape(value: &Value) -> Option<&[usize]> {
        match value {
            Value::Tensor(tensor) => Some(tensor.shape()),
            Value::LogicalArray(array) => Some(&array.shape),
            Value::GpuTensor(handle) => Some(handle.shape()),
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => Some(&[1, 1]),
            _ => None,
        }
    }

    let (Some(x_shape), Some(y_shape)) = (known_shape(x), known_shape(y)) else {
        return Ok(());
    };
    let element_count = |shape: &[usize]| {
        shape
            .iter()
            .copied()
            .try_fold(1usize, |count, dimension| count.checked_mul(dimension))
    };
    if element_count(x_shape) != element_count(y_shape) {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must contain the same number of elements"
        )));
    }
    if x_shape != y_shape {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must be the same size"
        )));
    }
    Ok(())
}

fn histcounts2_from_tensors(
    x_tensor: Tensor,
    y_tensor: Tensor,
    options: &Histcounts2Options,
) -> BuiltinResult<Histcounts2Evaluation> {
    if x_tensor.len() != y_tensor.len() {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must contain the same number of elements"
        )));
    }
    if x_tensor.shape() != y_tensor.shape() {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must be the same size"
        )));
    }

    // Histogram statistics and returned edges are defined in the floating
    // computation domain even when observations arrive in integer storage.
    let x_shape = x_tensor.shape().to_vec();
    let y_shape = y_tensor.shape().to_vec();
    let x_exact = x_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let y_exact = y_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let x_values = tensor::tensor_into_values_f64(x_tensor);
    let y_values = tensor::tensor_into_values_f64(y_tensor);
    let x_axis = collect_axis_data(&x_values);
    let y_axis = collect_axis_data(&y_values);

    let mut x_edges = compute_edges_for_axis(&x_axis, &options.x, "X")?;
    let mut y_edges = compute_edges_for_axis(&y_axis, &options.y, "Y")?;
    if options.x.explicit_edges.is_none() && options.x.bin_limits.is_none() {
        if let Some(exact) = x_exact.as_deref() {
            super::integer_bins::cover_exact_integer_range(&mut x_edges, exact);
        }
    }
    if options.y.explicit_edges.is_none() && options.y.bin_limits.is_none() {
        if let Some(exact) = y_exact.as_deref() {
            super::integer_bins::cover_exact_integer_range(&mut y_edges, exact);
        }
    }

    let x_bin_indices = compute_axis_bin_indices(
        &x_values,
        x_exact.as_deref(),
        &x_edges,
        options.x.exact_edges.as_deref(),
    );
    let y_bin_indices = compute_axis_bin_indices(
        &y_values,
        y_exact.as_deref(),
        &y_edges,
        options.y.exact_edges.as_deref(),
    );
    let counts = compute_histogram_counts(
        &x_bin_indices,
        &y_bin_indices,
        x_edges.len() - 1,
        y_edges.len() - 1,
    )?;
    let normalised = apply_normalization_2d(
        &counts,
        &x_edges,
        &y_edges,
        options.x.exact_edges.as_deref(),
        options.y.exact_edges.as_deref(),
        options.normalization,
        x_values.len(),
    );

    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let counts_tensor = Tensor::new(normalised, vec![x_bins, y_bins])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_edges_tensor = Tensor::new(x_edges.clone(), vec![1, x_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_edges_tensor = Tensor::new(y_edges.clone(), vec![1, y_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_bins_tensor =
        Tensor::new(x_bin_indices, x_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_bins_tensor =
        Tensor::new(y_bin_indices, y_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;

    Ok(Histcounts2Evaluation::new(
        counts_tensor,
        x_edges_tensor,
        y_edges_tensor,
        x_bins_tensor,
        y_bins_tensor,
    ))
}

fn compute_histogram_counts(
    x_indices: &[f64],
    y_indices: &[f64],
    x_bins: usize,
    y_bins: usize,
) -> BuiltinResult<Vec<f64>> {
    let cell_count = x_bins.checked_mul(y_bins).ok_or_else(|| {
        builtin_error("histcounts2: histogram dimensions exceed addressable size")
    })?;
    let mut counts = Vec::new();
    counts.try_reserve_exact(cell_count).map_err(|error| {
        builtin_error(format!(
            "histcounts2: cannot allocate histogram counts: {error}"
        ))
    })?;
    counts.resize(cell_count, 0.0);
    for (&x, &y) in x_indices.iter().zip(y_indices) {
        if x > 0.0 && y > 0.0 {
            let ix = x as usize - 1;
            let iy = y as usize - 1;
            let idx = ix + iy * x_bins;
            if idx < counts.len() {
                counts[idx] += 1.0;
            }
        }
    }
    Ok(counts)
}

fn compute_axis_bin_indices(
    values: &[f64],
    exact_values: Option<&[IntValue]>,
    edges: &[f64],
    exact_edges: Option<&[IntValue]>,
) -> Vec<f64> {
    if let Some(values) = exact_values {
        return values
            .iter()
            .map(|value| {
                let index = if let Some(exact) = exact_edges {
                    super::integer_bins::exact_edge_bin(value, exact)
                } else {
                    super::integer_bins::floating_edge_bin(value, edges)
                };
                index.map_or(0.0, |i| (i + 1) as f64)
            })
            .collect();
    }
    values
        .iter()
        .map(|&value| {
            let index = if let Some(exact) = exact_edges {
                super::integer_bins::floating_observation_exact_bin(value, exact)
            } else {
                find_bin_index(value, edges)
            };
            index.map_or(0.0, |i| (i + 1) as f64)
        })
        .collect()
}

#[derive(Clone)]
pub struct Histcounts2Evaluation {
    counts: Tensor,
    x_edges: Tensor,
    y_edges: Tensor,
    x_bins: Tensor,
    y_bins: Tensor,
}

impl Histcounts2Evaluation {
    fn new(
        counts: Tensor,
        x_edges: Tensor,
        y_edges: Tensor,
        x_bins: Tensor,
        y_bins: Tensor,
    ) -> Self {
        Self {
            counts,
            x_edges,
            y_edges,
            x_bins,
            y_bins,
        }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        (Value::Tensor(counts), Value::Tensor(x_edges))
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        let y_edges = self.y_edges;
        (
            Value::Tensor(counts),
            Value::Tensor(x_edges),
            Value::Tensor(y_edges),
        )
    }

    pub fn into_all(self) -> (Value, Value, Value, Value, Value) {
        (
            Value::Tensor(self.counts),
            Value::Tensor(self.x_edges),
            Value::Tensor(self.y_edges),
            Value::Tensor(self.x_bins),
            Value::Tensor(self.y_bins),
        )
    }
}

#[cfg(test)]
mod tests;
