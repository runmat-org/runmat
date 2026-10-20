//! Hierarchical agglomerative clustering compatibility surface.

use runmat_builtins::{
    matrix_size_for_condensed_length, BuiltinErrorDescriptor, DistanceMetricKind,
    LINKAGE_ERROR_INTERNAL, LINKAGE_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "linkage";
const EPS: f64 = 1.0e-12;
const MAX_OBSERVATIONS: usize = 700;
const MAX_CONDENSED_DISTANCES: usize = MAX_OBSERVATIONS * (MAX_OBSERVATIONS - 1) / 2;

mod admission;
mod compute;
mod geometry;
mod options;
#[cfg(test)]
mod tests;

use super::pairwise_metric::{condensed_distances, parse_metric_kind, MetricCaller};
use admission::{
    ensure_exact_metric_integer_boundaries, ensure_extensions, preflight_input_geometry,
    validate_distances, validate_observation_count, value_to_tensor,
};
use compute::compute_linkage;
use geometry::validate_euclidean_condensed;
use options::{parse_options, validate_savememory_for_matrix, LinkageMetric};

fn linkage_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    linkage_error(message, &LINKAGE_ERROR_INVALID_ARGUMENT)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    linkage_error(message, &LINKAGE_ERROR_INTERNAL)
}

fn with_linkage_context(mut error: RuntimeError) -> RuntimeError {
    if error.context.builtin.is_none() {
        error.context = error.context.with_builtin(NAME);
    }
    error
}

#[runtime_builtin(
    name = "linkage",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::linkage"
)]
async fn linkage_builtin(
    input: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let options = parse_options(rest.view())?;
    ensure_extensions(&input, &options)?;
    preflight_input_geometry(&input, &options)?;
    let tensor = value_to_tensor(input).await?;
    if let Some(metric) = &options.metric {
        ensure_exact_metric_integer_boundaries(&metric.parameters).await?;
    }
    let shape = tensor.shape();
    let rows = match shape {
        [] => 0,
        [_] => 1,
        [rows, ..] => *rows,
    };
    let is_condensed_vector = rows == 1 || shape.len() == 1;
    let (observations, distances) = if is_condensed_vector {
        if options.metric.is_some() {
            return Err(invalid(
                "linkage: condensed distance vector input does not accept a distance metric",
            ));
        }
        if options.savememory == Some(true) {
            return Err(invalid(
                "linkage: SaveMemory 'on' is only supported for observation matrix input",
            ));
        }
        let distances = tensor::tensor_values_f64(&tensor);
        let observations = matrix_size_for_condensed_length(distances.len())
            .filter(|observations| *observations >= 2)
            .ok_or_else(|| {
                invalid("linkage: condensed distance vector length must be n*(n-1)/2")
            })?;
        validate_observation_count(observations)?;
        validate_distances(&distances)?;
        if options.method.requires_euclidean_distances() {
            validate_euclidean_condensed(&distances, observations)?;
        }
        (observations, distances)
    } else {
        if shape.len() > 2 || rows < 2 {
            return Err(invalid(
                "linkage: X must be a numeric matrix with at least two observations in rows",
            ));
        }
        validate_savememory_for_matrix(&options)?;
        validate_observation_count(rows)?;
        let metric = options.metric.unwrap_or(LinkageMetric {
            kind: DistanceMetricKind::Euclidean,
            parameters: Vec::new(),
        });
        let metric = parse_metric_kind(
            MetricCaller::Linkage,
            &tensor,
            metric.kind,
            crate::call::FunctionArgumentView::from_slice(&metric.parameters),
        )
        .await?;
        let distances = condensed_distances(MetricCaller::Linkage, &tensor, &metric)?;
        let distances = distances.materialize_f64();
        validate_distances(&distances)?;
        (rows, distances)
    };
    let output = compute_linkage(observations, distances, options.method)?;
    Ok(Value::Tensor(output))
}
