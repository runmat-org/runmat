//! Typed metric-name admission and preparation of parameterized metrics.

use runmat_builtins::DistanceMetricKind;
use runmat_value::{Tensor, Value};

use super::compute::observation_matrix_dimensions;
use super::geometry::{covariance_matrix, invert_spd_matrix, variance_scale};
use super::{distance_error, DistanceMetric, MetricCaller};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::{gather_if_needed_async, BuiltinResult};

pub(in crate::builtins::stats::ml) fn parse_metric_name(
    caller: MetricCaller,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DistanceMetricKind> {
    let Some(first) = args.first() else {
        return Ok(DistanceMetricKind::Euclidean);
    };
    let text = keyword_of(first)?
        .ok_or_else(|| distance_error(caller, format!("{caller}: distance metric must be text")))?;
    DistanceMetricKind::from_keyword(&text).ok_or_else(|| {
        distance_error(
            caller,
            format!("{caller}: unsupported distance metric '{text}'"),
        )
    })
}

pub(in crate::builtins::stats::ml) async fn parse_metric_kind(
    caller: MetricCaller,
    x: &Tensor,
    kind: DistanceMetricKind,
    parameters: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<DistanceMetric> {
    match kind {
        DistanceMetricKind::Euclidean => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Euclidean)
        }
        DistanceMetricKind::SquareEuclidean => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::SquareEuclidean)
        }
        DistanceMetricKind::Cityblock => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Cityblock)
        }
        DistanceMetricKind::Chebychev => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Chebychev)
        }
        DistanceMetricKind::Minkowski => {
            let p = if parameters.is_empty() {
                2.0
            } else {
                scalar_parameter(caller, &parameters[0], "minkowski exponent").await?
            };
            if parameters.len() > 1 || !p.is_finite() || p <= 0.0 {
                return Err(distance_error(
                    caller,
                    format!("{caller}: Minkowski exponent must be a positive finite scalar"),
                ));
            }
            Ok(DistanceMetric::Minkowski(p))
        }
        DistanceMetricKind::Seuclidean => {
            let columns = observation_matrix_dimensions(x).1;
            let scale = if parameters.is_empty() {
                variance_scale(caller, x)?
            } else {
                vector_parameter(caller, parameters[0].clone(), columns, "seuclidean scale")
                    .await?
                    .into_iter()
                    .map(|value| value * value)
                    .collect()
            };
            if parameters.len() > 1
                || scale
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(distance_error(
                    caller,
                    format!("{caller}: seuclidean scale values must be positive and finite"),
                ));
            }
            Ok(DistanceMetric::Seuclidean(scale))
        }
        DistanceMetricKind::Mahalanobis => {
            let columns = observation_matrix_dimensions(x).1;
            let covariance = if parameters.is_empty() {
                covariance_matrix(caller, x)?
            } else {
                matrix_parameter(
                    caller,
                    parameters[0].clone(),
                    columns,
                    "mahalanobis covariance",
                )
                .await?
            };
            if parameters.len() > 1 {
                return Err(distance_error(
                    caller,
                    format!("{caller}: mahalanobis accepts at most one covariance matrix"),
                ));
            }
            Ok(DistanceMetric::Mahalanobis(invert_spd_matrix(
                caller,
                &covariance,
                columns,
                "mahalanobis covariance",
            )?))
        }
        DistanceMetricKind::Cosine => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Cosine)
        }
        DistanceMetricKind::Correlation => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Correlation)
        }
        DistanceMetricKind::Hamming => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Hamming)
        }
        DistanceMetricKind::Jaccard => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Jaccard)
        }
        DistanceMetricKind::Spearman => {
            no_parameter(caller, parameters, kind).map(|()| DistanceMetric::Spearman)
        }
    }
}

fn no_parameter(
    caller: MetricCaller,
    parameters: crate::call::FunctionArgumentView<'_>,
    kind: DistanceMetricKind,
) -> BuiltinResult<()> {
    if parameters.is_empty() {
        Ok(())
    } else {
        Err(distance_error(
            caller,
            format!(
                "{caller}: {} distance does not accept a distance parameter",
                kind.canonical_name()
            ),
        ))
    }
}

pub(in crate::builtins::stats::ml) async fn scalar_parameter(
    caller: MetricCaller,
    value: &Value,
    label: &str,
) -> BuiltinResult<f64> {
    let gathered = gather_parameter(caller, value.clone()).await?;
    match gathered {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            Ok(tensor::tensor_value_f64(&tensor, 0))
        }
        other => Err(distance_error(
            caller,
            format!("{caller}: {label} must be a numeric scalar, got {other:?}"),
        )),
    }
}

async fn vector_parameter(
    caller: MetricCaller,
    value: Value,
    expected_len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = parameter_matrix(caller, value).await?;
    if tensor::tensor_element_len(&tensor) != expected_len {
        return Err(distance_error(
            caller,
            format!("{caller}: {label} length must match the number of columns"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

async fn matrix_parameter(
    caller: MetricCaller,
    value: Value,
    expected_size: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = parameter_matrix(caller, value).await?;
    let (rows, columns) = observation_matrix_dimensions(&tensor);
    if rows != expected_size || columns != expected_size {
        return Err(distance_error(
            caller,
            format!("{caller}: {label} must be a square matrix matching the number of columns"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

async fn parameter_matrix(caller: MetricCaller, value: Value) -> BuiltinResult<Tensor> {
    let gathered = gather_parameter(caller, value).await?;
    let tensor = tensor::value_into_tensor_for(caller.name(), gathered)
        .map_err(|error| distance_error(caller, format!("{caller}: {error}")))?;
    if tensor.shape().len() > 2 {
        return Err(distance_error(
            caller,
            format!("{caller}: input must be a numeric vector or 2-D matrix"),
        ));
    }
    Ok(tensor)
}

async fn gather_parameter(caller: MetricCaller, value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|mut error| {
        if error.context.builtin.is_none() {
            error.context = error.context.with_builtin(caller.name());
        }
        error
    })
}
