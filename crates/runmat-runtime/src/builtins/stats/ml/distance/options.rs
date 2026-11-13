//! Typed selection and KNN controls at the distance builtin boundary.

use runmat_builtins::{
    DistanceMetricKind, DistanceSelectionKind, KnnCacheSizeKeyword, KnnMetric, KnnOption,
    KnnSearchMethod,
};
use runmat_value::Value;

use super::{
    distance_error, distance_keyword, KnnOptions, Selection, EPS, KNNSEARCH_NAME, PDIST2_NAME,
};
use crate::builtins::common::tensor;
use crate::builtins::stats::ml::pairwise_metric::{scalar_parameter, MetricCaller};
use crate::BuiltinResult;

pub(super) async fn split_pdist2_options(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<(crate::call::BuiltinArguments, Option<Selection>)> {
    let mut selected_pair = None;
    let mut selection = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let selector = distance_keyword(PDIST2_NAME, &args[idx])?
            .as_deref()
            .and_then(DistanceSelectionKind::from_keyword);
        if let Some(selector) = selector {
            if selection.is_some() || idx + 1 >= args.len() {
                return Err(distance_error(
                    PDIST2_NAME,
                    "pdist2: Smallest or Largest must be a single name-value pair",
                ));
            }
            let k = parse_positive_integer(PDIST2_NAME, &args[idx + 1], "pdist2 selection count")?;
            selection = Some(match selector {
                DistanceSelectionKind::Smallest => Selection::Smallest(k),
                DistanceSelectionKind::Largest => Selection::Largest(k),
            });
            selected_pair = Some(idx);
            idx += 2;
        } else {
            idx += 1;
        }
    }
    let Some(selected_pair) = selected_pair else {
        return Ok((args, selection));
    };
    let metric_args = args
        .try_map_preserving_roots_async(args.len() - 2, runtime, |index| {
            std::future::ready(Ok(args[if index < selected_pair {
                index
            } else {
                index + 2
            }]
            .clone()))
        })
        .await?;
    Ok((metric_args, selection))
}

pub(super) fn parse_positive_integer(
    name: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                distance_error(
                    name,
                    format!("{name}: {label} must be a positive integer scalar"),
                )
            });
    }
    let raw = match value {
        Value::Num(value) => *value,
        Value::Bool(value) => {
            if *value {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => f64::from(array.data[0] != 0),
        other => {
            return Err(distance_error(
                name,
                format!("{name}: {label} must be a positive integer scalar, got {other:?}"),
            ))
        }
    };
    if !raw.is_finite()
        || raw < 1.0
        || raw.fract().abs() > EPS
        || raw > usize::MAX as f64
        || (usize::BITS == 64 && raw == usize::MAX as f64)
    {
        return Err(distance_error(
            name,
            format!("{name}: {label} must be a positive integer scalar"),
        ));
    }
    Ok(raw as usize)
}

pub(super) struct ParsedKnnSearch {
    pub metric: KnnMetric,
    pub metric_parameter: Option<Value>,
    pub numeric_cache_sizes: Vec<Value>,
    pub options: KnnOptions,
}

pub(super) fn parse_knnsearch_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ParsedKnnSearch> {
    let mut k = 1usize;
    let mut include_ties = false;
    let mut sort_indices = true;
    let mut metric = KnnMetric {
        kind: DistanceMetricKind::Euclidean,
        fast_alias: false,
    };
    let mut metric_parameter: Option<(KnnOption, Value)> = None;
    let mut numeric_cache_sizes = Vec::new();
    let mut ns_method: Option<KnnSearchMethod> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let name = distance_keyword(KNNSEARCH_NAME, &args[idx])?.ok_or_else(|| {
            distance_error(
                KNNSEARCH_NAME,
                "knnsearch: options must be supplied as name-value pairs",
            )
        })?;
        if idx + 1 >= args.len() {
            return Err(distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: missing value for option '{name}'"),
            ));
        }
        let value = args[idx + 1].clone();
        let option = KnnOption::from_keyword(&name).ok_or_else(|| {
            distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: unsupported option '{name}'"),
            )
        })?;
        match option {
            KnnOption::K => k = parse_positive_integer(KNNSEARCH_NAME, &value, "K")?,
            KnnOption::IncludeTies => include_ties = parse_logical_scalar(&value, "IncludeTies")?,
            KnnOption::SortIndices => sort_indices = parse_logical_scalar(&value, "SortIndices")?,
            KnnOption::Distance => {
                let distance = distance_keyword(KNNSEARCH_NAME, &value)?.ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: Distance must be text")
                })?;
                metric = KnnMetric::from_keyword(&distance).ok_or_else(|| {
                    distance_error(
                        KNNSEARCH_NAME,
                        format!("knnsearch: unsupported distance metric '{distance}'"),
                    )
                })?;
            }
            KnnOption::P | KnnOption::Cov | KnnOption::Scale => {
                if metric_parameter.is_some() {
                    return Err(distance_error(
                        KNNSEARCH_NAME,
                        "knnsearch: only one distance parameter may be specified",
                    ));
                }
                metric_parameter = Some((option, value));
            }
            KnnOption::NsMethod => {
                let method = distance_keyword(KNNSEARCH_NAME, &value)?.ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: NSMethod must be text")
                })?;
                ns_method = Some(KnnSearchMethod::from_keyword(&method).ok_or_else(|| {
                    distance_error(
                        KNNSEARCH_NAME,
                        format!("knnsearch: unsupported NSMethod '{method}'"),
                    )
                })?);
            }
            KnnOption::CacheSize => {
                if distance_keyword(KNNSEARCH_NAME, &value)?
                    .as_deref()
                    .and_then(KnnCacheSizeKeyword::from_keyword)
                    .is_none()
                {
                    preflight_numeric_cache_size(&value)?;
                    numeric_cache_sizes.push(value);
                }
            }
            KnnOption::BucketSize => {
                parse_positive_integer(KNNSEARCH_NAME, &value, "BucketSize")?;
            }
        }
        idx += 2;
    }
    if ns_method == Some(KnnSearchMethod::KdTree) && !metric.allows_kdtree() {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: NSMethod='kdtree' is only valid for euclidean, cityblock, chebychev, or minkowski distances",
        ));
    }
    let _ = sort_indices;
    let metric_parameter = match metric_parameter {
        Some((kind, value)) if metric_parameter_matches(metric.kind, kind) => Some(value),
        Some(_) => {
            return Err(distance_error(
                KNNSEARCH_NAME,
                "knnsearch: P, Cov, or Scale must match the selected Distance metric",
            ));
        }
        None => None,
    };
    Ok(ParsedKnnSearch {
        metric,
        metric_parameter,
        numeric_cache_sizes,
        options: KnnOptions { k, include_ties },
    })
}

fn metric_parameter_matches(metric: DistanceMetricKind, option: KnnOption) -> bool {
    matches!(
        (metric, option),
        (DistanceMetricKind::Minkowski, KnnOption::P)
            | (DistanceMetricKind::Seuclidean, KnnOption::Scale)
            | (DistanceMetricKind::Mahalanobis, KnnOption::Cov)
    )
}

pub(super) fn preflight_knn_metric_parameter(
    metric: DistanceMetricKind,
    parameter: Option<&Value>,
    columns: usize,
) -> BuiltinResult<()> {
    let Some(parameter) = parameter else {
        return Ok(());
    };
    let Some((rows, cols, len)) = parameter_geometry(parameter) else {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: distance parameter must be a numeric scalar, vector, or matrix",
        ));
    };
    match metric {
        DistanceMetricKind::Minkowski
            if len == 1 && !matches!(parameter, Value::Bool(_) | Value::LogicalArray(_)) =>
        {
            Ok(())
        }
        DistanceMetricKind::Seuclidean if len == columns => Ok(()),
        DistanceMetricKind::Mahalanobis if rows == columns && cols == columns => Ok(()),
        _ => Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: distance parameter shape does not match the selected metric",
        )),
    }
}

fn parameter_geometry(value: &Value) -> Option<(usize, usize, usize)> {
    let shape = match value {
        Value::Tensor(tensor) => tensor.shape(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle)
            if handle.physical().storage() == runmat_accelerate_api::GpuTensorStorage::Real =>
        {
            handle.shape()
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => return Some((1, 1, 1)),
        _ => return None,
    };
    match shape {
        [] => Some((1, 1, 1)),
        [rows] => Some((*rows, 1, *rows)),
        [rows, columns] => Some((*rows, *columns, rows.checked_mul(*columns)?)),
        _ => None,
    }
}

fn preflight_numeric_cache_size(value: &Value) -> BuiltinResult<()> {
    let number = match value {
        Value::Num(number) => Some(*number),
        Value::Int(integer) => Some(integer.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        Value::GpuTensor(handle)
            if handle
                .shape()
                .iter()
                .try_fold(1usize, |product, extent| product.checked_mul(*extent))
                == Some(1) =>
        {
            None
        }
        _ => {
            return Err(distance_error(
                KNNSEARCH_NAME,
                "knnsearch: CacheSize must be a positive numeric scalar",
            ));
        }
    };
    if number.is_some_and(|number| !number.is_finite() || number <= 0.0) {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: CacheSize must be positive",
        ));
    }
    Ok(())
}

pub(super) async fn validate_numeric_cache_sizes(values: Vec<Value>) -> BuiltinResult<()> {
    for value in values {
        let raw = scalar_parameter(MetricCaller::Knnsearch, &value, "CacheSize").await?;
        if !raw.is_finite() || raw <= 0.0 {
            return Err(distance_error(
                KNNSEARCH_NAME,
                "knnsearch: CacheSize must be positive",
            ));
        }
    }
    Ok(())
}

pub(super) fn parse_logical_scalar(value: &Value, label: &str) -> BuiltinResult<bool> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(distance_error(
                KNNSEARCH_NAME,
                format!("knnsearch: {label} must be logical scalar, got {value:?}"),
            )),
        };
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            let raw = tensor::tensor_value_f64(tensor, 0);
            if raw == 0.0 || raw == 1.0 {
                Ok(raw != 0.0)
            } else {
                Err(distance_error(
                    KNNSEARCH_NAME,
                    format!("knnsearch: {label} must be logical scalar, got {value:?}"),
                ))
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(distance_error(
            KNNSEARCH_NAME,
            format!("knnsearch: {label} must be logical scalar, got {other:?}"),
        )),
    }
}
