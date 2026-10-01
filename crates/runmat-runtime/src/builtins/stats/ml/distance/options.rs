//! Typed admission of distance metrics and public name-value controls.

use runmat_builtins::{
    DistanceMetricKind, DistanceSelectionKind, KnnMetric, KnnOption, KnnSearchMethod,
};
use runmat_value::{Tensor, Value};

use super::metric_geometry::{covariance_matrix, invert_spd_matrix, variance_scale};
use super::{
    distance_error, gather_value, value_to_matrix, DistanceMetric, KnnOptions, Selection, EPS,
    KNNSEARCH_NAME, PDIST2_NAME,
};
use crate::builtins::common::{random_args::keyword_of, tensor};
use crate::BuiltinResult;

pub(super) fn parse_metric_name(
    name: &'static str,
    args: &[Value],
) -> BuiltinResult<DistanceMetricKind> {
    let Some(first) = args.first() else {
        return Ok(DistanceMetricKind::Euclidean);
    };
    let text = keyword_of(first)
        .ok_or_else(|| distance_error(name, format!("{name}: distance metric must be text")))?;
    DistanceMetricKind::from_keyword(&text).ok_or_else(|| {
        distance_error(
            name,
            format!("{name}: unsupported distance metric '{text}'"),
        )
    })
}

pub(super) async fn parse_metric(
    name: &'static str,
    x: &Tensor,
    _y: Option<&Tensor>,
    args: Vec<Value>,
) -> BuiltinResult<DistanceMetric> {
    let kind = parse_metric_name(name, &args)?;
    let rest = if args.is_empty() {
        Vec::new()
    } else {
        args[1..].to_vec()
    };
    match kind {
        DistanceMetricKind::Euclidean => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Euclidean)
        }
        DistanceMetricKind::SquareEuclidean => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::SquareEuclidean)
        }
        DistanceMetricKind::Cityblock => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Cityblock)
        }
        DistanceMetricKind::Chebychev => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Chebychev)
        }
        DistanceMetricKind::Minkowski => {
            let p = if rest.is_empty() {
                2.0
            } else {
                scalar_parameter(name, &rest[0], "minkowski exponent").await?
            };
            if rest.len() > 1 || !p.is_finite() || p <= 0.0 {
                return Err(distance_error(
                    name,
                    format!("{name}: Minkowski exponent must be a positive finite scalar"),
                ));
            }
            Ok(DistanceMetric::Minkowski(p))
        }
        DistanceMetricKind::Seuclidean => {
            let scale = if rest.is_empty() {
                variance_scale(name, x)?
            } else {
                let stddev =
                    vector_parameter(name, rest[0].clone(), x.cols, "seuclidean scale").await?;
                stddev
                    .into_iter()
                    .map(|value| value * value)
                    .collect::<Vec<_>>()
            };
            if rest.len() > 1 || scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(distance_error(
                    name,
                    format!("{name}: seuclidean scale values must be positive and finite"),
                ));
            }
            Ok(DistanceMetric::Seuclidean(scale))
        }
        DistanceMetricKind::Mahalanobis => {
            let covariance = if rest.is_empty() {
                covariance_matrix(name, x)?
            } else {
                matrix_parameter(name, rest[0].clone(), x.cols, "mahalanobis covariance").await?
            };
            if rest.len() > 1 {
                return Err(distance_error(
                    name,
                    format!("{name}: mahalanobis accepts at most one covariance matrix"),
                ));
            }
            Ok(DistanceMetric::Mahalanobis(invert_spd_matrix(
                name,
                &covariance,
                x.cols,
                "mahalanobis covariance",
            )?))
        }
        DistanceMetricKind::Cosine => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Cosine)
        }
        DistanceMetricKind::Correlation => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Correlation)
        }
        DistanceMetricKind::Hamming => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Hamming)
        }
        DistanceMetricKind::Jaccard => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Jaccard)
        }
        DistanceMetricKind::Spearman => {
            ensure_no_metric_parameter(name, &rest, kind.canonical_name())
                .map(|()| DistanceMetric::Spearman)
        }
    }
}

fn ensure_no_metric_parameter(
    name: &'static str,
    args: &[Value],
    metric: &str,
) -> BuiltinResult<()> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(distance_error(
            name,
            format!("{name}: {metric} distance does not accept a distance parameter"),
        ))
    }
}

async fn scalar_parameter(name: &'static str, value: &Value, label: &str) -> BuiltinResult<f64> {
    let gathered = gather_value(name, value.clone()).await?;
    match gathered {
        Value::Num(value) => Ok(value),
        Value::Int(value) => Ok(value.to_f64()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
            Ok(tensor::tensor_value_f64(&tensor, 0))
        }
        other => Err(distance_error(
            name,
            format!("{name}: {label} must be a numeric scalar, got {other:?}"),
        )),
    }
}

async fn vector_parameter(
    name: &'static str,
    value: Value,
    expected_len: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_matrix(name, value).await?;
    if tensor::tensor_element_len(&tensor) != expected_len {
        return Err(distance_error(
            name,
            format!("{name}: {label} length must match the number of columns"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

async fn matrix_parameter(
    name: &'static str,
    value: Value,
    expected_size: usize,
    label: &str,
) -> BuiltinResult<Vec<f64>> {
    let tensor = value_to_matrix(name, value).await?;
    if tensor.rows != expected_size || tensor.cols != expected_size {
        return Err(distance_error(
            name,
            format!("{name}: {label} must be a square matrix matching the number of columns"),
        ));
    }
    Ok(tensor::tensor_into_values_f64(tensor))
}

pub(super) fn split_pdist2_options(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, Option<Selection>)> {
    let mut metric_args = Vec::with_capacity(args.len());
    let mut selection = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let selector = keyword_of(&args[idx])
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
            idx += 2;
        } else {
            metric_args.push(args[idx].clone());
            idx += 1;
        }
    }
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

pub(super) async fn parse_knnsearch_options(
    args: Vec<Value>,
) -> BuiltinResult<(Vec<Value>, KnnOptions)> {
    let mut k = 1usize;
    let mut include_ties = false;
    let mut sort_indices = true;
    let mut metric_name: Option<Value> = None;
    let mut metric = KnnMetric {
        kind: DistanceMetricKind::Euclidean,
        fast_alias: false,
    };
    let mut metric_parameter: Option<Value> = None;
    let mut ns_method: Option<KnnSearchMethod> = None;
    let mut idx = 0usize;
    while idx < args.len() {
        let name = keyword_of(&args[idx]).ok_or_else(|| {
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
                let distance = keyword_of(&value).ok_or_else(|| {
                    distance_error(KNNSEARCH_NAME, "knnsearch: Distance must be text")
                })?;
                metric = KnnMetric::from_keyword(&distance).ok_or_else(|| {
                    distance_error(
                        KNNSEARCH_NAME,
                        format!("knnsearch: unsupported distance metric '{distance}'"),
                    )
                })?;
                metric_name = Some(Value::from(metric.kind.canonical_name()));
            }
            KnnOption::P | KnnOption::Cov | KnnOption::Scale => {
                if metric_parameter.is_some() {
                    return Err(distance_error(
                        KNNSEARCH_NAME,
                        "knnsearch: only one distance parameter may be specified",
                    ));
                }
                metric_parameter = Some(value);
            }
            KnnOption::NsMethod => {
                let method = keyword_of(&value).ok_or_else(|| {
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
                if keyword_of(&value).as_deref() != Some("maximal") {
                    let raw = scalar_parameter(KNNSEARCH_NAME, &value, "CacheSize").await?;
                    if !raw.is_finite() || raw <= 0.0 {
                        return Err(distance_error(
                            KNNSEARCH_NAME,
                            "knnsearch: CacheSize must be positive",
                        ));
                    }
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
    let mut metric_args = Vec::new();
    if let Some(metric) = metric_name {
        metric_args.push(metric);
        if let Some(parameter) = metric_parameter {
            metric_args.push(parameter);
        }
    } else if metric_parameter.is_some() {
        return Err(distance_error(
            KNNSEARCH_NAME,
            "knnsearch: P, Cov, or Scale requires a Distance option",
        ));
    }
    Ok((metric_args, KnnOptions { k, include_ties }))
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
