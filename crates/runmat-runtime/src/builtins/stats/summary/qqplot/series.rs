//! Quantile pairing, interpolation and reference-line geometry.
use super::geometry::{quantile_from_sorted, reference_geometry};
use super::*;
use crate::builtins::stats::summary::distribution_math::standard_normal_inv;
use std::cmp::Ordering;

pub(super) fn normal_series_from_tensor(
    x: Tensor,
    pvec: Option<Vec<f64>>,
) -> BuiltinResult<Vec<Series>> {
    let shape = tensor::default_shape_for(x.shape(), tensor::tensor_element_len(&x));
    let columns = columns_from_tensor(x, &shape)?;
    columns
        .into_iter()
        .filter_map(|column| {
            let sorted = finite_sorted(column);
            if sorted.is_empty() {
                return None;
            }
            Some(normal_series(sorted, pvec.clone()))
        })
        .collect()
}

pub(super) fn two_sample_series_from_tensors(
    x: Tensor,
    y: Tensor,
    pvec: Option<Vec<f64>>,
) -> BuiltinResult<Vec<Series>> {
    let x_shape = tensor::default_shape_for(x.shape(), tensor::tensor_element_len(&x));
    let y_shape = tensor::default_shape_for(y.shape(), tensor::tensor_element_len(&y));
    let x_columns = columns_from_tensor(x, &x_shape)?;
    let y_columns = columns_from_tensor(y, &y_shape)?;
    if x_columns.is_empty() || y_columns.is_empty() {
        return Err(invalid("qqplot: both samples must contain values"));
    }
    let pairs = column_pairs(&x_columns, &y_columns)?;
    let mut out = Vec::with_capacity(pairs.len());
    for (x_col, y_col) in pairs {
        let x = finite_sorted(x_col);
        let y = finite_sorted(y_col);
        if x.is_empty() || y.is_empty() {
            continue;
        }
        let probabilities = pvec
            .clone()
            .unwrap_or_else(|| default_probabilities(x.len().min(y.len())));
        let theoretical = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&x, *p))
            .collect::<Vec<_>>();
        let observed = probabilities
            .iter()
            .map(|p| quantile_from_sorted(&y, *p))
            .collect::<Vec<_>>();
        out.push(series_from_pairs(theoretical, observed)?);
    }
    if out.is_empty() {
        return Err(invalid("qqplot: both samples must contain finite values"));
    }
    Ok(out)
}

fn normal_series(sorted: Vec<f64>, pvec: Option<Vec<f64>>) -> BuiltinResult<Series> {
    let explicit_probabilities = pvec.is_some();
    let probabilities = pvec.unwrap_or_else(|| default_probabilities(sorted.len()));
    let theoretical = probabilities
        .iter()
        .map(|p| standard_normal_inv(*p))
        .collect::<Vec<_>>();
    let observed = if explicit_probabilities {
        probabilities
            .iter()
            .map(|p| quantile_from_sorted(&sorted, *p))
            .collect::<Vec<_>>()
    } else {
        sorted
    };
    series_from_pairs(theoretical, observed)
}

fn series_from_pairs(theoretical: Vec<f64>, observed: Vec<f64>) -> BuiltinResult<Series> {
    let geometry = reference_geometry(&theoretical, &observed)?;
    Ok(Series {
        theoretical,
        observed,
        reference_x: geometry.reference_x,
        reference_y: geometry.reference_y,
        quartile_x: geometry.quartile_x,
        quartile_y: geometry.quartile_y,
    })
}

pub(super) fn columns_from_tensor(tensor: Tensor, shape: &[usize]) -> BuiltinResult<Vec<Vec<f64>>> {
    let data = tensor_values_f64(&tensor);
    if shape.is_empty() {
        return Ok(vec![data]);
    }
    if shape.iter().filter(|dim| **dim > 1).count() <= 1 {
        return Ok(vec![data]);
    }
    if shape.len() > 2 {
        return Err(invalid("qqplot: input must be a vector or 2-D matrix"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut values = Vec::with_capacity(rows);
        for row in 0..rows {
            values.push(data[row + col * rows]);
        }
        out.push(values);
    }
    Ok(out)
}

fn column_pairs(x: &[Vec<f64>], y: &[Vec<f64>]) -> BuiltinResult<Vec<(Vec<f64>, Vec<f64>)>> {
    if x.len() == y.len() {
        return Ok(x.iter().cloned().zip(y.iter().cloned()).collect());
    }
    if x.len() == 1 {
        return Ok(y
            .iter()
            .cloned()
            .map(|y_col| (x[0].clone(), y_col))
            .collect());
    }
    if y.len() == 1 {
        return Ok(x
            .iter()
            .cloned()
            .map(|x_col| (x_col, y[0].clone()))
            .collect());
    }
    Err(invalid(
        "qqplot: x and y matrices must have the same number of columns",
    ))
}

fn finite_sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values
}

pub(super) fn probabilities_from_tensor(tensor: Tensor) -> BuiltinResult<Vec<f64>> {
    let data = tensor_values_f64(&tensor);
    if data.is_empty() {
        return Err(invalid("qqplot: pvec must not be empty"));
    }
    let mut probabilities = Vec::with_capacity(data.len());
    for value in data {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(invalid(
                "qqplot: pvec values must be finite percentages in the interval [0, 100]",
            ));
        }
        probabilities.push(value / 100.0);
    }
    probabilities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(probabilities)
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

fn default_probabilities(n: usize) -> Vec<f64> {
    (0..n).map(|idx| (idx as f64 + 0.5) / n as f64).collect()
}
