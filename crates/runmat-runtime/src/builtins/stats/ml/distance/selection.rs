//! Selection and output construction after pairwise metric evaluation.

use std::cmp::Ordering;

use runmat_value::{CellArray, Tensor, Value};

use super::super::pairwise_metric::{
    observation_matrix_dimensions, row_distance, row_value, MetricCaller,
};
use super::{internal_error, DistanceMetric, KnnOptions, Selection, KNNSEARCH_NAME, PDIST2_NAME};
use crate::BuiltinResult;

pub(super) fn selected_distance_outputs(
    distances: &Tensor,
    selection: Selection,
) -> BuiltinResult<Value> {
    let (selected_distances, selected_indices) = select_distances(distances, selection)?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![
                Value::Tensor(selected_distances),
                Value::Tensor(selected_indices),
            ],
        )),
        None => Ok(Value::Tensor(selected_distances)),
    }
}

fn select_distances(distances: &Tensor, selection: Selection) -> BuiltinResult<(Tensor, Tensor)> {
    let requested = match selection {
        Selection::Smallest(k) | Selection::Largest(k) => k,
    };
    let (rows, cols) = observation_matrix_dimensions(distances);
    let k = requested.min(rows);
    let len = k
        .checked_mul(cols)
        .ok_or_else(|| internal_error(PDIST2_NAME, "pdist2: selected output size overflow"))?;
    let mut out = Vec::new();
    let mut indices = Vec::new();
    out.try_reserve(len)
        .map_err(|_| internal_error(PDIST2_NAME, "pdist2: selected output allocation failed"))?;
    indices
        .try_reserve(len)
        .map_err(|_| internal_error(PDIST2_NAME, "pdist2: selected index allocation failed"))?;
    for col in 0..cols {
        let mut values = (0..rows)
            .map(|row| (row_value(distances, row, col), row + 1))
            .collect::<Vec<_>>();
        match selection {
            Selection::Smallest(_) => {
                values.sort_by(|a, b| distance_order_ascending(a.0, a.1, b.0, b.1));
            }
            Selection::Largest(_) => {
                values.sort_by(|a, b| distance_order_descending(a.0, a.1, b.0, b.1));
            }
        }
        for (distance, index) in values.iter().take(k) {
            out.push(*distance);
            indices.push(*index as f64);
        }
    }
    let distance_tensor = Tensor::new(out, vec![k, cols])
        .map_err(|err| internal_error(PDIST2_NAME, format!("pdist2: {err}")))?;
    let index_tensor = Tensor::new(indices, vec![k, cols])
        .map_err(|err| internal_error(PDIST2_NAME, format!("pdist2: {err}")))?;
    Ok((distance_tensor, index_tensor))
}

pub(super) fn knnsearch_outputs(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    options: KnnOptions,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if requested == Some(0) {
        for query in 0..observation_matrix_dimensions(y).0 {
            sorted_neighbors(x, y, metric, query)?;
        }
        return Ok(Value::output_list(Vec::new()));
    }
    let include_distances = requested.is_some_and(|count| count >= 2);
    if options.include_ties {
        let (indices, distances) = knnsearch_tie_cells(x, y, metric, options.k, include_distances)?;
        return match requested {
            Some(out_count) => {
                let mut values = vec![Value::Cell(indices)];
                values.extend(distances.map(Value::Cell));
                Ok(crate::output_count::output_list_with_padding(
                    out_count, values,
                ))
            }
            None => Ok(Value::Cell(indices)),
        };
    }

    let (indices, distances) =
        knnsearch_numeric_outputs(x, y, metric, options.k, include_distances)?;
    match requested {
        Some(out_count) => {
            let mut values = vec![Value::Tensor(indices)];
            values.extend(distances.map(Value::Tensor));
            Ok(crate::output_count::output_list_with_padding(
                out_count, values,
            ))
        }
        None => Ok(Value::Tensor(indices)),
    }
}

fn knnsearch_numeric_outputs(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    k: usize,
    include_distances: bool,
) -> BuiltinResult<(Tensor, Option<Tensor>)> {
    let query_rows = observation_matrix_dimensions(y).0;
    let len = query_rows
        .checked_mul(k)
        .ok_or_else(|| internal_error(KNNSEARCH_NAME, "knnsearch: output size overflow"))?;
    let mut indices = Vec::new();
    let mut distances = include_distances.then(Vec::new);
    indices
        .try_reserve(len)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: index allocation failed"))?;
    if let Some(distances) = &mut distances {
        distances
            .try_reserve(len)
            .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: distance allocation failed"))?;
        distances.resize(len, 0.0);
    }
    indices.resize(len, 0.0);
    for query in 0..query_rows {
        let values = sorted_neighbors(x, y, metric, query)?;
        for (rank, (distance, index)) in values.iter().take(k).enumerate() {
            let pos = rank * query_rows + query;
            indices[pos] = *index as f64;
            if let Some(distances) = &mut distances {
                distances[pos] = *distance;
            }
        }
    }
    let index_tensor = Tensor::new(indices, vec![query_rows, k])
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    let distance_tensor = distances
        .map(|distances| Tensor::new(distances, vec![query_rows, k]))
        .transpose()
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    Ok((index_tensor, distance_tensor))
}

fn knnsearch_tie_cells(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    k: usize,
    include_distances: bool,
) -> BuiltinResult<(CellArray, Option<CellArray>)> {
    let query_rows = observation_matrix_dimensions(y).0;
    let mut index_cells = Vec::new();
    let mut distance_cells = include_distances.then(Vec::new);
    index_cells
        .try_reserve(query_rows)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: cell allocation failed"))?;
    if let Some(distance_cells) = &mut distance_cells {
        distance_cells
            .try_reserve(query_rows)
            .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: cell allocation failed"))?;
    }
    for query in 0..query_rows {
        let values = sorted_neighbors(x, y, metric, query)?;
        let cutoff = values[k - 1].0;
        let mut selected = Vec::new();
        for (distance, index) in values {
            let include = if cutoff.is_nan() {
                distance.is_nan()
            } else {
                distance == cutoff
            };
            if include || selected.len() < k {
                selected.push((distance, index));
            }
            if !include && selected.len() >= k {
                break;
            }
        }
        let len = selected.len();
        let idx_tensor = Tensor::new(
            selected.iter().map(|(_, index)| *index as f64).collect(),
            vec![1, len],
        )
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
        index_cells.push(Value::Tensor(idx_tensor));
        if let Some(distance_cells) = &mut distance_cells {
            let dist_tensor = Tensor::new(
                selected.iter().map(|(distance, _)| *distance).collect(),
                vec![1, len],
            )
            .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
            distance_cells.push(Value::Tensor(dist_tensor));
        }
    }
    let indices = CellArray::new(index_cells, query_rows, 1)
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    let distances = distance_cells
        .map(|cells| CellArray::new(cells, query_rows, 1))
        .transpose()
        .map_err(|err| internal_error(KNNSEARCH_NAME, format!("knnsearch: {err}")))?;
    Ok((indices, distances))
}

fn sorted_neighbors(
    x: &Tensor,
    y: &Tensor,
    metric: &DistanceMetric,
    query: usize,
) -> BuiltinResult<Vec<(f64, usize)>> {
    let observation_rows = observation_matrix_dimensions(x).0;
    let mut values = Vec::new();
    values
        .try_reserve(observation_rows)
        .map_err(|_| internal_error(KNNSEARCH_NAME, "knnsearch: neighbor allocation failed"))?;
    for row in 0..observation_rows {
        values.push((
            row_distance(MetricCaller::Knnsearch, x, row, y, query, metric)?,
            row + 1,
        ));
    }
    values.sort_by(|a, b| distance_order_ascending(a.0, a.1, b.0, b.1));
    Ok(values)
}

fn distance_order_ascending(
    left_distance: f64,
    left_index: usize,
    right_distance: f64,
    right_index: usize,
) -> Ordering {
    match (left_distance.is_nan(), right_distance.is_nan()) {
        (true, true) => left_index.cmp(&right_index),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left_distance
            .partial_cmp(&right_distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(&right_index)),
    }
}

fn distance_order_descending(
    left_distance: f64,
    left_index: usize,
    right_distance: f64,
    right_index: usize,
) -> Ordering {
    match (left_distance.is_nan(), right_distance.is_nan()) {
        (true, true) => left_index.cmp(&right_index),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => right_distance
            .partial_cmp(&left_distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_index.cmp(&right_index)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn selection_preserves_rank_zero_and_rank_one_observation_geometry() {
        let scalar = Tensor::new(vec![7.0], vec![]).expect("valid scalar tensor");
        let (distances, indices) = select_distances(&scalar, Selection::Smallest(1))
            .expect("rank-zero distance selection");
        assert_eq!(distances.shape(), &[0, 0]);
        assert_eq!(indices.shape(), &[0, 0]);
        assert!(distances.materialize_f64().is_empty());
        assert!(indices.materialize_f64().is_empty());

        let row = Tensor::new(vec![3.0, 1.0, 2.0], vec![3]).expect("valid rank-one tensor");
        let (distances, indices) =
            select_distances(&row, Selection::Smallest(1)).expect("rank-one distance selection");
        assert_eq!(distances.shape(), &[1, 3]);
        assert_eq!(indices.shape(), &[1, 3]);
        assert_eq!(distances.materialize_f64(), vec![3.0, 1.0, 2.0]);
        assert_eq!(indices.materialize_f64(), vec![1.0, 1.0, 1.0]);
    }
}
