use std::collections::BTreeMap;

use runmat_value::Value;

use crate::BuiltinResult;

use super::{callback, data, error, output, provider::MaterializedRequest, shape, subscripts};

pub(super) async fn run(request: MaterializedRequest) -> BuiltinResult<Value> {
    let index_rows = subscripts::parse(request.subscripts)?;
    let data = data::column(request.data, index_rows.len())?;
    let output_shape = shape::resolve(&index_rows, request.options.first())?;
    let output_len = shape::element_count(&output_shape)?;
    let groups = build_groups(&index_rows, &output_shape)?;
    let function = request
        .options
        .get(1)
        .filter(|value| !shape::is_empty(value))
        .cloned();
    let fill = request
        .options
        .get(2)
        .filter(|value| !shape::is_empty(value))
        .cloned();
    let sparse = request
        .options
        .get(3)
        .map(shape::binary_flag)
        .transpose()?
        .unwrap_or(false);
    if let Some(function) = function {
        return callback::evaluate(
            data,
            groups,
            output_shape,
            output_len,
            function,
            fill,
            sparse,
        )
        .await;
    }
    default_sum(data, groups, output_shape, output_len, fill, sparse)
}

fn build_groups(
    index_rows: &[Vec<usize>],
    shape: &[usize],
) -> BuiltinResult<BTreeMap<usize, Vec<usize>>> {
    let mut groups = BTreeMap::new();
    for (row, subscripts) in index_rows.iter().enumerate() {
        groups
            .entry(shape::linear_index(subscripts, shape)?)
            .or_insert_with(Vec::new)
            .push(row);
    }
    Ok(groups)
}

fn default_sum(
    data: Value,
    groups: BTreeMap<usize, Vec<usize>>,
    shape: Vec<usize>,
    output_len: usize,
    fill: Option<Value>,
    sparse: bool,
) -> BuiltinResult<Value> {
    if sparse && !data::is_double(&data) {
        return Err(error::invalid(
            "accumarray: sparse output requires double input data",
        ));
    }
    if fill.as_ref().is_some_and(data::is_integer) {
        return Err(error::invalid(if sparse {
            "accumarray: sparse output requires a double zero fill value"
        } else {
            "accumarray: fill value class must match the double default sum output"
        }));
    }
    let rows = groups.values().map(|indices| indices.len()).sum();
    let values = data::default_sum_values(data, rows)?;
    let fill = fill
        .as_ref()
        .map(|value| data::numeric_scalar(value, "accumarray fill value"))
        .transpose()?
        .unwrap_or(0.0);
    if sparse && fill != 0.0 {
        return Err(error::invalid(
            "accumarray: sparse output requires a zero fill value",
        ));
    }
    let mut accumulated = vec![fill; output_len];
    for (linear, indices) in groups {
        accumulated[linear] = indices.into_iter().map(|index| values[index]).sum();
    }
    output::dense_or_sparse_f64(accumulated, shape, sparse)
}
