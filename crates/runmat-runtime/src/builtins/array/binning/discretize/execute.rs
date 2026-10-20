use runmat_value::{Tensor, Value};

use crate::BuiltinResult;

use super::{arguments, edges, error, numeric};
use crate::builtins::array::binning::numeric as numeric_order;

pub(super) async fn apply(
    x: Value,
    edges_or_count: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let inputs = arguments::gather(x, edges_or_count, rest).await?;
    let shape = shape_of(&inputs.x);
    let values = numeric::values(&inputs.x, "X")?;
    let plan = edges::plan(&values, inputs.edges_or_count, inputs.rest)?;
    if crate::output_count::current_output_count() == Some(0) {
        return Ok(Value::output_list(Vec::new()));
    }
    let bins = values
        .iter()
        .map(|value| numeric_order::assign(*value, &plan.edges, plan.included_right))
        .collect::<Vec<_>>();
    let edge_output = if plan.computed && crate::output_count::current_output_count() == Some(2) {
        Some(Value::Tensor(
            Tensor::new(
                plan.edges
                    .iter()
                    .map(|value| value.materialize_f64())
                    .collect(),
                vec![1, plan.edges.len()],
            )
            .map_err(error::internal)?,
        ))
    } else {
        None
    };
    let output = match plan.labels {
        Some(labels) => labels.materialize(&bins, shape)?,
        None => Value::Tensor(
            Tensor::new(
                bins.into_iter()
                    .map(|bin| bin.map(|index| index as f64).unwrap_or(f64::NAN))
                    .collect(),
                shape,
            )
            .map_err(error::internal)?,
        ),
    };
    select_outputs(output, edge_output)
}

fn select_outputs(output: Value, edges: Option<Value>) -> BuiltinResult<Value> {
    match crate::output_count::current_output_count() {
        None => Ok(output),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![output])),
        Some(2) => edges
            .map(|edges| Value::output_list(vec![output, edges]))
            .ok_or_else(|| {
                error::invalid("discretize: the second edge output requires a scalar bin count")
            }),
        Some(_) => Err(error::invalid("discretize: too many output arguments")),
    }
}

fn shape_of(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(value) => value.shape().to_vec(),
        Value::LogicalArray(value) => value.shape.to_vec(),
        Value::SparseTensor(value) => vec![value.rows, value.cols],
        _ => vec![1, 1],
    }
}
