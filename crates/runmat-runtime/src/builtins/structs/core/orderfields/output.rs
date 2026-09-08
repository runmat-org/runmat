use crate::builtins::common::tensor;
use runmat_value::{Tensor, Value};
use std::collections::HashMap;

pub(super) struct Evaluation {
    ordered: Value,
    permutation: Vec<f64>,
}

impl Evaluation {
    pub(super) fn new(
        ordered: Value,
        original: &[String],
        order: &[String],
    ) -> crate::BuiltinResult<Self> {
        let positions = original
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index + 1))
            .collect::<HashMap<_, _>>();
        let permutation = order
            .iter()
            .map(|name| {
                positions
                    .get(name.as_str())
                    .copied()
                    .map(|position| position as f64)
                    .ok_or_else(|| super::error::unknown_field(name))
            })
            .collect::<crate::BuiltinResult<Vec<_>>>()?;
        Ok(Self {
            ordered,
            permutation,
        })
    }

    pub(super) fn finish(self) -> crate::BuiltinResult<Value> {
        let Some(count) = crate::output_count::current_output_count() else {
            return Ok(self.ordered);
        };
        if count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let Self {
            ordered,
            permutation,
        } = self;
        let mut outputs = vec![ordered];
        if count >= 2 {
            outputs.push(permutation_value(permutation)?);
        }
        Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        ))
    }
}

fn permutation_value(permutation: Vec<f64>) -> crate::BuiltinResult<Value> {
    let rows = permutation.len();
    let tensor = Tensor::new(permutation, vec![rows, 1]).map_err(super::error::rebuild)?;
    Ok(tensor::tensor_into_value(tensor))
}
