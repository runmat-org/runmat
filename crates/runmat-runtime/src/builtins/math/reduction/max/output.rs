use super::*;

/// Evaluation artifact returned by `max` that carries both values and indices.
#[derive(Debug, Clone)]
pub struct MaxEvaluation {
    pub(super) values: Value,
    pub(super) indices: Value,
}

impl MaxEvaluation {
    /// Consume the evaluation and return only the maximum values (single-output call).
    pub fn into_value(self) -> Value {
        self.values
    }

    /// Consume the evaluation and return both maxima and indices.
    pub fn into_pair(self) -> (Value, Value) {
        (self.values, self.indices)
    }

    /// Peek at the indices without consuming.
    pub fn indices_value(&self) -> Value {
        self.indices.clone()
    }
}

pub(super) fn reject_pairwise_multiple_outputs(
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if crate::output_count::current_output_count().unwrap_or(1) > 1
        && rest
            .first()
            .is_some_and(|value| !is_empty_placeholder(value))
    {
        return Err(max_invalid_argument(
            "max: pairwise element-wise form has exactly one output",
        ));
    }
    Ok(())
}

pub(super) fn builtin_demand() -> super::extrema_provider::ExtremaDemand {
    if crate::output_count::current_output_count().unwrap_or(1) > 1 {
        super::extrema_provider::ExtremaDemand::ValuesAndIndices
    } else {
        super::extrema_provider::ExtremaDemand::ValuesOnly
    }
}

pub(super) fn evaluation_to_value(
    eval: super::extrema_provider::ExtremaResult,
) -> BuiltinResult<Value> {
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.into_value()]));
        }
        let (values, indices) = eval.into_pair(super::extrema_provider::ExtremaKind::Max)?;
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![values, indices],
        ));
    }
    Ok(eval.into_value())
}
