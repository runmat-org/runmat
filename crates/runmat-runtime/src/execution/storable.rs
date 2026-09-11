use runmat_value::Value;

use crate::{runtime_error::semantic_error, RuntimeError};

/// Reject executor-only value sequences before they enter durable aggregate or
/// workspace state. Callers retain ownership, so rejection is transactional.
pub fn validate_storable_value(value: &Value) -> Result<(), RuntimeError> {
    runmat_value::validate_no_transient_sequence(value).map_err(|_| {
        semantic_error(
            "TransientSequenceNotStorable",
            "transient output sequences cannot be stored as language values",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_transient_sequence_is_not_storable() {
        let value = Value::Cell(
            runmat_value::CellArray::new(vec![Value::OutputList(vec![Value::Num(1.0)])], 1, 1)
                .unwrap(),
        );
        assert_eq!(
            validate_storable_value(&value).unwrap_err().identifier(),
            Some("RunMat:TransientSequenceNotStorable")
        );

        let closure = Value::Closure(runmat_value::Closure {
            function_name: "synthetic_callback".into(),
            bound_function: None,
            captures: vec![Value::OutputList(Vec::new())],
        });
        assert_eq!(
            validate_storable_value(&closure).unwrap_err().identifier(),
            Some("RunMat:TransientSequenceNotStorable")
        );
    }
}
