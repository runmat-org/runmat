use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{SequenceRegister, SequenceState};

impl SequenceState {
    fn stage_window(
        slot: &mut Option<SequenceRegister>,
        stack: &mut Vec<Value>,
        sequence: runmat_value::ValueSequence,
        overwrite_message: &'static str,
    ) -> Result<(), RuntimeError> {
        if slot.is_some() {
            return Err(state_error(overwrite_message));
        }
        let values = sequence.into_values();
        let window = SequenceRegister {
            start: stack.len(),
            len: values.len(),
        };
        stack.extend(values);
        *slot = Some(window);
        Ok(())
    }

    pub(super) fn stage_call_output_sequence(
        &mut self,
        stack: &mut Vec<Value>,
        sequence: runmat_value::ValueSequence,
    ) -> Result<(), RuntimeError> {
        Self::stage_window(
            &mut self.call_outputs,
            stack,
            sequence,
            "call output sequence was overwritten before capture",
        )
    }

    pub(super) fn capture_call_outputs(&mut self) -> Result<(), RuntimeError> {
        if self.assignment.is_some() {
            return Err(state_error(
                "sequence register was overwritten before consumption",
            ));
        }
        self.assignment = self.call_outputs.take();
        if self.assignment.is_none() {
            return Err(state_error(
                "call output capture has no pending value sequence",
            ));
        }
        Ok(())
    }
}

fn state_error(message: &'static str) -> RuntimeError {
    crate::interpreter::errors::mex("RunMat:CommaSeparatedListState", message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::dispatch::object::take_sequence_register;
    use runmat_value::Value;

    #[test]
    fn capture_quarantines_zero_one_and_many_outputs() {
        let cases = [
            (Value::OutputList(Vec::new()), Vec::new()),
            (Value::Num(7.0), vec![Value::Num(7.0)]),
            (
                Value::OutputList(vec![Value::Num(1.0), Value::Num(2.0)]),
                vec![Value::Num(1.0), Value::Num(2.0)],
            ),
        ];
        for (legacy, expected) in cases {
            let mut stack = vec![Value::Num(99.0)];
            let mut state = SequenceState::default();
            let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(legacy)
                .expect("adapt builtin result");
            state
                .stage_call_output_sequence(&mut stack, sequence)
                .expect("stage call outputs");
            assert!(stack
                .iter()
                .all(|value| !matches!(value, Value::OutputList(_))));
            state.capture_call_outputs().expect("capture call outputs");
            let sequence = take_sequence_register(&mut stack, &mut state).expect("take sequence");
            assert_eq!(sequence.into_values(), expected);
            assert_eq!(stack, vec![Value::Num(99.0)]);
        }
    }

    #[test]
    fn nested_legacy_carrier_never_enters_the_register() {
        let stack: Vec<Value> = Vec::new();
        let mut state = SequenceState::default();
        let error =
            runmat_runtime::call::arguments::adapt_legacy_builtin_result(Value::OutputList(vec![
                Value::OutputList(vec![Value::Num(1.0)]),
            ]))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NestedLegacyOutputList"));
        assert!(stack.is_empty());
        assert!(state.capture_call_outputs().is_err());
    }
}
