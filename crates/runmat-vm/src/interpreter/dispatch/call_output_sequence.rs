use runmat_runtime::RuntimeError;

use super::{SequenceRegister, SequenceState};

impl SequenceState {
    pub(super) fn stage_rooted_call_output_sequence(
        &mut self,
        stack: &mut crate::interpreter::stack::InterpreterStack,
        sequence: runmat_runtime::execution::RootedValueSequence,
    ) -> Result<(), RuntimeError> {
        if self.call_outputs.is_some() {
            return Err(state_error(
                "call output sequence was overwritten before capture",
            ));
        }
        let window = SequenceRegister {
            start: stack.len(),
            len: sequence.len(),
        };
        // Existing rooted iteration moves unique slots or projects canonical
        // shared slots while retaining the original roots and physical claim.
        // try_extend preflights exact stack capacity before consuming a slot.
        stack.try_extend(sequence.try_into_values()?)?;
        self.call_outputs = Some(window);
        Ok(())
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
            (Value::output_list(Vec::new()), Vec::new()),
            (Value::Num(7.0), vec![Value::Num(7.0)]),
            (
                Value::output_list(vec![Value::Num(1.0), Value::Num(2.0)]),
                vec![Value::Num(1.0), Value::Num(2.0)],
            ),
        ];
        for (legacy, expected) in cases {
            let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
            stack.try_push(Value::Num(99.0)).expect("seed test operand");
            let mut state = SequenceState::default();
            let sequence = runmat_runtime::call::arguments::adapt_legacy_builtin_result(legacy)
                .expect("adapt builtin result");
            let sequence = runmat_runtime::execution::RootedValueSequence::new(sequence)
                .expect("retain adapted output roots");
            state
                .stage_rooted_call_output_sequence(&mut stack, sequence)
                .expect("stage call outputs");
            assert!(stack
                .iter()
                .all(|value| !matches!(value, Value::OutputList(_))));
            state.capture_call_outputs().expect("capture call outputs");
            let sequence = take_sequence_register(&mut stack, &mut state).expect("take sequence");
            assert_eq!(sequence.as_slice(), expected.as_slice());
            assert_eq!(stack.as_slice(), &[Value::Num(99.0)]);
        }
    }

    #[test]
    fn rooted_shared_prefix_stages_without_unwrapping_its_original_backing() {
        let sequence = runmat_runtime::execution::RootedValueSequence::new(
            runmat_value::ValueSequence::comma_separated(vec![
                Value::Num(1.0),
                Value::Num(2.0),
                Value::Num(3.0),
            ])
            .unwrap(),
        )
        .unwrap();
        let alias = sequence.clone();
        let prefix = sequence
            .resolve_sequence(
                runmat_types::SequenceUse::SelectPrefix { count: 2 },
                Default::default(),
            )
            .unwrap();
        let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
        stack.try_push(Value::Num(99.0)).unwrap();
        let mut state = SequenceState::default();
        state
            .stage_rooted_call_output_sequence(&mut stack, prefix)
            .unwrap();
        assert_eq!(
            stack.as_slice(),
            &[Value::Num(99.0), Value::Num(1.0), Value::Num(2.0)]
        );
        assert_eq!(
            alias.as_slice(),
            &[Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)]
        );
        state.capture_call_outputs().unwrap();
        let output = take_sequence_register(&mut stack, &mut state).unwrap();
        assert_eq!(output.as_slice(), &[Value::Num(1.0), Value::Num(2.0)]);
        assert_eq!(stack.as_slice(), &[Value::Num(99.0)]);
    }

    #[test]
    fn nested_legacy_carrier_never_enters_the_register() {
        let stack = crate::interpreter::stack::InterpreterStack::unbounded();
        let mut state = SequenceState::default();
        let error =
            runmat_runtime::call::arguments::adapt_legacy_builtin_result(Value::output_list(vec![
                Value::output_list(vec![Value::Num(1.0)]),
            ]))
            .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NestedLegacyOutputList"));
        assert!(stack.is_empty());
        assert!(state.capture_call_outputs().is_err());
    }
}
