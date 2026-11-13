//! Future outputs are immutable retained results, not unique mutable slots.
use super::{RuntimeError, Value};
use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_runtime::sequence::{ResolveValueSequence, SequenceResolutionContext};
use runmat_types::SequenceUse;

pub(super) fn select_outputs(
    sequence: RootedValueSequence,
    count: usize,
) -> Result<RootedValueSequence, RuntimeError> {
    let selected = sequence.resolve_sequence(
        SequenceUse::SelectPrefix { count },
        SequenceResolutionContext::default(),
    )?;
    if count == 1 {
        selected.try_into_single()?.try_into_sequence()
    } else {
        Ok(selected)
    }
}

pub(super) fn split_outputs(
    sequence: RootedValueSequence,
    count: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<RootedValues, RuntimeError> {
    sequence.with_owned_sequence(|sequence| {
        let values = sequence
            .resolve(
                SequenceUse::SelectPrefix { count },
                SequenceResolutionContext::default(),
            )
            .map_err(|_| {
                crate::interpreter::errors::mex(
                    "OutputCountMismatch",
                    "future result does not contain the requested number of outputs",
                )
            })?;
        RootedValues::from_output_storage_in_context(values, runtime)
    })
}

// Standard successor publication occurs while the caller retains source roots.
// Singleton projection then transfers that successor's existing root forest.
pub(super) fn root_join_outputs(
    outputs: runmat_value::OutputListStorage,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<RootedValueSequence, RuntimeError> {
    let count = outputs.len();
    let sequence = runmat_value::ValueSequence::from_output_storage(outputs)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
    let rooted = RootedValueSequence::in_context(sequence, runtime)?;
    select_outputs(rooted, count)
}

pub(super) fn fetch_next_outputs(
    index: usize,
    result: RootedValueSequence,
    requested_outputs: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<RootedValueSequence, RuntimeError> {
    if requested_outputs == 0 {
        return Ok(RootedValueSequence::empty());
    }
    let prefix = Value::Num((index + 1) as f64);
    if requested_outputs == 1 {
        return runmat_runtime::execution::RootedValue::in_context(prefix, runtime)?
            .try_into_sequence();
    }
    let result = result.resolve_sequence(
        SequenceUse::SelectPrefix {
            count: requested_outputs - 1,
        },
        SequenceResolutionContext::default(),
    )?;
    let selected = result.with_owned_sequence(|sequence| {
        RootedValues::from_output_storage_in_context(sequence.into_output_storage(), runtime)
    })?;
    selected
        .prepend_in_context(prefix, runtime)?
        .with_owned_output_storage(|storage| root_join_outputs(storage, runtime))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_fetch_refetch_await_and_index_prefix_share_original_payloads() {
        let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let _selected = runtime.enter();
        let tensor = runmat_value::Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let pointer = tensor.host_buffer().as_f64_slice().unwrap().as_ptr();
        let cell = runmat_value::CellArray::new(vec![Value::Tensor(tensor)], 1, 1).unwrap();
        let cell_pointer = cell.data().as_ptr();
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("nested", Value::Cell(cell));
        let source =
            runmat_value::OutputListStorage::shared_unbounded(vec![Value::Struct(fields.finish())]);
        let slots = source.as_slice().as_ptr();
        let completed = RootedValueSequence::new(
            runmat_value::ValueSequence::from_output_storage(source).unwrap(),
        )
        .unwrap();
        let check = |value: &Value| {
            let Value::Struct(record) = value else {
                panic!("original struct");
            };
            let Value::Cell(cell) = &record.fields["nested"] else {
                panic!("original cell");
            };
            assert_eq!(cell.data().as_ptr(), cell_pointer);
            let Value::Tensor(tensor) = &cell.data()[0] else {
                panic!("original numeric owner");
            };
            assert_eq!(
                tensor.host_buffer().as_f64_slice().unwrap().as_ptr(),
                pointer
            );
        };
        for _ in 0..2 {
            let fetched = select_outputs(completed.clone(), 1).unwrap();
            check(&fetched.as_slice()[0]);
            let awaited = completed.clone().try_into_single().unwrap();
            check(awaited.value());
            let next = fetch_next_outputs(3, completed.clone(), 2, &runtime).unwrap();
            assert_eq!(next.as_slice()[0], Value::Num(4.0));
            check(&next.as_slice()[1]);
            assert_eq!(completed.as_slice().as_ptr(), slots);
        }
        assert!(select_outputs(completed.clone(), 2).is_err());
        assert!(select_outputs(completed.clone(), 0).unwrap().is_empty());
        let joined = completed
            .clone()
            .with_owned_sequence(|sequence| {
                root_join_outputs(sequence.into_output_storage(), &runtime)
            })
            .unwrap();
        check(&joined.as_slice()[0]);
        check(&completed.as_slice()[0]);
    }
}
