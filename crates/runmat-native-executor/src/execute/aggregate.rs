use runmat_mir::{MirAggregateElement, MirAggregateKind, MirOperand};
use runmat_runtime::native::NativeValueRef;
use runmat_types::{MemberName, QualifiedName};
use runmat_value::{CellArray, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

fn adopt_tensor_result(
    state: &mut HostState,
    value: runmat_runtime::execution::RootedValue,
) -> NativeExecutorResult<NativeValueRef> {
    value.with_owned(|value| state.arena.insert(value, Some(&state.runtime)))
}

pub(super) fn evaluate(
    state: &mut HostState,
    kind: &MirAggregateKind,
    row_lengths: &[usize],
    elements: &[MirAggregateElement],
) -> NativeExecutorResult<NativeValueRef> {
    let embedded = if matches!(kind, MirAggregateKind::Tensor) {
        let embedded = state.current_embedded_operation()?;
        if let Some(value) = super::call_suspension::take_completed_semantic_value_operation(
            state,
            embedded.as_ref(),
            super::call_suspension::SemanticOperationKind::TensorAggregate,
        )? {
            return adopt_tensor_result(state, value);
        }
        embedded
    } else {
        None
    };
    let rows = row_lengths.len();
    let expected_elements = row_lengths
        .iter()
        .try_fold(0usize, |total, length| total.checked_add(*length));
    if expected_elements != Some(elements.len()) {
        return Err(NativeExecutorError::Host(
            "aggregate row boundaries do not match the element plan".into(),
        ));
    }
    let mut realized_rows = Vec::with_capacity(rows);
    let mut offset = 0usize;
    for &row_length in row_lengths {
        let end = offset
            .checked_add(row_length)
            .ok_or_else(|| NativeExecutorError::Host("aggregate row boundary overflow".into()))?;
        let mut row = Vec::new();
        for element in &elements[offset..end] {
            match element {
                MirAggregateElement::Single(operand) => {
                    row.push(materialize_operand(state, operand)?);
                }
                MirAggregateElement::CapturedSequence(sequence) => {
                    row.extend(super::call::take_captured_sequence(state, *sequence)?);
                }
            }
        }
        realized_rows.push(row);
        offset = end;
    }
    match kind {
        MirAggregateKind::Cell => {
            let columns = realized_rows.first().map_or(0, Vec::len);
            if realized_rows.iter().any(|row| row.len() != columns) {
                return Err(NativeExecutorError::Host(
                    "cell literal rows realize different widths after comma-separated-list expansion"
                        .into(),
                ));
            }
            let value =
                CellArray::new(realized_rows.into_iter().flatten().collect(), rows, columns)
                    .map(Value::Cell)
                    .map_err(NativeExecutorError::Host)?;
            state.arena.insert(value, Some(&state.runtime))
        }
        MirAggregateKind::Tensor => {
            if runmat_runtime::host_matrix_can_run_inline(&state.runtime, &realized_rows)
                .map_err(NativeExecutorError::from)?
            {
                let value = runmat_runtime::create_host_matrix_from_values(realized_rows)
                    .map_err(NativeExecutorError::from)?;
                return adopt_tensor_result(state, value);
            }
            let input_roots = runmat_runtime::execution::RootedValueLease::for_values(
                realized_rows.iter().flat_map(|row| row.iter()),
            )
            .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
            let future = async move {
                let _input_roots = input_roots;
                runmat_runtime::create_matrix_from_values(&realized_rows)
                    .await
                    .map_err(NativeExecutorError::from)
            };
            let value = super::call_suspension::begin_semantic_value_operation(
                state,
                embedded,
                super::call_suspension::SemanticOperationKind::TensorAggregate,
                Vec::new(),
                move || future,
            )?;
            adopt_tensor_result(state, value)
        }
    }
}

pub(super) fn structure(
    state: &mut HostState,
    fields: &[(MemberName, MirOperand)],
) -> NativeExecutorResult<runmat_runtime::execution::RootedValue> {
    let runtime = state.runtime.clone();
    // Operand values remain rooted by the native arena while the admitted
    // collection is initialized and registered. No intermediate Vec is built.
    let values = runmat_runtime::execution::RootedValues::try_initialize_in_context(
        fields.len(),
        &runtime,
        |index| materialize_operand(state, &fields[index].1),
    )?;
    runmat_runtime::execution::construct_struct_in_context(
        &runtime,
        fields.iter().map(|(name, _)| name.0.as_str()),
        values,
    )
    .map_err(NativeExecutorError::from)
}

pub(super) fn object(
    state: &mut HostState,
    class_name: &QualifiedName,
    fields: &[(MemberName, MirOperand)],
) -> NativeExecutorResult<runmat_runtime::execution::RootedValue> {
    let class_name = class_name.display_name().ok_or_else(|| {
        NativeExecutorError::Host("object literal has an empty class name".into())
    })?;
    let runtime = state.runtime.clone();
    let values = runmat_runtime::execution::RootedValues::try_initialize_in_context(
        fields.len(),
        &runtime,
        |index| materialize_operand(state, &fields[index].1),
    )?;
    runmat_runtime::execution::construct_object_in_context(
        &runtime,
        &class_name,
        fields.iter().map(|(name, _)| name.0.as_str()),
        values,
    )
    .map_err(NativeExecutorError::from)
}
