use runmat_mir::{MirAggregateElement, MirAggregateKind, MirOperand};
use runmat_types::{MemberName, QualifiedName};
use runmat_value::{CellArray, ObjectInstance, StructValue, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    kind: &MirAggregateKind,
    row_lengths: &[usize],
    elements: &[MirAggregateElement],
) -> NativeExecutorResult<Value> {
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
            CellArray::new(realized_rows.into_iter().flatten().collect(), rows, columns)
                .map(Value::Cell)
                .map_err(NativeExecutorError::Host)
        }
        MirAggregateKind::Tensor => super::sync::complete(
            &state.runtime,
            runmat_runtime::create_matrix_from_values(&realized_rows),
            "tensor aggregate construction",
        ),
    }
}

pub(super) fn structure(
    state: &mut HostState,
    fields: &[(MemberName, MirOperand)],
) -> NativeExecutorResult<Value> {
    let mut structure = StructValue::new();
    for (name, operand) in fields {
        structure.insert(name.0.clone(), materialize_operand(state, operand)?);
    }
    Ok(Value::Struct(structure))
}

pub(super) fn object(
    state: &mut HostState,
    class_name: &QualifiedName,
    fields: &[(MemberName, MirOperand)],
) -> NativeExecutorResult<Value> {
    let class_name = class_name.display_name().ok_or_else(|| {
        NativeExecutorError::Host("object literal has an empty class name".into())
    })?;
    let mut object = ObjectInstance::new(class_name);
    for (name, operand) in fields {
        object
            .properties
            .insert(name.0.clone(), materialize_operand(state, operand)?);
    }
    Ok(Value::Object(object))
}
