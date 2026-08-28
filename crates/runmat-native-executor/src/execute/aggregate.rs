use runmat_mir::{MirAggregateKind, MirOperand};
use runmat_types::{MemberName, QualifiedName};
use runmat_value::{CellArray, ObjectInstance, StructValue, Value};

use crate::{NativeExecutorError, NativeExecutorResult};

use super::operand::materialize_operand;
use super::state::HostState;

pub(super) fn evaluate(
    state: &mut HostState,
    kind: &MirAggregateKind,
    rows: usize,
    cols: usize,
    elements: &[MirOperand],
) -> NativeExecutorResult<Value> {
    let values = elements
        .iter()
        .map(|operand| materialize_operand(state, operand))
        .collect::<NativeExecutorResult<Vec<_>>>()?;
    match kind {
        MirAggregateKind::Cell => CellArray::new(values, rows, cols)
            .map(Value::Cell)
            .map_err(NativeExecutorError::Host),
        MirAggregateKind::Tensor => {
            if rows.checked_mul(cols) != Some(values.len()) {
                return Err(NativeExecutorError::Host(
                    "MIR tensor aggregate shape does not match its elements".into(),
                ));
            }
            let matrix_rows = values
                .chunks(cols.max(1))
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            super::sync::complete(
                &state.runtime,
                runmat_runtime::create_matrix_from_values(&matrix_rows),
                "tensor aggregate construction",
            )
        }
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
