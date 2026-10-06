use super::super::*;

pub(super) async fn execute(
    meta: DispatchMeta<'_>,
    state: DispatchState<'_>,
    hooks: DispatchHooks<'_>,
) -> Result<Option<DispatchHandled>, RuntimeError> {
    let instr = meta.instr;
    let _ = meta;
    let DispatchState { stack, .. } = state;
    let _ = hooks;
    match instr {
        Instr::PackToRow(count) => {
            pack_to_row(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::PackToCol(count) => {
            pack_to_col(stack, *count)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::Unpack(out_count) => {
            if *out_count > 0 {
                unpack(stack, *out_count)?;
            }
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMatrix(rows, cols) => {
            create_matrix(stack, *rows, *cols)?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateMatrixDynamic(num_rows) => {
            create_matrix_dynamic(stack, *num_rows, |rows_data| async move {
                runmat_runtime::create_matrix_from_values(&rows_data).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateRange(has_step) => {
            create_range(stack, *has_step, |args| async move {
                runmat_runtime::call_builtin_async("colon", &args).await
            })
            .await?;
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateCell2D(rows, cols) => {
            let mut elems = Vec::with_capacity(*rows * *cols);
            for _ in 0..(*rows * *cols) {
                elems.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            elems.reverse();
            stack.push(runmat_runtime::object::cell::create_cell_2d(
                elems, *rows, *cols,
            )?);
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateStructLiteral(fields) => {
            let values = pop_aggregate_literal_values(stack, fields.len())?;
            let mut struct_value = StructValue::new();
            for (field, value) in fields.iter().zip(values) {
                struct_value.fields.insert(field.clone(), value);
            }
            stack.push(Value::Struct(struct_value));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        Instr::CreateObjectLiteral { class_name, fields } => {
            let values = pop_aggregate_literal_values(stack, fields.len())?;
            let mut object = initialize_object_with_defaults(class_name);
            for (field, value) in fields.iter().zip(values) {
                object.properties.insert(field.clone(), value);
            }
            stack.push(Value::Object(object));
            Ok(Some(DispatchHandled::Generic(
                DispatchDecision::FallThrough,
            )))
        }
        _ => Ok(None),
    }
}
