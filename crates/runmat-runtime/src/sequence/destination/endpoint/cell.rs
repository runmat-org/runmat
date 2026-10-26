//! The existing cell/object endpoint write owner.
use super::*;

pub(crate) async fn assign_cell_contents_with_context(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: runmat_value::ExactArrayStorage<ObjectIndexComponent>,
    values: runmat_value::ValueSequence,
) -> Result<RootedValue, RuntimeError> {
    assign_cell_contents_with_access(context, base, indices, values, None).await
}

pub(crate) async fn assign_cell_contents_with_access(
    context: Option<&crate::context::RuntimeContext>,
    base: Value,
    indices: runmat_value::ExactArrayStorage<ObjectIndexComponent>,
    values: runmat_value::ValueSequence,
    access: Option<&crate::object::protocol::ObjectAccessContext>,
) -> Result<RootedValue, RuntimeError> {
    let input_values =
        std::iter::once(&base)
            .chain(values.iter())
            .chain(indices.iter().filter_map(|component| match component {
                ObjectIndexComponent::Value(value) => Some(value),
                ObjectIndexComponent::Colon | ObjectIndexComponent::End => None,
            }));
    let _inputs = match context {
        Some(context) => RootedValueLease::for_values_iterator_in_context(input_values, context)?,
        None => RootedValueLease::for_values(input_values)
            .map_err(|source| source.into_runtime_error())?,
    };
    match base {
        Value::Cell(cell) => {
            let positions = crate::object::cell::resolve_cell_assignment_components_in_context(
                context, &cell, &indices,
            )?;
            crate::object::cell::assign_cell_value_multi_in_context(
                context,
                cell,
                &positions,
                values.as_slice(),
                runmat_gc::gc_record_write,
            )
            .and_then(|value| RootedValue::in_optional_context(value, context))
        }
        base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
            let resolution = crate::object::dispatch::resolve_object_index_protocol_with_access(
                &base,
                crate::object::indexing::ObjectIndexOp::Subsasgn,
                None,
                access,
            )?;
            let Some(method) = resolution else {
                return Err(semantic_error(
                    "MissingObjectProtocol",
                    "brace assignment object does not define subsasgn",
                ));
            };
            let step = ObjectSubscript::braces(ObjectIndexSelector::IndexValues {
                components: indices,
            });
            let path = match context {
                Some(context) => ObjectSubscriptPath::single_in_context(Some(context), step)?,
                None => ObjectSubscriptPath::single(step),
            };
            match context {
                Some(context) => {
                    crate::object::protocol::invoke_resolved_object_assignment_sequence_in_context(
                        context, &method, base, path, values,
                    )
                    .await
                }
                None => {
                    crate::object::protocol::invoke_resolved_object_assignment_sequence(
                        &method, base, path, values,
                    )
                    .await
                }
            }
        }
        _ => Err(semantic_error(
            "CellAssignmentOnNonCell",
            "brace sequence assignment requires a cell array or indexing object",
        )),
    }
}
