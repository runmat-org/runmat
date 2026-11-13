//! Object-array member projection into one admitted output owner.

use runmat_value::{ObjectArray, Utf8Text, Value};

use crate::context::{host_allocation::host_allocation_runtime_error, RuntimeContext};
use crate::execution::{RootedValueChildren, RootedValueMember, RootedValueSequence, RootedValues};
use crate::sequence::ValueSequence;
use crate::RuntimeError;

// Retire output slots before their only construction roots on errors or
// cancellation, including while an element getter is suspended.
struct ProjectionOutput<'a> {
    slots: runmat_value::OutputListStorage,
    children: RootedValueChildren<'a>,
}

pub(super) async fn gather_object_members(
    runtime: Option<&RuntimeContext>,
    array: ObjectArray,
    field: Utf8Text,
    allow_init: bool,
    caller: super::MemberCaller<'_>,
    dispatch: super::ProtocolDispatch,
) -> Result<RootedValueSequence, RuntimeError> {
    // Validate cardinality and admit the final slots before invoking a getter.
    runmat_value::validate_output_count(array.len())
        .map_err(crate::sequence::sequence_error_to_runtime)?;
    let slots = match runtime {
        Some(runtime) => RootedValues::try_initialize_in_context(array.len(), runtime, |_| {
            Ok::<_, RuntimeError>(Value::Num(0.0))
        })?,
        None => RootedValues::new(vec![Value::Num(0.0); array.len()])
            .map_err(|error| error.into_runtime_error())?,
    };
    slots
        .with_owned_output_storage_async(move |slots| async move {
            let allocation = |error| {
                host_allocation_runtime_error(
                    runtime,
                    "RunMat:MemberRead",
                    "object-array member root admission failed",
                    error,
                )
            };
            let children = match runtime {
                Some(runtime) => RootedValueChildren::in_context(runtime).map_err(allocation)?,
                None => RootedValueChildren::new(),
            };
            let mut output = ProjectionOutput { slots, children };
            let values = output.slots.try_as_mut_slice_unique().ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "RunMat:MemberRead",
                    "new object-array projection storage is not uniquely owned",
                )
            })?;
            for (slot, value) in values.iter_mut().zip(array.data()) {
                // The source array remains rooted by the enclosing member read.
                // Managed recursion uses the existing factory-before-box owner.
                let selected = match runtime {
                    Some(runtime) => {
                        crate::execution::admit_executable_sequence_with(runtime, || {
                            super::read::read_member_sequence_text_with_dispatch(
                                Some(runtime),
                                value.clone(),
                                field.clone(),
                                allow_init,
                                caller,
                                dispatch,
                            )
                        })?
                        .await?
                    }
                    None => {
                        Box::pin(super::read::read_member_sequence_text_with_dispatch(
                            None,
                            value.clone(),
                            field.clone(),
                            allow_init,
                            caller,
                            dispatch,
                        ))
                        .await?
                    }
                };
                let selected = selected.try_into_single()?;
                *slot = RootedValueMember::from_rooted(selected)
                    .into_child(&mut output.children)
                    .map_err(allocation)?;
            }
            let ProjectionOutput { slots, children } = output;
            let values = children.try_publish_values(slots).map_err(allocation)?;
            values.with_owned_output_storage(|storage| {
                let sequence = ValueSequence::from_output_storage(storage)
                    .map_err(crate::sequence::sequence_error_to_runtime)?;
                match runtime {
                    Some(runtime) => RootedValueSequence::in_context(sequence, runtime),
                    None => RootedValueSequence::new(sequence)
                        .map_err(|error| error.into_runtime_error()),
                }
            })
        })
        .await
}
