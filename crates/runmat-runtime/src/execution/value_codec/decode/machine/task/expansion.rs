use super::aggregate::{CallableTask, CellTask, ImmutableTask, StructTask};
use super::durable::{ObjectArrayTask, PlainObjectTask, StructArrayTask};
use super::text::TextCopy;
use super::{Task, *};
use runmat_execution::value::{CallableValue, DurableObjectArray, DurablePlainObject};
use runmat_execution::value::{DurableStructArray, RegisteredData, StructField, ValuePayload};

pub(in crate::execution::value_codec::decode::machine) enum Expansion<'a> {
    Cell {
        shape: &'a [u64],
        values: &'a [ValuePayload],
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
    Struct {
        fields: &'a [StructField],
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
    Callable {
        callable: &'a CallableValue,
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
    Immutable {
        value: &'a RegisteredData,
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        class_name: Option<TextCopy<'a>>,
        class_identity: Option<runmat_types::ClassIdentity>,
        path: ControlPath<'a>,
    },
    PlainObject {
        value: &'a DurablePlainObject,
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
    StructArray {
        value: &'a DurableStructArray,
        field: usize,
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
    ObjectArray {
        value: &'a DurableObjectArray,
        index: usize,
        children: ControlVec<'a, PartialValue>,
        awaiting: bool,
        path: ControlPath<'a>,
    },
}

pub(super) fn advance<'a>(
    task: Expansion<'a>,
    machine: &mut DecodeMachine<'a>,
    maximum_work: usize,
) -> Result<usize, ValueCodecError> {
    match task {
        Expansion::Cell {
            shape,
            values,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(child) = values.get(index) {
                machine.tasks.push(Task::Expand(Expansion::Cell {
                    shape,
                    values,
                    index: index + 1,
                    children,
                    awaiting: true,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Decode {
                    payload: child,
                    path: path.extended(format_args!("[{index}]"))?,
                })?;
            } else {
                machine.tasks.push(Task::Cell(machine.boxed(CellTask::new(
                    shape,
                    TakenValues::from_partials(children, machine.runtime())?,
                    path,
                    machine.runtime(),
                )?)?))?;
            }
        }
        Expansion::Struct {
            fields,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(field) = fields.get(index) {
                machine.tasks.push(Task::Expand(Expansion::Struct {
                    fields,
                    index: index + 1,
                    children,
                    awaiting: true,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Decode {
                    payload: &field.value,
                    path: path.extended(format_args!(".{}", field.name))?,
                })?;
            } else {
                machine
                    .tasks
                    .push(Task::Struct(machine.boxed(StructTask::new(
                        fields,
                        TakenValues::from_partials(children, machine.runtime())?,
                        path,
                        machine.runtime,
                    )?)?))?;
            }
        }
        Expansion::Callable {
            callable,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(capture) = callable.captures.get(index) {
                machine.tasks.push(Task::Expand(Expansion::Callable {
                    callable,
                    index: index + 1,
                    children,
                    awaiting: true,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Decode {
                    payload: capture,
                    path: path.extended(format_args!(".capture[{index}]"))?,
                })?;
            } else {
                let source_resolver = machine.source_resolver();
                machine
                    .tasks
                    .push(Task::Callable(machine.boxed(CallableTask::new(
                        callable,
                        TakenValues::from_partials(children, machine.runtime())?,
                        source_resolver,
                        path,
                        machine.runtime(),
                    )?)?))?;
            }
        }
        Expansion::Immutable {
            value,
            index,
            mut children,
            awaiting,
            mut class_name,
            mut class_identity,
            path,
        } => {
            if class_identity.is_none() {
                let copy = class_name.as_mut().expect("pending class identity");
                let completed = copy.advance(maximum_work);
                if !copy.is_complete() {
                    machine.tasks.push(Task::Expand(Expansion::Immutable {
                        value,
                        index,
                        children,
                        awaiting,
                        class_name,
                        class_identity,
                        path,
                    }))?;
                    return Ok(completed);
                }
                let name = class_name
                    .take()
                    .expect("completed class identity")
                    .finish();
                let registered =
                    crate::execution::value_codec::immutable_class::schema_for_type_identity(
                        &name,
                    )?;
                let identity = runmat_types::ClassIdentity::new(
                    registered
                        .as_ref()
                        .map_or(name, |schema| schema.class_name().to_owned()),
                )
                .map_err(|error| ValueCodecError::invalid(&path, error.to_string()))?;
                if registered.is_none()
                    && !crate::parallel::codistributor::is_supported_class(&identity)
                {
                    return Err(ValueCodecError::unsupported(
                        &path,
                        format!(
                            "unknown immutable value-class type identity `{}`",
                            value.type_identity,
                        ),
                    ));
                }
                class_identity = Some(identity);
                // Account the terminal text chunk before expanding a field.
                // Otherwise this branch could copy `maximum_work` bytes and
                // still perform the next task's unit under the same charge.
                machine.tasks.push(Task::Expand(Expansion::Immutable {
                    value,
                    index,
                    children,
                    awaiting,
                    class_name,
                    class_identity,
                    path,
                }))?;
                return Ok(completed);
            }
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(field) = value.fields.get(index) {
                machine.tasks.push(Task::Expand(Expansion::Immutable {
                    value,
                    index: index + 1,
                    children,
                    awaiting: true,
                    class_name,
                    class_identity,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Decode {
                    payload: &field.value,
                    path: path.extended(format_args!(".{}", field.name))?,
                })?;
            } else {
                machine
                    .tasks
                    .push(Task::Immutable(machine.boxed(ImmutableTask::new(
                        value,
                        class_identity.expect("validated class identity"),
                        TakenValues::from_partials(children, machine.runtime())?,
                        path,
                        machine.runtime(),
                    )?)?))?;
            }
        }
        Expansion::PlainObject {
            value,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(field) = value.properties.get(index) {
                machine.tasks.push(Task::Expand(Expansion::PlainObject {
                    value,
                    index: index + 1,
                    children,
                    awaiting: true,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Decode {
                    payload: &field.value,
                    path: path.extended(format_args!(".{}", field.name))?,
                })?;
            } else {
                machine
                    .tasks
                    .push(Task::PlainObject(machine.boxed(PlainObjectTask::new(
                        value,
                        TakenValues::from_partials(children, machine.runtime())?,
                        path,
                        machine.runtime(),
                    )?)?))?;
            }
        }
        Expansion::StructArray {
            value,
            field,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(column) = value.fields.get(field) {
                if let Some(child) = column.values.get(index) {
                    machine.tasks.push(Task::Expand(Expansion::StructArray {
                        value,
                        field,
                        index: index + 1,
                        children,
                        awaiting: true,
                        path: path.try_clone()?,
                    }))?;
                    machine.tasks.push(Task::Decode {
                        payload: child,
                        path: path.extended(format_args!(".{}[{index}]", column.name))?,
                    })?;
                } else {
                    machine.tasks.push(Task::Expand(Expansion::StructArray {
                        value,
                        field: field + 1,
                        index: 0,
                        children,
                        awaiting: false,
                        path,
                    }))?;
                }
            } else {
                machine
                    .tasks
                    .push(Task::StructArray(machine.boxed(StructArrayTask::new(
                        value,
                        TakenValues::from_partials(children, machine.runtime())?,
                        path,
                    )?)?))?;
            }
        }
        Expansion::ObjectArray {
            value,
            index,
            mut children,
            awaiting,
            path,
        } => {
            retain_completed_child(machine, &mut children, awaiting, &path)?;
            if let Some(object) = value.elements.get(index) {
                machine.tasks.push(Task::Expand(Expansion::ObjectArray {
                    value,
                    index: index + 1,
                    children,
                    awaiting: true,
                    path: path.try_clone()?,
                }))?;
                machine.tasks.push(Task::Expand(Expansion::PlainObject {
                    value: object,
                    index: 0,
                    children: machine.control.vector(object.properties.len(), &path)?,
                    awaiting: false,
                    path: path.extended(format_args!("[{index}]"))?,
                }))?;
            } else {
                machine
                    .tasks
                    .push(Task::ObjectArray(machine.boxed(ObjectArrayTask::new(
                        value,
                        TakenValues::from_partials(children, machine.runtime())?,
                        path,
                        machine.runtime(),
                    )?)?))?;
            }
        }
    }
    Ok(1)
}

fn retain_completed_child<'a>(
    machine: &mut DecodeMachine<'_>,
    children: &mut ControlVec<'a, PartialValue>,
    awaiting: bool,
    path: &str,
) -> Result<(), ValueCodecError> {
    if awaiting {
        children.push(machine.pop_partial(), path)?;
    }
    Ok(())
}
