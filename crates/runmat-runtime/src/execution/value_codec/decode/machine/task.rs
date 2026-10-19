use super::*;
use runmat_execution::value::{InlineValue, ValuePayload};

#[path = "task/aggregate.rs"]
mod aggregate;
#[path = "task/dense.rs"]
mod dense;
#[path = "task/durable.rs"]
mod durable;
#[path = "task/exception.rs"]
mod exception;
#[path = "task/expansion.rs"]
mod expansion;
#[path = "task/sparse.rs"]
mod sparse;
#[path = "task/text.rs"]
mod text;
#[path = "task/text_value.rs"]
mod text_value;

use aggregate::{CallableTask, CellTask, ImmutableTask, StructTask};
use dense::DenseTask;
use durable::{ObjectArrayTask, PlainObjectTask, StructArrayTask};
use exception::ExceptionTask;
use sparse::SparseTask;
use text::TextCopy;
use text_value::TextValueTask;

pub(super) enum Task<'a> {
    Decode {
        payload: &'a ValuePayload,
        path: String,
    },
    Expand(expansion::Expansion<'a>),
    Struct(Box<StructTask<'a>>),
    Cell(CellTask<'a>),
    Callable(Box<CallableTask<'a>>),
    Immutable(Box<ImmutableTask<'a>>),
    PlainObject(Box<PlainObjectTask<'a>>),
    StructArray(Box<StructArrayTask<'a>>),
    ObjectArray(Box<ObjectArrayTask<'a>>),
    Dense(DenseTask<'a>),
    Sparse(Box<SparseTask<'a>>),
    Text(TextValueTask<'a>),
    Exception(Box<ExceptionTask<'a>>),
}

pub(super) fn advance<'a>(
    task: Task<'a>,
    machine: &mut DecodeMachine<'a>,
    maximum_work: usize,
) -> Result<usize, ValueCodecError> {
    match task {
        Task::Decode { payload, path } => decode_task(payload, path, machine),
        Task::Expand(task) => expansion::advance(task, machine, maximum_work),
        Task::Struct(mut task) => {
            let (completed, complete) = task.advance(maximum_work);
            if complete {
                let value = task.finish();
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::Struct(task));
            }
            Ok(completed)
        }
        Task::Cell(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::Cell(task));
            }
            Ok(completed)
        }
        Task::Callable(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::Callable(task));
            }
            Ok(completed)
        }
        Task::Immutable(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::Immutable(task));
            }
            Ok(completed)
        }
        Task::PlainObject(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::PlainObject(task));
            }
            Ok(completed)
        }
        Task::StructArray(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::StructArray(task));
            }
            Ok(completed)
        }
        Task::ObjectArray(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let children = task.take_children();
                machine.push_aggregate(value, children);
            } else {
                machine.tasks.push(Task::ObjectArray(task));
            }
            Ok(completed)
        }
        Task::Dense(mut task) => {
            let completed = task.advance(maximum_work)?;
            if task.is_complete() {
                let path = task.path().to_owned();
                machine.push_value(task.finish()?, &path)?;
            } else {
                machine.tasks.push(Task::Dense(task));
            }
            Ok(completed)
        }
        Task::Sparse(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let path = task.path().to_owned();
                machine.push_value(value, &path)?;
            } else {
                machine.tasks.push(Task::Sparse(task));
            }
            Ok(completed)
        }
        Task::Text(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let path = task.path().to_owned();
                machine.push_value(value, &path)?;
            } else {
                machine.tasks.push(Task::Text(task));
            }
            Ok(completed)
        }
        Task::Exception(mut task) => {
            let (completed, value) = task.advance(maximum_work)?;
            if let Some(value) = value {
                let path = task.path().to_owned();
                machine.push_value(value, &path)?;
            } else {
                machine.tasks.push(Task::Exception(task));
            }
            Ok(completed)
        }
    }
}

fn decode_task<'a>(
    payload: &'a ValuePayload,
    path: String,
    machine: &mut DecodeMachine<'a>,
) -> Result<usize, ValueCodecError> {
    let value = match payload {
        ValuePayload::Inline(value) => value,
        ValuePayload::Distributed(handle) => {
            machine.push_value(Value::Distributed(handle.clone()), &path)?;
            return Ok(1);
        }
        ValuePayload::Composite(handle) => {
            machine.push_value(Value::Composite(handle.clone()), &path)?;
            return Ok(1);
        }
        ValuePayload::Object(_) => {
            return Err(ValueCodecError::unsupported(
                &path,
                "object references require an execution object-store decoder",
            ))
        }
    };
    match value.as_ref() {
        InlineValue::Null => {
            return Err(ValueCodecError::unsupported(
                &path,
                "MATLAB has no null runtime value",
            ))
        }
        InlineValue::Logical(value) => machine.push_value(Value::Bool(*value), &path)?,
        InlineValue::F64Bits(value) => {
            machine.push_value(Value::Num(f64::from_bits(*value)), &path)?
        }
        InlineValue::I8(value) => machine.push_value(Value::Int(IntValue::I8(*value)), &path)?,
        InlineValue::I16(value) => machine.push_value(Value::Int(IntValue::I16(*value)), &path)?,
        InlineValue::I32(value) => machine.push_value(Value::Int(IntValue::I32(*value)), &path)?,
        InlineValue::I64(value) => machine.push_value(Value::Int(IntValue::I64(*value)), &path)?,
        InlineValue::U8(value) => machine.push_value(Value::Int(IntValue::U8(*value)), &path)?,
        InlineValue::U16(value) => machine.push_value(Value::Int(IntValue::U16(*value)), &path)?,
        InlineValue::U32(value) => machine.push_value(Value::Int(IntValue::U32(*value)), &path)?,
        InlineValue::U64(value) => machine.push_value(Value::Int(IntValue::U64(*value)), &path)?,
        InlineValue::ComplexF64Bits { real, imaginary } => machine.push_value(
            Value::Complex(f64::from_bits(*real), f64::from_bits(*imaginary)),
            &path,
        )?,
        InlineValue::String(value) => {
            let runtime = machine.runtime();
            let task = TextValueTask::string(value, path, runtime)?;
            machine.tasks.push(Task::Text(task));
        }
        InlineValue::Char { shape, code_points } => machine.tasks.push(Task::Text(
            TextValueTask::character(shape, code_points, path),
        )),
        InlineValue::StringArray { shape, values } => machine
            .tasks
            .push(Task::Text(TextValueTask::string_array(shape, values, path))),
        InlineValue::Dense(value) => machine
            .tasks
            .push(Task::Dense(DenseTask::new(value, path)?)),
        InlineValue::Sparse(value) => machine
            .tasks
            .push(Task::Sparse(Box::new(SparseTask::new(value, path)?))),
        InlineValue::Symbolic(_) => {
            return Err(ValueCodecError::unsupported(
                &path,
                "the registered symbolic codec is unavailable in this runtime",
            ))
        }
        InlineValue::Cell { shape, values } => {
            machine.tasks.push(Task::Expand(expansion::Expansion::Cell {
                shape,
                values,
                index: 0,
                children: Vec::with_capacity(values.len()),
                awaiting: false,
                path,
            }));
        }
        InlineValue::Struct(fields) | InlineValue::DurableOrderedStruct(fields) => {
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::Struct {
                    fields,
                    index: 0,
                    children: Vec::with_capacity(fields.len()),
                    awaiting: false,
                    path,
                }));
        }
        InlineValue::Exception(value) => {
            let frame_sources = machine.frame_sources();
            machine
                .tasks
                .push(Task::Exception(Box::new(ExceptionTask::new(
                    value,
                    path,
                    frame_sources,
                ))));
        }
        InlineValue::Callable(callable) => {
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::Callable {
                    callable,
                    index: 0,
                    children: Vec::with_capacity(callable.captures.len()),
                    awaiting: false,
                    path,
                }));
        }
        InlineValue::ImmutableValueClass(value) => {
            if value.schema_version != 1 {
                return Err(ValueCodecError::unsupported(
                    &path,
                    "the immutable value-class schema version is unavailable in this runtime",
                ));
            }
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::Immutable {
                    value,
                    index: 0,
                    children: Vec::with_capacity(value.fields.len()),
                    awaiting: false,
                    class_name: Some(TextCopy::new(&value.type_identity)),
                    class_identity: None,
                    path,
                }));
        }
        InlineValue::DurablePlainObject(value) => {
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::PlainObject {
                    value,
                    index: 0,
                    children: Vec::with_capacity(value.properties.len()),
                    awaiting: false,
                    path,
                }))
        }
        InlineValue::DurableStructArray(value) => {
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::StructArray {
                    value,
                    field: 0,
                    index: 0,
                    children: Vec::with_capacity(
                        value
                            .fields
                            .len()
                            .checked_mul(value.fields.first().map_or(0, |field| field.values.len()))
                            .expect("validated structure-array child count"),
                    ),
                    awaiting: false,
                    path,
                }));
        }
        InlineValue::DurableObjectArray(value) => {
            machine
                .tasks
                .push(Task::Expand(expansion::Expansion::ObjectArray {
                    value,
                    index: 0,
                    children: Vec::with_capacity(value.elements.len()),
                    awaiting: false,
                    path,
                }));
        }
    }
    Ok(1)
}
