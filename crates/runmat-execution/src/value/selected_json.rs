//! Incremental JSON presentation of the canonical sessionless capture graph.
//! The cursor borrows typed payloads and keeps only traversal frames. It does
//! not create Values, resolve code, validate source authority, or own an executor.
use super::*;
use std::num::NonZeroUsize;
mod exception;
mod leaves;
mod nodes;

#[derive(Debug, thiserror::Error)]
pub enum SelectedGraphJsonError {
    #[error("selected graph JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("selected graph JSON allocation: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    #[error("selected graph JSON: {0}")]
    Contract(#[from] crate::ContractError),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectedGraphJsonProgress {
    Pending { completed_work: usize },
    Complete { completed_work: usize },
}
trait Atom {
    fn write(&self, target: &mut Vec<u8>) -> Result<(), serde_json::Error>;
}
impl<T: serde::Serialize> Atom for T {
    fn write(&self, target: &mut Vec<u8>) -> Result<(), serde_json::Error> {
        serde_json::to_writer(target, self)
    }
}
enum Task<'a> {
    Raw(&'static [u8]),
    Key(&'static str),
    Atom(&'a dyn Atom),
    Text(&'a str),
    TextChunk(&'a str, usize),
    OptionalText(Option<&'a str>),
    Array(Array<'a>),
    Items(Array<'a>, usize),
    Node(&'a GraphNode),
    Field(&'a GraphField),
    Binding(&'a GraphBinding),
    Dynamic(&'a DurableDynamicProperty),
    Payload(&'a ValuePayload),
    Inline(&'a InlineValue),
    Exception(&'a ExceptionValue),
    Runtime(&'a ExceptionRuntimeMetadata),
    CallFrame(&'a ExceptionCallFrame),
}
#[derive(Clone, Copy)]
enum Array<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U64(&'a [u64]),
    Refs(&'a [GraphNodeRef]),
    Nodes(&'a [GraphNode]),
    Fields(&'a [GraphField]),
    Bindings(&'a [GraphBinding]),
    Dynamic(&'a [DurableDynamicProperty]),
    Texts(&'a [String]),
    Strings(&'a [StringElement]),
    Exceptions(&'a [ExceptionValue]),
    Frames(&'a [ExceptionCallFrame]),
}
impl<'a> Array<'a> {
    fn len(self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::Refs(v) => v.len(),
            Self::Nodes(v) => v.len(),
            Self::Fields(v) => v.len(),
            Self::Bindings(v) => v.len(),
            Self::Dynamic(v) => v.len(),
            Self::Texts(v) => v.len(),
            Self::Strings(v) => v.len(),
            Self::Exceptions(v) => v.len(),
            Self::Frames(v) => v.len(),
        }
    }
    fn item(self, index: usize) -> Task<'a> {
        match self {
            Self::U8(v) => Task::Atom(&v[index]),
            Self::U16(v) => Task::Atom(&v[index]),
            Self::U64(v) => Task::Atom(&v[index]),
            Self::Refs(v) => Task::Atom(&v[index]),
            Self::Nodes(v) => Task::Node(&v[index]),
            Self::Fields(v) => Task::Field(&v[index]),
            Self::Bindings(v) => Task::Binding(&v[index]),
            Self::Dynamic(v) => Task::Dynamic(&v[index]),
            Self::Texts(v) => Task::Text(&v[index]),
            Self::Strings(v) => Task::OptionalText(v[index].as_str()),
            Self::Exceptions(v) => Task::Exception(&v[index]),
            Self::Frames(v) => Task::CallFrame(&v[index]),
        }
    }
}

pub struct SelectedGraphJsonCursor<'a> {
    tasks: Vec<Task<'a>>,
}
impl<'a> SelectedGraphJsonCursor<'a> {
    pub fn new(graph: &'a PayloadGraph) -> Result<Self, SelectedGraphJsonError> {
        let mut cursor = Self { tasks: Vec::new() };
        cursor.object([
            ("roots", Task::Array(Array::Refs(&graph.roots))),
            ("nodes", Task::Array(Array::Nodes(&graph.nodes))),
        ])?;
        Ok(cursor)
    }
    pub fn advance(
        &mut self,
        target: &mut Vec<u8>,
        maximum: NonZeroUsize,
    ) -> Result<SelectedGraphJsonProgress, SelectedGraphJsonError> {
        let mut completed = 0;
        while completed < maximum.get() {
            let Some(task) = self.tasks.pop() else {
                return Ok(SelectedGraphJsonProgress::Complete {
                    completed_work: completed,
                });
            };
            match task {
                Task::Raw(bytes) => append(target, bytes)?,
                Task::Key(name) => {
                    append(target, b"\"")?;
                    append(target, name.as_bytes())?;
                    append(target, b"\":")?;
                }
                Task::Atom(value) => {
                    target.try_reserve(512)?;
                    value.write(target)?;
                }
                Task::Text(value) => {
                    append(target, b"\"")?;
                    self.push(Task::TextChunk(value, 0))?;
                }
                Task::TextChunk(value, first) => {
                    if first == value.len() {
                        append(target, b"\"")?;
                    } else {
                        let mut end = value.len().min(first.saturating_add(1024));
                        while !value.is_char_boundary(end) {
                            end -= 1;
                        }
                        let bytes = serde_json::to_vec(&value[first..end])?;
                        append(target, &bytes[1..bytes.len() - 1])?;
                        self.push(Task::TextChunk(value, end))?;
                    }
                }
                Task::OptionalText(value) => match value {
                    Some(value) => self.push(Task::Text(value))?,
                    None => append(target, b"null")?,
                },
                Task::Array(values) => {
                    append(target, b"[")?;
                    self.push(Task::Items(values, 0))?;
                }
                Task::Items(values, index) => {
                    if index == values.len() {
                        append(target, b"]")?;
                    } else {
                        if index != 0 {
                            append(target, b",")?;
                        }
                        self.push(Task::Items(values, index + 1))?;
                        self.push(values.item(index))?;
                    }
                }
                Task::Node(value) => self.node(value)?,
                Task::Field(value) => self.object([
                    ("name", Task::Text(&value.name)),
                    ("value", Task::Atom(&value.value)),
                ])?,
                Task::Binding(value) => self.object([
                    ("binding", Task::Atom(&value.binding)),
                    ("value", Task::Atom(&value.value)),
                ])?,
                Task::Dynamic(value) => self.dynamic(value)?,
                Task::Payload(value) => self.payload(value)?,
                Task::Inline(value) => self.inline(value)?,
                Task::Exception(value) => self.exception(value)?,
                Task::Runtime(value) => self.runtime(value)?,
                Task::CallFrame(value) => self.call_frame(value)?,
            }
            completed += 1;
        }
        Ok(if self.tasks.is_empty() {
            SelectedGraphJsonProgress::Complete {
                completed_work: completed,
            }
        } else {
            SelectedGraphJsonProgress::Pending {
                completed_work: completed,
            }
        })
    }
    fn push(&mut self, task: Task<'a>) -> Result<(), SelectedGraphJsonError> {
        self.tasks.try_reserve(1)?;
        self.tasks.push(task);
        Ok(())
    }
    fn object<const N: usize>(
        &mut self,
        fields: [(&'static str, Task<'a>); N],
    ) -> Result<(), SelectedGraphJsonError> {
        self.tasks
            .try_reserve(N.saturating_mul(3).saturating_add(2))?;
        self.tasks.push(Task::Raw(b"}"));
        for (index, (key, value)) in fields.into_iter().enumerate().rev() {
            self.tasks.push(value);
            self.tasks.push(Task::Key(key));
            if index != 0 {
                self.tasks.push(Task::Raw(b","));
            }
        }
        self.tasks.push(Task::Raw(b"{"));
        Ok(())
    }
    fn invalid(reason: &'static str) -> SelectedGraphJsonError {
        crate::ContractError::invalid("selected graph JSON", reason).into()
    }
}
fn append(target: &mut Vec<u8>, bytes: &[u8]) -> Result<(), SelectedGraphJsonError> {
    target.try_reserve(bytes.len())?;
    target.extend_from_slice(bytes);
    Ok(())
}
