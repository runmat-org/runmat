use crate::bytecode::ArgSpec;
use crate::object::{BRACE_SELECTOR_NAME, MEMBER_SELECTOR_NAME, PAREN_SELECTOR_NAME};
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;
use std::future::Future;

pub fn expand_cell_indices(
    cell: &runmat_builtins::CellArray,
    indices: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    crate::ops::cells::expand_cell_indices(cell, indices)
}

pub fn expand_all_cell(cell: &runmat_builtins::CellArray) -> Result<Vec<Value>, RuntimeError> {
    crate::ops::cells::expand_all_cell_values(cell)
}

#[derive(Clone, Copy)]
pub(crate) enum ObjectIndexOp {
    Subsref,
    Subsasgn,
}

impl ObjectIndexOp {
    pub(crate) fn protocol_name(self) -> &'static str {
        match self {
            Self::Subsref => "subsref",
            Self::Subsasgn => "subsasgn",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ObjectIndexKind {
    Paren,
    Brace,
    Member,
}

impl ObjectIndexKind {
    pub(crate) fn protocol_name(self) -> &'static str {
        match self {
            Self::Paren => PAREN_SELECTOR_NAME,
            Self::Brace => BRACE_SELECTOR_NAME,
            Self::Member => MEMBER_SELECTOR_NAME,
        }
    }
}

pub(crate) enum ObjectIndexSelector {
    Empty {
        context: &'static str,
    },
    ScalarIndices {
        indices: Vec<usize>,
        context: &'static str,
    },
    IndexValues {
        values: Vec<Value>,
        context: &'static str,
    },
    Member(String),
}

pub(crate) struct ObjectIndexDescriptor {
    base: Value,
    op: ObjectIndexOp,
    kind: ObjectIndexKind,
    selector: ObjectIndexSelector,
    rhs: Option<Value>,
}

impl ObjectIndexDescriptor {
    pub(crate) fn subsref_paren(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: None,
        }
    }

    pub(crate) fn subsref_brace(base: Value, selector: ObjectIndexSelector) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: None,
        }
    }

    pub(crate) fn subsasgn_paren(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Paren,
            selector,
            rhs: Some(rhs),
        }
    }

    pub(crate) fn subsasgn_brace(base: Value, selector: ObjectIndexSelector, rhs: Value) -> Self {
        Self {
            base,
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Brace,
            selector,
            rhs: Some(rhs),
        }
    }

    pub(crate) fn member(
        base: Value,
        op: ObjectIndexOp,
        field: String,
        rhs: Option<Value>,
    ) -> Self {
        Self {
            base,
            op,
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member(field),
            rhs,
        }
    }

    fn into_runtime_method_args(self) -> Result<Vec<Value>, RuntimeError> {
        let selector = match self.selector {
            ObjectIndexSelector::Empty { context } => {
                build_protocol_index_cell(Vec::new(), context)?
            }
            ObjectIndexSelector::ScalarIndices { indices, context } => {
                let values = indices
                    .into_iter()
                    .map(|index| Value::Num(index as f64))
                    .collect();
                build_protocol_index_cell(values, context)?
            }
            ObjectIndexSelector::IndexValues { values, context } => {
                build_protocol_index_cell(values, context)?
            }
            ObjectIndexSelector::Member(field) => Value::String(field),
        };
        let mut args = vec![
            self.base,
            Value::String(self.op.protocol_name().to_string()),
            Value::String(self.kind.protocol_name().to_string()),
            selector,
        ];
        if let Some(rhs) = self.rhs {
            args.push(rhs);
        }
        Ok(args)
    }
}

fn build_protocol_index_cell(values: Vec<Value>, context: &str) -> Result<Value, RuntimeError> {
    let cols = values.len();
    let cell =
        runmat_builtins::CellArray::new(values, 1, cols).map_err(|e| format!("{context}: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn call_runtime_method(args: &[Value]) -> Result<Value, RuntimeError> {
    runmat_runtime::call_method_async(args).await
}

pub(crate) async fn call_object_member_method(
    base: Value,
    op: ObjectIndexOp,
    field: String,
    rhs: Option<Value>,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method(ObjectIndexDescriptor::member(base, op, field, rhs)).await
}

pub(crate) async fn call_object_index_descriptor_method(
    descriptor: ObjectIndexDescriptor,
) -> Result<Value, RuntimeError> {
    call_runtime_method(&descriptor.into_runtime_method_args()?).await
}

pub async fn build_expanded_args_from_specs<ExpandObjectAll, ExpandObjectIndices, FutAll, FutIdx>(
    stack: &mut Vec<Value>,
    specs: &[ArgSpec],
    invalid_expand_all_msg: &str,
    invalid_expand_msg: &str,
    mut expand_object_all: ExpandObjectAll,
    mut expand_object_indices: ExpandObjectIndices,
) -> Result<Vec<Value>, RuntimeError>
where
    ExpandObjectAll: FnMut(Value) -> FutAll,
    ExpandObjectIndices: FnMut(Value, Vec<Value>) -> FutIdx,
    FutAll: Future<Output = Result<Vec<Value>, RuntimeError>>,
    FutIdx: Future<Output = Result<Vec<Value>, RuntimeError>>,
{
    let mut temp: Vec<Value> = Vec::new();
    for spec in specs.iter().rev() {
        if spec.is_expand {
            let mut indices = Vec::with_capacity(spec.num_indices);
            for _ in 0..spec.num_indices {
                indices.push(stack.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
                })?);
            }
            indices.reverse();
            let base = stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?;

            let expanded = if spec.expand_all {
                match base {
                    Value::OutputList(outputs) => outputs,
                    Value::Cell(ca) => expand_all_cell(&ca)?,
                    other @ Value::Object(_) => expand_object_all(other).await?,
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "InvalidExpandAllTarget",
                            invalid_expand_all_msg,
                        ))
                    }
                }
            } else {
                match (base, indices.len()) {
                    (Value::Cell(ca), 1) | (Value::Cell(ca), 2) => {
                        expand_cell_indices(&ca, &indices)?
                    }
                    (other @ Value::Object(_), _) => expand_object_indices(other, indices).await?,
                    _ => {
                        return Err(crate::interpreter::errors::mex(
                            "ExpandError",
                            invalid_expand_msg,
                        ))
                    }
                }
            };
            temp.extend(expanded.into_iter().rev());
        } else {
            temp.push(stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?);
        }
    }
    temp.reverse();
    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::{ObjectIndexDescriptor, ObjectIndexOp, ObjectIndexSelector};
    use runmat_builtins::Value;

    #[test]
    fn object_index_descriptor_serializes_protocol_args_once() {
        let descriptor = ObjectIndexDescriptor::subsref_brace(
            Value::Num(1.0),
            ObjectIndexSelector::IndexValues {
                values: vec![Value::Num(2.0)],
                context: "test subsref build error",
            },
        );

        let args = descriptor
            .into_runtime_method_args()
            .expect("descriptor args");
        assert_eq!(args[1], Value::String("subsref".to_string()));
        assert_eq!(args[2], Value::String("{}".to_string()));
        match &args[3] {
            Value::Cell(cell) => assert_eq!((*cell.data[0]).clone(), Value::Num(2.0)),
            other => panic!("expected selector cell, got {other:?}"),
        }
    }

    #[test]
    fn object_member_descriptor_carries_rhs() {
        let descriptor = ObjectIndexDescriptor::member(
            Value::Num(1.0),
            ObjectIndexOp::Subsasgn,
            "field".to_string(),
            Some(Value::Num(9.0)),
        );

        let args = descriptor
            .into_runtime_method_args()
            .expect("descriptor args");
        assert_eq!(args[1], Value::String("subsasgn".to_string()));
        assert_eq!(args[2], Value::String(".".to_string()));
        assert_eq!(args[3], Value::String("field".to_string()));
        assert_eq!(args[4], Value::Num(9.0));
    }
}
