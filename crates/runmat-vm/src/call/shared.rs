use crate::bytecode::ArgSpec;
use crate::compiler::CompileError;
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

pub fn subsref_paren_index_cell(indices: &[Value]) -> Result<Value, RuntimeError> {
    Ok(Value::Cell(
        runmat_builtins::CellArray::new(indices.to_vec(), 1, indices.len())
            .map_err(|e| CompileError::new(format!("subsref build error: {e}")))?,
    ))
}

pub fn subsref_brace_index_cell_raw(indices: &[Value]) -> Result<Value, RuntimeError> {
    Ok(Value::Cell(
        runmat_builtins::CellArray::new(indices.to_vec(), 1, indices.len())
            .map_err(|e| CompileError::new(format!("subsref build error: {e}")))?,
    ))
}

pub fn subsref_empty_brace_cell() -> Result<Value, RuntimeError> {
    Ok(Value::Cell(
        runmat_builtins::CellArray::new(vec![], 1, 0)
            .map_err(|e| CompileError::new(format!("subsref build error: {e}")))?,
    ))
}

#[derive(Clone, Copy)]
pub enum ObjectIndexOp {
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
pub enum ObjectIndexKind {
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

pub enum ObjectIndexSelector {
    Indices(Value),
    Member(String),
}

pub struct ObjectIndexDescriptor {
    pub base: Value,
    pub op: ObjectIndexOp,
    pub kind: ObjectIndexKind,
    pub selector: ObjectIndexSelector,
    pub rhs: Option<Value>,
}

impl ObjectIndexDescriptor {
    fn into_runtime_method_args(self) -> Vec<Value> {
        let selector = match self.selector {
            ObjectIndexSelector::Indices(value) => value,
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
        args
    }
}

pub fn object_protocol_index_cell(
    values: Vec<Value>,
    context: &str,
) -> Result<Value, RuntimeError> {
    let cols = values.len();
    let cell =
        runmat_builtins::CellArray::new(values, 1, cols).map_err(|e| format!("{context}: {e}"))?;
    Ok(Value::Cell(cell))
}

pub async fn call_runtime_method(args: &[Value]) -> Result<Value, RuntimeError> {
    runmat_runtime::call_method_async(args).await
}

pub async fn call_object_index_method(
    base: Value,
    op: ObjectIndexOp,
    kind: ObjectIndexKind,
    cell: Value,
    rhs: Option<Value>,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method(ObjectIndexDescriptor {
        base,
        op,
        kind,
        selector: ObjectIndexSelector::Indices(cell),
        rhs,
    })
    .await
}

pub async fn call_object_member_method(
    base: Value,
    op: ObjectIndexOp,
    field: String,
    rhs: Option<Value>,
) -> Result<Value, RuntimeError> {
    call_object_index_descriptor_method(ObjectIndexDescriptor {
        base,
        op,
        kind: ObjectIndexKind::Member,
        selector: ObjectIndexSelector::Member(field),
        rhs,
    })
    .await
}

pub async fn call_object_index_descriptor_method(
    descriptor: ObjectIndexDescriptor,
) -> Result<Value, RuntimeError> {
    call_runtime_method(&descriptor.into_runtime_method_args()).await
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
    use super::{ObjectIndexDescriptor, ObjectIndexKind, ObjectIndexOp, ObjectIndexSelector};
    use runmat_builtins::Value;

    #[test]
    fn object_index_descriptor_serializes_protocol_args_once() {
        let descriptor = ObjectIndexDescriptor {
            base: Value::Num(1.0),
            op: ObjectIndexOp::Subsref,
            kind: ObjectIndexKind::Brace,
            selector: ObjectIndexSelector::Indices(Value::Num(2.0)),
            rhs: None,
        };

        let args = descriptor.into_runtime_method_args();
        assert_eq!(args[1], Value::String("subsref".to_string()));
        assert_eq!(args[2], Value::String("{}".to_string()));
        assert_eq!(args[3], Value::Num(2.0));
    }

    #[test]
    fn object_member_descriptor_carries_rhs() {
        let descriptor = ObjectIndexDescriptor {
            base: Value::Num(1.0),
            op: ObjectIndexOp::Subsasgn,
            kind: ObjectIndexKind::Member,
            selector: ObjectIndexSelector::Member("field".to_string()),
            rhs: Some(Value::Num(9.0)),
        };

        let args = descriptor.into_runtime_method_args();
        assert_eq!(args[1], Value::String("subsasgn".to_string()));
        assert_eq!(args[2], Value::String(".".to_string()));
        assert_eq!(args[3], Value::String("field".to_string()));
        assert_eq!(args[4], Value::Num(9.0));
    }
}
