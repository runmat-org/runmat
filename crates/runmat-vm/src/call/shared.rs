use crate::bytecode::{ArgSpec, UserFunction};
use crate::compiler::CompileError;
use runmat_builtins::{Type, Value};
use runmat_hir::{remapping, HirProgram, VarId};
use runmat_runtime::RuntimeError;
use std::collections::HashMap;
use std::future::Future;

pub struct PreparedUserCall {
    pub func: UserFunction,
    pub var_map: HashMap<VarId, VarId>,
    pub func_program: HirProgram,
    pub func_vars: Vec<Value>,
}

pub fn lookup_user_function(
    name: &str,
    functions: &HashMap<String, UserFunction>,
) -> Result<UserFunction, RuntimeError> {
    functions.get(name).cloned().ok_or_else(|| {
        crate::interpreter::errors::mex("UndefinedFunction", &format!("Undefined function: {name}"))
    })
}

pub fn validate_user_function_arity(
    name: &str,
    func: &UserFunction,
    arg_count: usize,
) -> Result<(), RuntimeError> {
    if !func.has_varargin {
        if arg_count < func.params.len() {
            return Err(crate::interpreter::errors::mex(
                "NotEnoughInputs",
                &format!(
                    "Function '{name}' expects {} inputs, got {arg_count}",
                    func.params.len()
                ),
            ));
        }
        if arg_count > func.params.len() {
            return Err(crate::interpreter::errors::mex(
                "TooManyInputs",
                &format!(
                    "Function '{name}' expects {} inputs, got {arg_count}",
                    func.params.len()
                ),
            ));
        }
    } else {
        let min_args = func.params.len().saturating_sub(1);
        if arg_count < min_args {
            return Err(crate::interpreter::errors::mex(
                "NotEnoughInputs",
                &format!("Function '{name}' expects at least {min_args} inputs, got {arg_count}"),
            ));
        }
    }
    Ok(())
}

pub fn prepare_user_call(
    func: UserFunction,
    args: &[Value],
    vars: &[Value],
) -> Result<PreparedUserCall, CompileError> {
    let var_map =
        remapping::create_complete_function_var_map(&func.params, &func.outputs, &func.body);
    let local_var_count = var_map.len();
    let remapped_body = remapping::remap_function_body(&func.body, &var_map);
    let func_vars_count = local_var_count.max(func.params.len());
    let mut func_vars = vec![Value::Num(0.0); func_vars_count];

    if func.has_varargin {
        let fixed = func.params.len().saturating_sub(1);
        for i in 0..fixed {
            if i < args.len() && i < func_vars.len() {
                func_vars[i] = args[i].clone();
            }
        }
        let mut rest: Vec<Value> = if args.len() > fixed {
            args[fixed..].to_vec()
        } else {
            Vec::new()
        };
        let cell = runmat_builtins::CellArray::new(
            std::mem::take(&mut rest),
            1,
            if args.len() > fixed {
                args.len() - fixed
            } else {
                0
            },
        )
        .map_err(|e| CompileError::new(format!("varargin: {e}")))?;
        if fixed < func_vars.len() {
            func_vars[fixed] = Value::Cell(cell);
        }
    } else {
        for (i, _param_id) in func.params.iter().enumerate() {
            if i < args.len() && i < func_vars.len() {
                func_vars[i] = args[i].clone();
            }
        }
    }

    for (original_var_id, local_var_id) in &var_map {
        let local_index = local_var_id.0;
        let global_index = original_var_id.0;
        if local_index < func_vars.len() && global_index < vars.len() {
            let is_parameter = func
                .params
                .iter()
                .any(|param_id| param_id == original_var_id);
            if !is_parameter {
                func_vars[local_index] = vars[global_index].clone();
            }
        }
    }

    if func.has_varargout {
        if let Some(varargout_oid) = func.outputs.last() {
            if let Some(local_id) = var_map.get(varargout_oid) {
                if local_id.0 < func_vars.len() {
                    let empty = runmat_builtins::CellArray::new(vec![], 1, 0)
                        .map_err(|e| CompileError::new(format!("varargout init: {e}")))?;
                    func_vars[local_id.0] = Value::Cell(empty);
                }
            }
        }
    }

    let mut func_var_types = func.var_types.clone();
    if func_var_types.len() < local_var_count {
        func_var_types.resize(local_var_count, Type::Unknown);
    }
    let func_program = HirProgram {
        body: remapped_body,
        var_types: func_var_types,
    };

    Ok(PreparedUserCall {
        func,
        var_map,
        func_program,
        func_vars,
    })
}

pub fn first_output_value(
    func: &UserFunction,
    var_map: &HashMap<VarId, VarId>,
    func_result_vars: &[Value],
) -> Value {
    if func.outputs.is_empty() {
        return Value::Num(0.0);
    }
    if func.has_varargout {
        let total_named = func.outputs.len().saturating_sub(1);
        if total_named > 0 {
            if let Some(oid) = func.outputs.first() {
                if let Some(local_id) = var_map.get(oid) {
                    if let Some(value) = func_result_vars.get(local_id.0) {
                        return value.clone();
                    }
                }
            }
        }
        if let Some(varargout_oid) = func.outputs.last() {
            if let Some(local_id) = var_map.get(varargout_oid) {
                if let Some(Value::Cell(ca)) = func_result_vars.get(local_id.0) {
                    if let Some(first) = ca.data.first() {
                        return (**first).clone();
                    }
                }
            }
        }
        return Value::Num(0.0);
    }
    let Some(output_id) = func.outputs.first() else {
        return Value::Num(0.0);
    };
    let Some(local_id) = var_map.get(output_id) else {
        return Value::Num(0.0);
    };
    func_result_vars
        .get(local_id.0)
        .cloned()
        .unwrap_or(Value::Num(0.0))
}

pub fn collect_multi_outputs(
    name: &str,
    func: &UserFunction,
    var_map: &HashMap<VarId, VarId>,
    func_result_vars: &[Value],
    out_count: usize,
) -> Result<Vec<Value>, RuntimeError> {
    let mut outputs = Vec::with_capacity(out_count);
    if func.has_varargout {
        let total_named = func.outputs.len().saturating_sub(1);
        let mut pushed = 0usize;
        for i in 0..total_named.min(out_count) {
            if let Some(oid) = func.outputs.get(i) {
                if let Some(local_id) = var_map.get(oid) {
                    let idx = local_id.0;
                    let v = func_result_vars
                        .get(idx)
                        .cloned()
                        .unwrap_or(Value::Num(0.0));
                    outputs.push(v);
                    pushed += 1;
                }
            }
        }
        if pushed < out_count {
            if let Some(varargout_oid) = func.outputs.last() {
                if let Some(local_id) = var_map.get(varargout_oid) {
                    if let Some(Value::Cell(ca)) = func_result_vars.get(local_id.0) {
                        let available = ca.data.len();
                        let need = out_count - pushed;
                        if need > available {
                            return Err(crate::interpreter::errors::mex(
                                "VarargoutMismatch",
                                &format!(
                                    "Function '{name}' returned {available} varargout values, {need} requested"
                                ),
                            ));
                        }
                        for vi in 0..need {
                            outputs.push((*ca.data[vi]).clone());
                        }
                    }
                }
            }
        }
    } else {
        let defined = func.outputs.len();
        if out_count > defined {
            return Err(crate::interpreter::errors::mex(
                "TooManyOutputs",
                &format!("Function '{name}' defines {defined} outputs, {out_count} requested"),
            ));
        }
        for i in 0..out_count {
            let v = func
                .outputs
                .get(i)
                .and_then(|oid| var_map.get(oid))
                .map(|lid| lid.0)
                .and_then(|idx| func_result_vars.get(idx))
                .cloned()
                .unwrap_or(Value::Num(0.0));
            outputs.push(v);
        }
    }
    Ok(outputs)
}

pub fn expand_cell_indices(
    cell: &runmat_builtins::CellArray,
    indices: &[Value],
) -> Result<Vec<Value>, RuntimeError> {
    match indices.len() {
        1 => match &indices[0] {
            Value::Num(n) => {
                let idx = *n as usize;
                if idx == 0 || idx > cell.data.len() {
                    return Err(crate::interpreter::errors::mex(
                        "CellIndexOutOfBounds",
                        "Cell index out of bounds",
                    ));
                }
                Ok(vec![(*cell.data[idx - 1]).clone()])
            }
            Value::Int(i) => {
                let idx = i.to_i64() as usize;
                if idx == 0 || idx > cell.data.len() {
                    return Err(crate::interpreter::errors::mex(
                        "CellIndexOutOfBounds",
                        "Cell index out of bounds",
                    ));
                }
                Ok(vec![(*cell.data[idx - 1]).clone()])
            }
            Value::Tensor(t) => {
                let mut out = Vec::with_capacity(t.data.len());
                for &val in &t.data {
                    let idx = val as usize;
                    if idx == 0 || idx > cell.data.len() {
                        return Err(crate::interpreter::errors::mex(
                            "CellIndexOutOfBounds",
                            "Cell index out of bounds",
                        ));
                    }
                    out.push((*cell.data[idx - 1]).clone());
                }
                Ok(out)
            }
            _ => Err(crate::interpreter::errors::mex(
                "CellIndexType",
                "Unsupported cell index type",
            )),
        },
        2 => {
            let r: f64 = (&indices[0]).try_into()?;
            let c: f64 = (&indices[1]).try_into()?;
            let (ir, ic) = (r as usize, c as usize);
            if ir == 0 || ir > cell.rows || ic == 0 || ic > cell.cols {
                return Err(crate::interpreter::errors::mex(
                    "CellSubscriptOutOfBounds",
                    "Cell subscript out of bounds",
                ));
            }
            Ok(vec![(*cell.data[(ir - 1) * cell.cols + (ic - 1)]).clone()])
        }
        _ => Err(crate::interpreter::errors::mex(
            "CellIndexType",
            "Unsupported cell index type",
        )),
    }
}

pub fn expand_all_cell(cell: &runmat_builtins::CellArray) -> Vec<Value> {
    cell.data.iter().map(|p| (*(*p)).clone()).collect()
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

pub fn subsref_brace_numeric_index_values(indices: &[Value]) -> Vec<Value> {
    indices
        .iter()
        .map(|v| Value::Num((v).try_into().unwrap_or(0.0)))
        .collect()
}

pub fn subsref_empty_brace_cell() -> Result<Value, RuntimeError> {
    Ok(Value::Cell(
        runmat_builtins::CellArray::new(vec![], 1, 0)
            .map_err(|e| CompileError::new(format!("subsref build error: {e}")))?,
    ))
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
                    Value::Cell(ca) => expand_all_cell(&ca),
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
            temp.extend(expanded);
        } else {
            temp.push(stack.pop().ok_or_else(|| {
                crate::interpreter::errors::mex("StackUnderflow", "stack underflow")
            })?);
        }
    }
    temp.reverse();
    Ok(temp)
}
