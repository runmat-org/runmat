use super::*;

pub async fn paren_index_value(
    base: Value,
    raw_indices: Vec<Value>,
    requested_outputs: usize,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<Value, RuntimeError> {
    match &base {
        Value::ObjectArray(array) => {
            let selectors =
                build_slice_selectors(raw_indices.len(), 0, 0, &raw_indices, array.shape()).await?;
            let plan = build_index_plan(&selectors, raw_indices.len(), array.shape())?;
            gather_object_array_with_plan(array, &plan)
        }
        Value::Object(_) | Value::HandleObject(_) => {
            if let Some(err) = missing_member_index_overload_error(&base, ObjectIndexOp::Subsref) {
                return Err(err);
            }
            let descriptor = ObjectIndexDescriptor::subsref_paren(
                base,
                ObjectIndexSelector::IndexValues {
                    values: raw_indices,
                },
            );
            call_object_index_descriptor_method_with_outputs(descriptor, requested_outputs).await
        }
        Value::Cell(ca) => {
            let selectors = build_cell_scalar_selectors(&raw_indices).await?;
            let plan = build_index_plan(&selectors, raw_indices.len(), &ca.shape)?;
            Ok(gather_cell_with_plan(ca, &plan)?)
        }
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => match crate::call::feval::execute_feval(
            base,
            raw_indices,
            requested_outputs,
            function_registry,
        )
        .await?
        {
            crate::call::feval::FevalDispatch::Completed(value) => Ok(value),
        },
        Value::Symbolic(expr) => {
            if let Some((name, parameters)) = expr.function_reference_signature() {
                apply_symbolic_function_reference(name, parameters, &raw_indices)
            } else {
                let indices = linear_index_values(&raw_indices).await?;
                if indices == [1] {
                    Ok(Value::Symbolic(expr.clone()))
                } else {
                    idx_read_linear::generic_index(&Value::Symbolic(expr.clone()), &indices).await
                }
            }
        }
        _ => {
            let indices = linear_index_values(&raw_indices).await?;
            idx_read_linear::generic_index(&base, &indices).await
        }
    }
}

pub(super) fn symbolic_scalar_from_value(value: &Value) -> Result<SymbolicExpr, RuntimeError> {
    match value {
        Value::Symbolic(expr) => Ok(expr.clone()),
        Value::Num(value) => Ok(SymbolicExpr::constant(*value)),
        Value::Int(value) => Ok(SymbolicExpr::constant(value.to_f64())),
        Value::Bool(value) => Ok(SymbolicExpr::constant(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor)
            if runmat_runtime::builtins::common::tensor::is_scalar_tensor(tensor) =>
        {
            Ok(SymbolicExpr::constant(tensor_value_f64(tensor, 0)))
        }
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(SymbolicExpr::constant(if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }))
        }
        other => Err(crate::interpreter::errors::mex(
            "InvalidSymbolicFunctionArgument",
            &format!("symbolic function arguments must be scalar values, got {other:?}"),
        )),
    }
}

pub(super) fn apply_symbolic_function_reference(
    name: &str,
    parameters: &[String],
    raw_indices: &[Value],
) -> Result<Value, RuntimeError> {
    if raw_indices.len() != parameters.len() {
        return Err(crate::interpreter::errors::mex(
            "SymbolicFunctionArity",
            &format!(
                "symbolic function {name} expects {} argument(s), got {}",
                parameters.len(),
                raw_indices.len()
            ),
        ));
    }
    let args = raw_indices
        .iter()
        .map(symbolic_scalar_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Symbolic(SymbolicExpr::function_call(
        name.to_string(),
        args,
    )))
}

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    function_registry: &crate::bytecode::FunctionRegistry,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::Index(num_indices) => {
            let mut raw_indices = Vec::with_capacity(*num_indices);
            for _ in 0..*num_indices {
                raw_indices.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            raw_indices.reverse();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            stack.push(paren_index_value(base, raw_indices, 1, function_registry).await?);
            Ok(true)
        }
        _ => Ok(false),
    }
}
