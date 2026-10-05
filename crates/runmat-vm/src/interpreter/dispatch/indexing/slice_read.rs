use super::{
    apply_symbolic_function_reference, build_index_plan, build_slice_selectors,
    gather_cell_with_plan, gather_struct_array_with_plan, idx_read_slice,
    invoke_resolved_object_index_path_method_in_context, logical_value_from_tensor,
    map_slice_shape_error, object_slice_selector, read_scalar_like_slice,
    resolve_object_index_protocol, resolve_single_sequence, ObjectIndexOp, ObjectSubscript,
    ObjectSubscriptPath, RuntimeError, Value,
};

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut Vec<Value>,
    _vars: &mut [Value],
    function_registry: &crate::bytecode::FunctionRegistry,
    _pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    _sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
            let mut numeric: Vec<Value> = Vec::with_capacity(*numeric_count);
            for _ in 0..*numeric_count {
                numeric.push(stack.pop().ok_or(crate::interpreter::errors::mex(
                    "StackUnderflow",
                    "stack underflow",
                ))?);
            }
            numeric.reverse();
            let mut base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            let mut logical_base = false;
            base = match base {
                Value::LogicalArray(la) => {
                    logical_base = true;
                    let data: Vec<f64> = la
                        .data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    let tensor = runmat_value::Tensor::new(data, la.shape.clone())
                        .map_err(|e| map_slice_shape_error("slice", e))?;
                    Value::Tensor(tensor)
                }
                other => other,
            };
            match base {
                base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
                    let resolution =
                        resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
                    if matches!(
                        resolution,
                        runmat_runtime::object::protocol::ProtocolResolution::Method(_)
                    ) {
                        let path = ObjectSubscriptPath::single(ObjectSubscript::parentheses(
                            object_slice_selector(*dims, *colon_mask, *end_mask, &numeric)?,
                        ));
                        let sequence = invoke_resolved_object_index_path_method_in_context(
                            runtime,
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await?;
                        stack.push(resolve_single_sequence(sequence)?);
                    } else {
                        let shape = runmat_runtime::indexing::object::shape(&base)
                            .expect("object-like values have an object-array shape");
                        let selectors =
                            build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &shape)
                                .await?;
                        let plan = build_index_plan(&selectors, *dims, &shape)?;
                        stack.push(runmat_runtime::indexing::object::read_with_plan(
                            &base, &plan,
                        )?);
                    }
                }
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
                | Value::Closure(_) => {
                    if *colon_mask != 0 || *end_mask != 0 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedFunctionHandleSelector",
                            "Function handle call does not support colon or end selector syntax",
                        ));
                    }
                    let args = numeric;
                    match crate::call::feval::execute_feval(
                        runtime,
                        base,
                        args,
                        1,
                        function_registry,
                    )
                    .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(sequence) => {
                            stack.push(resolve_single_sequence(sequence)?)
                        }
                    }
                }
                Value::Symbolic(expr) => {
                    if let Some((name, parameters)) = expr.function_reference_signature() {
                        if *colon_mask != 0 || *end_mask != 0 {
                            return Err(crate::interpreter::errors::mex(
                                "UnsupportedSymbolicFunctionSelector",
                                "symbolic function calls do not support colon or end selector syntax",
                            ));
                        }
                        stack.push(apply_symbolic_function_reference(
                            name, parameters, &numeric,
                        )?);
                    } else {
                        stack.push(
                            read_scalar_like_slice(
                                &Value::Symbolic(expr),
                                *dims,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        );
                    }
                }
                Value::Tensor(t) => {
                    if *dims == 1 {
                        stack.push(
                            idx_read_slice::read_tensor_slice_1d(
                                &t,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        )
                    } else {
                        stack.push(
                            idx_read_slice::read_tensor_slice_nd(
                                &t,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                &numeric,
                            )
                            .await?,
                        )
                    }
                }
                Value::ComplexTensor(ct) => stack.push(
                    idx_read_slice::read_complex_slice(
                        &ct,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )
                    .await?,
                ),
                Value::SparseTensor(st) => stack.push(
                    idx_read_slice::read_sparse_slice(&st, *dims, *colon_mask, *end_mask, &numeric)
                        .await?,
                ),
                Value::GpuTensor(handle) => stack.push(
                    idx_read_slice::read_gpu_slice(
                        &handle,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                    )
                    .await?,
                ),
                Value::StringArray(sa) => stack.push(
                    idx_read_slice::read_string_slice(&sa, *dims, *colon_mask, *end_mask, &numeric)
                        .await?,
                ),
                Value::CharArray(chars) => {
                    let selectors = build_slice_selectors(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        chars.shape(),
                    )
                    .await?;
                    let plan = build_index_plan(&selectors, *dims, chars.shape())?;
                    stack.push(runmat_runtime::indexing::value::read_with_plan(
                        Value::CharArray(chars),
                        &plan,
                    )?);
                }
                Value::Cell(ca) => {
                    let selectors =
                        build_slice_selectors(*dims, *colon_mask, *end_mask, &numeric, &ca.shape)
                            .await?;
                    let plan = build_index_plan(&selectors, *dims, &ca.shape)?;
                    stack.push(gather_cell_with_plan(&ca, &plan)?);
                }
                Value::StructArray(array) => {
                    let selectors = build_slice_selectors(
                        *dims,
                        *colon_mask,
                        *end_mask,
                        &numeric,
                        array.shape(),
                    )
                    .await?;
                    let plan = build_index_plan(&selectors, *dims, array.shape())?;
                    stack.push(gather_struct_array_with_plan(&array, &plan)?);
                }
                other => {
                    stack.push(
                        read_scalar_like_slice(&other, *dims, *colon_mask, *end_mask, &numeric)
                            .await?,
                    );
                }
            }
            if logical_base {
                let result = stack.pop().ok_or(crate::interpreter::errors::mex(
                    "SliceNonTensor",
                    "logical slice missing result",
                ))?;
                let converted = match result {
                    Value::Tensor(t) => logical_value_from_tensor(t)?,
                    Value::Num(n) => Value::Bool(n != 0.0),
                    Value::Bool(_) | Value::LogicalArray(_) => result,
                    other => other,
                };
                stack.push(converted);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
