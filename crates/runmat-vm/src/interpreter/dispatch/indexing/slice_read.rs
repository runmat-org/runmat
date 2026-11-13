use super::{
    idx_read_slice, invoke_resolved_object_index_path_method_in_context, object_slice_selector,
    read_scalar_like_slice, resolve_object_index_protocol, resolve_single_sequence, ObjectIndexOp,
    ObjectSubscript, ObjectSubscriptPath, RuntimeError, Value,
};
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::indexing::symbolic::apply_function_reference as apply_symbolic_function_reference;

pub(super) async fn dispatch(
    instr: &crate::bytecode::Instr,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    _vars: &mut InterpreterVars,
    function_registry: &crate::bytecode::FunctionRegistry,
    _pc: usize,
    runtime: &runmat_runtime::context::RuntimeContext,
    _sequence_register: &mut super::super::SequenceState,
) -> Result<bool, RuntimeError> {
    match instr {
        crate::bytecode::Instr::IndexSlice(dims, numeric_count, colon_mask, end_mask) => {
            let (numeric_owner, _operand_roots) =
                super::extract_selector_operands(stack, runtime, *numeric_count, false)?;
            let numeric = numeric_owner.as_slice();
            let base = stack.pop().ok_or(crate::interpreter::errors::mex(
                "StackUnderflow",
                "stack underflow",
            ))?;
            match base {
                Value::LogicalArray(value) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &value.shape,
                        )
                        .await?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime,
                        &selectors,
                        *dims,
                        &value.shape,
                    )?;
                    let selected = runmat_runtime::indexing::value::read_with_plan_in_context(
                        Some(runtime),
                        Value::LogicalArray(value),
                        &plan,
                    )?;
                    stack.try_push(match selected {
                        Value::LogicalArray(value)
                            if value.len() == 1
                                && runmat_runtime::builtins::common::shape::is_scalar_shape(
                                    &value.shape,
                                ) =>
                        {
                            Value::Bool(value.data[0] != 0)
                        }
                        other => other,
                    })?;
                }
                base @ (Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_)) => {
                    let resolution =
                        resolve_object_index_protocol(&base, ObjectIndexOp::Subsref, None)?;
                    if resolution.is_some() {
                        let path = ObjectSubscriptPath::single_in_context(
                            Some(runtime),
                            ObjectSubscript::parentheses(object_slice_selector(
                                runtime,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                numeric,
                            )?),
                        )?;
                        let sequence = invoke_resolved_object_index_path_method_in_context(
                            runtime,
                            &resolution,
                            base,
                            path,
                            1,
                        )
                        .await?;
                        resolve_single_sequence(sequence)?
                            .with_owned(|value| stack.try_push(value))?;
                    } else {
                        let shape = runmat_runtime::indexing::value::shape_borrowed(&base)
                            .expect("object-like values have an object-array shape");
                        let selectors =
                            runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                                runtime,
                                *dims,
                                *colon_mask,
                                *end_mask,
                                numeric,
                                shape.as_slice(),
                            )
                            .await?;
                        let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                            runtime,
                            &selectors,
                            *dims,
                            shape.as_slice(),
                        )?;
                        stack.try_push(
                            runmat_runtime::indexing::object::read_with_plan_in_context(
                                Some(runtime),
                                &base,
                                &plan,
                            )?,
                        )?;
                    }
                }
                Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
                | Value::ScopedFunctionHandle(_) => {
                    if *colon_mask != 0 || *end_mask != 0 {
                        return Err(crate::interpreter::errors::mex(
                            "UnsupportedFunctionHandleSelector",
                            "Function handle call does not support colon or end selector syntax",
                        ));
                    }
                    let arguments =
                        runmat_runtime::execution::RootedValues::from_output_storage_in_context(
                            numeric_owner,
                            runtime,
                        )?;
                    match crate::call::feval::execute_feval_retained(
                        runtime,
                        base,
                        arguments.into(),
                        1,
                        function_registry,
                    )
                    .await?
                    {
                        crate::call::feval::FevalDispatch::Completed(sequence) => {
                            resolve_single_sequence(sequence)?
                                .with_owned(|value| stack.try_push(value))?;
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
                        stack.try_push(apply_symbolic_function_reference(
                            name, parameters, numeric,
                        )?)?;
                    } else {
                        stack.try_push(
                            read_scalar_like_slice(
                                runtime,
                                &Value::Symbolic(expr),
                                *dims,
                                *colon_mask,
                                *end_mask,
                                numeric,
                            )
                            .await?,
                        )?;
                    }
                }
                Value::Tensor(t) => {
                    stack.try_push(
                        idx_read_slice::read_tensor_slice_nd_in_context(
                            Some(runtime),
                            &t,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                        )
                        .await?,
                    )?;
                }
                Value::ComplexTensor(ct) => stack.try_push(
                    idx_read_slice::read_complex_slice_in_context(
                        Some(runtime),
                        &ct,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                    )
                    .await?,
                )?,
                Value::SparseTensor(st) => stack.try_push(
                    idx_read_slice::read_sparse_slice_in_context(
                        Some(runtime),
                        &st,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                    )
                    .await?,
                )?,
                Value::GpuTensor(handle) => stack.try_push(
                    idx_read_slice::read_gpu_slice_in_context(
                        Some(runtime),
                        &handle,
                        *dims,
                        *colon_mask,
                        *end_mask,
                        numeric,
                    )
                    .await?,
                )?,
                Value::StringArray(sa) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &sa.shape,
                        )
                        .await?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime, &selectors, *dims, &sa.shape,
                    )?;
                    stack.try_push(idx_read_slice::gather_string_slice_in_context(
                        Some(runtime),
                        &sa,
                        &plan,
                    )?)?;
                }
                Value::CharArray(chars) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            chars.shape(),
                        )
                        .await?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime,
                        &selectors,
                        *dims,
                        chars.shape(),
                    )?;
                    stack.try_push(runmat_runtime::indexing::value::read_with_plan_in_context(
                        Some(runtime),
                        Value::CharArray(chars),
                        &plan,
                    )?)?;
                }
                Value::Cell(ca) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            &ca.shape,
                        )
                        .await?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime, &selectors, *dims, &ca.shape,
                    )?;
                    stack.try_push(runmat_runtime::indexing::value::read_with_plan_in_context(
                        Some(runtime),
                        Value::Cell(ca),
                        &plan,
                    )?)?;
                }
                Value::StructArray(array) => {
                    let selectors =
                        runmat_runtime::indexing::selectors::build_slice_selectors_in_context(
                            runtime,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                            array.shape(),
                        )
                        .await?;
                    let plan = runmat_runtime::indexing::plan::build_index_plan_in_context(
                        runtime,
                        &selectors,
                        *dims,
                        array.shape(),
                    )?;
                    stack.try_push(
                        runmat_runtime::indexing::structure::read_with_plan_in_context(
                            Some(runtime),
                            &array,
                            &plan,
                        )?,
                    )?;
                }
                other => {
                    stack.try_push(
                        read_scalar_like_slice(
                            runtime,
                            &other,
                            *dims,
                            *colon_mask,
                            *end_mask,
                            numeric,
                        )
                        .await?,
                    )?;
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
