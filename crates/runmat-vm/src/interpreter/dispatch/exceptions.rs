use crate::interpreter::state::ActiveTryHandler;
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::RuntimeError;
use runmat_value::{MException, Value};

pub enum ExceptionHandling {
    Caught,
    Uncaught(Box<RuntimeError>),
}

pub fn redirect_exception_to_catch(
    err: RuntimeError,
    try_stack: &mut Vec<ActiveTryHandler>,
    vars: &mut InterpreterVars,
    last_exception: &mut Option<MException>,
    pc: &mut usize,
    stack: &mut crate::interpreter::stack::InterpreterStack,
) -> ExceptionHandling {
    if let Some(handler) = try_stack.pop() {
        if stack.len() < handler.stack_depth {
            return ExceptionHandling::Uncaught(Box::new(
                runmat_runtime::runtime_error::semantic_error_with_source(
                    "TryFrameStackUnderflow",
                    "exception unwinding cannot restore its enclosing operand stack",
                    err,
                ),
            ));
        }
        // Retire only temporaries created inside this try region. Repeated
        // catches must not retain a failed expression's partial operands.
        stack.truncate(handler.stack_depth);
        let mex = runmat_runtime::runtime_error::exception_from_error(err);
        *last_exception = Some(mex.clone());
        if let Some(var_idx) = handler.catch_var {
            let value = Value::MException(mex);
            if let Err(error) =
                crate::runtime::workspace::commit_workspace_slot_value(vars, var_idx, None, value)
            {
                return ExceptionHandling::Uncaught(Box::new(error));
            }
        }
        *pc = handler.catch_pc;
        ExceptionHandling::Caught
    } else {
        ExceptionHandling::Uncaught(Box::new(err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn catch_retires_partial_operands_without_losing_the_enclosing_stack() {
        let mut stack = crate::interpreter::stack::InterpreterStack::unbounded();
        stack.try_push(Value::Num(11.0)).expect("enclosing operand");
        let mut vars = InterpreterVars::from_owned(Vec::new());
        let mut last_exception = None;
        let mut pc = 0;
        for _ in 0..4 {
            let mut handlers = vec![ActiveTryHandler {
                scope: 1,
                catch_pc: 9,
                catch_var: None,
                stack_depth: stack.len(),
            }];
            stack.try_push(Value::Num(22.0)).expect("partial operand");
            let error = runmat_runtime::build_runtime_error("failed expression")
                .with_identifier("RunMat:Test:PartialExpression")
                .build();
            assert!(matches!(
                redirect_exception_to_catch(
                    error,
                    &mut handlers,
                    &mut vars,
                    &mut last_exception,
                    &mut pc,
                    &mut stack,
                ),
                ExceptionHandling::Caught
            ));
            assert_eq!(&*stack, &[Value::Num(11.0)]);
            assert_eq!(pc, 9);
            assert!(handlers.is_empty());
            assert_eq!(
                last_exception.as_ref().expect("caught error").identifier,
                "RunMat:Test:PartialExpression"
            );
        }
    }
}
