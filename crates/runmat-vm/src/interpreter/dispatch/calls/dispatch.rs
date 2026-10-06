//! One call-family boundary for VM instructions that can suspend. Each child
//! owns a bounded subset of the call grammar so its polling frame does not
//! become part of the general instruction dispatch frame.

use super::super::*;
use std::future::Future;
use std::pin::Pin;

#[path = "dispatch/builtin.rs"]
mod builtin;
#[path = "dispatch/expanded.rs"]
mod expanded;
#[path = "dispatch/method.rs"]
mod method;
#[path = "dispatch/user.rs"]
mod user;
#[path = "dispatch/workspace.rs"]
mod workspace;

pub(in crate::interpreter::dispatch) struct CallDispatchContext<'a> {
    pub instr: &'a Instr,
    pub stack: &'a mut Vec<Value>,
    pub vars: &'a mut Vec<Value>,
    pub sequence_register: &'a mut SequenceState,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub try_stack: &'a mut Vec<crate::interpreter::state::ActiveTryHandler>,
    pub last_exception: &'a mut Option<runmat_value::MException>,
    pub imports: &'a Vec<(Vec<String>, bool)>,
    pub pc: &'a mut usize,
    pub source_id: Option<runmat_hir::SourceId>,
    pub call_arg_spans: Option<Vec<runmat_hir::Span>>,
    pub invocation_arity: runmat_runtime::context::RuntimeInvocationArity,
    pub function_registry: &'a crate::bytecode::FunctionRegistry,
    pub current_function_name: &'a str,
}

pub(in crate::interpreter::dispatch) fn dispatch_call<'a>(
    ctx: CallDispatchContext<'a>,
) -> Pin<Box<dyn Future<Output = Result<Option<DispatchHandled>, RuntimeError>> + 'a>> {
    match ctx.instr {
        Instr::CallBuiltinMulti(..)
        | Instr::CallBuiltinMultiUsingOutputSlot(..)
        | Instr::CallSuperConstructorMulti { .. }
        | Instr::CallSuperMethodMulti { .. }
        | Instr::CallFevalMulti(..)
        | Instr::CallFevalMultiUsingOutputSlot(..)
        | Instr::CallFevalExpandMultiOutput(..)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..) => Box::pin(builtin::dispatch(ctx)),
        Instr::CallSemanticFunctionMulti(..)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(..)
        | Instr::CallSemanticNestedFunctionMulti { .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
        | Instr::CallFunctionMulti { .. }
        | Instr::CallFunctionMultiUsingOutputSlot { .. } => Box::pin(user::dispatch(ctx)),
        Instr::CallWorkspaceFirstMulti { .. }
        | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. } => {
            Box::pin(workspace::dispatch(ctx))
        }
        Instr::CallBuiltinExpandMultiOutput(..)
        | Instr::CallSuperConstructorExpandMultiOutput { .. }
        | Instr::CallSuperMethodExpandMultiOutput { .. }
        | Instr::CallFunctionExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
        | Instr::CallSemanticFunctionExpandMultiOutput(..)
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. } => {
            Box::pin(expanded::dispatch(ctx))
        }
        Instr::CallMethodOrMemberIndexMulti { .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. } => Box::pin(method::dispatch(ctx)),
        _ => unreachable!("call dispatcher received a non-call instruction"),
    }
}
