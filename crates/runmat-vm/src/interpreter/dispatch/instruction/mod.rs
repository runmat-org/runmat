use super::*;
use futures::future::{ready, Either, LocalBoxFuture, Ready};
mod aggregates;
mod builtins;
mod control;
mod deferred;
mod named;
mod objects;
mod semantic;
mod values;

type InstructionDispatch<'a> = Either<
    Ready<Result<Option<DispatchHandled>, RuntimeError>>,
    LocalBoxFuture<'a, Result<Option<DispatchHandled>, RuntimeError>>,
>;

pub(super) fn dispatch<'a>(
    meta: DispatchMeta<'a>,
    state: DispatchState<'a>,
    hooks: DispatchHooks<'a>,
) -> InstructionDispatch<'a> {
    match meta.instr {
        Instr::EmitStackTop { .. }
        | Instr::EmitVar { .. }
        | Instr::LoadConst(_)
        | Instr::LoadInt(_)
        | Instr::LoadComplex(_, _)
        | Instr::LoadBool(_)
        | Instr::LoadString(_)
        | Instr::LoadCharRow(_)
        | Instr::LoadLocal(_)
        | Instr::LoadVar(_)
        | Instr::LoadVarForIndexAssignment(_)
        | Instr::StoreVar(_)
        | Instr::StoreLocal(_)
        | Instr::Swap
        | Instr::Pop => Either::Right(Box::pin(values::execute(meta, state, hooks))),
        Instr::AndAnd(_)
        | Instr::OrOr(_)
        | Instr::JumpIfFalse(_)
        | Instr::Jump(_)
        | Instr::EnterTry { .. }
        | Instr::LeaveTry(_)
        | Instr::ReturnValue
        | Instr::Return
        | Instr::EnterScope(_)
        | Instr::ExitScope(_)
        | Instr::RegisterImport { .. }
        | Instr::DeclareGlobal(_)
        | Instr::DeclareGlobalNamed(_, _)
        | Instr::DeclarePersistent(_)
        | Instr::DeclarePersistentNamed(_, _) => {
            Either::Right(Box::pin(control::execute(meta, state, hooks)))
        }
        Instr::PackToRow(_)
        | Instr::PackToCol(_)
        | Instr::Unpack(_)
        | Instr::CreateMatrix(_, _)
        | Instr::CreateMatrixDynamic(_)
        | Instr::CreateRange(_)
        | Instr::CreateCell2D(_, _)
        | Instr::CreateStructLiteral(_)
        | Instr::CreateObjectLiteral { .. } => {
            Either::Right(Box::pin(aggregates::execute(meta, state, hooks)))
        }
        Instr::CallBuiltinMulti(_, _, _)
        | Instr::CallBuiltinMultiUsingOutputSlot(_, _, _)
        | Instr::CallSuperConstructorMulti { .. }
        | Instr::CallSuperMethodMulti { .. }
        | Instr::CallFevalMulti(_, _)
        | Instr::CallFevalMultiUsingOutputSlot(_, _)
        | Instr::CallFevalExpandMultiOutput(_, _)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(_, _)
        | Instr::CallBuiltinExpandMultiOutput(_, _, _)
        | Instr::CallSuperConstructorExpandMultiOutput { .. }
        | Instr::CallSuperMethodExpandMultiOutput { .. } => {
            Either::Right(Box::pin(builtins::execute(meta, state, hooks)))
        }
        Instr::CreateSemanticFuture(_, _, _)
        | Instr::CreateSemanticFutureExpandMultiOutput(_, _, _)
        | Instr::Spawn
        | Instr::Await => Either::Right(Box::pin(deferred::execute(meta, state, hooks))),
        Instr::CallSemanticFunctionMulti(_, _, _)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, _)
        | Instr::CallSemanticNestedFunctionMulti { .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
        | Instr::CallSemanticFunctionExpandMultiOutput(_, _, _)
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. } => {
            Either::Right(Box::pin(semantic::execute(meta, state, hooks)))
        }
        Instr::CallFunctionMulti { .. }
        | Instr::CallFunctionMultiUsingOutputSlot { .. }
        | Instr::CallWorkspaceFirstMulti { .. }
        | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
        | Instr::CallFunctionExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. } => {
            Either::Right(Box::pin(named::execute(meta, state, hooks)))
        }
        Instr::CallMethodOrMemberIndexMulti { .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. }
        | Instr::LoadMethod(_)
        | Instr::CreateClosure(_, _)
        | Instr::CreateSemanticClosure(_, _, _)
        | Instr::CreateFunctionHandle(_)
        | Instr::CreateExternalFunctionHandle(_)
        | Instr::CreateMethodFunctionHandle(_)
        | Instr::CreateBoundFunctionHandle(_, _)
        | Instr::CreateExternalBoundFunctionHandle(_, _)
        | Instr::LoadStaticProperty(_, _)
        | Instr::LoadWorkspaceFirstStaticProperty { .. }
        | Instr::RegisterClass { .. } => {
            Either::Right(Box::pin(objects::execute(meta, state, hooks)))
        }
        _ => Either::Left(ready(Ok(None))),
    }
}
