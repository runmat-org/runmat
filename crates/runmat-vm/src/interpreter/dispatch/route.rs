//! Exhaustive ownership of the instruction-to-dispatch-family boundary.
//!
//! Keeping this match exhaustive makes a new bytecode instruction choose its
//! polling domain deliberately. Semantic implementations remain in their
//! existing family modules.

use crate::bytecode::Instr;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InstructionDispatchFamily {
    Immediate,
    SemanticClosure,
    NameHandle,
    Parallel,
    PreparedOutput,
    Call,
    Indexing,
    Object,
    Arithmetic,
    Output,
    Logical,
    Aggregate,
}

pub(super) fn family(instruction: &Instr) -> InstructionDispatchFamily {
    use InstructionDispatchFamily::*;
    match instruction {
        Instr::CreateSemanticClosure(..) => SemanticClosure,
        Instr::CreateFunctionHandle(..) | Instr::CreateExternalFunctionHandle(..) => NameHandle,
        Instr::CreateSemanticFuture(..)
        | Instr::CreateSemanticFutureExpandMultiOutput(..)
        | Instr::CreateExternalFuture(..)
        | Instr::ScheduleFeval { .. }
        | Instr::Spawn
        | Instr::SpawnOn
        | Instr::Await
        | Instr::FetchOutputs { .. }
        | Instr::FetchNext { .. }
        | Instr::EnsurePool(..)
        | Instr::CurrentPool(..)
        | Instr::ExecuteParfor { .. }
        | Instr::ExecuteSpmd { .. }
        | Instr::Distributed(..)
        | Instr::Collective { .. } => Parallel,

        Instr::BeginOutputAssignment { .. }
        | Instr::BeginFixedAssignment { .. }
        | Instr::CommitFixedAssignment { .. }
        | Instr::PrepareFixedOutputTarget
        | Instr::PrepareDiscardOutputTarget
        | Instr::PrepareMemberSequenceOutputTarget { .. }
        | Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
        | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
        | Instr::PrepareCellContentsSequenceOutputTarget { .. }
        | Instr::BeginSequenceOutputTarget { .. }
        | Instr::PrepareMemberPathStep { .. }
        | Instr::PrepareDynamicMemberPathStep { .. }
        | Instr::BeginPreparedIndexSelectors { .. }
        | Instr::LoadPreparedIndexEnd { .. }
        | Instr::BeginContextualIndexSelectors { .. }
        | Instr::LoadContextualIndexEnd { .. }
        | Instr::FinishContextualIndexSelectors { .. }
        | Instr::PrepareParenthesesPathStep { .. }
        | Instr::PrepareBracesPathStep { .. }
        | Instr::FinishMemberSequenceOutputTarget(..)
        | Instr::FinishDynamicMemberSequenceOutputTarget
        | Instr::FinishCellContentsSequenceOutputTarget { .. }
        | Instr::FinishFixedAssignmentTarget { .. }
        | Instr::LoadPreparedOutputCardinality
        | Instr::CommitPreparedOutputTargets { .. } => PreparedOutput,

        Instr::CallBuiltinMulti(..)
        | Instr::CallBuiltinMultiUsingOutputSlot(..)
        | Instr::CallSuperConstructorMulti { .. }
        | Instr::CallSuperMethodMulti { .. }
        | Instr::CallFunctionMulti { .. }
        | Instr::CallFunctionMultiUsingOutputSlot { .. }
        | Instr::CallWorkspaceFirstMulti { .. }
        | Instr::CallWorkspaceFirstMultiUsingOutputSlot { .. }
        | Instr::CallSemanticFunctionMulti(..)
        | Instr::CallSemanticFunctionMultiUsingOutputSlot(..)
        | Instr::CallSemanticNestedFunctionMulti { .. }
        | Instr::CallSemanticNestedFunctionMultiUsingOutputSlot { .. }
        | Instr::CallFunctionExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutput { .. }
        | Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot { .. }
        | Instr::CallSemanticFunctionExpandMultiOutput(..)
        | Instr::CallSemanticNestedFunctionExpandMultiOutput { .. }
        | Instr::CallBuiltinExpandMultiOutput(..)
        | Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(..)
        | Instr::CallSuperConstructorExpandMultiOutput { .. }
        | Instr::CallSuperMethodExpandMultiOutput { .. }
        | Instr::CallFevalMulti(..)
        | Instr::CallFevalMultiUsingOutputSlot(..)
        | Instr::CallFevalExpandMultiOutput(..)
        | Instr::CallFevalExpandMultiOutputUsingOutputSlot(..)
        | Instr::CallMethodOrMemberIndexMulti { .. }
        | Instr::CallMethodOrMemberIndexExpandMultiOutput { .. } => Call,

        Instr::Index(..)
        | Instr::IndexSlice(..)
        | Instr::IndexCell { .. }
        | Instr::IndexCellExpand { .. }
        | Instr::IndexCellList { .. }
        | Instr::StoreIndex(..)
        | Instr::StoreIndexCell { .. }
        | Instr::StoreIndexDelete(..)
        | Instr::StoreIndexCellDelete { .. }
        | Instr::StoreSlice(..)
        | Instr::StoreSliceDelete(..) => Indexing,

        Instr::LoadMember(..)
        | Instr::LoadMemberOrInit(..)
        | Instr::LoadMemberDynamic
        | Instr::LoadMemberDynamicOrInit
        | Instr::LoadMemberSequence { .. }
        | Instr::LoadMemberDynamicSequence { .. }
        | Instr::MemberSequenceCardinality
        | Instr::LoadMemberSequenceUsingOutputSlot { .. }
        | Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. }
        | Instr::CaptureCallOutputSequence
        | Instr::CaptureScalarSequence
        | Instr::CaptureMemberSequence { .. }
        | Instr::CaptureMemberDynamicSequence { .. }
        | Instr::CaptureCellContentsSequence { .. }
        | Instr::CaptureReturnedOutputsSequence { .. }
        | Instr::MoveCapturedSequence { .. }
        | Instr::ReadSubscriptPath { .. }
        | Instr::CaptureSubscriptPath { .. }
        | Instr::BeginSubscriptEndReceiver { .. }
        | Instr::LoadSubscriptEnd { .. }
        | Instr::FinishSubscriptEndReceiver { .. }
        | Instr::FinishDottedInvokeReceiver { .. }
        | Instr::StoreMemberSequence(..)
        | Instr::StoreMemberDynamicSequence
        | Instr::StoreMember(..)
        | Instr::StoreMemberOrInit(..)
        | Instr::StoreMemberDynamic
        | Instr::StoreMemberDynamicOrInit => Object,

        Instr::Add
        | Instr::Sub
        | Instr::Mul
        | Instr::RightDiv
        | Instr::LeftDiv
        | Instr::Pow
        | Instr::Neg
        | Instr::UPlus
        | Instr::Transpose
        | Instr::ConjugateTranspose
        | Instr::ElemMul
        | Instr::ElemDiv
        | Instr::ElemPow
        | Instr::ElemLeftDiv
        | Instr::LessEqual
        | Instr::Less
        | Instr::Greater
        | Instr::GreaterEqual
        | Instr::Equal
        | Instr::NotEqual
        | Instr::LogicalNot
        | Instr::LogicalAnd
        | Instr::LogicalOr => Arithmetic,

        Instr::EmitStackTop { .. } | Instr::EmitVar { .. } => Output,
        Instr::AndAnd(..) | Instr::OrOr(..) | Instr::JumpIfFalse(..) => Logical,
        Instr::CreateMatrixFromSequences { .. } | Instr::CreateRange(..) => Aggregate,

        Instr::LoadConst(..)
        | Instr::LoadInt(..)
        | Instr::LoadComplex(..)
        | Instr::LoadBool(..)
        | Instr::LoadString(..)
        | Instr::LoadCharRow(..)
        | Instr::LoadVar(..)
        | Instr::LoadVarForIndexAssignment(..)
        | Instr::LoadVarForCellAssignment(..)
        | Instr::LoadVarForStructAssignment(..)
        | Instr::StoreVar(..)
        | Instr::StochasticEvolution { .. }
        | Instr::Jump(..)
        | Instr::Pop
        | Instr::Unpack(..)
        | Instr::CreateMatrix(..)
        | Instr::CreateCell2D(..)
        | Instr::CreateCellFromSequences { .. }
        | Instr::CreateStructLiteral(..)
        | Instr::CreateObjectLiteral { .. }
        | Instr::LoadMethod(..)
        | Instr::CreateMethodFunctionHandle(..)
        | Instr::CreateBoundFunctionHandle(..)
        | Instr::CreateExternalBoundFunctionHandle(..)
        | Instr::CreateClosure(..)
        | Instr::LoadStaticProperty(..)
        | Instr::LoadWorkspaceFirstStaticProperty { .. }
        | Instr::RegisterClass { .. }
        | Instr::Swap
        | Instr::EnterTry { .. }
        | Instr::LeaveTry(..)
        | Instr::Return
        | Instr::ReturnValue
        | Instr::PackToRow(..)
        | Instr::PackToCol(..)
        | Instr::EnterScope(..)
        | Instr::ExitScope(..)
        | Instr::LoadLocal(..)
        | Instr::StoreLocal(..)
        | Instr::RegisterImport { .. }
        | Instr::DeclareGlobal(..)
        | Instr::DeclarePersistent(..)
        | Instr::DeclareGlobalNamed(..)
        | Instr::DeclarePersistentNamed(..) => Immediate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representative_routes_separate_ready_and_suspending_families() {
        assert_eq!(
            family(&Instr::LoadConst(1.0)),
            InstructionDispatchFamily::Immediate
        );
        assert_eq!(family(&Instr::Return), InstructionDispatchFamily::Immediate);
        assert_eq!(family(&Instr::Add), InstructionDispatchFamily::Arithmetic);
        assert_eq!(
            family(&Instr::CallBuiltinMulti("javaObject".into(), 0, 1)),
            InstructionDispatchFamily::Call
        );
        assert_eq!(family(&Instr::Await), InstructionDispatchFamily::Parallel);
        assert_eq!(
            family(&Instr::PrepareDynamicMemberPathStep {
                child_creation: runmat_types::AssignmentChildCreation::ExistingOnly
            }),
            InstructionDispatchFamily::PreparedOutput
        );
    }
}
