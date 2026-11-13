macro_rules! with_instruction_function_references {
    ($instruction:expr, $context:expr, $direct:ident, $external_direct:ident, $identity:ident, $steps:ident) => {
        match $instruction {
            crate::bytecode::Instr::CreateSemanticClosure(function, ..)
            | crate::bytecode::Instr::CreateBoundFunctionHandle(function, ..)
            | crate::bytecode::Instr::CreateSemanticFuture(function, ..)
            | crate::bytecode::Instr::CreateSemanticFutureExpandMultiOutput(function, ..)
            | crate::bytecode::Instr::CallSemanticFunctionMulti(function, ..)
            | crate::bytecode::Instr::CallSemanticFunctionMultiUsingOutputSlot(function, ..)
            | crate::bytecode::Instr::CallSemanticFunctionExpandMultiOutput(function, ..)
            | crate::bytecode::Instr::CallSemanticNestedFunctionMulti { function, .. }
            | crate::bytecode::Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
                function,
                ..
            }
            | crate::bytecode::Instr::CallSemanticNestedFunctionExpandMultiOutput {
                function,
                ..
            } => {
                $direct($context, function);
            }
            crate::bytecode::Instr::CreateExternalBoundFunctionHandle(function, ..)
            | crate::bytecode::Instr::CreateExternalFuture(function, ..) => {
                $external_direct($context, function);
            }
            crate::bytecode::Instr::CallFunctionMulti { identity, .. }
            | crate::bytecode::Instr::CallFunctionMultiUsingOutputSlot { identity, .. }
            | crate::bytecode::Instr::CallFunctionExpandMultiOutput { identity, .. }
            | crate::bytecode::Instr::CallWorkspaceFirstMulti { identity, .. }
            | crate::bytecode::Instr::CallWorkspaceFirstMultiUsingOutputSlot { identity, .. }
            | crate::bytecode::Instr::CallWorkspaceFirstExpandMultiOutput { identity, .. }
            | crate::bytecode::Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
                identity,
                ..
            }
            | crate::bytecode::Instr::CallMethodOrMemberIndexMulti { identity, .. }
            | crate::bytecode::Instr::CallMethodOrMemberIndexExpandMultiOutput {
                identity, ..
            } => {
                $identity($context, identity);
            }
            crate::bytecode::Instr::ReadSubscriptPath { steps, .. }
            | crate::bytecode::Instr::CaptureSubscriptPath { steps, .. } => {
                $steps($context, steps);
            }
            crate::bytecode::Instr::BeginSubscriptEndReceiver { prefix } => {
                $steps($context, prefix);
            }
            crate::bytecode::Instr::LoadConst(..)
            | crate::bytecode::Instr::LoadInt(..)
            | crate::bytecode::Instr::LoadComplex(..)
            | crate::bytecode::Instr::LoadBool(..)
            | crate::bytecode::Instr::LoadString(..)
            | crate::bytecode::Instr::LoadCharRow(..)
            | crate::bytecode::Instr::LoadVar(..)
            | crate::bytecode::Instr::LoadVarForIndexAssignment(..)
            | crate::bytecode::Instr::LoadVarForCellAssignment(..)
            | crate::bytecode::Instr::StoreVar(..)
            | crate::bytecode::Instr::Add
            | crate::bytecode::Instr::Sub
            | crate::bytecode::Instr::Mul
            | crate::bytecode::Instr::RightDiv
            | crate::bytecode::Instr::LeftDiv
            | crate::bytecode::Instr::Pow
            | crate::bytecode::Instr::Neg
            | crate::bytecode::Instr::UPlus
            | crate::bytecode::Instr::Transpose
            | crate::bytecode::Instr::ConjugateTranspose
            | crate::bytecode::Instr::ElemMul
            | crate::bytecode::Instr::ElemDiv
            | crate::bytecode::Instr::ElemPow
            | crate::bytecode::Instr::ElemLeftDiv
            | crate::bytecode::Instr::LessEqual
            | crate::bytecode::Instr::Less
            | crate::bytecode::Instr::Greater
            | crate::bytecode::Instr::GreaterEqual
            | crate::bytecode::Instr::Equal
            | crate::bytecode::Instr::NotEqual
            | crate::bytecode::Instr::LogicalNot
            | crate::bytecode::Instr::LogicalAnd
            | crate::bytecode::Instr::LogicalOr
            | crate::bytecode::Instr::AndAnd(..)
            | crate::bytecode::Instr::OrOr(..)
            | crate::bytecode::Instr::JumpIfFalse(..)
            | crate::bytecode::Instr::Jump(..)
            | crate::bytecode::Instr::Pop
            | crate::bytecode::Instr::Unpack(..)
            | crate::bytecode::Instr::StochasticEvolution { .. }
            | crate::bytecode::Instr::CreateMatrix(..)
            | crate::bytecode::Instr::CreateMatrixFromSequences { .. }
            | crate::bytecode::Instr::CreateRange(..)
            | crate::bytecode::Instr::Index(..)
            | crate::bytecode::Instr::IndexSlice(..)
            | crate::bytecode::Instr::CreateCell2D(..)
            | crate::bytecode::Instr::CreateCellFromSequences { .. }
            | crate::bytecode::Instr::CreateStructLiteral(..)
            | crate::bytecode::Instr::CreateObjectLiteral { .. }
            | crate::bytecode::Instr::IndexCell { .. }
            | crate::bytecode::Instr::IndexCellExpand { .. }
            | crate::bytecode::Instr::IndexCellList { .. }
            | crate::bytecode::Instr::StoreIndex(..)
            | crate::bytecode::Instr::StoreIndexCell { .. }
            | crate::bytecode::Instr::StoreIndexDelete(..)
            | crate::bytecode::Instr::StoreIndexCellDelete { .. }
            | crate::bytecode::Instr::StoreSlice(..)
            | crate::bytecode::Instr::StoreSliceDelete(..)
            | crate::bytecode::Instr::LoadMember(..)
            | crate::bytecode::Instr::LoadMemberOrInit(..)
            | crate::bytecode::Instr::LoadMemberDynamic
            | crate::bytecode::Instr::LoadMemberDynamicOrInit
            | crate::bytecode::Instr::LoadMemberSequence { .. }
            | crate::bytecode::Instr::LoadMemberDynamicSequence { .. }
            | crate::bytecode::Instr::MemberSequenceCardinality
            | crate::bytecode::Instr::LoadMemberSequenceUsingOutputSlot { .. }
            | crate::bytecode::Instr::LoadMemberDynamicSequenceUsingOutputSlot { .. }
            | crate::bytecode::Instr::CaptureCallOutputSequence
            | crate::bytecode::Instr::CaptureScalarSequence
            | crate::bytecode::Instr::CaptureMemberSequence { .. }
            | crate::bytecode::Instr::CaptureMemberDynamicSequence { .. }
            | crate::bytecode::Instr::CaptureCellContentsSequence { .. }
            | crate::bytecode::Instr::CaptureReturnedOutputsSequence { .. }
            | crate::bytecode::Instr::MoveCapturedSequence { .. }
            | crate::bytecode::Instr::LoadSubscriptEnd { .. }
            | crate::bytecode::Instr::FinishSubscriptEndReceiver { .. }
            | crate::bytecode::Instr::FinishDottedInvokeReceiver { .. }
            | crate::bytecode::Instr::BeginOutputAssignment { .. }
            | crate::bytecode::Instr::BeginFixedAssignment { .. }
            | crate::bytecode::Instr::CommitFixedAssignment { .. }
            | crate::bytecode::Instr::PrepareFixedOutputTarget
            | crate::bytecode::Instr::PrepareDiscardOutputTarget
            | crate::bytecode::Instr::PrepareMemberSequenceOutputTarget { .. }
            | crate::bytecode::Instr::PrepareMemberDynamicSequenceOutputTarget { .. }
            | crate::bytecode::Instr::PrepareIndexedMemberSequenceOutputTarget { .. }
            | crate::bytecode::Instr::PrepareIndexedMemberDynamicSequenceOutputTarget { .. }
            | crate::bytecode::Instr::PrepareCellContentsSequenceOutputTarget { .. }
            | crate::bytecode::Instr::BeginSequenceOutputTarget { .. }
            | crate::bytecode::Instr::PrepareMemberPathStep { .. }
            | crate::bytecode::Instr::PrepareDynamicMemberPathStep { .. }
            | crate::bytecode::Instr::BeginPreparedIndexSelectors { .. }
            | crate::bytecode::Instr::LoadPreparedIndexEnd { .. }
            | crate::bytecode::Instr::BeginContextualIndexSelectors { .. }
            | crate::bytecode::Instr::LoadContextualIndexEnd { .. }
            | crate::bytecode::Instr::FinishContextualIndexSelectors { .. }
            | crate::bytecode::Instr::PrepareParenthesesPathStep { .. }
            | crate::bytecode::Instr::PrepareBracesPathStep { .. }
            | crate::bytecode::Instr::FinishMemberSequenceOutputTarget(..)
            | crate::bytecode::Instr::FinishDynamicMemberSequenceOutputTarget
            | crate::bytecode::Instr::FinishCellContentsSequenceOutputTarget { .. }
            | crate::bytecode::Instr::FinishFixedAssignmentTarget { .. }
            | crate::bytecode::Instr::LoadPreparedOutputCardinality
            | crate::bytecode::Instr::CommitPreparedOutputTargets { .. }
            | crate::bytecode::Instr::StoreMemberSequence(..)
            | crate::bytecode::Instr::StoreMemberDynamicSequence
            | crate::bytecode::Instr::StoreMember(..)
            | crate::bytecode::Instr::StoreMemberOrInit(..)
            | crate::bytecode::Instr::StoreMemberDynamic
            | crate::bytecode::Instr::StoreMemberDynamicOrInit
            | crate::bytecode::Instr::LoadMethod(..)
            | crate::bytecode::Instr::CreateFunctionHandle(..)
            | crate::bytecode::Instr::CreateExternalFunctionHandle(..)
            | crate::bytecode::Instr::CreateMethodFunctionHandle(..)
            | crate::bytecode::Instr::CreateClosure(..)
            | crate::bytecode::Instr::LoadStaticProperty(..)
            | crate::bytecode::Instr::LoadWorkspaceFirstStaticProperty { .. }
            | crate::bytecode::Instr::RegisterClass { .. }
            | crate::bytecode::Instr::CallFevalMulti(..)
            | crate::bytecode::Instr::CallFevalMultiUsingOutputSlot(..)
            | crate::bytecode::Instr::CallFevalExpandMultiOutput(..)
            | crate::bytecode::Instr::CallFevalExpandMultiOutputUsingOutputSlot(..)
            | crate::bytecode::Instr::ScheduleFeval { .. }
            | crate::bytecode::Instr::Spawn
            | crate::bytecode::Instr::SpawnOn
            | crate::bytecode::Instr::Await
            | crate::bytecode::Instr::FetchOutputs { .. }
            | crate::bytecode::Instr::FetchNext { .. }
            | crate::bytecode::Instr::EnsurePool(..)
            | crate::bytecode::Instr::CurrentPool(..)
            | crate::bytecode::Instr::ExecuteParfor { .. }
            | crate::bytecode::Instr::ExecuteSpmd { .. }
            | crate::bytecode::Instr::Distributed(..)
            | crate::bytecode::Instr::Collective { .. }
            | crate::bytecode::Instr::Swap
            | crate::bytecode::Instr::EnterTry { .. }
            | crate::bytecode::Instr::LeaveTry(..)
            | crate::bytecode::Instr::Return
            | crate::bytecode::Instr::ReturnValue
            | crate::bytecode::Instr::CallBuiltinMulti(..)
            | crate::bytecode::Instr::CallBuiltinMultiUsingOutputSlot(..)
            | crate::bytecode::Instr::CallSuperConstructorMulti { .. }
            | crate::bytecode::Instr::CallSuperMethodMulti { .. }
            | crate::bytecode::Instr::CallBuiltinExpandMultiOutput(..)
            | crate::bytecode::Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(..)
            | crate::bytecode::Instr::CallSuperConstructorExpandMultiOutput { .. }
            | crate::bytecode::Instr::CallSuperMethodExpandMultiOutput { .. }
            | crate::bytecode::Instr::PackToRow(..)
            | crate::bytecode::Instr::PackToCol(..)
            | crate::bytecode::Instr::EnterScope(..)
            | crate::bytecode::Instr::ExitScope(..)
            | crate::bytecode::Instr::LoadLocal(..)
            | crate::bytecode::Instr::StoreLocal(..)
            | crate::bytecode::Instr::RegisterImport { .. }
            | crate::bytecode::Instr::DeclareGlobal(..)
            | crate::bytecode::Instr::DeclarePersistent(..)
            | crate::bytecode::Instr::DeclareGlobalNamed(..)
            | crate::bytecode::Instr::DeclarePersistentNamed(..)
            | crate::bytecode::Instr::EmitStackTop { .. }
            | crate::bytecode::Instr::EmitVar { .. }
            | crate::bytecode::Instr::LoadVarForStructAssignment(..) => {}
        }
    };
}

pub(super) use with_instruction_function_references;
