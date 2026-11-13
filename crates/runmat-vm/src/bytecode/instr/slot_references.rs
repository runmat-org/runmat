//! Exhaustive typed VM-slot references for private temporary use admission.
//! Adding an Instr variant requires an explicit classification here.

use crate::bytecode::{
    AggregateElementSpec, BytecodeDottedInvokeArgument, BytecodeSubscriptSelector,
    BytecodeSubscriptStep, EmitLabel, Instr, StochasticSourceOperand,
};
use runmat_runtime::call::arguments::{ArgumentExpansionSpec, ArgumentSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VmSlotAccess {
    StoreVar,
    LoadVar,
    Other,
}

impl Instr {
    pub(crate) fn visit_vm_slot_accesses(&self, mut visit: impl FnMut(usize, VmSlotAccess)) {
        match self {
            Instr::StoreVar(index) => visit(*index, VmSlotAccess::StoreVar),
            Instr::LoadVar(index) => visit(*index, VmSlotAccess::LoadVar),
            Instr::MoveCapturedSequence {
                source_slot,
                destination_slot,
            } => {
                visit(*source_slot, VmSlotAccess::Other);
                visit(*destination_slot, VmSlotAccess::Other);
            }
            Instr::StochasticEvolution {
                iteration_slot,
                noise_slot,
                state_slot,
                steps,
                drift,
                scale,
                randn_args,
                ..
            } => {
                for slot in [*iteration_slot, *noise_slot, *state_slot] {
                    visit(slot, VmSlotAccess::StoreVar);
                }
                visit(*state_slot, VmSlotAccess::LoadVar);
                for operand in std::iter::once(steps)
                    .chain(std::iter::once(drift))
                    .chain(std::iter::once(scale))
                    .chain(randn_args)
                {
                    if let StochasticSourceOperand::Slot(slot) = operand {
                        visit(*slot, VmSlotAccess::LoadVar);
                    }
                }
            }
            Instr::LoadVarForIndexAssignment(index)
            | Instr::LoadVarForCellAssignment(index)
            | Instr::LoadVarForStructAssignment(index)
            | Instr::LoadLocal(index)
            | Instr::StoreLocal(index)
            | Instr::CallFevalMultiUsingOutputSlot(_, index)
            | Instr::CallBuiltinMultiUsingOutputSlot(_, _, index)
            | Instr::CallSemanticFunctionMultiUsingOutputSlot(_, _, index) => {
                visit(*index, VmSlotAccess::Other);
            }
            Instr::LoadMemberSequenceUsingOutputSlot {
                output_count_slot: index,
                ..
            }
            | Instr::LoadMemberDynamicSequenceUsingOutputSlot {
                output_count_slot: index,
            }
            | Instr::CallFunctionMultiUsingOutputSlot {
                out_count_slot: index,
                ..
            }
            | Instr::CallWorkspaceFirstMultiUsingOutputSlot {
                out_count_slot: index,
                ..
            } => {
                visit(*index, VmSlotAccess::Other);
            }
            Instr::CaptureMemberSequence {
                sequence_slot: index,
                ..
            }
            | Instr::CaptureMemberDynamicSequence {
                sequence_slot: index,
            }
            | Instr::CaptureCellContentsSequence {
                sequence_slot: index,
                ..
            }
            | Instr::CaptureReturnedOutputsSequence {
                sequence_slot: index,
            } => {
                visit(*index, VmSlotAccess::Other);
            }
            Instr::PrepareMemberSequenceOutputTarget {
                root_slot: index, ..
            }
            | Instr::PrepareMemberDynamicSequenceOutputTarget { root_slot: index }
            | Instr::PrepareIndexedMemberSequenceOutputTarget {
                root_slot: index, ..
            }
            | Instr::PrepareIndexedMemberDynamicSequenceOutputTarget {
                root_slot: index, ..
            }
            | Instr::PrepareCellContentsSequenceOutputTarget {
                root_slot: index, ..
            }
            | Instr::BeginSequenceOutputTarget {
                root_slot: index, ..
            }
            | Instr::BeginFixedAssignment {
                root_slot: index, ..
            }
            | Instr::CommitFixedAssignment {
                root_slot: index, ..
            } => {
                visit(*index, VmSlotAccess::Other);
            }
            Instr::CallSemanticNestedFunctionMulti { capture_slots, .. } => {
                for &slot in capture_slots {
                    visit(slot, VmSlotAccess::Other);
                }
            }
            Instr::CallSemanticNestedFunctionMultiUsingOutputSlot {
                capture_slots,
                out_count_slot,
                ..
            } => {
                for &slot in capture_slots {
                    visit(slot, VmSlotAccess::Other);
                }
                visit(*out_count_slot, VmSlotAccess::Other);
            }
            Instr::CallSemanticNestedFunctionExpandMultiOutput {
                capture_slots,
                specs,
                ..
            } => {
                for &slot in capture_slots {
                    visit(slot, VmSlotAccess::Other);
                }
                visit_specs(specs, &mut visit);
            }
            Instr::CallFevalExpandMultiOutput(specs, _)
            | Instr::CreateSemanticFutureExpandMultiOutput(_, specs, _)
            | Instr::CreateExternalFuture(_, specs, _)
            | Instr::CallSemanticFunctionExpandMultiOutput(_, specs, _)
            | Instr::CallBuiltinExpandMultiOutput(_, specs, _)
            | Instr::CallSuperConstructorExpandMultiOutput { specs, .. }
            | Instr::CallSuperMethodExpandMultiOutput { specs, .. }
            | Instr::CallMethodOrMemberIndexExpandMultiOutput { specs, .. }
            | Instr::CallFunctionExpandMultiOutput { specs, .. }
            | Instr::CallWorkspaceFirstExpandMultiOutput { specs, .. } => {
                visit_specs(specs, &mut visit);
            }
            Instr::CreateMatrixFromSequences { elements, .. }
            | Instr::CreateCellFromSequences { elements, .. } => {
                for element in elements {
                    match element {
                        AggregateElementSpec::CapturedSequence { slot } => {
                            visit(*slot, VmSlotAccess::Other);
                        }
                        AggregateElementSpec::Single => {}
                    }
                }
            }
            Instr::ReadSubscriptPath { steps, .. }
            | Instr::BeginSubscriptEndReceiver { prefix: steps } => {
                visit_steps(steps, &mut visit);
            }
            Instr::FinishFixedAssignmentTarget { endpoint, .. } => {
                visit_steps(std::slice::from_ref(endpoint), &mut visit)
            }
            Instr::PrepareParenthesesPathStep { selectors, .. } => {
                for selector in selectors {
                    if let BytecodeSubscriptSelector::Expansion(spec) = selector {
                        visit_specs(std::slice::from_ref(spec), &mut visit);
                    }
                }
            }
            Instr::CaptureSubscriptPath {
                steps,
                sequence_slot,
                ..
            } => {
                visit(*sequence_slot, VmSlotAccess::Other);
                visit_steps(steps, &mut visit);
            }
            Instr::EnterTry { catch_var, .. } => {
                if let Some(slot) = catch_var {
                    visit(*slot, VmSlotAccess::Other);
                }
            }
            Instr::DeclareGlobal(slots)
            | Instr::DeclarePersistent(slots)
            | Instr::DeclareGlobalNamed(slots, _)
            | Instr::DeclarePersistentNamed(slots, _) => {
                for &slot in slots {
                    visit(slot, VmSlotAccess::Other);
                }
            }
            Instr::EmitVar { var_index, label } => {
                visit(*var_index, VmSlotAccess::Other);
                visit_label(label, &mut visit);
            }
            Instr::EmitStackTop { label } => {
                visit_label(label, &mut visit);
            }
            Instr::CallWorkspaceFirstExpandMultiOutputUsingOutputSlot {
                specs,
                out_count_slot,
                ..
            } => {
                visit(*out_count_slot, VmSlotAccess::Other);
                visit_specs(specs, &mut visit);
            }
            Instr::CallFevalExpandMultiOutputUsingOutputSlot(specs, index) => {
                visit(*index, VmSlotAccess::Other);
                visit_specs(specs, &mut visit);
            }
            Instr::CallBuiltinExpandMultiOutputUsingOutputSlot(_, specs, index) => {
                visit(*index, VmSlotAccess::Other);
                visit_specs(specs, &mut visit);
            }
            Instr::LoadConst(..)
            | Instr::LoadInt(..)
            | Instr::LoadComplex(..)
            | Instr::LoadBool(..)
            | Instr::LoadString(..)
            | Instr::LoadCharRow(..)
            | Instr::Add
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
            | Instr::LogicalOr
            | Instr::AndAnd(..)
            | Instr::OrOr(..)
            | Instr::JumpIfFalse(..)
            | Instr::Jump(..)
            | Instr::Pop
            | Instr::Unpack(..)
            | Instr::CreateMatrix(..)
            | Instr::CreateRange(..)
            | Instr::Index(..)
            | Instr::IndexSlice(..)
            | Instr::CreateCell2D(..)
            | Instr::CreateStructLiteral(..)
            | Instr::CreateObjectLiteral { .. }
            | Instr::IndexCell { .. }
            | Instr::IndexCellExpand { .. }
            | Instr::IndexCellList { .. }
            | Instr::StoreIndex(..)
            | Instr::StoreIndexCell { .. }
            | Instr::StoreIndexDelete(..)
            | Instr::StoreIndexCellDelete { .. }
            | Instr::StoreSlice(..)
            | Instr::StoreSliceDelete(..)
            | Instr::LoadMember(..)
            | Instr::LoadMemberOrInit(..)
            | Instr::LoadMemberDynamic
            | Instr::LoadMemberDynamicOrInit
            | Instr::LoadMemberSequence { .. }
            | Instr::LoadMemberDynamicSequence { .. }
            | Instr::MemberSequenceCardinality
            | Instr::CaptureCallOutputSequence
            | Instr::CaptureScalarSequence
            | Instr::LoadSubscriptEnd { .. }
            | Instr::FinishSubscriptEndReceiver { .. }
            | Instr::FinishDottedInvokeReceiver { .. }
            | Instr::BeginOutputAssignment { .. }
            | Instr::PrepareFixedOutputTarget
            | Instr::PrepareDiscardOutputTarget
            | Instr::PrepareMemberPathStep { .. }
            | Instr::PrepareDynamicMemberPathStep { .. }
            | Instr::BeginPreparedIndexSelectors { .. }
            | Instr::LoadPreparedIndexEnd { .. }
            | Instr::BeginContextualIndexSelectors { .. }
            | Instr::LoadContextualIndexEnd { .. }
            | Instr::FinishContextualIndexSelectors { .. }
            | Instr::PrepareBracesPathStep { .. }
            | Instr::FinishMemberSequenceOutputTarget(..)
            | Instr::FinishDynamicMemberSequenceOutputTarget
            | Instr::FinishCellContentsSequenceOutputTarget { .. }
            | Instr::LoadPreparedOutputCardinality
            | Instr::CommitPreparedOutputTargets { .. }
            | Instr::StoreMemberSequence(..)
            | Instr::StoreMemberDynamicSequence
            | Instr::StoreMember(..)
            | Instr::StoreMemberOrInit(..)
            | Instr::StoreMemberDynamic
            | Instr::StoreMemberDynamicOrInit
            | Instr::LoadMethod(..)
            | Instr::CallMethodOrMemberIndexMulti { .. }
            | Instr::CreateFunctionHandle(..)
            | Instr::CreateExternalFunctionHandle(..)
            | Instr::CreateMethodFunctionHandle(..)
            | Instr::CreateBoundFunctionHandle(..)
            | Instr::CreateExternalBoundFunctionHandle(..)
            | Instr::CreateClosure(..)
            | Instr::CreateSemanticClosure(..)
            | Instr::LoadStaticProperty(..)
            | Instr::LoadWorkspaceFirstStaticProperty { .. }
            | Instr::RegisterClass { .. }
            | Instr::CallFevalMulti(..)
            | Instr::CreateSemanticFuture(..)
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
            | Instr::Collective { .. }
            | Instr::Swap
            | Instr::LeaveTry(..)
            | Instr::Return
            | Instr::ReturnValue
            | Instr::CallBuiltinMulti(..)
            | Instr::CallSuperConstructorMulti { .. }
            | Instr::CallSuperMethodMulti { .. }
            | Instr::CallFunctionMulti { .. }
            | Instr::CallWorkspaceFirstMulti { .. }
            | Instr::CallSemanticFunctionMulti(..)
            | Instr::PackToRow(..)
            | Instr::PackToCol(..)
            | Instr::EnterScope(..)
            | Instr::ExitScope(..)
            | Instr::RegisterImport { .. } => {}
        }
    }
}

fn visit_specs(specs: &[ArgumentSpec], visit: &mut impl FnMut(usize, VmSlotAccess)) {
    for spec in specs {
        match spec {
            ArgumentSpec::CapturedSequence { slot } => visit(*slot, VmSlotAccess::Other),
            ArgumentSpec::Single => {}
            ArgumentSpec::Expansion(expansion) => match expansion {
                ArgumentExpansionSpec::CellContents { .. }
                | ArgumentExpansionSpec::ReturnedOutputs
                | ArgumentExpansionSpec::Member(_)
                | ArgumentExpansionSpec::DynamicMember => {}
            },
        }
    }
}

fn visit_label(label: &EmitLabel, visit: &mut impl FnMut(usize, VmSlotAccess)) {
    match label {
        EmitLabel::Var(slot) => visit(*slot, VmSlotAccess::Other),
        EmitLabel::Ans | EmitLabel::InternalCapture => {}
    }
}

fn visit_steps(steps: &[BytecodeSubscriptStep], visit: &mut impl FnMut(usize, VmSlotAccess)) {
    for step in steps {
        match step {
            BytecodeSubscriptStep::DottedInvoke { arguments, .. } => {
                for argument in arguments {
                    match argument {
                        BytecodeDottedInvokeArgument::Call(spec) => {
                            visit_specs(std::slice::from_ref(spec), visit);
                        }
                        BytecodeDottedInvokeArgument::Colon => {}
                    }
                }
            }
            BytecodeSubscriptStep::Parentheses { selectors }
            | BytecodeSubscriptStep::Braces { selectors } => {
                for selector in selectors {
                    match selector {
                        BytecodeSubscriptSelector::Value | BytecodeSubscriptSelector::Colon => {}
                        BytecodeSubscriptSelector::Expansion(spec) => {
                            visit_specs(std::slice::from_ref(spec), visit);
                        }
                    }
                }
            }
            BytecodeSubscriptStep::Member(_) | BytecodeSubscriptStep::DynamicMember => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_roles_in_one_instruction_are_two_slot_accesses() {
        let instruction = Instr::EmitVar {
            var_index: 7,
            label: EmitLabel::Var(7),
        };
        let mut accesses = Vec::new();
        instruction.visit_vm_slot_accesses(|slot, access| accesses.push((slot, access)));
        assert_eq!(
            accesses,
            [(7, VmSlotAccess::Other), (7, VmSlotAccess::Other)]
        );
    }
}
