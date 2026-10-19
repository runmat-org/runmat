use crate::{
    MirDiagnostic, MirDottedInvokeArgument, MirExpressionStep, MirLocalId, MirSubscriptChain,
};
use runmat_hir::Span;

use super::{
    diagnose_call_argument_reads, diagnose_expansion_source_reads, diagnose_indexing_reads,
    diagnose_operand_read, diagnose_rvalue_reads, transfer_region_workspace_lookup, InitFact,
};

pub(super) fn diagnose_subscript_chain_reads(
    chain: &MirSubscriptChain,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    diagnose_operand_read(&chain.root, state, span, diagnostics);
    for step in &chain.steps {
        match step {
            crate::MirSubscriptStep::Index(indexing) => {
                diagnose_indexing_reads(indexing, state, current_bindings, span, diagnostics)
            }
            crate::MirSubscriptStep::DottedInvoke { arguments, .. } => {
                for argument in arguments {
                    diagnose_dotted_argument_reads(
                        argument,
                        state,
                        current_bindings,
                        span,
                        diagnostics,
                    );
                }
            }
            crate::MirSubscriptStep::DynamicMember(member) => {
                diagnose_operand_read(member, state, span, diagnostics)
            }
            crate::MirSubscriptStep::PreparedDynamicMember(region) => {
                let mut nested_state = state.to_vec();
                diagnose_expression_steps(
                    region.steps(),
                    &mut nested_state,
                    current_bindings,
                    diagnostics,
                );
                diagnose_operand_read(region.result(), &nested_state, span, diagnostics);
                transfer_region_workspace_lookup(state, &nested_state, current_bindings);
            }
            crate::MirSubscriptStep::Member(_) => {}
        }
    }
}

fn diagnose_dotted_argument_reads(
    argument: &MirDottedInvokeArgument,
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    span: Span,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    match argument {
        MirDottedInvokeArgument::Colon => {}
        MirDottedInvokeArgument::Contextual(region) => {
            let mut nested_state = state.to_vec();
            diagnose_expression_steps(
                region.steps(),
                &mut nested_state,
                current_bindings,
                diagnostics,
            );
            diagnose_operand_read(region.result(), &nested_state, span, diagnostics);
            transfer_region_workspace_lookup(state, &nested_state, current_bindings);
        }
        MirDottedInvokeArgument::Call(region) => {
            let mut nested_state = state.to_vec();
            diagnose_expression_steps(
                &region.steps,
                &mut nested_state,
                current_bindings,
                diagnostics,
            );
            diagnose_call_argument_reads(
                &region.argument,
                &mut nested_state,
                current_bindings,
                region.span,
                diagnostics,
            );
            transfer_region_workspace_lookup(state, &nested_state, current_bindings);
        }
    }
}

pub(super) fn diagnose_expression_steps(
    steps: &[MirExpressionStep],
    state: &mut [InitFact],
    current_bindings: &[MirLocalId],
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for step in steps {
        match step {
            MirExpressionStep::Let { local, value, span } => {
                diagnose_rvalue_reads(value, state, current_bindings, *span, diagnostics);
                state[local.0] = InitFact::DefinitelyAssigned;
            }
            MirExpressionStep::CaptureSequence { source, span, .. } => {
                diagnose_expansion_source_reads(
                    source,
                    state,
                    current_bindings,
                    *span,
                    diagnostics,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MirCallArg, MirCallArgumentRegion, MirConstant, MirExpansionSource, MirLocalId};

    #[test]
    fn expansion_subscript_chain_respects_argument_region_order() {
        let span = Span { start: 3, end: 9 };
        let chain = MirSubscriptChain {
            root: crate::MirOperand::Local(MirLocalId(0)),
            steps: vec![crate::MirSubscriptStep::DottedInvoke {
                member: runmat_types::MemberName("method".into()),
                arguments: vec![MirDottedInvokeArgument::Call(MirCallArgumentRegion {
                    steps: vec![MirExpressionStep::Let {
                        local: MirLocalId(1),
                        value: crate::MirRvalue::Use(crate::MirOperand::Constant(
                            MirConstant::Number("42".into()),
                        )),
                        span,
                    }],
                    argument: MirCallArg::Single(crate::MirOperand::Local(MirLocalId(1))),
                    span,
                })],
                semantic_fallback: None,
            }],
            sequence_use: runmat_types::SequenceUse::RequireSingle,
            context: runmat_types::ObjectIndexingContext::Expression,
        };
        let source = MirExpansionSource::SubscriptChain(chain);
        let mut state = [InitFact::DefinitelyAssigned, InitFact::Unassigned];
        let mut diagnostics = Vec::new();

        super::super::diagnose_expansion_source_reads(
            &source,
            &mut state,
            &[],
            span,
            &mut diagnostics,
        );

        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        assert_eq!(state[1], InitFact::Unassigned);
    }
}
