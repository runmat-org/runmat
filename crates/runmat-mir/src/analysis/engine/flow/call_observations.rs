use std::collections::BTreeMap;

use runmat_hir::{FunctionId, Span};
use runmat_types::ValueFact;

use crate::analysis::inference::{infer_rvalue, operand_fact, rvalue_literal, FunctionSummary};
use crate::{MirRvalue, MirStmt, MirStmtKind};

use super::{transfer_short_circuit_temps, transfer_statement, FlowState};

#[derive(Debug, Clone)]
pub(crate) struct CallObservation {
    pub callee: FunctionId,
    pub arguments: Vec<ValueFact>,
    pub span: Span,
    pub argument_spans: Vec<Span>,
}

pub(super) fn collect(
    statement: &MirStmt,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    calls: &mut Vec<CallObservation>,
) {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::SequenceAssign { value, .. }
        | MirStmtKind::Expr(value) => collect_value(value, state, statement.span, summaries, calls),
        MirStmtKind::CaptureSequence { source, .. } => {
            collect_expansion(source, state, summaries, calls)
        }
        _ => {}
    }
}

fn collect_expansion(
    source: &crate::MirExpansionSource,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    calls: &mut Vec<CallObservation>,
) {
    source.visit_direct_expression_regions_dyn(&mut |region| {
        collect_region(region, state, summaries, calls)
    });
}

fn collect_value(
    value: &MirRvalue,
    state: &FlowState,
    span: Span,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    calls: &mut Vec<CallObservation>,
) {
    match value {
        MirRvalue::Call(call) => {
            let callee = match &call.callee {
                crate::MirCallee::Static(
                    runmat_hir::CallableIdentity::BoundFunction(function)
                    | runmat_hir::CallableIdentity::AnonymousFunction(function)
                    | runmat_hir::CallableIdentity::ExternalFunction { function, .. },
                ) => Some(*function),
                _ => None,
            };
            if let Some(callee) = callee {
                calls.push(CallObservation {
                    callee,
                    arguments: call
                        .args
                        .iter()
                        .filter_map(|argument| argument.operand())
                        .map(|operand| operand_fact(operand, state))
                        .collect(),
                    span,
                    argument_spans: call.arg_spans.clone(),
                });
            }
        }
        MirRvalue::Future { function, args, .. } => calls.push(CallObservation {
            callee: *function,
            arguments: args
                .iter()
                .filter_map(|argument| argument.operand())
                .map(|operand| operand_fact(operand, state))
                .collect(),
            span,
            argument_spans: Vec::new(),
        }),
        MirRvalue::ShortCircuit { right_temps, .. } => {
            collect_statement_sequence(right_temps, state, summaries, calls)
        }
        _ => {}
    }
    if !matches!(value, MirRvalue::ShortCircuit { .. }) {
        value.visit_direct_expression_regions_dyn(&mut |region| {
            collect_region(region, state, summaries, calls)
        });
    }
}

fn collect_statement_sequence(
    statements: &[MirStmt],
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    calls: &mut Vec<CallObservation>,
) {
    let mut nested = state.clone();
    let mut pending_mutation = None;
    let mut diagnostics = Vec::new();
    for statement in statements {
        collect(statement, &nested, summaries, calls);
        transfer_statement(
            statement,
            &mut nested,
            &mut pending_mutation,
            summaries,
            &mut diagnostics,
        );
    }
}

fn collect_region(
    region: &crate::MirExpressionRegion,
    state: &FlowState,
    summaries: &BTreeMap<FunctionId, FunctionSummary>,
    calls: &mut Vec<CallObservation>,
) {
    let mut nested = state.clone();
    for step in region.steps() {
        match step {
            crate::MirExpressionStep::Let { local, value, span } => {
                collect_value(value, &nested, *span, summaries, calls);
                let mut diagnostics = Vec::new();
                transfer_short_circuit_temps(value, &mut nested, summaries, &mut diagnostics);
                let fact = infer_rvalue(value, &mut nested, summaries, *span, &mut diagnostics);
                let literal = rvalue_literal(value, &nested);
                if let Some(slot) = nested.locals.get_mut(local.0) {
                    slot.set(fact, literal);
                }
            }
            crate::MirExpressionStep::CaptureSequence { source, .. } => {
                collect_expansion(source, &nested, summaries, calls);
            }
        }
    }
}
