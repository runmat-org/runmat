//! Suspension-origin evidence collected under the canonical flow facts. This
//! never removes effects and never supplies runtime callable authority.
use super::{transfer_short_circuit_temps, transfer_statement, FlowState};
use crate::analysis::inference::{infer_rvalue, rvalue_literal, FunctionSummary};
use crate::{MirCallArg, MirCallee, MirOperand, MirRvalue, MirStmt, MirStmtKind};
use runmat_types::{
    CallableFallbackPolicy, CallableIdentity, EffectKind, FunctionId, ValueKindFact,
};
use std::collections::{BTreeMap, BTreeSet};

fn numeric_operand(operand: &MirOperand, state: &FlowState) -> bool {
    match operand {
        MirOperand::Local(local) => state
            .locals
            .get(local.0)
            .and_then(|slot| slot.fact.as_ref())
            .is_some_and(|fact| {
                matches!(
                    fact.kind,
                    ValueKindFact::Numeric(_) | ValueKindFact::Logical
                )
            }),
        MirOperand::Constant(constant) => matches!(
            crate::analysis::dataflow::constant_fact(constant).kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ),
        MirOperand::FunctionHandle(_) => false,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NumericalSuspensionEvidence {
    pub local: bool,
    pub callees: BTreeSet<FunctionId>,
}
impl Default for NumericalSuspensionEvidence {
    fn default() -> Self {
        Self {
            local: true,
            callees: BTreeSet::new(),
        }
    }
}
impl NumericalSuspensionEvidence {
    pub(super) fn statement(
        &mut self,
        statement: &MirStmt,
        state: &FlowState,
        summaries: &BTreeMap<FunctionId, FunctionSummary>,
    ) {
        let mut evaluated = None;
        statement
            .kind
            .visit_statement_expression_steps_dyn(&mut |steps| {
                if !steps.is_empty() {
                    evaluated =
                        Some(self.steps(steps, evaluated.as_ref().unwrap_or(state), summaries));
                }
            });
        let state = evaluated.as_ref().unwrap_or(state);
        if let Some(value) = crate::operation_contract::statement_rvalue(&statement.kind) {
            self.value(value, state, statement.span, summaries);
        }
        if let MirStmtKind::CaptureSequence { source, .. } = &statement.kind {
            source.visit_operands(|operand| {
                self.local &= numeric_operand(operand, state);
            });
        }
    }
    pub(super) fn value(
        &mut self,
        value: &MirRvalue,
        state: &FlowState,
        _span: runmat_types::Span,
        summaries: &BTreeMap<FunctionId, FunctionSummary>,
    ) {
        let mut evaluated = None;
        if !matches!(value, MirRvalue::ShortCircuit { .. }) {
            value.visit_direct_expression_steps_dyn(&mut |steps| {
                if !steps.is_empty() {
                    evaluated =
                        Some(self.steps(steps, evaluated.as_ref().unwrap_or(state), summaries));
                }
            });
        }
        let state = evaluated.as_ref().unwrap_or(state);
        let numeric = |operand: &MirOperand| numeric_operand(operand, state);
        match value {
            MirRvalue::Call(call) => {
                // A canonical builtin name does not prove that an object
                // overload cannot run. Require exact positional numeric facts
                // at this operation, including nested expression preparation.
                let positional_numeric = call.args.iter().all(|arg| {
                    matches!(arg,
                    MirCallArg::Single(operand) if numeric(operand))
                });
                if !positional_numeric
                    || call.fallback_policy != CallableFallbackPolicy::None
                    || call.workspace_first_name.is_some()
                {
                    self.local = false;
                }
                match &call.callee {
                    MirCallee::Static(CallableIdentity::Builtin(id)) => {
                        let valid = runmat_builtins::builtin_catalog_entry_by_name(&id.0)
                            .is_some_and(|entry| {
                                use runmat_builtins::{
                                    BuiltinAsyncBehavior as A, BuiltinContractMaturity as M,
                                    BuiltinPurity as P,
                                };
                                // Incomplete value/shape inference does not invent
                                // suspension proof: only the explicit reviewed
                                // async-kind certifies local numerical suspension.
                                !matches!(entry.contract.maturity, M::LegacyResolver)
                                    && entry.contract.purity == P::Pure
                                    && call.purity == P::Pure
                                    && matches!(
                                        entry.contract.async_behavior,
                                        A::NeverSuspends | A::MaySuspendLocally
                                    )
                                    && entry.contract.capabilities.is_empty()
                                    && entry.contract.effects.iter().all(|effect| {
                                        matches!(
                                            effect,
                                            EffectKind::MayThrow | EffectKind::MaySuspend
                                        )
                                    })
                            });
                        self.local &= valid;
                    }
                    MirCallee::Static(CallableIdentity::BoundFunction(function)) => {
                        self.callees.insert(*function);
                    }
                    _ => self.local = false,
                }
            }
            MirRvalue::Unary(_, operand) => self.local &= numeric(operand),
            MirRvalue::Binary(left, _, right) => self.local &= numeric(left) && numeric(right),
            MirRvalue::Future { .. }
            | MirRvalue::ExternalFuture { .. }
            | MirRvalue::Spawn(_)
            | MirRvalue::Distributed(_)
            | MirRvalue::Collective(_)
            | MirRvalue::Member { .. }
            | MirRvalue::DynamicMember { .. }
            | MirRvalue::WorkspaceFirstStaticProperty { .. }
            | MirRvalue::ObjectLiteral { .. }
            | MirRvalue::SubscriptChain(_) => self.local = false,
            MirRvalue::Index { .. } | MirRvalue::Range { .. } | MirRvalue::Aggregate { .. } => {
                value.visit_operands(|operand| self.local &= numeric(operand));
            }
            MirRvalue::ShortCircuit {
                left,
                right_temps,
                right,
                ..
            } => {
                self.local &= numeric(left);
                if right_temps.is_empty() {
                    self.local &= numeric(right);
                    return;
                }
                let mut nested = state.clone();
                let mut pending = None;
                for statement in right_temps {
                    self.statement(statement, &nested, summaries);
                    transfer_statement(
                        statement,
                        &mut nested,
                        &mut pending,
                        summaries,
                        &mut Vec::new(),
                    );
                }
                self.local &= numeric_operand(right, &nested);
            }
            _ => {}
        }
    }
    fn steps(
        &mut self,
        steps: &[crate::MirExpressionStep],
        state: &FlowState,
        summaries: &BTreeMap<FunctionId, FunctionSummary>,
    ) -> FlowState {
        let mut nested = state.clone();
        for step in steps {
            match step {
                crate::MirExpressionStep::Let { local, value, span } => {
                    self.value(value, &nested, *span, summaries);
                    let mut diagnostics = Vec::new();
                    transfer_short_circuit_temps(value, &mut nested, summaries, &mut diagnostics);
                    let fact = infer_rvalue(value, &mut nested, summaries, *span, &mut diagnostics);
                    let literal = rvalue_literal(value, &nested);
                    if let Some(slot) = nested.locals.get_mut(local.0) {
                        slot.set(fact, literal);
                    }
                }
                crate::MirExpressionStep::CaptureSequence { source, .. } => {
                    source.visit_direct_expression_steps_dyn(&mut |steps| {
                        if !steps.is_empty() {
                            nested = self.steps(steps, &nested, summaries);
                        }
                    });
                    source.visit_operands(|operand| {
                        self.local &= numeric_operand(operand, &nested);
                    });
                }
            }
        }
        nested
    }
}
