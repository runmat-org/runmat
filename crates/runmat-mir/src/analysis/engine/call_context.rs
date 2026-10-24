//! Additional diagnostics from concrete call inputs. These facts are deliberately
//! not published into AnalysisStore, executable contracts, or generic summaries.
use std::collections::BTreeSet;

use runmat_hir::{DiagnosticExecution, FunctionId, HirDiagnosticSeverity, HirDiagnosticSpan};
use runmat_types::{ProgramFunctionId, ProgramSourceId, ValueFact};

use super::{call_graph, flow, interprocedural};
use crate::{MirAssembly, MirDiagnostic, MirLocalKind};

// This is a precision budget, not a new executable analysis limit. Exhausting
// it withholds contextual diagnostics and leaves all generic facts untouched.
const MAX_CONTEXTS: usize = interprocedural::MAX_SUMMARY_UPDATES;

struct ContextFacts {
    function: FunctionId,
    arguments: Vec<ValueFact>,
    diagnostics: Vec<MirDiagnostic>,
    calls: Vec<flow::CallObservation>,
}

enum Observation<'a> {
    Root(&'a flow::CallObservation),
    Cached { context: usize, call: usize },
}

enum Resolution {
    Existing(usize),
    New(ContextFacts),
    BudgetExhausted,
    PrecisionWithheld,
}

/// Diagnostic-only precision accounting, never executable specialization.
#[derive(Debug)]
pub struct CallContextDiagnostics {
    pub diagnostics: Vec<MirDiagnostic>,
    pub budget_exhausted: bool,
    pub precision_withheld: bool,
}

/// Check statically bound calls using the same body transfer and typed operation
/// contracts as ordinary MIR analysis. Public functions remain polymorphic.
/// Unknown dispatch, captures, argument conversion/expansion and recursive
/// callees are conservative boundaries, never guessed concrete inputs.
pub fn diagnose_call_contexts(assembly: &MirAssembly) -> CallContextDiagnostics {
    let generic = interprocedural::analyze_program(assembly);
    let graph = call_graph::call_graph(assembly);
    let recursive = call_graph::strongly_connected_components(&graph)
        .into_iter()
        .filter(|component| {
            component.len() > 1
                || component
                    .first()
                    .is_some_and(|function| graph[function].contains(function))
        })
        .flatten()
        .collect();
    let mut checker = Checker {
        assembly,
        generic: &generic,
        recursive,
        contexts: Vec::new(),
        diagnostics: Vec::new(),
        budget_exhausted: false,
        precision_withheld: false,
    };
    for (function, body) in &assembly.bodies {
        let Ok(ordinal) = u32::try_from(function.0) else {
            continue;
        };
        let flow = flow::analyze_body(
            body,
            ProgramFunctionId(ordinal),
            generic.parameters.get(function).map_or(&[], Vec::as_slice),
            generic.captures.get(function).map_or(&[], Vec::as_slice),
            super::super::constructor_entry::instance_output_local(assembly, body),
            &generic.summaries,
        );
        // A path/cwd mutation can replace the runtime callable at a later
        // lookup. Without a per-edge binding proof, withhold contextual calls
        // from this entire body rather than guess which branch executed.
        if !flow.converged || has_environment_mutation(&flow) {
            checker.precision_withheld = true;
            continue;
        }
        let Some(metadata) = assembly.functions.get(function) else {
            continue;
        };
        for call in &flow.calls {
            let site = HirDiagnosticSpan {
                span: call.span,
                label: Some("these call inputs expose an incompatible operation".into()),
                source: Some(metadata.source),
            };
            checker.check(call, &site);
        }
    }
    checker.diagnostics.sort_by_key(|diagnostic| {
        (
            diagnostic.primary.source,
            diagnostic.primary.span.start,
            diagnostic.primary.span.end,
            diagnostic.code.clone(),
        )
    });
    checker.diagnostics.dedup();
    CallContextDiagnostics {
        diagnostics: checker.diagnostics,
        budget_exhausted: checker.budget_exhausted,
        precision_withheld: checker.precision_withheld,
    }
}

struct Checker<'a> {
    assembly: &'a MirAssembly,
    generic: &'a interprocedural::ProgramAnalysis,
    recursive: BTreeSet<FunctionId>,
    contexts: Vec<ContextFacts>,
    diagnostics: Vec<MirDiagnostic>,
    budget_exhausted: bool,
    precision_withheld: bool,
}

impl Checker<'_> {
    fn check(&mut self, call: &flow::CallObservation, site: &HirDiagnosticSpan) {
        let Some(first) = self.context(Observation::Root(call)) else {
            return;
        };
        let mut pending = vec![first];
        let mut visited = BTreeSet::new();
        while let Some(index) = pending.pop() {
            if !visited.insert(index) {
                continue;
            }
            let context = &self.contexts[index];
            let source = self.assembly.functions[&context.function].source;
            for diagnostic in &context.diagnostics {
                self.diagnostics
                    .push(at_call_site(diagnostic, source, site));
            }
            let calls = context.calls.len();
            for call in 0..calls {
                if let Some(next) = self.context(Observation::Cached {
                    context: index,
                    call,
                }) {
                    if !visited.contains(&next) && !pending.contains(&next) {
                        pending.push(next);
                    }
                }
            }
        }
    }

    fn context(&mut self, observation: Observation<'_>) -> Option<usize> {
        // Borrow the call in its original flow or cached context. Finish all
        // immutable analysis before extending the cache; no call/facts vector
        // is cloned merely to recurse into another diagnostic context.
        let resolution = {
            let call = match observation {
                Observation::Root(call) => call,
                Observation::Cached { context, call } => &self.contexts[context].calls[call],
            };
            self.resolve(call)
        };
        match resolution {
            Resolution::Existing(index) => Some(index),
            Resolution::New(context) => {
                let index = self.contexts.len();
                self.contexts.push(context);
                Some(index)
            }
            Resolution::BudgetExhausted => {
                self.budget_exhausted = true;
                self.precision_withheld = true;
                None
            }
            Resolution::PrecisionWithheld => {
                self.precision_withheld = true;
                None
            }
        }
    }

    fn resolve(&self, call: &flow::CallObservation) -> Resolution {
        if !call.complete_arguments || self.recursive.contains(&call.callee) {
            return Resolution::PrecisionWithheld;
        }
        let Some(body) = self.assembly.bodies.get(&call.callee) else {
            return Resolution::PrecisionWithheld;
        };
        let Some(metadata) = self.assembly.functions.get(&call.callee) else {
            return Resolution::PrecisionWithheld;
        };
        if call.arguments.iter().any(|fact| {
            matches!(
                fact.kind,
                runmat_types::ValueKindFact::Unknown | runmat_types::ValueKindFact::Never
            ) || fact
                .shape
                .known_dims()
                .is_none_or(|dims| dims.iter().any(Option::is_none))
        }) {
            return Resolution::PrecisionWithheld;
        }
        // Declared arguments can convert inputs; captures need the actual
        // lexical environment. Neither is replaced by a guessed caller fact.
        if !metadata.argument_validations.is_empty()
            || !metadata.captures.is_empty()
            || metadata.enclosing_class.is_some()
            || body
                .locals
                .iter()
                .any(|local| matches!(local.kind, MirLocalKind::Capture))
            || body.abi.varargin.is_some()
            // analyze_body has no contextual argument literals or requested
            // output count. Unknown branch guards (including ABI arity reads)
            // must not make a conditional failure a concrete caller error.
            || body.blocks.iter().any(|block| {
                !matches!(
                    block.terminator.kind,
                    crate::MirTerminatorKind::Goto(_)
                        | crate::MirTerminatorKind::Return(_)
                        | crate::MirTerminatorKind::Unreachable
                )
            })
            || body
                .locals
                .iter()
                .filter(|local| matches!(local.kind, MirLocalKind::Parameter))
                .count()
                != call.arguments.len()
        {
            return Resolution::PrecisionWithheld;
        }
        if let Some(index) = self.contexts.iter().position(|context| {
            context.function == call.callee && context.arguments == call.arguments
        }) {
            return Resolution::Existing(index);
        }
        if self.contexts.len() >= MAX_CONTEXTS {
            return Resolution::BudgetExhausted;
        }
        let Ok(ordinal) = u32::try_from(call.callee.0) else {
            return Resolution::PrecisionWithheld;
        };
        let result = flow::analyze_body(
            body,
            ProgramFunctionId(ordinal),
            &call.arguments,
            &[],
            None,
            &self.generic.summaries,
        );
        if !result.converged || has_environment_mutation(&result) {
            return Resolution::PrecisionWithheld;
        }
        Resolution::New(ContextFacts {
            function: call.callee,
            arguments: call.arguments.clone(),
            diagnostics: result
                .diagnostics
                .into_iter()
                .filter(|diagnostic| {
                    diagnostic.severity == HirDiagnosticSeverity::Error
                        && diagnostic.execution == DiagnosticExecution::RuntimeCheck
                })
                .collect(),
            calls: result.calls,
        })
    }
}

fn has_environment_mutation(flow: &flow::BodyFlow) -> bool {
    flow.operation_contracts.iter().any(|contract| {
        contract.site.construct == crate::MirConstructKind::EnvironmentEffect
            || contract
                .effects
                .0
                .contains(&runmat_types::EffectKind::EnvironmentWrite)
    })
}

fn at_call_site(
    diagnostic: &MirDiagnostic,
    source: ProgramSourceId,
    site: &HirDiagnosticSpan,
) -> MirDiagnostic {
    let mut diagnostic = diagnostic.clone().with_source(source);
    let mut operation = diagnostic.primary.clone();
    operation.label = Some("incompatible operation in the called function".into());
    diagnostic.secondary.insert(0, operation);
    diagnostic.primary = site.clone();
    // A callee-local suggestion cannot safely be applied at the caller.
    diagnostic.suggestions.clear();
    diagnostic
}

#[cfg(test)]
#[path = "call_context/tests.rs"]
mod tests;
