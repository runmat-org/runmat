//! The assignment is installed dynamically by the execution service.

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-CATALOG-CURRENT-EXECUTION-ARITY",
            format!("{} does not accept input arguments", entry.identity.name),
            0,
        )]
    };
    // The same call produces a specific projection object inside an assigned
    // invocation and [] outside it. CallRequest has no execution-scope fact,
    // so neither the object class nor an empty shape can be proven here.
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        diagnostics,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::parallel::current_execution::{
        GET_CURRENT_JOB_CATALOG_ENTRY, GET_CURRENT_TASK_CATALOG_ENTRY,
        GET_CURRENT_WORKER_CATALOG_ENTRY,
    };
    use runmat_types::{
        CapabilityRequirement, EffectKind, LiteralContext, OutputSelection, RequestedOutputCount,
        ValueKindFact,
    };

    fn request(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
        }
    }

    #[test]
    fn assignment_dependent_results_do_not_claim_an_object_or_empty_matrix() {
        for entry in [
            &GET_CURRENT_JOB_CATALOG_ENTRY,
            &GET_CURRENT_TASK_CATALOG_ENTRY,
            &GET_CURRENT_WORKER_CATALOG_ENTRY,
        ] {
            let result = infer(&request(Vec::new(), 1), entry);
            assert_eq!(result.outputs.len(), 1, "{}", entry.identity.name);
            assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
            assert!(result.effects.0.contains(&EffectKind::MayThrow));
            assert!(result
                .capabilities
                .0
                .contains(&CapabilityRequirement::ParallelRuntime));
            assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
        }
    }

    #[test]
    fn source_arity_and_destination_count_are_checked_for_each_identity() {
        for entry in [
            &GET_CURRENT_JOB_CATALOG_ENTRY,
            &GET_CURRENT_TASK_CATALOG_ENTRY,
            &GET_CURRENT_WORKER_CATALOG_ENTRY,
        ] {
            assert!(infer(&request(Vec::new(), 0), entry).outputs.is_empty());
            let bad_input = infer(
                &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 1),
                entry,
            );
            assert!(bad_input
                .diagnostics
                .iter()
                .any(|diagnostic| { diagnostic.code == "RM-CATALOG-CURRENT-EXECUTION-ARITY" }));
            assert!(!infer(&request(Vec::new(), 2), entry).diagnostics.is_empty());
        }
    }
}
