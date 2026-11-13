//! Session-pool facts are owned by the pool identities, not the parallel-data router.

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, ExecutionFact, ValueFact, ValueKindFact,
};

fn pool() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Pool))
}

pub(super) fn infer_parpool(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() > 2 {
        vec![argument_error(
            "RM-CATALOG-PARPOOL-ARITY",
            "parpool accepts at most a pool kind and worker count",
            2,
        )]
    } else {
        Vec::new()
    };
    finish_fixed(entry, request, pool(), diagnostics)
}

pub(super) fn infer_gcp(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() > 1 {
        vec![argument_error(
            "RM-CATALOG-GCP-ARITY",
            "gcp accepts no arguments or the option 'nocreate'",
            1,
        )]
    } else if request.arguments.len() == 1
        && request
            .literals
            .literal_string_at(0)
            .is_some_and(|option| option.trim() != "nocreate")
    {
        vec![argument_error(
            "RM-CATALOG-GCP-OPTION",
            "gcp accepts only the option 'nocreate'",
            0,
        )]
    } else {
        Vec::new()
    };
    // With an argument, the only valid result depends on session state: a pool
    // handle or an empty double matrix. No single exact fact represents both.
    let output = if request.arguments.is_empty() {
        pool()
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::entries::parallel::pools::{GCP_CATALOG_ENTRY, PARPOOL_CATALOG_ENTRY};
    use runmat_types::{
        EffectKind, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
    };

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    #[test]
    fn pool_creation_has_an_exact_handle_and_inspection_stays_dynamic() {
        let created = infer_parpool(&request(Vec::new(), Vec::new()), &PARPOOL_CATALOG_ENTRY);
        assert_eq!(
            created.outputs[0].kind,
            ValueKindFact::Execution(ExecutionFact::Pool)
        );
        let automatic = infer_gcp(&request(Vec::new(), Vec::new()), &GCP_CATALOG_ENTRY);
        assert_eq!(
            automatic.outputs[0].kind,
            ValueKindFact::Execution(ExecutionFact::Pool)
        );
        assert!(automatic.effects.0.contains(&EffectKind::EnvironmentWrite));
        let inspected = infer_gcp(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![LiteralValue::String("nocreate".into())],
            ),
            &GCP_CATALOG_ENTRY,
        );
        assert_eq!(inspected.outputs[0].kind, ValueKindFact::Unknown);
        assert!(inspected.diagnostics.is_empty());
    }

    #[test]
    fn known_invalid_pool_forms_are_diagnosed() {
        let unknown = ValueFact::unknown(DynamicReason::RuntimeValue);
        assert!(!infer_parpool(
            &request(
                vec![unknown.clone(), unknown.clone(), unknown.clone()],
                Vec::new()
            ),
            &PARPOOL_CATALOG_ENTRY,
        )
        .diagnostics
        .is_empty());
        assert!(!infer_gcp(
            &request(vec![unknown], vec![LiteralValue::String("create".into())]),
            &GCP_CATALOG_ENTRY,
        )
        .diagnostics
        .is_empty());
    }
}
