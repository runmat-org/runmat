use super::*;
use runmat_types::*;

fn call(arguments: Vec<ValueFact>, outputs: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(outputs)),
    }
}

#[test]
fn all_predicates_bind_exactly_once_and_document_asserted_examples() {
    for entry in ENTRIES {
        assert_eq!(
            entry.bindings, &REQUIRED_DEFAULT_BINDING,
            "{}",
            entry.identity.name
        );
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert!(entry.documentation.examples.iter().all(|example| matches!(
            example.verification,
            BuiltinExampleVerification::Assertions { .. }
        )));
        assert_eq!(entry.placement.residency, BuiltinResidencyPolicy::Host);
        assert_eq!(
            entry.placement.distributed,
            BuiltinDistributedPolicy::InspectHandles
        );
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::NeverSuspends
        );
    }
}

#[test]
fn predicates_infer_one_host_logical_scalar_for_known_and_unknown_input() {
    for entry in ENTRIES {
        for input in [
            ValueFact::unknown(DynamicReason::RuntimeValue),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
        ] {
            let result = infer_catalog_call(entry, &call(vec![input], 1));
            assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
            assert_eq!(result.outputs[0], ValueFact::scalar(ValueKindFact::Logical));
        }
        assert!(!infer_catalog_call(entry, &call(vec![], 1))
            .diagnostics
            .is_empty());
        assert!(!infer_catalog_call(
            entry,
            &call(vec![ValueFact::unknown(DynamicReason::RuntimeValue)], 2)
        )
        .diagnostics
        .is_empty());
    }
}
