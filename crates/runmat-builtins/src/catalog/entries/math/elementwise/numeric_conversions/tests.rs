use runmat_types::{
    CallRequest, DynamicReason, LiteralContext, NumericClass, NumericDomain, NumericFact,
    OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

use super::ENTRIES;
use crate::{infer_catalog_call, BuiltinInferenceRule, MathInferenceRule};

#[test]
fn every_integer_conversion_has_canonical_documentation_and_typed_inference() {
    let expected = [
        ("int8", NumericClass::Int8),
        ("int16", NumericClass::Int16),
        ("int32", NumericClass::Int32),
        ("int64", NumericClass::Int64),
        ("uint8", NumericClass::UInt8),
        ("uint16", NumericClass::UInt16),
        ("uint32", NumericClass::UInt32),
        ("uint64", NumericClass::UInt64),
    ];

    for (entry, (name, class)) in ENTRIES.iter().zip(expected) {
        assert_eq!(entry.identity.name, name);
        assert_eq!(entry.documentation.title, Some(name));
        assert_eq!(entry.documentation.examples.len(), 3);
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(class))
        );
    }
}

fn request(input: ValueFact) -> CallRequest {
    CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    }
}

#[test]
fn integer_conversion_facts_match_scalar_and_sparse_runtime_representations() {
    for entry in ENTRIES {
        for (domain, storage, expected) in [
            (NumericDomain::Real, StorageFact::Dense, StorageFact::Scalar),
            (
                NumericDomain::Complex,
                StorageFact::Dense,
                StorageFact::Dense,
            ),
            (
                NumericDomain::Real,
                StorageFact::Sparse,
                StorageFact::Sparse,
            ),
        ] {
            let input = ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain,
                }),
                ShapeFact::from(vec![Some(1), Some(1)]),
                storage,
            );
            let result = infer_catalog_call(entry, &request(input.clone()));
            assert!(result.diagnostics.is_empty(), "{}", entry.identity.name);
            assert_eq!(
                result.outputs[0].storage, expected,
                "{}",
                entry.identity.name
            );
            assert_eq!(result.outputs[0].shape, input.shape);
            assert_eq!(result.outputs[0].numeric().unwrap().domain, domain);
        }
    }
}

#[test]
fn integer_conversion_rejects_only_proven_complex_sparse_inputs() {
    for entry in ENTRIES {
        let mut input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Sparse,
        );
        let rejected = infer_catalog_call(entry, &request(input.clone()));
        assert_eq!(rejected.diagnostics.len(), 1, "{}", entry.identity.name);
        assert_eq!(rejected.outputs[0].kind, ValueKindFact::Unknown);
        input.kind = ValueKindFact::Unknown;
        let unresolved = infer_catalog_call(entry, &request(input));
        assert!(unresolved.diagnostics.is_empty(), "{}", entry.identity.name);
        assert_eq!(unresolved.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(unresolved.outputs[0].storage, StorageFact::Sparse);
    }
}

#[test]
fn integer_conversion_unknown_domain_keeps_shape_and_placement_without_real_claim() {
    for entry in ENTRIES {
        let mut input = ValueFact::unknown(DynamicReason::ExternalData);
        input.shape = ShapeFact::from(vec![Some(2), Some(3)]);
        input.storage = StorageFact::Dense;
        input.residency = ResidencyFact::Host;
        let result = infer_catalog_call(entry, &request(input.clone()));
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].kind, ValueKindFact::Unknown);
        assert_eq!(result.outputs[0].shape, input.shape);
        assert_eq!(result.outputs[0].residency, input.residency);
    }
}
