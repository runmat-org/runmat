use super::*;
use crate::{catalog::inference::infer_catalog_call, BuiltinPortability};
use runmat_types::*;

fn request(arguments: Vec<ValueFact>, count: usize) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(count)),
    }
}

fn infer(request: CallRequest, entry: &crate::BuiltinCatalogEntry) -> CallInference {
    infer_catalog_call(entry, &request)
}

#[test]
fn all_three_native_mex_identities_have_one_default_binding() {
    let names: std::collections::BTreeSet<_> =
        ENTRIES.iter().map(|entry| entry.identity.name).collect();
    assert_eq!(names, ["mex", "mexcuda", "mexext"].into_iter().collect());
    for entry in ENTRIES {
        assert_eq!(entry.placement.portability, BuiltinPortability::NativeOnly);
        assert_eq!(entry.bindings.len(), 1);
        assert!(entry
            .contract
            .effects
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        assert!(entry
            .contract
            .capabilities
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
    }
}

#[test]
fn build_inference_retains_accepted_scalar_coercions_and_zero_outputs() {
    for descriptor in [&MEX_DESCRIPTOR, &MEXCUDA_DESCRIPTOR] {
        assert_eq!(
            descriptor.signatures[0].inputs[0].ty,
            crate::BuiltinParamType::Any
        );
    }
    let numeric = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    for entry in [&MEX_CATALOG_ENTRY, &MEXCUDA_CATALOG_ENTRY] {
        let result = infer(
            request(
                vec![numeric.clone(), ValueFact::scalar(ValueKindFact::Logical)],
                0,
            ),
            entry,
        );
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert!(result.outputs.is_empty());
        assert!(result
            .capabilities
            .0
            .contains(&CapabilityRequirement::NativeCode));
        let missing = infer(request(vec![], 0), entry);
        assert!(missing
            .diagnostics
            .iter()
            .any(|item| item.code == "RM-MEX-SOURCE"));
        let output = infer(
            request(vec![ValueFact::scalar(ValueKindFact::String)], 1),
            entry,
        );
        assert!(!output.diagnostics.is_empty());
    }
}

#[test]
fn build_inference_rejects_only_definitely_unconvertible_facts() {
    let cell = ValueFact::unknown(DynamicReason::RuntimeValue);
    let unknown = infer(request(vec![cell], 0), &MEX_CATALOG_ENTRY);
    assert!(unknown.diagnostics.is_empty(), "{:?}", unknown.diagnostics);
    let matrix = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Dense,
    );
    let rejected = infer(request(vec![matrix], 0), &MEX_CATALOG_ENTRY);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-MEX-ARGUMENT"));
    let scalar_integer_array = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    let rejected = infer(request(vec![scalar_integer_array], 0), &MEX_CATALOG_ENTRY);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-MEX-ARGUMENT"));
    let scalar_integer = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt64,
        domain: NumericDomain::Real,
    }));
    let accepted = infer(request(vec![scalar_integer], 0), &MEX_CATALOG_ENTRY);
    assert!(
        accepted.diagnostics.is_empty(),
        "{:?}",
        accepted.diagnostics
    );
    let logical_array = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    let rejected = infer(request(vec![logical_array], 0), &MEX_CATALOG_ENTRY);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-MEX-ARGUMENT"));
    let char_3d = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), Some(4), Some(1)]),
        StorageFact::Dense,
    );
    let rejected = infer(request(vec![char_3d], 0), &MEX_CATALOG_ENTRY);
    assert!(rejected
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-MEX-ARGUMENT"));
}

#[test]
fn mexext_is_a_one_string_result_without_compiler_capability() {
    let extension = infer(request(vec![], 1), &MEXEXT_CATALOG_ENTRY);
    assert!(
        extension.diagnostics.is_empty(),
        "{:?}",
        extension.diagnostics
    );
    assert_eq!(extension.outputs.len(), 1);
    assert_eq!(extension.outputs[0].kind, ValueKindFact::String);
    assert!(!extension
        .capabilities
        .0
        .contains(&CapabilityRequirement::NativeCode));
    let extra = infer(
        request(vec![ValueFact::scalar(ValueKindFact::String)], 1),
        &MEXEXT_CATALOG_ENTRY,
    );
    assert!(extra
        .diagnostics
        .iter()
        .any(|item| item.code == "RM-MEXEXT-ARITY"));
    let too_many = infer(request(vec![], 2), &MEXEXT_CATALOG_ENTRY);
    assert!(!too_many.diagnostics.is_empty());
}
