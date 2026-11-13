use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, ObjectFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};
use std::collections::BTreeMap;

fn object(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            // The current runtime stores these as ObjectInstance values. A
            // class registration alone does not establish handle semantics.
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn is_equal_to(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.len() == 1 {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-IS-EQUAL-TO-ARITY",
            "IsEqualTo currently requires exactly one expected value",
            0,
        )]
    };
    let expected = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue));
    let properties = BTreeMap::from([("Expected".into(), expected)]);
    finish_fixed(
        entry,
        request,
        object(runmat_types::standard::UNIT_TEST_IS_EQUAL_TO, properties),
        diagnostics,
    )
}

fn boolean_constraint(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-BOOLEAN-CONSTRAINT-ARITY",
            "IsTrue and IsFalse accept no inputs",
            0,
        )]
    };
    finish_fixed(entry, request, object(class, BTreeMap::new()), diagnostics)
}

pub(super) fn is_true(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    boolean_constraint(request, entry, runmat_types::standard::UNIT_TEST_IS_TRUE)
}

pub(super) fn is_false(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    boolean_constraint(request, entry, runmat_types::standard::UNIT_TEST_IS_FALSE)
}
