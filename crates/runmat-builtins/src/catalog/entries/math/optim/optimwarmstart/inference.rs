use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn object(x: ValueFact, options: ValueFact) -> ValueFact {
    let mut properties = std::collections::BTreeMap::new();
    properties.insert("X".into(), x);
    properties.insert("Options".into(), options);
    for name in ["MaxLinearEqualities", "MaxLinearInequalities"] {
        properties.insert(
            name.into(),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::Scalar,
                StorageFact::Scalar,
            ),
        );
    }
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            // Solver values are runtime data, not literal facts of the option
            // field names. Both public classes have these property facts.
            runtime_class: None,
            properties,
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}
pub(super) fn constructor(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !matches!(request.arguments.len(), 2 | 6) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIMWARMSTART-ARITY",
            "expected x0,options and optionally both memory-limit pairs",
            0,
        ));
    }
    let shape = request
        .arguments
        .first()
        .and_then(|x| x.shape.element_count());
    let x = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![shape, Some(1)]),
        StorageFact::Dense,
    );
    let options = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    finish_fixed(entry, request, object(x, options), diagnostics)
}
pub(super) fn read(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        Vec::new(),
    )
}
pub(super) fn write(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    finish_fixed(
        entry,
        request,
        ValueFact::unknown(DynamicReason::RuntimeValue),
        Vec::new(),
    )
}
