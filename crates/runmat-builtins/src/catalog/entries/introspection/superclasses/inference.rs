use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-INTROSPECTION-ARITY",
            "expected 1 inputs",
            0,
        ));
    }

    let mut element = ValueFact::proven(
        ValueKindFact::Character,
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    element
        .invalidation
        .0
        .insert(InvalidationCause::ClassHierarchyChanged);
    let mut output = ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(element),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    output
        .invalidation
        .0
        .insert(InvalidationCause::ClassHierarchyChanged);
    if let Some(input) = request.arguments.first() {
        if input.residency == ResidencyFact::Host
            && !matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::String
                    | ValueKindFact::Character
                    | ValueKindFact::Object(_)
                    | ValueKindFact::ClassReference(_)
                    | ValueKindFact::Exception(_)
                    | ValueKindFact::Execution(_)
            )
        {
            diagnostics.push(argument_error(
                "RM-SUPERCLASSES-INPUT",
                "expected class-name text or an object",
                0,
            ));
        }
    }
    let mut contract = CallContract::fixed(vec![output]);

    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
