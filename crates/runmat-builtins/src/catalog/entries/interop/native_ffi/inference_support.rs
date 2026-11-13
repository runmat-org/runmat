use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer_with(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    outputs: Vec<ValueFact>,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_dynamic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::RuntimeValue);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn owned_pointer() -> ValueFact {
    let mut fact = ValueFact::proven(
        ValueKindFact::Foreign(ForeignFact {
            family: "native-ffi".into(),
            type_name: None,
            type_version: None,
            ownership: ForeignOwnershipFact::Owned,
            affinity: ForeignAffinityFact::OriginThread,
            lifetime: ForeignLifetimeFact::Session,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
}

pub(super) fn string_cell_row() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Cell(CellFact {
            element: Box::new(ValueFact::scalar(ValueKindFact::String)),
            elements: Vec::new(),
            elements_complete: false,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Opaque,
    )
}

pub(super) fn text_at(
    request: &CallRequest,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if let Some(fact) = request.arguments.get(index) {
        // The retained Value-to-String ABI also accepts primitive numeric and
        // logical scalars. A one-element Numeric/Logical fact alone does not
        // prove whether the runtime representation is primitive or an array.
        let definitely_invalid = match &fact.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::String | ValueKindFact::Logical => fact.shape.is_definitely_nonscalar(),
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Complex || fact.shape.is_definitely_nonscalar()
            }
            ValueKindFact::Character => {
                fact.shape.rank().is_some_and(|rank| rank > 2)
                    || fact.shape.known_dims().is_some_and(|dims| {
                        dims.first()
                            .copied()
                            .flatten()
                            .is_some_and(|rows| rows != 1)
                    })
            }
            _ => true,
        };
        if definitely_invalid {
            diagnostics.push(argument_error(
                "RM-NATIVE-FFI-TEXT",
                "native library path and alias arguments must be text",
                index,
            ));
        }
    }
}

pub(super) fn arity(
    request: &CallRequest,
    minimum: usize,
    maximum: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let count = request.arguments.len();
    if count < minimum || maximum.is_some_and(|maximum| count > maximum) {
        diagnostics.push(argument_error(
            "RM-NATIVE-FFI-ARITY",
            "native library call has an invalid number of arguments",
            count,
        ));
    }
}
