use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::{
    standard, CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-MACD-ARITY",
            "macd expects one price-data argument",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-MACD-OUTPUTS",
            "macd supports at most two outputs",
            0,
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    if let Some(data) = request.arguments.first() {
        if matches!(data.residency, ResidencyFact::Remote { .. }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-MACD-PLACEMENT",
                "distributed price data is not supported",
                0,
            ));
        }
        match &data.kind {
            ValueKindFact::Numeric(numeric) => {
                if numeric.domain == NumericDomain::Complex || data.storage == StorageFact::Sparse {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-MACD-CLASS",
                        "raw price data must be a dense real numeric matrix",
                        0,
                    ));
                }
                if data
                    .shape
                    .known_dims()
                    .is_some_and(|dims| dims.len() != 2 || dims[1].is_some_and(|n| n != 4))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-MACD-SHAPE",
                        "raw price data must have four columns",
                        0,
                    ));
                }
                let rows = data
                    .shape
                    .known_dims()
                    .and_then(|dims| dims.first().copied().flatten());
                output = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![rows, Some(1)]),
                    StorageFact::Dense,
                );
            }
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class)
                    if class == &standard::TABLE.owned()
                        || class == &standard::TIMETABLE.owned() =>
                {
                    output = ValueFact::proven(
                        ValueKindFact::Object(ObjectFact {
                            class: None,
                            runtime_class: Some(class.clone()),
                            properties: BTreeMap::new(),
                            properties_complete: false,
                            handle_semantics: Some(false),
                        }),
                        ShapeFact::Scalar,
                        StorageFact::Opaque,
                    );
                }
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-MACD-CLASS",
                    "object data must be a table or timetable",
                    0,
                )),
                None if object.class.is_some() => {}
                None => {}
            },
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-MACD-CLASS",
                "data must be an M-by-4 numeric matrix, table, or timetable",
                0,
            )),
        }
    }
    finish_fixed_outputs(entry, request, vec![output.clone(), output], diagnostics)
}
