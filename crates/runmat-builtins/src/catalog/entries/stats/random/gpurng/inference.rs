use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GPURNG-ARITY",
            "gpurng accepts no more than two inputs",
            2,
        ));
    }
    let fields = BTreeMap::from([
        ("Type".into(), ValueFact::scalar(ValueKindFact::String)),
        (
            "Seed".into(),
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
        ),
        (
            "State".into(),
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt32,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(17), Some(1)]),
                StorageFact::Dense,
            ),
        ),
    ]);
    let output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    finish_fixed(entry, request, output, diagnostics)
}
