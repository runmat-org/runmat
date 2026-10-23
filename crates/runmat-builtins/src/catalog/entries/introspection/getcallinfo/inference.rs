use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-GETCALLINFO-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    let outputs = vec![ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(
            std::collections::BTreeMap::from([
                ("name".into(), ValueFact::scalar(ValueKindFact::String)),
                ("file".into(), ValueFact::scalar(ValueKindFact::String)),
                (
                    "line".into(),
                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    })),
                ),
                (
                    "stack".into(),
                    ValueFact::proven(
                        ValueKindFact::Struct(StructFact::array(
                            std::collections::BTreeMap::from([
                                ("file".into(), ValueFact::scalar(ValueKindFact::String)),
                                ("name".into(), ValueFact::scalar(ValueKindFact::String)),
                                (
                                    "line".into(),
                                    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                                        class: NumericClass::Double,
                                        domain: NumericDomain::Real,
                                    })),
                                ),
                            ]),
                            true,
                            vec![],
                            false,
                        )),
                        ShapeFact::from(vec![None, Some(1)]),
                        StorageFact::Dense,
                    ),
                ),
            ]),
            true,
        ),
    ))];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
