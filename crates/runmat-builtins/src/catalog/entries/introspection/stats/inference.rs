use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STATS-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    if let Some(input) = request.arguments.first() {
        if crate::catalog::inference::known_invalid_scalar_object_handle(
            input,
            &[crate::MEMOIZED_FUNCTION_CLASS],
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STATS-INPUT",
                "expected a scalar MemoizedFunction handle",
                0,
            ));
        }
    }
    let outputs = vec![ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(
            std::collections::BTreeMap::from([(
                "Cache".into(),
                ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(
                    std::collections::BTreeMap::from([
                        (
                            "Inputs".into(),
                            ValueFact::proven(
                                ValueKindFact::Cell(CellFact {
                                    element: Box::new(ValueFact::unknown(
                                        DynamicReason::RuntimeValue,
                                    )),
                                    elements: vec![],
                                    elements_complete: false,
                                }),
                                ShapeFact::from(vec![Some(1), None]),
                                StorageFact::Dense,
                            ),
                        ),
                        (
                            "Nargout".into(),
                            ValueFact::proven(
                                ValueKindFact::Numeric(NumericFact {
                                    class: NumericClass::Double,
                                    domain: NumericDomain::Real,
                                }),
                                ShapeFact::from(vec![Some(1), None]),
                                StorageFact::Dense,
                            ),
                        ),
                        (
                            "Outputs".into(),
                            ValueFact::proven(
                                ValueKindFact::Cell(CellFact {
                                    element: Box::new(ValueFact::unknown(
                                        DynamicReason::RuntimeValue,
                                    )),
                                    elements: vec![],
                                    elements_complete: false,
                                }),
                                ShapeFact::from(vec![Some(1), None]),
                                StorageFact::Dense,
                            ),
                        ),
                        (
                            "HitCount".into(),
                            ValueFact::proven(
                                ValueKindFact::Numeric(NumericFact {
                                    class: NumericClass::Double,
                                    domain: NumericDomain::Real,
                                }),
                                ShapeFact::from(vec![Some(1), None]),
                                StorageFact::Dense,
                            ),
                        ),
                        (
                            "TotalHits".into(),
                            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                                class: NumericClass::Double,
                                domain: NumericDomain::Real,
                            })),
                        ),
                        (
                            "TotalMisses".into(),
                            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                                class: NumericClass::Double,
                                domain: NumericDomain::Real,
                            })),
                        ),
                    ]),
                    true,
                ))),
            )]),
            true,
        ),
    ))];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
