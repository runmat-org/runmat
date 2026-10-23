use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-DBSTACK-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    for (index, input) in request.arguments.iter().enumerate() {
        let valid_control = match input.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::String | ValueKindFact::Character => {
                !crate::catalog::inference::known_invalid_text_scalar(input)
            }
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) => input.shape.element_count().is_none_or(|n| n == 1),
            _ => false,
        };
        if !valid_control {
            diagnostics.push(argument_error(
                "RM-CATALOG-DBSTACK-CONTROL",
                "expected -completenames or a nonnegative integer scalar",
                index,
            ));
        }
    }
    let outputs = vec![
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
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
    ];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
