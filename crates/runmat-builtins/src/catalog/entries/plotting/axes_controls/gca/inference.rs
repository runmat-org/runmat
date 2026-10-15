use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-GCA-ARITY",
            "gca accepts at most one argument",
            1,
        ));
    }
    let mut output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    if let Some(argument) = request.arguments.first() {
        let invalid = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Numeric(fact) => {
                fact.domain == NumericDomain::Complex
                    || argument
                        .shape
                        .element_count()
                        .is_some_and(|count| count != 1)
            }
            ValueKindFact::String => {
                !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
                    || crate::catalog::inference::known_invalid_text_scalar(argument)
            }
            ValueKindFact::Character => false,
            ValueKindFact::Object(_) => {
                crate::catalog::inference::known_invalid_scalar_object_handle(
                    argument,
                    &[standard::FIGURE],
                )
            }
            _ => true,
        };
        if invalid
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-GCA-INPUT",
                "gca requires a host figure handle or 'struct' token",
                0,
            ));
        }
        if let Some(mode) = request.literals.literal_string_at(0) {
            if mode.trim().eq_ignore_ascii_case("struct") {
                let value = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }));
                let fields = BTreeMap::from([
                    ("handle".to_owned(), value.clone()),
                    ("figure".to_owned(), value.clone()),
                    ("rows".to_owned(), value.clone()),
                    ("cols".to_owned(), value.clone()),
                    ("index".to_owned(), value),
                ]);
                output = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
            } else {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GCA-MODE",
                    "only 'struct' is a supported gca mode",
                    0,
                ));
            }
        } else if matches!(
            argument.kind,
            ValueKindFact::String | ValueKindFact::Character
        ) {
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.residency = ResidencyFact::Host;
        }
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}
