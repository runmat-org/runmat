use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest, ShapeFact, StorageFact, ValueFact, ValueKindFact};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MAT2STR-ARITY",
            "mat2str accepts one to three inputs",
            0,
        ));
    }
    if let Some(option) = request.literals.literal_string_at(1) {
        if request.arguments.len() == 2 && !option.eq_ignore_ascii_case("class") {
            diagnostics.push(argument_error(
                "RM-CATALOG-MAT2STR-OPTION",
                "only final 'class' is a text option",
                1,
            ));
        }
    }
    finish_fixed(
        entry,
        request,
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        ),
        diagnostics,
    )
}
