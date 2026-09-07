mod output;
mod representation;
mod selection;
mod shape;
mod source;

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

pub(in crate::catalog) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-TYPECAST-ARITY",
            "typecast requires two inputs or the three-input like form",
            request.arguments.len().min(2),
        ));
    }
    let Some(source) = request.arguments.first() else {
        return finish_fixed(entry, request, dynamic(), diagnostics);
    };
    source::validate(source, &mut diagnostics);

    let target = match request.arguments.as_slice() {
        [_, _] => selection::from_literal(request, &mut diagnostics),
        [_, _, prototype] => {
            selection::from_prototype(request, source, prototype, &mut diagnostics)
        }
        _ => None,
    };
    let output = target.map_or_else(dynamic, |target| {
        output::fact(source, target, &mut diagnostics)
    });
    finish_fixed(entry, request, output, diagnostics)
}

fn dynamic() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}
