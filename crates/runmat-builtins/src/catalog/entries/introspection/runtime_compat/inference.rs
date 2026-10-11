use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

fn fixed(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    kind: ValueKindFact,
    arity: usize,
    code: &'static str,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(kind)]);
    contract.effects = entry.contract.effect_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != arity {
        result
            .diagnostics
            .push(argument_error(code, "unsupported argument count", 0));
    }
    for (index, arg) in request.arguments.iter().enumerate() {
        if !matches!(
            arg.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || crate::catalog::inference::known_invalid_text_scalar(arg)
        {
            result
                .diagnostics
                .push(argument_error(code, "expected text scalar", index));
        }
    }
    result
}
pub(super) fn infer_isdeployed(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    fixed(
        request,
        entry,
        ValueKindFact::Logical,
        0,
        "RM.ISDEPLOYED.INVALID_INPUT",
    )
}
pub(super) fn infer_ver_less_than(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    fixed(
        request,
        entry,
        ValueKindFact::Logical,
        2,
        "RM.VER_LESS_THAN.INVALID_INPUT",
    )
}
pub(super) fn infer_version(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut result = fixed(
        request,
        entry,
        ValueKindFact::String,
        request.arguments.len().min(1),
        "RM.VERSION.INVALID_INPUT",
    );
    if request.arguments.len() > 1 {
        result.diagnostics.push(argument_error(
            "RM.VERSION.INVALID_INPUT",
            "expected at most one option",
            0,
        ));
    }
    result
}
