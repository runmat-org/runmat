use super::{TabularWriteKind, TabularWriteOption};
use crate::{
    catalog::inference::{argument_error, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;
fn text(request: &CallRequest, index: usize) -> Option<&str> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Character(v) | LiteralValue::String(v) | LiteralValue::Keyword(v) => Some(v),
        _ => None,
    }
}
fn invalid_text(value: &ValueFact) -> bool {
    !matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
    ) || known_invalid_text_scalar(value)
}
pub(super) fn infer(
    kind: TabularWriteKind,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    let code = kind.argument_error().code;
    if request.arguments.len() < 2 || !request.arguments.len().is_multiple_of(2) {
        result.diagnostics.push(argument_error(
            code,
            "writer requires an exact tabular object, filename, and complete option pairs",
            0,
        ));
    }
    if let Some(value) = request.arguments.first() {
        let invalid = match &value.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Object(object) => object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class != &kind.class().owned()),
            _ => true,
        };
        if invalid {
            result.diagnostics.push(argument_error(
                code,
                format!("expected an exact {} object", kind.class().display_name()),
                0,
            ));
        }
    }
    if request.arguments.get(1).is_some_and(invalid_text)
        || text(request, 1).is_some_and(str::is_empty)
    {
        result.diagnostics.push(argument_error(
            code,
            "filename must be nonempty scalar text",
            1,
        ));
    }
    for index in (2..request.arguments.len().saturating_sub(1)).step_by(2) {
        if invalid_text(&request.arguments[index]) {
            result.diagnostics.push(argument_error(
                code,
                "option names must be scalar text",
                index,
            ));
            continue;
        }
        if let Some(name) = text(request, index) {
            match TabularWriteOption::parse(name) {
                Some(TabularWriteOption::Delimiter) => {
                    if invalid_text(&request.arguments[index + 1]) {
                        result.diagnostics.push(argument_error(
                            code,
                            "Delimiter must be scalar text",
                            index + 1,
                        ));
                    }
                }
                Some(TabularWriteOption::VariableNames) => {
                    let value = &request.arguments[index + 1];
                    if value.shape.element_count().is_some_and(|n| n != 1)
                        && !matches!(
                            value.kind,
                            ValueKindFact::Character | ValueKindFact::Unknown
                        )
                    {
                        result.diagnostics.push(argument_error(
                            code,
                            "WriteVariableNames must be scalar",
                            index + 1,
                        ));
                    }
                }
                None => result.diagnostics.push(argument_error(
                    code,
                    "unsupported table write option",
                    index,
                )),
            }
        }
    }
    result
}
