use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

use super::{which_option_token, WhichOptionToken};

pub(super) fn infer_which(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let all = request.literals.literal_args.iter().any(|literal| {
        crate::catalog::inference::literal_text(literal).and_then(|text| which_option_token(&text))
            == Some(WhichOptionToken::All)
    });
    // The literal vector is position-aligned with arguments: a dynamic operand
    // is represented by LiteralValue::Unknown, not necessarily by a short list.
    // Any such operand in a multi-argument call might decode to "-all".
    let unknown_option = request.arguments.len() > 1
        && (0..request.arguments.len()).any(|index| {
            request
                .literals
                .literal_args
                .get(index)
                .and_then(crate::catalog::inference::literal_text)
                .is_none()
        });
    // Even a source-known "-all" returns a char row on no match and a cell
    // only when matches exist. Resolution is runtime-dependent.
    let mut output = if all || unknown_option {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), None]),
            StorageFact::Dense,
        )
    };
    output.residency = ResidencyFact::Host;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.is_empty() {
        result
            .diagnostics
            .push(argument_error("RM.WHICH.ARG_COUNT", "expected a name", 0));
    }
    let mut known_name_before = false;
    for (index, arg) in request.arguments.iter().enumerate() {
        let invalid = !matches!(
            arg.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || crate::catalog::inference::known_invalid_text_scalar(arg);
        if invalid {
            let (code, message) = if known_name_before {
                ("RM.WHICH.OPTION_ARG_TYPE", "expected text option")
            } else {
                ("RM.WHICH.NAME_ARG_TYPE", "expected text name or flag")
            };
            result
                .diagnostics
                .push(argument_error(code, message, index));
            continue;
        }
        if request
            .literals
            .literal_args
            .get(index)
            .and_then(crate::catalog::inference::literal_text)
            .is_some_and(|text| which_option_token(&text).is_none())
        {
            known_name_before = true;
        }
    }
    result
}
