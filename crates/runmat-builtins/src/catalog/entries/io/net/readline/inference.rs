use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    // The remote peer and timeout determine whether the host result is a
    // string scalar or the documented 0-by-0 double timeout sentinel.
    let mut output = ValueFact::unknown(DynamicReason::ExternalData);
    output.residency = ResidencyFact::Host;
    output.alias = AliasFact::Unique;
    output.mutation = MutationFact::ValueSemantics;
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if request.arguments.len() != 1 {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-READLINE-ARITY",
            "readline requires exactly one tcpclient value",
            0,
        ));
    }
    if request.arguments.first().is_some_and(|input| {
        !super::super::receiver_inference::may_be_receiver(input, super::super::TCPCLIENT_CLASS)
    }) {
        result.diagnostics.push(argument_error(
            "RM-CATALOG-READLINE-CLIENT",
            "readline requires a tcpclient value",
            0,
        ));
    }
    result
}
