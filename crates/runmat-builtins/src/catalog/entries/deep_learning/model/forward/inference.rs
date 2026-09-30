use crate::{
    catalog::inference::{
        argument_error,
        deep_learning::{
            infer_network_output, known_state_rows, network_possible, NetworkInferenceCaller,
        },
        finish_fixed_outputs,
    },
    BuiltinCatalogEntry,
};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-ARITY",
            "forward requires exactly a network and one input array; options are not supported",
            request.arguments.len().saturating_sub(1),
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|net| !network_possible(net))
    {
        diagnostics.push(argument_error(
            "RM-DL-FORWARD-NETWORK",
            "network must be a dlnetwork or supported trained network object",
            0,
        ));
    }
    let output = infer_network_output(
        request.arguments.first(),
        request.arguments.get(1),
        NetworkInferenceCaller::Forward,
    );
    diagnostics.extend(output.diagnostics());
    let state_rows = known_state_rows(request.arguments.first());
    let state = super::super::dlnetwork::state_rows_table_fact(state_rows);
    finish_fixed_outputs(entry, request, vec![output.output, state], diagnostics)
}
