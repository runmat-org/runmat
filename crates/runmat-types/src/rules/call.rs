use crate::{
    CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic, ValueFact,
};

pub fn infer_call(contract: &CallContract, request: &CallRequest) -> CallInference {
    let mut diagnostics = Vec::new();
    diagnostics.extend(
        request
            .outputs
            .validate()
            .err()
            .map(|message| InferenceDiagnostic::error("RM-TYPE-CALL-DISCARD", message)),
    );
    let Some(requested) = request.outputs.requested.known_count() else {
        return CallInference {
            outputs: contract.outputs.clone(),
            dynamic_outputs: true,
            discarded: request.outputs.discarded.iter().copied().collect(),
            effects: contract.effects.clone(),
            capabilities: contract.capabilities.clone(),
            diagnostics,
        };
    };
    if contract
        .maximum_outputs
        .is_some_and(|maximum| requested > maximum)
    {
        diagnostics.push(InferenceDiagnostic::error(
            "RM-TYPE-CALL-ARITY",
            format!(
                "call requests {requested} outputs but at most {} are available",
                contract.maximum_outputs.unwrap_or(0)
            ),
        ));
    }
    let outputs = (0..requested)
        .map(|index| {
            contract
                .outputs
                .get(index)
                .cloned()
                .or_else(|| contract.variadic_output.as_deref().cloned())
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::UnsupportedRepresentation))
        })
        .collect();
    CallInference {
        outputs,
        dynamic_outputs: false,
        discarded: request.outputs.discarded.iter().copied().collect(),
        effects: contract.effects.clone(),
        capabilities: contract.capabilities.clone(),
        diagnostics,
    }
}
