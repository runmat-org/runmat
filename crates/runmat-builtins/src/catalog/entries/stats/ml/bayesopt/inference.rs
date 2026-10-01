use super::{BayesoptAcquisition, BayesoptOption};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, InferenceDiagnostic,
    ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 || !(request.arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BAYESOPT-ARITY",
            "bayesopt requires an objective, variables and paired name-value options",
            request.arguments.len(),
        ));
    }
    if request.arguments.first().is_some_and(|fact| {
        !matches!(
            fact.kind,
            ValueKindFact::Callable(_) | ValueKindFact::Unknown
        )
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-BAYESOPT-OBJECTIVE",
            "objective must be callable",
            0,
        ));
    }
    if let Some(variables) = request.arguments.get(1) {
        match &variables.kind {
            ValueKindFact::Object(object) => {
                if object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| !class.is(standard::OPTIMIZABLE_VARIABLE))
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-BAYESOPT-VARIABLES",
                        "variables must be optimizableVariable objects",
                        1,
                    ));
                }
            }
            ValueKindFact::Cell(_) | ValueKindFact::Unknown | ValueKindFact::OutputList(_) => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-BAYESOPT-VARIABLES",
                "variables must be optimizableVariable objects",
                1,
            )),
        }
    }
    diagnose_options(request, &mut diagnostics);
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::BAYESIAN_OPTIMIZATION.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn diagnose_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    for index in (2..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(option) = BayesoptOption::from_public_name(&name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-BAYESOPT-OPTION",
                "unknown bayesopt option",
                index,
            ));
            continue;
        };
        if option.always_unsupported() {
            diagnostics.push(argument_error(
                "RM-CATALOG-BAYESOPT-UNSUPPORTED-OPTION",
                "warm-start options are not supported",
                index,
            ));
        }
        if option == BayesoptOption::AcquisitionFunctionName {
            if let Some(value) = request.literals.literal_string_at(index + 1) {
                if BayesoptAcquisition::from_public_name(&value).is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-BAYESOPT-ACQUISITION",
                        "unsupported acquisition function",
                        index + 1,
                    ));
                }
            }
        }
    }
}
