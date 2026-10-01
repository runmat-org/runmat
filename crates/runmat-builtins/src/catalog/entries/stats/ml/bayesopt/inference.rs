use super::{BayesoptAcquisition, BayesoptOption};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, LiteralContext, ObjectFact, OutputSelection, RequestedOutputCount,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
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
    propagate_callback_facts(request, &mut result);
    if !result.diagnostics.is_empty()
        || (2..request.arguments.len())
            .step_by(2)
            .any(|index| request.literals.literal_string_at(index).is_none())
        || request
            .arguments
            .iter()
            .any(|argument| matches!(argument.kind, ValueKindFact::Unknown))
    {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result
}

fn propagate_callback_facts(request: &CallRequest, result: &mut CallInference) {
    let objective = request.arguments.first();
    let constraint = (2..request.arguments.len())
        .step_by(2)
        .filter_map(|index| {
            request
                .literals
                .literal_string_at(index)
                .and_then(|name| BayesoptOption::from_public_name(&name))
                .filter(|option| *option == BayesoptOption::XConstraintFcn)
                .and_then(|_| request.arguments.get(index + 1))
        })
        .next_back();
    for callable in [objective, constraint]
        .into_iter()
        .flatten()
        .filter_map(|fact| {
            let ValueKindFact::Callable(callable) = &fact.kind else {
                return None;
            };
            Some(callable)
        })
    {
        let inferred = crate::infer_callable_call(
            callable,
            &CallRequest {
                arguments: vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        result.effects.0.extend(inferred.effects.0);
        result.capabilities.0.extend(inferred.capabilities.0);
        for mut diagnostic in inferred.diagnostics {
            diagnostic.argument = Some(0);
            result.diagnostics.push(diagnostic);
        }
    }
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
        if option == BayesoptOption::UseParallel
            && request.literals.literal_bool_at(index + 1) == Some(true)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BAYESOPT-PARALLEL",
                "UseParallel=true is not supported",
                index + 1,
            ));
        }
        if option == BayesoptOption::MaxObjectiveEvaluations
            && request
                .literals
                .literal_args
                .get(index + 1)
                .and_then(runmat_types::LiteralContext::numeric_from_literal)
                .is_some_and(|value| {
                    !value.is_finite() || value.fract() != 0.0 || !(1.0..=2048.0).contains(&value)
                })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-BAYESOPT-EVALUATIONS",
                "MaxObjectiveEvaluations must be an integer from 1 through 2048",
                index + 1,
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
