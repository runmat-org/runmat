use super::{OptimizableVariableOption, OptimizationVariableTransform, OptimizationVariableType};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason,
    InferenceDiagnostic, ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 || !(request.arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIMIZABLE-VARIABLE-ARITY",
            "optimizableVariable requires Name, Range and paired name-value options",
            request.arguments.len(),
        ));
    }
    if let Some(name) = request.arguments.first() {
        if !matches!(
            name.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) || matches!(name.kind, ValueKindFact::String)
            && name.shape.element_count().is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMIZABLE-VARIABLE-NAME",
                "Name must be a text scalar",
                0,
            ));
        }
    }
    if let Some(range) = request.arguments.get(1) {
        if !matches!(
            range.kind,
            ValueKindFact::Numeric(_)
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMIZABLE-VARIABLE-RANGE",
                "Range must contain numeric bounds or categorical text",
                1,
            ));
        }
        if matches!(range.kind, ValueKindFact::Numeric(_))
            && range.shape.element_count().is_some_and(|count| count != 2)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMIZABLE-VARIABLE-RANGE-SIZE",
                "numeric Range must have exactly two elements",
                1,
            ));
        }
    }
    diagnose_options(request, &mut diagnostics);
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::OPTIMIZABLE_VARIABLE.owned()),
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

fn diagnose_options(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    for index in (2..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        let Some(option) = OptimizableVariableOption::from_public_name(&name) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMIZABLE-VARIABLE-OPTION",
                "unknown optimizableVariable option",
                index,
            ));
            continue;
        };
        let value = request.arguments.get(index + 1);
        let known_invalid_class = value.is_some_and(|value| match option {
            OptimizableVariableOption::Type | OptimizableVariableOption::Transform => !matches!(
                value.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ),
            OptimizableVariableOption::Optimize => {
                matches!(
                    value.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Object(_)
                ) || value.shape.element_count().is_some_and(|count| count != 1)
            }
        });
        let invalid_literal = request
            .literals
            .literal_string_at(index + 1)
            .is_some_and(|value| match option {
                OptimizableVariableOption::Type => {
                    OptimizationVariableType::from_public_name(&value).is_none()
                }
                OptimizableVariableOption::Transform => {
                    OptimizationVariableTransform::from_public_name(&value).is_none()
                }
                OptimizableVariableOption::Optimize => true,
            });
        if known_invalid_class || invalid_literal {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIMIZABLE-VARIABLE-OPTION-VALUE",
                "unsupported Type or Transform value",
                index + 1,
            ));
        }
    }
}
