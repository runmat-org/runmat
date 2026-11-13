use super::{project_regex_options, RegexOutput, RegexParseMode};
use crate::{
    catalog::inference::{argument_error, numeric_kind},
    BuiltinCatalogEntry,
};
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CertaintyFact, DynamicReason,
    LiteralValue, NumericClass, NumericDomain, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::borrow::Cow;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_for(request, entry, RegexBuiltin::Regexp)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegexBuiltin {
    Regexp,
    Regexpi,
}

impl RegexBuiltin {
    fn name(self) -> &'static str {
        match self {
            Self::Regexp => "regexp",
            Self::Regexpi => "regexpi",
        }
    }

    fn arity_code(self) -> &'static str {
        match self {
            Self::Regexp => "RM-CATALOG-REGEXP-ARITY",
            Self::Regexpi => "RM-CATALOG-REGEXPI-ARITY",
        }
    }

    fn text_code(self) -> &'static str {
        match self {
            Self::Regexp => "RM-CATALOG-REGEXP-TEXT",
            Self::Regexpi => "RM-CATALOG-REGEXPI-TEXT",
        }
    }
}

pub(crate) fn infer_for(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    builtin: RegexBuiltin,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            builtin.arity_code(),
            format!("{} requires subject and expression", builtin.name()),
            request.arguments.len(),
        ));
    }
    for (index, argument) in request.arguments.iter().enumerate() {
        if !matches!(
            &argument.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Cell(_)
        ) {
            diagnostics.push(argument_error(
                builtin.text_code(),
                format!(
                    "{} accepts host text inputs and option flags",
                    builtin.name()
                ),
                index,
            ));
        }
    }
    let mut outputs = Vec::new();
    let requested = request.outputs.requested.known_count().unwrap_or(1).max(1);
    let option_len = request.arguments.len().saturating_sub(2);
    let projection = project_regex_options(
        option_len,
        |index| match request.literals.literal_args.get(index + 2) {
            Some(
                LiteralValue::Character(text)
                | LiteralValue::String(text)
                | LiteralValue::Keyword(text),
            ) => Some(Cow::Borrowed(text.as_str())),
            _ => None,
        },
        RegexParseMode::Inference,
    )
    .expect("inference projection does not reject unknown literals");
    let specs = projection.options.outputs_for_multi();
    for index in 0..requested.min(7) {
        let output = if !projection.complete {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        } else {
            match specs.get(index) {
                Some(RegexOutput::Start | RegexOutput::End) => index_output(
                    request.arguments.first(),
                    projection.options.force_cell_output,
                ),
                Some(RegexOutput::Match) if projection.options.once => {
                    ValueFact::unknown(DynamicReason::RuntimeValue)
                }
                Some(RegexOutput::Match | RegexOutput::Tokens | RegexOutput::Split) => cell_output(
                    request.arguments.first(),
                    projection.options.force_cell_output,
                ),
                _ => ValueFact::unknown(DynamicReason::RuntimeValue),
            }
        };
        outputs.push(output);
    }
    let mut contract = CallContract::fixed(outputs);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    diagnostics.append(&mut inferred.diagnostics);
    inferred.diagnostics = diagnostics;
    inferred
}

fn index_output(subject: Option<&ValueFact>, force_cell: bool) -> ValueFact {
    let Some(subject) = subject else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if !matches!(
        &subject.kind,
        ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_)
    ) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let scalar = match &subject.kind {
        ValueKindFact::String => subject.shape.element_count() == Some(1),
        ValueKindFact::Character => subject
            .shape
            .known_dims()
            .is_some_and(|dims| dims.first().copied().flatten() == Some(1)),
        _ => false,
    };
    if !scalar || force_cell {
        return cell_output(Some(subject), force_cell);
    }
    let mut output = ValueFact::proven(
        numeric_kind(NumericClass::Double, NumericDomain::Real),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    );
    output.certainty = CertaintyFact::Dynamic(DynamicReason::RuntimeValue);
    output
}
fn cell_output(subject: Option<&ValueFact>, force_cell: bool) -> ValueFact {
    if subject.is_none_or(|value| {
        !matches!(
            &value.kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Cell(_)
        )
    }) {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Cell(runmat_types::CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: vec![],
        elements_complete: false,
    });
    if let Some(subject) = subject {
        let scalar = !force_cell
            && match &subject.kind {
                ValueKindFact::String => subject.shape.element_count() == Some(1),
                ValueKindFact::Character => subject
                    .shape
                    .known_dims()
                    .is_some_and(|dims| dims.first().copied().flatten() == Some(1)),
                _ => false,
            };
        output.shape = if scalar {
            ShapeFact::from(vec![Some(1), None])
        } else {
            match &subject.kind {
                ValueKindFact::String | ValueKindFact::Cell(_) => subject.shape.clone(),
                ValueKindFact::Character => subject
                    .shape
                    .known_dims()
                    .map_or(ShapeFact::Unknown, |dims| {
                        ShapeFact::from(vec![dims.first().copied().flatten(), Some(1)])
                    }),
                _ => ShapeFact::Unknown,
            }
        };
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::{LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount};
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn scalar_indices_are_double_with_dynamic_match_count() {
        let request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXP_CATALOG_ENTRY);
        assert_eq!(
            result.outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
    }
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn array_results_remain_containers_and_invalid_inputs_diagnose() {
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::proven(
                    ValueKindFact::String,
                    ShapeFact::from(vec![Some(2), Some(1)]),
                    StorageFact::Dense,
                ),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::default(),
        };
        let result = infer(&request, &super::super::REGEXP_CATALOG_ENTRY);
        assert!(matches!(result.outputs[0].kind, ValueKindFact::Cell(_)));
        request.arguments[1] = ValueFact::scalar(ValueKindFact::Logical);
        assert!(infer(&request, &super::super::REGEXP_CATALOG_ENTRY)
            .diagnostics
            .iter()
            .any(|d| d.code == "RM-CATALOG-REGEXP-TEXT"));
        request.arguments[0] = ValueFact::scalar(ValueKindFact::Logical);
        assert_eq!(
            infer(&request, &super::super::REGEXP_CATALOG_ENTRY).outputs[0].kind,
            ValueKindFact::Unknown
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn consumed_warning_selector_and_scalar_match_shape_follow_runtime_projection() {
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Unknown,
                LiteralValue::String("warnings".into()),
                LiteralValue::String("match".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        };
        let entry = &super::super::REGEXP_CATALOG_ENTRY;
        assert_eq!(
            infer(&request, entry).outputs[0].kind,
            numeric_kind(NumericClass::Double, NumericDomain::Real)
        );
        request.literals.literal_args[2] = LiteralValue::String("match".into());
        request.literals.literal_args[3] = LiteralValue::String("once".into());
        assert_eq!(
            infer(&request, entry).outputs[0].kind,
            ValueKindFact::Unknown
        );
        request.literals.literal_args[3] = LiteralValue::String("ignorecase".into());
        let match_inferred = infer(&request, entry);
        let match_output = &match_inferred.outputs[0];
        assert!(matches!(match_output.kind, ValueKindFact::Cell(_)));
        assert_eq!(match_output.shape, ShapeFact::from(vec![Some(1), None]));
        request.literals.literal_args[2] = LiteralValue::String("forceCellOutput".into());
        request.literals.literal_args[3] = LiteralValue::String("start".into());
        let forced_output = &infer(&request, entry).outputs[0];
        assert!(matches!(forced_output.kind, ValueKindFact::Cell(_)));
        assert_eq!(forced_output.shape, request.arguments[0].shape);
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn seventh_default_split_and_unknown_options_are_conservative() {
        let mut request = CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(7)),
        };
        let entry = &super::super::REGEXP_CATALOG_ENTRY;
        let inferred = infer(&request, entry);
        assert_eq!(inferred.outputs.len(), 7);
        assert!(matches!(inferred.outputs[6].kind, ValueKindFact::Cell(_)));
        assert_eq!(
            inferred.outputs[6].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        request.outputs = OutputSelection::new(RequestedOutputCount::Exactly(8));
        let padded = infer(&request, entry);
        assert_eq!(padded.outputs.len(), 8);
        assert_eq!(padded.outputs[7].kind, ValueKindFact::Unknown);
        request
            .arguments
            .push(ValueFact::scalar(ValueKindFact::String));
        request.literals.literal_args = vec![LiteralValue::Unknown; 3];
        assert_eq!(
            infer(&request, entry).outputs[0].kind,
            ValueKindFact::Unknown
        );
    }
}
