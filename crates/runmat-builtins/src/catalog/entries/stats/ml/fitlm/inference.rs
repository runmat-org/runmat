use super::FitlmOption;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, InferenceDiagnostic,
    ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let first = request.arguments.first();
    if first.is_none() {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITLM-ARITY",
            "fitlm requires a predictor matrix or table",
            0,
        ));
    }
    if let Some(first) = first {
        match &first.kind {
            ValueKindFact::Unknown => {}
            ValueKindFact::Numeric(numeric)
                if numeric.domain == runmat_types::NumericDomain::Real =>
            {
                diagnose_matrix_form(request, first, &mut diagnostics);
            }
            ValueKindFact::Logical => diagnose_matrix_form(request, first, &mut diagnostics),
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) if standard::is_tabular(class) => {}
                None => {}
                Some(_) => diagnostics.push(argument_error(
                    "RM-CATALOG-FITLM-DATA",
                    "predictors must be a numeric matrix or table",
                    0,
                )),
            },
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-FITLM-DATA",
                "predictors must be a numeric matrix or table",
                0,
            )),
        }
    }
    diagnose_known_option_names(request, &mut diagnostics);
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::LINEAR_MODEL.owned()),
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

fn diagnose_matrix_form(
    request: &CallRequest,
    first: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITLM-RESPONSE",
            "matrix fitting requires a response",
            1,
        ));
    }
    if matches!(&first.shape, ShapeFact::Shaped { dims } if dims.len() > 2) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITLM-RANK",
            "predictors must be a vector or 2-D matrix",
            0,
        ));
    }
}

fn diagnose_known_option_names(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    // `fitlm` has a positional model specification and table response forms.
    // Only diagnose a source-known name when its name-value role is certain.
    let len = request.arguments.len();
    if len < 4 || !len.is_multiple_of(2) {
        return;
    }
    for index in (2..len).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            continue;
        };
        match FitlmOption::from_public_name(&name) {
            Some(FitlmOption::RobustOpts | FitlmOption::CategoricalVars) => diagnostics.push(
                argument_error("RM-CATALOG-FITLM-OPTION", "option is not supported", index),
            ),
            Some(_) => {}
            None => diagnostics.push(argument_error(
                "RM-CATALOG-FITLM-OPTION",
                "unknown name-value option",
                index,
            )),
        }
    }
}
