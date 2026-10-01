use super::FitctreeOption;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, NumericDomain, ObjectFact,
    ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let table_input = match request.arguments.first().map(|value| &value.kind) {
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITCTREE-ARITY",
                "predictors are required",
                0,
            ));
            None
        }
        Some(ValueKindFact::Unknown) => None,
        Some(ValueKindFact::Object(object)) => match object.runtime_class.as_ref() {
            Some(class) if standard::is_tabular(class) => Some(true),
            None => None,
            Some(_) => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FITCTREE-DATA",
                    "predictors must be a real numeric matrix or table",
                    0,
                ));
                None
            }
        },
        Some(ValueKindFact::Numeric(numeric)) if numeric.domain == NumericDomain::Real => {
            Some(false)
        }
        Some(ValueKindFact::Logical) => Some(false),
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITCTREE-DATA",
                "predictors must be a real numeric matrix or table",
                0,
            ));
            None
        }
    };
    if table_input == Some(false) && request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITCTREE-RESPONSE",
            "matrix fitting requires response labels",
            1,
        ));
    }
    if request.arguments.first().is_some_and(|fact| {
        table_input == Some(false)
            && matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.len() > 2)
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITCTREE-RANK",
            "X must be a 2-D matrix",
            0,
        ));
    }
    diagnose_option_names(request, table_input, &mut diagnostics);
    let output = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::CLASSIFICATION_TREE.owned()),
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

fn diagnose_option_names(
    request: &CallRequest,
    table_input: Option<bool>,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) {
    let Some(mut index) = (match table_input {
        Some(false) => Some(2),
        Some(true) => request.literals.literal_string_at(1).map(|name| {
            if FitctreeOption::from_public_name(&name).is_some() {
                1
            } else {
                2
            }
        }),
        None => None,
    }) else {
        return;
    };
    while index + 1 < request.arguments.len() {
        if let Some(name) = request.literals.literal_string_at(index) {
            match FitctreeOption::from_public_name(&name) {
                Some(option) if !option.supported() => diagnostics.push(argument_error(
                    "RM-CATALOG-FITCTREE-OPTION",
                    "option is not supported",
                    index,
                )),
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-FITCTREE-OPTION",
                    "unknown name-value option",
                    index,
                )),
            }
        }
        index += 2;
    }
}
