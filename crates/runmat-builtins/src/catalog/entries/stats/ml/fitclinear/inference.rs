use super::FitclinearOption;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, NumericClass, NumericDomain,
    NumericFact, ObjectFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let table_input = match request.arguments.first().map(|value| &value.kind) {
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FITCLINEAR-ARITY",
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
                    "RM-CATALOG-FITCLINEAR-DATA",
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
                "RM-CATALOG-FITCLINEAR-DATA",
                "predictors must be a real numeric matrix or table",
                0,
            ));
            None
        }
    };
    if table_input == Some(false) && request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITCLINEAR-RESPONSE",
            "matrix fitting requires response labels",
            1,
        ));
    }
    if request.arguments.first().is_some_and(|fact| {
        table_input == Some(false)
            && matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.len() > 2)
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-FITCLINEAR-RANK",
            "X must be a 2-D matrix",
            0,
        ));
    }
    diagnose_option_names(request, table_input, &mut diagnostics);
    let model = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::CLASSIFICATION_LINEAR.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let info = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(BTreeMap::new(), false)),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let mut contract = CallContract::fixed(vec![model, info]);
    // Legacy execution explicitly pads requests beyond FitInfo with numeric zero.
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(ValueFact::scalar(ValueKindFact::Numeric(
        NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        },
    ))));
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
            if FitclinearOption::from_public_name(&name).is_some() {
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
            match FitclinearOption::from_public_name(&name) {
                Some(option) if option.unsupported() => diagnostics.push(argument_error(
                    "RM-CATALOG-FITCLINEAR-OPTION",
                    "option is not supported",
                    index,
                )),
                Some(_) => {}
                None => diagnostics.push(argument_error(
                    "RM-CATALOG-FITCLINEAR-OPTION",
                    "unknown name-value option",
                    index,
                )),
            }
        }
        index += 2;
    }
}
