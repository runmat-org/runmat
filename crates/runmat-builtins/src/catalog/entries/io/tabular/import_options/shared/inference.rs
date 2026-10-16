use crate::catalog::inference::argument_error;
use runmat_types::{
    CallRequest, DynamicReason, ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity,
    StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

const VARIABLE_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("matlab.io.VariableImportOptions");

pub(crate) fn modified_options(request: &CallRequest) -> ValueFact {
    let Some(input) = request.arguments.first() else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let ValueKindFact::Object(object) = &input.kind else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: object.runtime_class.clone(),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    result
}

pub(crate) fn variable_options() -> ValueFact {
    let mut result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(VARIABLE_CLASS.owned()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Opaque,
    );
    result.residency = ResidencyFact::Host;
    result
}

pub(crate) fn common_diagnostics(
    request: &CallRequest,
    name: &str,
) -> Vec<runmat_types::InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    if request.arguments.first().is_some_and(|value| {
        !matches!(
            value.kind,
            ValueKindFact::Object(_) | ValueKindFact::Unknown
        )
    }) {
        diagnostics.push(argument_error(
            format!("RM-CATALOG-{name}-OPTIONS"),
            "first argument must be a typed import-options object",
            0,
        ));
    }
    diagnostics
}
