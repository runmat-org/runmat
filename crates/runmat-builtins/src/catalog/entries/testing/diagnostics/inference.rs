use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn object(class: StaticClassIdentity, properties: BTreeMap<String, ValueFact>) -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            // The current ObjectInstance representation is value-backed, but
            // these MATLAB class contracts are not yet fully modeled as
            // value or handle identities by RunMat's testing subsystem.
            handle_semantics: None,
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

pub(super) fn test_case(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-TEST-CASE-ARITY",
            "TestCase accepts no inputs",
            0,
        )]
    };
    let properties = BTreeMap::from([("Name".into(), ValueFact::scalar(ValueKindFact::String))]);
    finish_fixed(
        entry,
        request,
        object(runmat_types::standard::UNIT_TEST_CASE, properties),
        diagnostics,
    )
}

fn host_text(value: &ValueFact) -> bool {
    let shape_ok = match &value.kind {
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => value
            .shape
            .known_dims()
            .is_none_or(|dims| dims.first().copied().flatten().is_none_or(|rows| rows <= 1)),
        ValueKindFact::Unknown => true,
        _ => false,
    };
    shape_ok && !matches!(value.residency, runmat_types::ResidencyFact::Device { .. })
}

pub(super) fn string_diagnostic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-TESTING-STRING-DIAGNOSTIC-ARITY",
            "StringDiagnostic requires one text input",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !host_text(value))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-STRING-DIAGNOSTIC-TEXT",
            "StringDiagnostic requires host scalar text",
            0,
        ));
    }
    let text = ValueFact::scalar(ValueKindFact::String);
    let properties = BTreeMap::from([
        ("Identifier".into(), text.clone()),
        ("Message".into(), text.clone()),
        ("DiagnosticText".into(), text),
    ]);
    finish_fixed(
        entry,
        request,
        object(
            runmat_types::standard::UNIT_TEST_STRING_DIAGNOSTIC,
            properties,
        ),
        diagnostics,
    )
}

fn is_test_case(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        // An object with a different concrete class may still derive from
        // TestCase; CallRequest does not carry a proven class hierarchy.
        ValueKindFact::Object(_) => true,
        _ => false,
    }
}

fn is_diagnostic(value: &ValueFact) -> bool {
    match &value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String | ValueKindFact::Character => host_text(value),
        // Custom Diagnostic subclasses remain admissible until runtime
        // class-registry validation.
        ValueKindFact::Object(_) => true,
        _ => false,
    }
}

fn verbosity_literal_valid(value: &LiteralValue) -> bool {
    match value {
        LiteralValue::Number(value) => (1.0..=4.0).contains(value) && value.fract() == 0.0,
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|value| (1.0..=4.0).contains(&value) && value.fract() == 0.0),
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "terse" | "concise" | "detailed" | "verbose"
        ),
        LiteralValue::Unknown => true,
        _ => false,
    }
}

fn verbosity(value: &ValueFact) -> bool {
    let kind_ok = match &value.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) => true,
        ValueKindFact::String | ValueKindFact::Character => host_text(value),
        ValueKindFact::Unknown => true,
        _ => false,
    };
    kind_ok && value.shape.element_count().is_none_or(|count| count == 1)
}

pub(super) fn log(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-TESTING-LOG-ARITY",
            "log expects a TestCase, optional verbosity, and diagnostic",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !is_test_case(value))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-LOG-RECEIVER",
            "log requires a TestCase receiver",
            0,
        ));
    }
    if request.arguments.len() == 3
        && (!verbosity(&request.arguments[1])
            || request
                .literals
                .literal_args
                .get(1)
                .is_some_and(|literal| !verbosity_literal_valid(literal)))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-LOG-VERBOSITY",
            "verbosity must be an integer from 1 to 4 or a supported name",
            1,
        ));
    }
    if let Some(diagnostic) = request.arguments.last() {
        if !is_diagnostic(diagnostic) {
            diagnostics.push(argument_error(
                "RM-TESTING-LOG-DIAGNOSTIC",
                "diagnostic must be host scalar text or a Diagnostic object",
                request.arguments.len() - 1,
            ));
        }
    }
    // The runtime returns an empty double matrix, even when the call is used as a statement.
    let empty = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(0), Some(0)]),
        StorageFact::Dense,
    );
    finish_fixed(entry, request, empty, diagnostics)
}
