use std::collections::BTreeMap;

use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    AliasFact, CallInference, CallRequest, MutationFact, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StaticClassIdentity, StorageFact, ValueFact,
    ValueKindFact,
};

const PATH_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("matlab.unittest.fixtures.PathFixture");
const TEMP_CLASS: StaticClassIdentity =
    StaticClassIdentity::new("matlab.unittest.fixtures.TemporaryFolderFixture");

fn fixture(class: StaticClassIdentity, extra: (&str, ValueFact)) -> ValueFact {
    let mut properties =
        BTreeMap::from([("Name".into(), ValueFact::scalar(ValueKindFact::String))]);
    properties.insert(extra.0.into(), extra.1);
    let mut fact = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(class.owned()),
            properties,
            properties_complete: true,
            handle_semantics: Some(true),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    fact.alias = AliasFact::Identity;
    fact.mutation = MutationFact::HandleSemantics;
    fact
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
    shape_ok && !matches!(value.residency, ResidencyFact::Device { .. })
}

fn possible_object(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Object(_) | ValueKindFact::Unknown
    )
}

fn possible_callback(value: &ValueFact) -> bool {
    // Callable facts may be represented as a dedicated function kind or
    // unresolved at this point. The runtime validates exact handle variants.
    matches!(
        value.kind,
        ValueKindFact::Unknown | ValueKindFact::Callable(_)
    )
}

pub(super) fn path_fixture(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-TESTING-PATH-FIXTURE-ARITY",
            "PathFixture requires one folder",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !host_text(value))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-PATH-FIXTURE-FOLDER",
            "folder must be host scalar text",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        fixture(
            PATH_CLASS,
            ("Path", ValueFact::scalar(ValueKindFact::String)),
        ),
        diagnostics,
    )
}

pub(super) fn temporary_folder_fixture(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let diagnostics = if request.arguments.is_empty() {
        Vec::new()
    } else {
        vec![argument_error(
            "RM-TESTING-TEMP-FIXTURE-ARITY",
            "TemporaryFolderFixture currently accepts no options",
            0,
        )]
    };
    finish_fixed(
        entry,
        request,
        fixture(
            TEMP_CLASS,
            ("Folder", ValueFact::scalar(ValueKindFact::String)),
        ),
        diagnostics,
    )
}

pub(super) fn apply_fixture(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-TESTING-APPLY-FIXTURE-ARITY",
            "applyFixture requires a TestCase and fixture",
            0,
        ));
    }
    for (index, name) in [(0, "TestCase"), (1, "fixture")] {
        if request
            .arguments
            .get(index)
            .is_some_and(|value| !possible_object(value))
        {
            diagnostics.push(argument_error(
                format!("RM-TESTING-APPLY-FIXTURE-{index}"),
                format!("{name} must be an object"),
                index,
            ));
        }
    }
    // The fixture is returned after setup. Retain its concrete class and
    // handle identity; its Folder/Path value may change at runtime.
    let output = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue));
    finish_fixed(entry, request, output, diagnostics)
}

pub(super) fn add_teardown(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-TESTING-ADD-TEARDOWN-ARITY",
            "addTeardown requires a TestCase and callback",
            0,
        ));
    }
    if request
        .arguments
        .first()
        .is_some_and(|value| !possible_object(value))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-ADD-TEARDOWN-RECEIVER",
            "receiver must be a TestCase object",
            0,
        ));
    }
    if request
        .arguments
        .get(1)
        .is_some_and(|value| !possible_callback(value))
    {
        diagnostics.push(argument_error(
            "RM-TESTING-ADD-TEARDOWN-CALLBACK",
            "callback must be a function handle",
            1,
        ));
    }
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
