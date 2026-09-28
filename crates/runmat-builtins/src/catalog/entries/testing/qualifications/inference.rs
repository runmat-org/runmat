use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

#[derive(Clone, Copy)]
enum Predicate {
    Equal,
    LogicalTrue,
    LogicalFalse,
    Constraint,
    Empty,
    GreaterThan,
}

fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    predicate: Predicate,
) -> CallInference {
    let required = match predicate {
        Predicate::LogicalTrue | Predicate::LogicalFalse | Predicate::Empty => 2,
        Predicate::Equal | Predicate::Constraint | Predicate::GreaterThan => 3,
    };
    let mut diagnostics = Vec::new();
    if request.arguments.len() < required {
        diagnostics.push(argument_error(
            "RM-TESTING-QUALIFICATION-ARITY",
            format!(
                "{} requires {required} inputs before optional diagnostics",
                entry.identity.name
            ),
            request.arguments.len(),
        ));
    }
    if request.arguments.first().is_some_and(|receiver| {
        !matches!(
            &receiver.kind,
            ValueKindFact::Unknown | ValueKindFact::Object(_)
        )
    }) {
        diagnostics.push(argument_error(
            "RM-TESTING-QUALIFICATION-RECEIVER",
            "qualification requires a TestCase receiver",
            0,
        ));
    }
    if matches!(predicate, Predicate::Constraint)
        && request.arguments.get(2).is_some_and(|constraint| {
            !matches!(
                &constraint.kind,
                ValueKindFact::Unknown | ValueKindFact::Object(_)
            )
        })
    {
        diagnostics.push(argument_error(
            "RM-TESTING-QUALIFICATION-CONSTRAINT",
            "assertThat requires a Constraint object",
            2,
        ));
    }
    // Qualification outcome depends on runtime values and test context. A
    // failed call throws, while a successful one returns an empty double.
    // Neither a literal `true` nor a matching-looking expected value can
    // prove that user-defined equality or a constraint succeeds.
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

pub(super) fn assert_equal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::Equal)
}

pub(super) fn assert_that(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::Constraint)
}

pub(super) fn assert_true(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::LogicalTrue)
}

pub(super) fn assume_equal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::Equal)
}

pub(super) fn assume_true(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::LogicalTrue)
}

pub(super) fn fatal_assert_true(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer(request, entry, Predicate::LogicalTrue)
}

pub(super) fn verify_empty(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::Empty)
}

pub(super) fn verify_equal(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::Equal)
}

pub(super) fn verify_false(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer(request, entry, Predicate::LogicalFalse)
}

pub(super) fn verify_greater_than(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    infer(request, entry, Predicate::GreaterThan)
}
