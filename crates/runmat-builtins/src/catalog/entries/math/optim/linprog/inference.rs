use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(3..=8).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINPROG-ARITY",
            "linprog requires f, A, b and optional equality/bound operands and options",
            0,
        ));
    }
    for (index, argument) in request.arguments.iter().take(7).enumerate() {
        if matches!(argument.kind, ValueKindFact::Logical) {
            diagnostics.push(argument_error(
                "RM-CATALOG-LINPROG-CLASS",
                "linprog operands must be real numeric, not logical",
                index,
            ));
        }
    }
    facts::numeric(
        request,
        0,
        facts::OptimNumericInput::NonemptyVector,
        &mut diagnostics,
    );
    for index in [1, 3] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalSparseMatrix,
            &mut diagnostics,
        );
    }
    for index in [2, 4, 5, 6] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalVector,
            &mut diagnostics,
        );
    }
    super::super::nonlinear_facts::options(request.arguments.get(7), 7, true, &mut diagnostics);
    facts::constraint_geometry(request, 0, &[(1, 2), (3, 4)], &[5, 6], &mut diagnostics);
    if request.arguments.len() == 4
        && request.arguments[3]
            .shape
            .element_count()
            .is_some_and(|count| count != 0)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-LINPROG-EQUALITY",
            "Aeq requires a matching beq",
            3,
        ));
    }
    let mut outputs = facts::solver_outputs(false);
    let mut fields = std::collections::BTreeMap::new();
    for name in [
        "iterations",
        "constrviolation",
        "primalfeasibility",
        "dualfeasibility",
        "dualitygap",
        "firstorderopt",
    ] {
        fields.insert(
            name.into(),
            facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        );
    }
    for name in ["algorithm", "message"] {
        fields.insert(name.into(), ValueFact::scalar(ValueKindFact::String));
    }
    outputs[3] = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let mut fields = std::collections::BTreeMap::new();
    // Verified optimal multipliers are columns. Failure/stopped forms are [],
    // so a universally column-shaped proof would be false.
    for name in ["ineqlin", "eqlin", "lower", "upper"] {
        fields.insert(
            name.into(),
            facts::double(ShapeFact::Unknown, StorageFact::Dense),
        );
    }
    outputs.push(ValueFact::scalar(ValueKindFact::Struct(
        StructFact::scalar(fields, true),
    )));
    let mut result = facts::finish(entry, request, outputs, diagnostics);
    // Runtime's default Display is off. Existing facts can prove an absent
    // field, but do not carry nested string literal values: a present dynamic
    // Display field must retain the console/diary/forwarder upper bound.
    let default_display = match request.arguments.get(7) {
        None => true,
        Some(options) if options.shape.element_count() == Some(0) => true,
        Some(options) => matches!(&options.kind, ValueKindFact::Struct(options)
            if options.fields_complete && !options.fields.keys().any(|name|
                crate::OptimOptionField::parse(name)==Some(crate::OptimOptionField::Display))),
    };
    if default_display {
        for effect in [
            EffectKind::FilesystemWrite,
            EffectKind::UserInterface,
            EffectKind::HostCallback,
        ] {
            result.effects.0.remove(&effect);
        }
    }
    result
}
