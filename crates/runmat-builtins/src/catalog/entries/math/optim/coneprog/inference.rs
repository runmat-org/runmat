use super::super::program_facts as facts;
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() == 1 {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(problem)
                if matches!(
                    request.arguments[0].storage,
                    StorageFact::Scalar | StorageFact::Unknown
                ) && request.arguments[0]
                    .shape
                    .element_count()
                    .is_none_or(|count| count == 1) =>
            {
                let find = |names: &[&str]| {
                    names.iter().find_map(|name| {
                        problem
                            .fields
                            .iter()
                            .find(|(key, _)| key.eq_ignore_ascii_case(name))
                            .map(|(_, fact)| fact.clone())
                    })
                };
                let unknown = || ValueFact::unknown(DynamicReason::RuntimeValue);
                let empty =
                    || facts::double(ShapeFact::from(vec![Some(0), Some(0)]), StorageFact::Dense);
                if find(&["f"]).is_none() && problem.fields_complete {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONE-F",
                        "problem struct requires f",
                        0,
                    ));
                }
                let mut normalized = request.clone();
                normalized.arguments = vec![find(&["f"]).unwrap_or_else(unknown)];
                for names in [
                    &["socConstraints"][..],
                    &["Aineq", "A"][..],
                    &["bineq", "b"][..],
                    &["Aeq"][..],
                    &["beq"][..],
                    &["lb"][..],
                    &["ub"][..],
                ] {
                    normalized.arguments.push(find(names).unwrap_or_else(|| {
                        if problem.fields_complete {
                            empty()
                        } else {
                            unknown()
                        }
                    }));
                }
                let start = diagnostics.len();
                inspect_positional(&normalized, &mut diagnostics);
                for diagnostic in &mut diagnostics[start..] {
                    diagnostic.argument = Some(0);
                }
                if find(&["options"]).is_some_and(|fact| {
                    fact.shape.element_count() != Some(0)
                        && !matches!(fact.kind, ValueKindFact::Struct(_) | ValueKindFact::Unknown)
                }) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CONE-OPTIONS",
                        "problem options must be a scalar struct",
                        0,
                    ));
                }
            }
            ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-CONE-PROBLEM",
                "single-argument form requires a scalar problem struct",
                0,
            )),
        }
    } else {
        let has_options = request.arguments.len() > 2 && request.arguments.len() % 2 == 1;
        let numeric_count = request
            .arguments
            .len()
            .saturating_sub(usize::from(has_options));
        if !matches!(numeric_count, 2 | 4 | 6 | 8) || request.arguments.len() > 9 {
            // Unknown final metadata may still be an options struct.
            if !(matches!(request.arguments.len(), 3 | 5 | 7 | 9)
                && request
                    .arguments
                    .last()
                    .is_some_and(|fact| matches!(fact.kind, ValueKindFact::Unknown)))
            {
                diagnostics.push(argument_error("RM-CATALOG-CONE-ARITY", "expected f, cone constraints, complete linear/equality/bound pairs and optional options", 0));
            }
        }
        let mut positional = request.clone();
        if has_options
            || (request.arguments.len() % 2 == 1
                && request
                    .arguments
                    .last()
                    .is_some_and(|fact| matches!(fact.kind, ValueKindFact::Unknown)))
        {
            positional.arguments.pop();
        }
        inspect_positional(&positional, &mut diagnostics);
    }
    let mut outputs = facts::solver_outputs(false);
    let mut fields = std::collections::BTreeMap::new();
    for name in ["iterations", "constrviolation"] {
        fields.insert(
            name.into(),
            facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        );
    }
    for name in ["primalfeasibility", "dualfeasibility", "dualitygap"] {
        fields.insert(
            name.into(),
            facts::double(ShapeFact::Unknown, StorageFact::Unknown),
        );
    }
    for name in ["algorithm", "linearsolver"] {
        fields.insert(name.into(), ValueFact::unknown(DynamicReason::RuntimeValue));
    }
    fields.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    outputs[3] = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    // Optimum lambda is a struct of numeric columns; terminal failure is [].
    // No unconditional structure/cell or column-shape proof spans both forms.
    outputs.push(ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut result = facts::finish(entry, request, outputs, diagnostics);
    if default_display(request) {
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

fn inspect_positional(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    facts::numeric(
        request,
        0,
        facts::OptimNumericInput::NonemptyVector,
        diagnostics,
    );
    if let Some(cones) = request.arguments.get(1) {
        let admitted = match cones.kind {
            ValueKindFact::Struct(_) => true,
            ValueKindFact::Cell(_) | ValueKindFact::OutputList(_) | ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Real && cones.shape.element_count() == Some(0)
            }
            _ => false,
        };
        if !admitted {
            diagnostics.push(argument_error(
                "RM-CATALOG-CONE-CONSTRAINT",
                "cone constraints must be descriptors, descriptor containers or empty numeric data",
                1,
            ));
        } else {
            let objective = request
                .arguments
                .first()
                .and_then(|fact| fact.shape.element_count());
            match &cones.kind {
                ValueKindFact::Struct(_) => inspect_cone_element(cones, objective, diagnostics),
                ValueKindFact::Cell(cell) => {
                    for element in &cell.elements {
                        inspect_cone_element(element, objective, diagnostics);
                    }
                    if !cell.elements_complete
                        && cones.shape.element_count().is_some_and(|count| count > 0)
                    {
                        inspect_cone_element(&cell.element, objective, diagnostics);
                    }
                }
                ValueKindFact::OutputList(sequence) => {
                    for element in &sequence.outputs {
                        inspect_cone_element(element, objective, diagnostics);
                    }
                }
                _ => {}
            }
        }
    }
    for index in [2, 4] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalSparseMatrix,
            diagnostics,
        );
    }
    for index in [3, 5, 6, 7] {
        facts::numeric(
            request,
            index,
            facts::OptimNumericInput::OptionalVector,
            diagnostics,
        );
    }
    facts::constraint_geometry(request, 0, &[(2, 3), (4, 5)], &[6, 7], diagnostics);
}

fn inspect_cone_element(
    fact: &ValueFact,
    objective: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    match &fact.kind {
        ValueKindFact::Struct(descriptor) => {
            if descriptor.elements_complete && !descriptor.elements.is_empty() {
                for fields in &descriptor.elements {
                    super::super::secondordercone::inspect_descriptor(
                        &StructFact::scalar(fields.clone(), true),
                        objective,
                        diagnostics,
                    );
                }
            } else {
                super::super::secondordercone::inspect_descriptor(
                    descriptor,
                    objective,
                    diagnostics,
                );
            }
        }
        ValueKindFact::Unknown => {}
        _ => diagnostics.push(argument_error(
            "RM-CATALOG-CONE-DESCRIPTOR",
            "cone containers require constructor descriptors",
            1,
        )),
    }
}
fn default_display(request: &CallRequest) -> bool {
    if request.arguments.len() == 1 {
        let ValueKindFact::Struct(problem) = &request.arguments[0].kind else {
            return false;
        };
        let option = problem
            .fields
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("options"))
            .map(|(_, value)| value);
        return option.map_or(problem.fields_complete, quiet_options);
    }
    if request.arguments.len().is_multiple_of(2) {
        return true;
    }
    request.arguments.last().is_some_and(quiet_options)
}
fn quiet_options(options: &ValueFact) -> bool {
    options.shape.element_count() == Some(0)
        || matches!(&options.kind,ValueKindFact::Struct(fields)
        if fields.fields_complete && !fields.fields.keys().any(|name|crate::OptimOptionField::parse(name)==Some(crate::OptimOptionField::Display)))
}
