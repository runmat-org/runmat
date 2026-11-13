use super::super::program_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimOptionField, OptimSolver,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut arguments = request.arguments.clone();
    let problem = arguments.len() == 1;
    if problem {
        match &request.arguments[0].kind {
            ValueKindFact::Struct(s) => {
                if request.arguments[0]
                    .shape
                    .element_count()
                    .is_some_and(|n| n != 1)
                {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-MIP-PROBLEM",
                        "problem must be scalar",
                        0,
                    ));
                }
                for required in ["f", "intcon", "solver", "options"] {
                    if s.fields_complete && !s.fields.contains_key(required) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-MIP-PROBLEM",
                            "problem requires f,intcon,solver and options",
                            0,
                        ));
                    }
                }
                arguments = [
                    "f", "intcon", "Aineq", "bineq", "Aeq", "beq", "lb", "ub", "x0", "options",
                ]
                .into_iter()
                .map(|name| {
                    s.fields
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
                })
                .collect();
            }
            ValueKindFact::Unknown => arguments.clear(),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-PROBLEM",
                    "single argument requires a scalar problem struct",
                    0,
                ));
                arguments.clear();
            }
        }
    } else if !matches!(arguments.len(), 4 | 6 | 8 | 9 | 10) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MIP-ARITY",
            "expected complete positional pairs, optional x0/options, or a problem struct",
            0,
        ));
    }
    for (i, value) in arguments.iter().take(9).enumerate() {
        if i == 1
            && matches!(&value.kind,ValueKindFact::Object(object)
            if object.runtime_class.as_ref().is_none_or(|class| class.is(standard::INTEGER_CONSTRAINT)))
        {
            if value.shape.element_count().is_some_and(|count| count != 1) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-DOMAIN",
                    "IntegerConstraint must be scalar",
                    i,
                ));
            }
            // Writable property facts do not certify a domain. Runtime admits
            // the exact current value generation through the shared schema.
        } else if i == 3 {
            match &value.kind {
                ValueKindFact::Cell(cell) => {
                    if value.shape.element_count().is_some_and(|n| n != 2) {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-MIP-RANGE",
                            "range b must be a two-element cell {bl,bu}",
                            i,
                        ));
                    }
                    for side in &cell.elements {
                        numeric(side, i, true, false, &mut diagnostics);
                    }
                }
                _ => numeric(value, i, true, false, &mut diagnostics),
            }
        } else {
            numeric(value, i, i == 5, matches!(i, 2 | 4), &mut diagnostics);
        }
    }
    if let Some(ValueFact {
        kind: ValueKindFact::Struct(options),
        ..
    }) = arguments.get(9)
    {
        for (name, value) in &options.fields {
            if name.eq_ignore_ascii_case("Solver") {
                continue;
            }
            let field = OptimOptionField::parse_for_solver(name, OptimSolver::Intlinprog);
            let empty = value.shape.element_count() == Some(0);
            if OptimOptionField::parse(name).is_none() || field.is_none() && !empty {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-OPTION",
                    "unknown or unsupported nonempty intlinprog option",
                    9,
                ));
            }
            if field.is_some_and(|f| {
                matches!(f, OptimOptionField::OutputFcn | OptimOptionField::PlotFcn)
            }) && !super::super::milp_observers::callback_metadata(value)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-MIP-CALLBACK",
                    "MILP callbacks require handles, selected predefined names, cells of those callbacks, or []",
                    9,
                ));
            }
        }
    } else if arguments.get(9).is_some_and(|v| {
        !matches!(v.kind, ValueKindFact::Unknown) && v.shape.element_count() != Some(0)
    }) {
        diagnostics.push(argument_error(
            "RM-CATALOG-MIP-OPTION",
            "options must be a scalar struct or []",
            9,
        ));
    }
    if problem {
        for diagnostic in &mut diagnostics {
            diagnostic.argument = Some(0);
        }
    }
    let real = || facts::double(ShapeFact::Unknown, StorageFact::Unknown);
    let scalar = || facts::double(ShapeFact::Scalar, StorageFact::Scalar);
    let mut fields = std::collections::BTreeMap::new();
    for name in ["relativegap", "absolutegap", "numfeaspoints", "numnodes"] {
        fields.insert(name.into(), real());
    }
    for name in ["constrviolation", "lowerbound", "certifiedobjective"] {
        fields.insert(name.into(), scalar());
    }
    fields.insert("message".into(), ValueFact::scalar(ValueKindFact::String));
    let callbacks_empty = (!problem && arguments.len() < 10)
        || arguments
            .get(9)
            .is_some_and(|value| value.shape.element_count() == Some(0))
        || arguments.get(9).is_some_and(|value| match &value.kind {
            ValueKindFact::Struct(options) if options.fields_complete => {
                options.fields.iter().all(|(name, value)| {
                    !matches!(
                        OptimOptionField::parse_for_solver(name, OptimSolver::Intlinprog),
                        Some(OptimOptionField::OutputFcn | OptimOptionField::PlotFcn)
                    ) || value.shape.element_count() == Some(0)
                })
            }
            _ => false,
        });
    let mut result = facts::finish(
        entry,
        request,
        vec![
            facts::double(ShapeFact::Unknown, StorageFact::Dense),
            real(),
            scalar(),
            ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true))),
        ],
        diagnostics,
    );
    if callbacks_empty {
        result.effects.0.remove(&EffectKind::Unknown);
    }
    result
}
fn numeric(
    value: &ValueFact,
    index: usize,
    full_vector: bool,
    matrix: bool,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let real = matches!(value.kind, ValueKindFact::Unknown)
        || matches!(value.kind,ValueKindFact::Numeric(n) if n.domain==NumericDomain::Real);
    let invalid_shape = (full_vector || matrix) && value.shape.rank().is_some_and(|n| n > 2)
        || full_vector
            && value.shape.known_dims().is_some_and(|dims| {
                dims.len() == 2 && dims[0].is_some_and(|n| n > 1) && dims[1].is_some_and(|n| n > 1)
            });
    if !real || invalid_shape || full_vector && value.storage == StorageFact::Sparse {
        diagnostics.push(argument_error(
            "RM-CATALOG-MIP-NUMERIC",
            "unsupported real numeric class, storage or role geometry",
            index,
        ));
    }
}
