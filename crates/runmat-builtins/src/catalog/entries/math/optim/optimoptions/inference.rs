use super::super::program_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimAlgorithm, OptimDisplay,
    OptimOptionField as Field, OptimSolver,
};
use runmat_types::*;
use std::collections::BTreeMap;

mod annealing;
mod genetic;
mod quadratic;
mod surrogate;

const FIELDS: &[Field] = &Field::ALL;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut fields = BTreeMap::new();
    let mut complete = true;
    let mut explicit_solver = false;
    let mut solver = None;
    let mut quadratic = quadratic::Selection::default();
    match request.arguments.first() {
        None => diagnostics.push(argument_error(
            "RM-CATALOG-OPTIONS-ARITY",
            "expected a solver name or existing options struct",
            0,
        )),
        Some(ValueFact {
            kind: ValueKindFact::Struct(options),
            storage: StorageFact::Scalar | StorageFact::Unknown,
            ..
        }) => {
            quadratic.merge(options);
            let has_solver = options
                .fields
                .keys()
                .any(|name| name.eq_ignore_ascii_case("Solver"));
            if options.fields_complete && !has_solver {
                solver = Some(OptimSolver::Generic);
            }
            complete = solver.is_some();
            inspect_struct(options, solver, &mut fields, &mut diagnostics, 0);
            if !options.fields_complete {
                invalidate_fields(&mut fields, solver);
            }
        }
        Some(ValueFact {
            kind: ValueKindFact::Unknown,
            ..
        }) => {
            complete = false;
        }
        Some(ValueFact {
            kind: ValueKindFact::Callable(_),
            ..
        }) => {
            explicit_solver = true;
            complete = false;
            fields.insert("Solver".into(), ValueFact::scalar(ValueKindFact::String));
        }
        Some(first) if text(first) => {
            explicit_solver = true;
            solver = request
                .literals
                .literal_string_at(0)
                .and_then(|name| OptimSolver::parse(&name));
            if request.literals.literal_string_at(0).is_some() && solver.is_none() {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-SOLVER",
                    "unsupported optimization solver",
                    0,
                ));
            }
            if let Some(solver) = solver {
                for &field in FIELDS
                    .iter()
                    .filter(|&&field| solver.accepts(field) && field != Field::HessUpdate)
                {
                    let fact = if matches!(
                        solver,
                        OptimSolver::Lsqnonlin
                            | OptimSolver::Fmincon
                            | OptimSolver::Fminunc
                            | OptimSolver::Fseminf
                            | OptimSolver::Fminimax
                            | OptimSolver::Fgoalattain
                    ) && field == Field::MaxFunEvals
                    {
                        ValueFact::unknown(DynamicReason::RuntimeValue)
                    } else {
                        normalized(field, Some(solver), None)
                    };
                    fields.insert(field.name().into(), fact);
                }
            } else {
                complete = false;
            }
            fields.insert("Solver".into(), ValueFact::scalar(ValueKindFact::String));
        }
        Some(_) => diagnostics.push(argument_error(
            "RM-CATALOG-OPTIONS-SOLVER",
            "solver must be scalar text or an options struct",
            0,
        )),
    }
    let mut index = 1;
    while index < request.arguments.len() {
        if matches!(request.arguments[index].kind, ValueKindFact::Unknown) {
            quadratic.unknown();
            // Unknown kind can be a one-argument struct merge or a name/value
            // pair, so subsequent argument positions are not statically bound.
            if !explicit_solver {
                fields.clear();
            } else {
                invalidate_fields(&mut fields, solver);
            }
            complete = false;
            break;
        }
        if let ValueKindFact::Struct(options) = &request.arguments[index].kind {
            quadratic.merge(options);
            if !matches!(
                request.arguments[index].storage,
                StorageFact::Scalar | StorageFact::Unknown
            ) || request.arguments[index]
                .shape
                .element_count()
                .is_some_and(|count| count != 1)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-STRUCT",
                    "options must be a scalar struct",
                    index,
                ));
            }
            if !explicit_solver {
                // A later struct can select a new solver and replace defaults.
                fields.clear();
                solver = None;
                complete = false;
            }
            complete &= options.fields_complete;
            if !options.fields_complete {
                invalidate_fields(&mut fields, solver);
            }
            inspect_struct(options, solver, &mut fields, &mut diagnostics, index);
            if !options.fields_complete {
                // Unknown aliases may follow a known key in runtime field order.
                invalidate_fields(&mut fields, solver);
            } else if options
                .fields
                .keys()
                .any(|name| name.eq_ignore_ascii_case("Solver"))
            {
                // A named source solver's default-valued fields may be skipped
                // at merge time; metadata does not prove those payload values.
                for field in [Field::MaxIter, Field::MaxFunEvals] {
                    if let Some(fact) = fields.get_mut(field.name()) {
                        *fact = ValueFact::scalar(ValueKindFact::Unknown);
                    }
                }
                complete = false;
            }
            index += 1;
            continue;
        }
        if !text(&request.arguments[index]) {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIONS-NAME",
                "option names must be scalar text",
                index,
            ));
        }
        let Some(value) = request.arguments.get(index + 1) else {
            diagnostics.push(argument_error(
                "RM-CATALOG-OPTIONS-VALUE",
                "option name requires a value",
                index,
            ));
            break;
        };
        if let Some(name) = request.literals.literal_string_at(index) {
            match solver.map_or_else(
                || Field::parse(&name).filter(|field| *field != Field::HessUpdate),
                |solver| Field::parse_for_constructor(&name, solver),
            ) {
                Some(field) => {
                    quadratic.record(request, field, value, index + 1);
                    inspect_value(request, field, solver, value, index + 1, &mut diagnostics);
                    fields.insert(field.name().into(), normalized(field, solver, Some(value)));
                }
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-FIELD",
                    "option is not supported by the selected solver",
                    index,
                )),
            }
        } else {
            quadratic.unknown();
            complete = false;
            invalidate_fields(&mut fields, solver);
        }
        index += 2;
    }
    if matches!(
        solver,
        Some(OptimSolver::Fminunc | OptimSolver::Fmincon | OptimSolver::Lsqnonlin)
    ) {
        fields.remove(Field::TolFun.name());
    }
    if solver == Some(OptimSolver::Quadprog) {
        quadratic.finish(&mut fields, &mut complete);
    }
    value_facts::public_names(solver, &mut fields);
    facts::finish(
        entry,
        request,
        vec![ValueFact::scalar(ValueKindFact::Struct(
            StructFact::scalar(fields, complete),
        ))],
        diagnostics,
    )
}

fn invalidate_fields(fields: &mut BTreeMap<String, ValueFact>, solver: Option<OptimSolver>) {
    for field in FIELDS.iter().filter(|&&field| {
        solver.is_none_or(|solver| solver.accepts(field) && field != Field::HessUpdate)
    }) {
        fields.remove(field.name());
    }
}

fn inspect_struct(
    source: &StructFact,
    solver: Option<OptimSolver>,
    fields: &mut BTreeMap<String, ValueFact>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
    index: usize,
) {
    for (name, value) in &source.fields {
        if name.eq_ignore_ascii_case("Solver") {
            if !text(value) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-SOLVER",
                    "options Solver field must be scalar text",
                    index,
                ));
            }
            fields.insert("Solver".into(), ValueFact::scalar(ValueKindFact::String));
            continue;
        }
        match solver.map_or_else(
            || Field::parse(name).filter(|field| *field != Field::HessUpdate),
            |solver| Field::parse_for_constructor(name, solver),
        ) {
            Some(field) => {
                if !solver_value_metadata(field, solver, value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIONS-TYPE",
                        "option value has unsupported class or geometry",
                        index,
                    ));
                }
                fields.insert(field.name().into(), normalized(field, solver, Some(value)));
            }
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-OPTIONS-FIELD",
                "unsupported option field",
                index,
            )),
        }
    }
}

mod inspect;
mod pattern;
mod semi_infinite;
use inspect::inspect_value;
mod swarm;
mod value_facts;
use value_facts::{normalized, solver_value_metadata, text};
