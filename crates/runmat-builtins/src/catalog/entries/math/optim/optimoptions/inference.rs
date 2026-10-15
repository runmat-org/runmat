use super::super::program_facts as facts;
use crate::{
    catalog::inference::argument_error, BuiltinCatalogEntry, OptimAlgorithm, OptimDisplay,
    OptimOptionField as Field, OptimSolver,
};
use runmat_types::*;
use std::collections::BTreeMap;

const FIELDS: &[Field] = &[
    Field::TolX,
    Field::TolFun,
    Field::MaxIter,
    Field::MaxFunEvals,
    Field::Display,
    Field::Algorithm,
    Field::SpecifyObjectiveGradient,
];

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut fields = BTreeMap::new();
    let mut complete = true;
    let mut explicit_solver = false;
    let mut solver = None;
    match request.arguments.first() {
        None => diagnostics.push(argument_error(
            "RM-CATALOG-OPTIONS-ARITY",
            "expected a solver name or existing options struct",
            0,
        )),
        Some(ValueFact {
            kind: ValueKindFact::Struct(options),
            storage,
            ..
        }) if matches!(storage, StorageFact::Scalar | StorageFact::Unknown) => {
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
                for &field in FIELDS.iter().filter(|&&field| solver.accepts(field)) {
                    fields.insert(field.name().into(), normalized(field, None));
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
            match Field::parse(&name) {
                Some(field) if solver.is_none_or(|solver| solver.accepts(field)) => {
                    inspect_value(request, field, solver, value, index + 1, &mut diagnostics);
                    fields.insert(field.name().into(), normalized(field, Some(value)));
                }
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-FIELD",
                    "option is not supported by the selected solver",
                    index,
                )),
            }
        } else {
            complete = false;
            invalidate_fields(&mut fields, solver);
        }
        index += 2;
    }
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
    for field in FIELDS
        .iter()
        .filter(|&&field| solver.is_none_or(|solver| solver.accepts(field)))
    {
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
        match Field::parse(name) {
            Some(field) if solver.is_none_or(|solver| solver.accepts(field)) => {
                if !value_metadata(field, value) {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIONS-TYPE",
                        "option value has unsupported class or geometry",
                        index,
                    ));
                }
                fields.insert(field.name().into(), normalized(field, Some(value)));
            }
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-OPTIONS-FIELD",
                "unsupported option field",
                index,
            )),
        }
    }
}

fn inspect_value(
    request: &CallRequest,
    field: Field,
    solver: Option<OptimSolver>,
    value: &ValueFact,
    index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !value_metadata(field, value) {
        diagnostics.push(argument_error(
            "RM-CATALOG-OPTIONS-TYPE",
            "option value has unsupported class or geometry",
            index,
        ));
    }
    match field {
        Field::Display | Field::Algorithm => {
            if let Some(text) = request.literals.literal_string_at(index) {
                let valid = if field == Field::Display {
                    OptimDisplay::parse(&text).is_some_and(|display| {
                        solver.is_none_or(|solver| solver.accepts_display(display))
                    })
                } else {
                    OptimAlgorithm::parse(&text).is_some_and(|algorithm| {
                        solver.is_none_or(|solver| solver.accepts_algorithm(algorithm))
                    })
                };
                if !valid {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-OPTIONS-CONTROL",
                        "unsupported solver control value",
                        index,
                    ));
                }
            }
        }
        Field::TolX | Field::TolFun | Field::MaxIter | Field::MaxFunEvals => {
            if request.literals.numeric_at(index).is_some_and(|number| {
                !number.is_finite()
                    || number <= 0.0
                    || (matches!(field, Field::MaxIter | Field::MaxFunEvals)
                        && number.fract() != 0.0)
            }) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-RANGE",
                    "option must be a finite positive scalar and counts must be integral",
                    index,
                ));
            }
            if matches!(field, Field::TolX | Field::TolFun)
                && request
                    .literals
                    .literal_args
                    .get(index)
                    .is_some_and(|literal| literal.integer_is_exact_binary64() == Some(false))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-EXACT",
                    "integer tolerances must be exactly representable as binary64",
                    index,
                ));
            }
        }
        Field::SpecifyObjectiveGradient => {
            if request
                .literals
                .numeric_at(index)
                .is_some_and(|value| value != 0.0 && value != 1.0)
                || request
                    .literals
                    .literal_string_at(index)
                    .is_some_and(|text| crate::OptimSwitch::parse(&text).is_none())
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-OPTIONS-FLAG",
                    "gradient control must be logical 0/1 or a supported on/off spelling",
                    index,
                ));
            }
        }
    }
}

fn value_metadata(field: Field, value: &ValueFact) -> bool {
    if matches!(value.kind, ValueKindFact::Unknown) {
        return true;
    }
    if matches!(field, Field::Display | Field::Algorithm) {
        return text(value);
    }
    if field == Field::SpecifyObjectiveGradient && text(value) {
        return true;
    }
    value.shape.element_count().is_none_or(|count| count == 1)
        && value.storage != StorageFact::Sparse
        && match value.kind {
            ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
            ValueKindFact::Logical => {
                field == Field::SpecifyObjectiveGradient || value.storage != StorageFact::Scalar
            }
            _ => false,
        }
}

fn normalized(field: Field, source: Option<&ValueFact>) -> ValueFact {
    match field {
        Field::Display | Field::Algorithm => ValueFact::scalar(ValueKindFact::String),
        Field::SpecifyObjectiveGradient => ValueFact::scalar(ValueKindFact::Logical),
        Field::MaxIter | Field::MaxFunEvals => match source.map(|value| &value.kind) {
            Some(ValueKindFact::Numeric(numeric)) if numeric.class.integer_class().is_some() => {
                ValueFact::scalar(ValueKindFact::Numeric(*numeric))
            }
            Some(ValueKindFact::Unknown) => ValueFact::scalar(ValueKindFact::Unknown),
            _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
        },
        _ => facts::double(ShapeFact::Scalar, StorageFact::Scalar),
    }
}

fn text(value: &ValueFact) -> bool {
    match value.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::String => value.shape.element_count().is_none_or(|count| count == 1),
        ValueKindFact::Character => value.shape.known_dims().is_none_or(|dims| {
            dims.len() == 1
                || dims
                    .first()
                    .is_some_and(|rows| rows.is_none_or(|rows| rows == 1))
        }),
        _ => false,
    }
}
