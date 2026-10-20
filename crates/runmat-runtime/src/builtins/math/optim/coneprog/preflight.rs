//! Descriptive argument checks precede the owning numeric gather boundary.
use super::*;
use runmat_builtins::OptimNumericInput as Role;

pub(super) fn numeric(label: &str, value: &Value, role: Role) -> BuiltinResult<()> {
    if !role.accepts_metadata(&crate::value_fact::value_fact(value)) {
        return Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            format!("coneprog: {label} has unsupported real numeric class or geometry"),
        ));
    }
    Ok(())
}

pub(super) fn arguments(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if let Some(Value::Struct(problem)) = args.first().filter(|_| args.len() == 1) {
        numeric("f", &required_field(problem, "f")?, Role::NonemptyVector)?;
        if let Some(value) = optional_field(problem, "socConstraints") {
            cones(&value)?;
        }
        for (label, alias, role) in [
            ("Aineq", "A", Role::OptionalMatrix),
            ("bineq", "b", Role::OptionalVector),
            ("Aeq", "Aeq", Role::OptionalMatrix),
            ("beq", "beq", Role::OptionalVector),
            ("lb", "lb", Role::OptionalVector),
            ("ub", "ub", Role::OptionalVector),
        ] {
            if let Some(value) =
                optional_field(problem, label).or_else(|| optional_field(problem, alias))
            {
                numeric(label, &value, role)?;
            }
        }
        if let Some(value) = optional_field(problem, "options") {
            let Value::Struct(options) = value else {
                return Err(coneprog_error(
                    &ERROR_INVALID_INPUT,
                    "coneprog: options must be a struct",
                ));
            };
            controls(&options)?;
        }
        return Ok(());
    }
    if args.len() < 2 {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "coneprog: expected f and socConstraints, or a problem struct",
        ));
    }
    let mut end = args.len();
    if let Some(Value::Struct(options)) = args.last().filter(|_| args.len() > 2) {
        if is_options_struct(options) {
            controls(options)?;
            end -= 1;
        }
    }
    if !matches!(end, 2 | 4 | 6 | 8) {
        return Err(coneprog_error(
            &ERROR_INVALID_ARGUMENT,
            "coneprog: optional linear constraints must be A,b,Aeq,beq,lb,ub",
        ));
    }
    numeric(
        "f",
        args.get(0).expect("validated coneprog objective"),
        Role::NonemptyVector,
    )?;
    cones(args.get(1).expect("validated coneprog constraints"))?;
    for (index, label, role) in [
        (2, "A", Role::OptionalMatrix),
        (3, "b", Role::OptionalVector),
        (4, "Aeq", Role::OptionalMatrix),
        (5, "beq", Role::OptionalVector),
        (6, "lb", Role::OptionalVector),
        (7, "ub", Role::OptionalVector),
    ] {
        if let Some(value) = args.get(index).filter(|_| index < end) {
            numeric(label, value, role)?;
        }
    }
    Ok(())
}

fn cones(value: &Value) -> BuiltinResult<()> {
    if is_empty_value(value) {
        return Ok(());
    }
    match value {
        Value::Struct(constraint) => cone(constraint),
        Value::Cell(cell) => {
            for value in cell.data() {
                descriptor(value)?;
            }
            Ok(())
        }
        Value::OutputList(values) => {
            for value in values {
                descriptor(value)?;
            }
            Ok(())
        }
        // Only an empty numeric resident value can become the accepted [] form.
        Value::GpuTensor(_)
            if crate::value_fact::value_fact(value).shape.element_count() == Some(0) =>
        {
            numeric("socConstraints", value, Role::OptionalVector)
        }
        _ => Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "coneprog: socConstraints must be a secondordercone struct or cell array",
        )),
    }
}

fn descriptor(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Struct(constraint) => cone(constraint),
        _ => Err(coneprog_error(
            &ERROR_INVALID_INPUT,
            "coneprog: socConstraints entries must be secondordercone structs",
        )),
    }
}

fn cone(constraint: &StructValue) -> BuiltinResult<()> {
    for (field, role) in [
        ("A", Role::NonemptyMatrix),
        ("b", Role::OptionalVector),
        ("d", Role::OptionalVector),
        ("gamma", Role::Scalar),
    ] {
        numeric(field, &required_field(constraint, field)?, role)?;
    }
    Ok(())
}

fn controls(options: &StructValue) -> BuiltinResult<()> {
    // Validate every consumed host control now; resident controls are checked
    // descriptively here and decoded only after their authentic gather.
    for field in [
        runmat_builtins::OptimOptionField::TolFun,
        runmat_builtins::OptimOptionField::TolX,
        runmat_builtins::OptimOptionField::MaxIter,
    ] {
        if let Some(value) = lookup_option(options, field.name()) {
            if matches!(value, Value::GpuTensor(_)) {
                crate::builtins::math::optim::common::preflight_option_scalar(
                    CONEPROG, value, field,
                )?;
            } else if field == runmat_builtins::OptimOptionField::MaxIter {
                option_usize(CONEPROG, Some(options), field.name(), DEFAULT_MAX_ITER)?;
            } else {
                option_f64(CONEPROG, Some(options), field.name(), DEFAULT_TOL)?;
            }
        }
    }
    Ok(())
}
