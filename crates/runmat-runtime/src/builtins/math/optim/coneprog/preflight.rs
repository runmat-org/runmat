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
            ("Aineq", "A", Role::OptionalSparseMatrix),
            ("bineq", "b", Role::OptionalVector),
            ("Aeq", "Aeq", Role::OptionalSparseMatrix),
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
            if is_empty_value(&value) {
                return Ok(());
            }
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
    if args.len() > 2 && args.len() % 2 == 1 {
        match args.last().expect("nonempty arguments") {
            Value::Struct(options) => controls(options)?,
            value if is_empty_value(value) => {}
            _ => {
                return Err(coneprog_error(
                    &ERROR_INVALID_ARGUMENT,
                    "coneprog: options must be a scalar struct or []",
                ))
            }
        }
        end -= 1;
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
        (2, "A", Role::OptionalSparseMatrix),
        (3, "b", Role::OptionalVector),
        (4, "Aeq", Role::OptionalSparseMatrix),
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
        Value::StructArray(array) => {
            for element in array.elements() {
                cone(&element.to_owned())?;
            }
            Ok(())
        }
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
        ("A", Role::NonemptySparseMatrix),
        ("b", Role::OptionalVector),
        ("d", Role::OptionalVector),
        ("gamma", Role::Scalar),
    ] {
        numeric(field, &required_field(constraint, field)?, role)?;
    }
    Ok(())
}

fn controls(options: &StructValue) -> BuiltinResult<()> {
    options::Controls::validate(options)
}
