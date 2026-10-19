//! Preserve builder grammar and host diagnostics before any option readback.
use super::*;

pub(super) fn arguments(args: &[Value]) -> BuiltinResult<()> {
    let Some(first) = args.first() else {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions: expected a solver name or options struct",
        ));
    };
    let explicit = !matches!(first, Value::Struct(_));
    let mut solver = match first {
        Value::Struct(options) => {
            let solver = solver_from_options(options)?;
            fields(options, solver)?;
            solver
        }
        value => parse_solver(value)?,
    };
    let mut index = 1;
    while let Some(argument) = args.get(index) {
        if let Value::Struct(options) = argument {
            let next = solver_from_options(options)?;
            if !explicit && next != Solver::Generic {
                solver = next;
            }
            fields(options, solver)?;
            index += 1;
        } else {
            let name = expect_string_scalar(
                argument,
                "optimoptions: option names must be character vectors or string scalars",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
            )?;
            let Some(value) = args.get(index + 1) else {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
                    format!("optimoptions: missing value for option '{name}'"),
                ));
            };
            value_for_field(solver, admitted_option_field(solver, &name)?, value)?;
            index += 2;
        }
    }
    Ok(())
}

fn fields(options: &StructValue, solver: Solver) -> BuiltinResult<()> {
    for (name, value) in &options.fields {
        if name.eq_ignore_ascii_case("Solver") {
            continue;
        }
        value_for_field(solver, admitted_option_field(solver, name)?, value)?;
    }
    Ok(())
}

fn value_for_field(solver: Solver, field: OptimOptionField, value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        let fact = crate::value_fact::value_fact(value);
        let scalar_real = fact.shape.element_count() == Some(1)
            && (matches!(fact.kind, runmat_types::ValueKindFact::Numeric(numeric)
                if numeric.domain == runmat_types::NumericDomain::Real)
                || matches!(fact.kind, runmat_types::ValueKindFact::Logical));
        if !scalar_real
            || matches!(
                field,
                OptimOptionField::Display | OptimOptionField::Algorithm
            )
        {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!(
                    "optimoptions: option {} has unsupported scalar class or geometry",
                    field.name()
                ),
            ));
        }
    } else {
        normalized_option_value(solver, field, value)?;
    }
    Ok(())
}
