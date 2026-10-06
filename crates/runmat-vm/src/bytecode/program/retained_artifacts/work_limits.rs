use super::{RetainedCallableStageError, RetainedCallableStageLimits, RuntimeBoundExecutableUnit};

pub(super) fn validate(
    units: &[RuntimeBoundExecutableUnit],
    limits: RetainedCallableStageLimits,
) -> Result<(), RetainedCallableStageError> {
    check("callable_artifact.units", units.len(), limits.max_units)?;
    let (functions, instructions) =
        units
            .iter()
            .try_fold((0usize, 0usize), |(functions, instructions), unit| {
                let functions = checked_add(
                    "callable_artifact.functions",
                    functions,
                    unit.registry().functions.len(),
                    limits.max_functions,
                )?;
                let instructions = unit.registry().functions.values().try_fold(
                    instructions,
                    |instructions, function| {
                        checked_add(
                            "callable_artifact.instructions",
                            instructions,
                            function.instructions.len(),
                            limits.max_instructions,
                        )
                    },
                )?;
                Ok::<_, RetainedCallableStageError>((functions, instructions))
            })?;
    check(
        "callable_artifact.functions",
        functions,
        limits.max_functions,
    )?;
    check(
        "callable_artifact.instructions",
        instructions,
        limits.max_instructions,
    )
}

fn checked_add(
    field: &'static str,
    total: usize,
    increment: usize,
    limit: usize,
) -> Result<usize, RetainedCallableStageError> {
    let total = total
        .checked_add(increment)
        .ok_or(RetainedCallableStageError::Limit { field, limit })?;
    check(field, total, limit)?;
    Ok(total)
}

fn check(
    field: &'static str,
    actual: usize,
    limit: usize,
) -> Result<(), RetainedCallableStageError> {
    if actual > limit {
        return Err(RetainedCallableStageError::Limit { field, limit });
    }
    Ok(())
}
