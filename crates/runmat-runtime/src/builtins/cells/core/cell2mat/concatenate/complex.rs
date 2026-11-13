use runmat_value::{ComplexStorage, IntegerComplexStorage};

use super::destination;
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL, CELL2MAT_ERROR_INVALID_CONTENTS,
};
use crate::builtins::cells::core::cell2mat::input::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn copy_floating(
    entries: &[CellEntry],
    plan: &AssemblyPlan,
    output: &mut ComplexStorage,
) -> BuiltinResult<()> {
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::Complex(data) = &entry.data else {
            continue;
        };
        if data.numeric_dtype() != output.numeric_dtype() {
            return Err(cell2mat_error_with_message(
                "cell2mat: floating-point complex contents must share a class",
                &CELL2MAT_ERROR_INVALID_CONTENTS,
            ));
        }
        destination::for_each(entry, coordinates, plan, |source, destination| {
            match (&mut *output, data) {
                (ComplexStorage::F64(output), ComplexStorage::F64(input)) => {
                    output[destination] = input[source]
                }
                (ComplexStorage::F32(output), ComplexStorage::F32(input)) => {
                    output[destination] = input[source]
                }
                _ => {
                    return Err(cell2mat_error_with_message(
                        "cell2mat: floating-point complex contents must share a class",
                        &CELL2MAT_ERROR_INVALID_CONTENTS,
                    ))
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

pub(super) fn copy_integer(
    entries: &[CellEntry],
    plan: &AssemblyPlan,
    output: &mut IntegerComplexStorage,
) -> BuiltinResult<()> {
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::TypedComplexInteger(storage) = &entry.data else {
            continue;
        };
        destination::for_each(entry, coordinates, plan, |source, destination| {
            let real = storage
                .real()
                .value_at(source)
                .expect("validated real storage bounds");
            let imaginary = storage
                .imag()
                .value_at(source)
                .expect("validated imaginary storage bounds");
            output
                .set_exact_assignment(destination, real, imaginary)
                .map_err(|error| {
                    cell2mat_error_with_message(
                        format!("cell2mat: {error}"),
                        &CELL2MAT_ERROR_INTERNAL,
                    )
                })
        })?;
    }
    Ok(())
}
