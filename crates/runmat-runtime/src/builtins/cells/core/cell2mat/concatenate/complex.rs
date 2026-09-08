use runmat_value::IntegerComplexStorage;

use super::destination;
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL,
};
use crate::builtins::cells::core::cell2mat::input::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn copy_floating(
    entries: &[CellEntry],
    plan: &AssemblyPlan,
    output: &mut [(f64, f64)],
) -> BuiltinResult<()> {
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::Complex(data) = &entry.data else {
            continue;
        };
        destination::for_each(entry, coordinates, plan, |source, destination| {
            output[destination] = data[source];
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
                .real
                .value_at(source)
                .expect("validated real storage bounds");
            let imaginary = storage
                .imag
                .value_at(source)
                .expect("validated imaginary storage bounds");
            output
                .real
                .set_value(destination, real)
                .and_then(|_| output.imag.set_value(destination, imaginary))
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
