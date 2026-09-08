use runmat_value::Tensor;

use super::destination;
use crate::builtins::cells::core::cell2mat::error::{
    cell2mat_error_with_message, CELL2MAT_ERROR_INTERNAL,
};
use crate::builtins::cells::core::cell2mat::input::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn copy(
    entries: &[CellEntry],
    plan: &AssemblyPlan,
    output: &mut Tensor,
) -> BuiltinResult<()> {
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::Numeric(storage) = &entry.data else {
            continue;
        };
        destination::for_each(entry, coordinates, plan, |source, destination| {
            let value = storage
                .value_at(source)
                .expect("validated cell storage bounds");
            output
                .set_numeric_assignment_at(destination, value)
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
