use super::destination;
use crate::builtins::cells::core::cell2mat::input::{CellEntry, EntryData};
use crate::builtins::cells::core::cell2mat::plan::AssemblyPlan;
use crate::BuiltinResult;

pub(super) fn copy(
    entries: &[CellEntry],
    plan: &AssemblyPlan,
    output: &mut [u8],
) -> BuiltinResult<()> {
    for (entry, coordinates) in entries.iter().zip(&plan.cell_coordinates) {
        let EntryData::Logical(data) = &entry.data else {
            continue;
        };
        destination::for_each(entry, coordinates, plan, |source, destination| {
            output[destination] = data[source];
            Ok(())
        })?;
    }
    Ok(())
}
