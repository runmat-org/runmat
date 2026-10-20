use super::super::facts::*;
use crate::catalog::inference::finish_fixed;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut data = if request.arguments.is_empty() {
        empty_cell()
    } else {
        retained(request, 0)
    };
    // Runtime scans even-position name/value pairs and chooses the first Data
    // match; an unknown earlier name can change which later value wins.
    for index in (0..request.arguments.len().saturating_sub(1)).step_by(2) {
        match request.literals.literal_string_at(index) {
            Some(name) if name == "data" => {
                data = retained(request, index + 1);
                break;
            }
            Some(_) => {}
            None if matches!(
                request.arguments[index].kind,
                ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
            ) =>
            {
                data = ValueFact::unknown(DynamicReason::RuntimeValue);
                break;
            }
            None => {}
        }
    }
    finish_fixed(
        entry,
        request,
        object(
            StaticClassIdentity::new("uitable"),
            &[
                ("Data", data),
                ("ColumnName", empty_names()),
                ("RowName", empty_names()),
            ],
        ),
        Vec::new(),
    )
}
