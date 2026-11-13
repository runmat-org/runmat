mod load;
mod save;

pub(super) use load::{parse_hgload_args, parse_openfig_args};
pub(super) use save::{parse_saveas_args, parse_savefig_args};

use super::*;

fn typed_integer_handle_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn parse_archive_profile(
    value: &Value,
    builtin: &'static str,
) -> BuiltinResult<FigureArchiveProfile> {
    let token = scalar_string(value, builtin)?;
    match token.trim().to_ascii_lowercase().as_str() {
        "-v6" | "v6" => Ok(FigureArchiveProfile::V6),
        "-v7" | "v7" => Ok(FigureArchiveProfile::V7),
        "-v7.3" | "v7.3" => Ok(FigureArchiveProfile::V73),
        "compact" => Ok(FigureArchiveProfile::Compact),
        other => Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: unsupported FIG version option '{other}'"),
        )),
    }
}
