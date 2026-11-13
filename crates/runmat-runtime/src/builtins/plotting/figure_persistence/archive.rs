mod decode;
mod encode;

pub(super) use decode::decode_figure_bundle;
pub(super) use encode::encode_figure_bundle;

use super::*;
use runmat_builtins::BuiltinErrorDescriptor;

pub(super) const FLAG_COMPRESSED: u8 = 1;

fn profile_tag(profile: FigureArchiveProfile) -> u8 {
    match profile {
        FigureArchiveProfile::Default => 0,
        FigureArchiveProfile::V6 => 1,
        FigureArchiveProfile::V7 => 2,
        FigureArchiveProfile::V73 => 3,
        FigureArchiveProfile::Compact => 4,
    }
}

fn profile_from_tag(tag: u8, builtin: &'static str) -> BuiltinResult<FigureArchiveProfile> {
    match tag {
        0 => Ok(FigureArchiveProfile::Default),
        1 => Ok(FigureArchiveProfile::V6),
        2 => Ok(FigureArchiveProfile::V7),
        3 => Ok(FigureArchiveProfile::V73),
        4 => Ok(FigureArchiveProfile::Compact),
        _ => Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: FIG archive uses an unsupported encoding profile"),
        )),
    }
}

fn payload_too_large(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    persistence_error(
        descriptor,
        builtin,
        format!("{builtin}: FIG payload exceeds maximum scene payload size"),
    )
}

fn corrupt(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_IMPORT,
        builtin,
        format!("{builtin}: corrupt FIG bundle"),
    )
}

fn truncated(builtin: &'static str) -> RuntimeError {
    persistence_error(
        &ERROR_IMPORT,
        builtin,
        format!("{builtin}: truncated FIG bundle"),
    )
}
