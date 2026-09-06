use runmat_builtins::{
    BuiltinErrorDescriptor, MOVEFILE_RESULT_DEST_EXISTS, MOVEFILE_RESULT_DEST_MISSING,
    MOVEFILE_RESULT_DEST_NOT_DIR, MOVEFILE_RESULT_EMPTY_DEST, MOVEFILE_RESULT_EMPTY_SOURCE,
    MOVEFILE_RESULT_OS_ERROR, MOVEFILE_RESULT_PATTERN_ERROR, MOVEFILE_RESULT_SOURCE_NOT_FOUND,
};
use std::io;

use super::super::outcome::TransferOutcome;

pub(super) fn success() -> TransferOutcome {
    TransferOutcome::success()
}

pub(super) fn empty_source() -> TransferOutcome {
    failure(
        MOVEFILE_RESULT_EMPTY_SOURCE.message,
        &MOVEFILE_RESULT_EMPTY_SOURCE,
    )
}

pub(super) fn empty_destination() -> TransferOutcome {
    failure(
        MOVEFILE_RESULT_EMPTY_DEST.message,
        &MOVEFILE_RESULT_EMPTY_DEST,
    )
}

pub(super) fn source_not_found(path: &str) -> TransferOutcome {
    failure(
        format!("Source \"{path}\" does not exist."),
        &MOVEFILE_RESULT_SOURCE_NOT_FOUND,
    )
}

pub(super) fn destination_exists(path: &str) -> TransferOutcome {
    failure(
        format!("Cannot move to \"{path}\": destination already exists."),
        &MOVEFILE_RESULT_DEST_EXISTS,
    )
}

pub(super) fn destination_missing(path: &str) -> TransferOutcome {
    failure(
        format!("Destination folder \"{path}\" must exist when moving multiple sources."),
        &MOVEFILE_RESULT_DEST_MISSING,
    )
}

pub(super) fn destination_not_directory(path: &str) -> TransferOutcome {
    failure(
        format!("Destination \"{path}\" must refer to a folder."),
        &MOVEFILE_RESULT_DEST_NOT_DIR,
    )
}

pub(super) fn invalid_pattern(pattern: &str, reason: &str) -> TransferOutcome {
    failure(
        format!("Invalid source pattern \"{pattern}\": {reason}"),
        &MOVEFILE_RESULT_PATTERN_ERROR,
    )
}

pub(super) fn filesystem(source: &str, target: &str, error: &io::Error) -> TransferOutcome {
    failure(
        format!("Unable to move \"{source}\" to \"{target}\": {error}"),
        &MOVEFILE_RESULT_OS_ERROR,
    )
}

fn failure(message: impl Into<String>, error: &'static BuiltinErrorDescriptor) -> TransferOutcome {
    TransferOutcome::failure(message, error)
}
