use runmat_builtins::{
    BuiltinErrorDescriptor, COPYFILE_RESULT_DEST_EXISTS, COPYFILE_RESULT_DEST_MISSING,
    COPYFILE_RESULT_DEST_NOT_DIR, COPYFILE_RESULT_EMPTY_DEST, COPYFILE_RESULT_EMPTY_SOURCE,
    COPYFILE_RESULT_OS_ERROR, COPYFILE_RESULT_PATTERN_ERROR, COPYFILE_RESULT_SAME_PATH,
    COPYFILE_RESULT_SOURCE_NOT_FOUND,
};
use std::io;

use super::super::outcome::TransferOutcome;

pub(super) fn success() -> TransferOutcome {
    TransferOutcome::success()
}

pub(super) fn empty_source() -> TransferOutcome {
    failure(
        COPYFILE_RESULT_EMPTY_SOURCE.message,
        &COPYFILE_RESULT_EMPTY_SOURCE,
    )
}

pub(super) fn empty_destination() -> TransferOutcome {
    failure(
        COPYFILE_RESULT_EMPTY_DEST.message,
        &COPYFILE_RESULT_EMPTY_DEST,
    )
}

pub(super) fn source_not_found(path: &str) -> TransferOutcome {
    failure(
        format!("Source \"{path}\" does not exist."),
        &COPYFILE_RESULT_SOURCE_NOT_FOUND,
    )
}

pub(super) fn destination_exists(path: &str) -> TransferOutcome {
    failure(
        format!("Cannot copy to \"{path}\": destination already exists."),
        &COPYFILE_RESULT_DEST_EXISTS,
    )
}

pub(super) fn destination_missing(path: &str) -> TransferOutcome {
    failure(
        format!("Destination folder \"{path}\" must exist when copying multiple sources."),
        &COPYFILE_RESULT_DEST_MISSING,
    )
}

pub(super) fn destination_not_directory(path: &str) -> TransferOutcome {
    failure(
        format!("Destination \"{path}\" must refer to a folder."),
        &COPYFILE_RESULT_DEST_NOT_DIR,
    )
}

pub(super) fn same_path(path: &str) -> TransferOutcome {
    failure(
        format!("Cannot copy \"{path}\" onto itself."),
        &COPYFILE_RESULT_SAME_PATH,
    )
}

pub(super) fn invalid_pattern(pattern: &str, reason: &str) -> TransferOutcome {
    failure(
        format!("Invalid source pattern \"{pattern}\": {reason}"),
        &COPYFILE_RESULT_PATTERN_ERROR,
    )
}

pub(super) fn filesystem(source: &str, target: &str, error: &io::Error) -> TransferOutcome {
    failure(
        format!("Unable to copy \"{source}\" to \"{target}\": {error}"),
        &COPYFILE_RESULT_OS_ERROR,
    )
}

pub(super) fn failure(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> TransferOutcome {
    TransferOutcome::failure(message, error)
}
