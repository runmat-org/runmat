use std::path::PathBuf;

use runmat_builtins::{TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE, TEMPNAME_ERROR_TOO_MANY_INPUTS};
use runmat_value::Value;

use super::super::{error, output};
use super::{generator, input};

const IDENTITY: &str = "tempname";

pub(super) async fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let base = match args.as_slice() {
        [] => default_folder()?,
        [folder] => input::folder(folder)?,
        _ => return Err(error::builtin(IDENTITY, &TEMPNAME_ERROR_TOO_MANY_INPUTS)),
    };
    generator::unused_path(&base)
        .await
        .map(|path| output::character_path(&path))
}

fn default_folder() -> crate::BuiltinResult<PathBuf> {
    let folder = crate::builtins::common::env::temp_dir();
    if folder.as_os_str().is_empty() {
        Err(error::builtin(
            IDENTITY,
            &TEMPNAME_ERROR_TEMP_DIR_UNAVAILABLE,
        ))
    } else {
        Ok(folder)
    }
}
