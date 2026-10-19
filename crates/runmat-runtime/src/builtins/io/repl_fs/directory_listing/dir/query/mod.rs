use crate::builtins::common::fs::compare_names;
use crate::BuiltinResult;

use super::input::Input;
use super::record::Record;

mod authority;
mod directory;
mod selection;

use authority::DirFilesystem;
use directory::directory;
use selection::{folder_pattern, from_text};

pub(super) async fn execute(input: Input) -> BuiltinResult<Vec<Record>> {
    let fs = DirFilesystem::capture()?;
    let mut records = match input {
        Input::Current => directory(&fs, fs.current_directory(), true).await?,
        Input::Name(name) => from_text(&fs, &name).await?,
        Input::FolderPattern { folder, pattern } => folder_pattern(&fs, &folder, &pattern).await?,
    };
    fs.check()
        .map_err(|error| super::error::io(format!("dir: {error}"), error))?;
    records.sort_by(|left, right| compare_names(&left.name, &right.name));
    Ok(records)
}
