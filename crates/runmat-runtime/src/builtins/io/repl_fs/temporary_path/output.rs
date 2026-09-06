use std::path::Path;

use runmat_value::{CharArray, Value};

pub(super) fn character_path(path: &Path) -> Value {
    Value::CharArray(CharArray::new_row(
        &crate::builtins::common::fs::path_to_string(path),
    ))
}

pub(super) fn character_directory(path: &Path) -> Option<Value> {
    let mut text = crate::builtins::common::fs::path_to_string(path);
    if text.is_empty() {
        return None;
    }
    if !ends_with_separator(&text) {
        text.push(std::path::MAIN_SEPARATOR);
    }
    Some(Value::CharArray(CharArray::new_row(&text)))
}

fn ends_with_separator(text: &str) -> bool {
    text.chars().next_back().is_some_and(|character| {
        character == std::path::MAIN_SEPARATOR || (cfg!(windows) && matches!(character, '/' | '\\'))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;

    #[test]
    fn directory_output_adds_one_platform_separator() {
        let path = Path::new("runmat_tempdir_unit_path");
        let value = character_directory(path).expect("directory output");
        let text = String::try_from(&value).expect("character row");
        assert!(text.ends_with(std::path::MAIN_SEPARATOR));
        assert_eq!(
            text.trim_end_matches(std::path::MAIN_SEPARATOR),
            path.to_string_lossy()
        );
    }

    #[test]
    fn directory_output_rejects_an_empty_path() {
        assert!(character_directory(Path::new("")).is_none());
    }
}
