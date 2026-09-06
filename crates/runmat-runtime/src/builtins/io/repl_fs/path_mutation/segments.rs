use crate::builtins::common::path_state::PATH_LIST_SEPARATOR;

pub(in crate::builtins::io::repl_fs) fn split(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(PATH_LIST_SEPARATOR)
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_owned)
}

pub(in crate::builtins::io::repl_fs) fn join(segments: &[String]) -> String {
    segments.join(&PATH_LIST_SEPARATOR.to_string())
}

pub(in crate::builtins::io::repl_fs) fn identity(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('/', "\\").to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}
