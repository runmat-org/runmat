use crate::builtins::common::path_state::PATH_LIST_SEPARATOR;

pub(super) fn split(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(PATH_LIST_SEPARATOR)
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_owned)
}

pub(super) fn join(segments: &[String]) -> String {
    segments.join(&PATH_LIST_SEPARATOR.to_string())
}

pub(super) fn identity(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('/', "\\").to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}
