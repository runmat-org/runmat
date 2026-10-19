use std::path::Path;

pub(super) fn display(path: &Path) -> String {
    path.display().to_string()
}
