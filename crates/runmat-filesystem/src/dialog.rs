use std::io;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogFilter {
    pub patterns: Vec<String>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenFileDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
    pub filters: Vec<OpenFileDialogFilter>,
    pub multiselect: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenFileDialogSelection {
    pub paths: Vec<PathBuf>,
    pub filter_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaveFileDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
    pub filters: Vec<OpenFileDialogFilter>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveFileDialogSelection {
    pub path: PathBuf,
    pub filter_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectoryDialogRequest {
    pub title: Option<String>,
    pub default_path: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectoryDialogSelection {
    pub path: PathBuf,
}

pub fn resolve_open_file_dialog_request(
    request: &OpenFileDialogRequest,
) -> io::Result<OpenFileDialogRequest> {
    let mut resolved = request.clone();
    if let Some(default_path) = resolved.default_path.as_mut() {
        *default_path = crate::capture_provider_path(default_path)?.1;
    }
    Ok(resolved)
}

pub fn resolve_save_file_dialog_request(
    request: &SaveFileDialogRequest,
) -> io::Result<SaveFileDialogRequest> {
    let mut resolved = request.clone();
    if let Some(default_path) = resolved.default_path.as_mut() {
        *default_path = crate::capture_provider_path(default_path)?.1;
    }
    Ok(resolved)
}

pub fn resolve_directory_dialog_request(
    request: &DirectoryDialogRequest,
) -> io::Result<DirectoryDialogRequest> {
    let mut resolved = request.clone();
    if let Some(default_path) = resolved.default_path.as_mut() {
        *default_path = crate::capture_provider_path(default_path)?.1;
    }
    Ok(resolved)
}
