use super::{codec::decode_pcode_bytes, pcode_error_with_detail};
use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::path_search::{find_file_with_extensions, path_is_directory};
use crate::BuiltinResult;
use glob::Pattern;
use runmat_builtins::{
    PCODE_ERROR_FILE_NOT_FOUND, PCODE_ERROR_PATH_RESOLVE, PCODE_ERROR_UNSUPPORTED_EXTENSION,
};
use runmat_filesystem as vfs;
use std::{
    collections::HashSet,
    fmt, io,
    path::{Component, Path, PathBuf},
};
const BUILTIN_NAME: &str = "pcode";
const SOURCE_EXTENSIONS: &[&str] = &[".m"];

#[derive(Debug)]
pub enum PcodeSourceReadError {
    Io(io::Error),
    InvalidPcode(String),
}

impl fmt::Display for PcodeSourceReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PcodeSourceReadError::Io(err) => write!(f, "{err}"),
            PcodeSourceReadError::InvalidPcode(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for PcodeSourceReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PcodeSourceReadError::Io(err) => Some(err),
            PcodeSourceReadError::InvalidPcode(_) => None,
        }
    }
}

impl From<io::Error> for PcodeSourceReadError {
    fn from(err: io::Error) -> Self {
        PcodeSourceReadError::Io(err)
    }
}

pub async fn prefer_pcode_source_path(
    path: &Path,
) -> Result<PathBuf, crate::builtins::common::path_search::PathSearchError> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
    {
        let pcode_path = path.with_extension("p");
        if crate::builtins::common::path_search::path_is_file(&pcode_path).await? {
            return Ok(pcode_path);
        }
    }
    Ok(path.to_path_buf())
}

pub async fn read_source_text_async(path: &Path) -> Result<String, PcodeSourceReadError> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("p"))
    {
        let bytes = vfs::read_async(path).await?;
        decode_pcode_bytes(&bytes).map_err(PcodeSourceReadError::InvalidPcode)
    } else {
        vfs::read_to_string_async(path)
            .await
            .map_err(PcodeSourceReadError::Io)
    }
}

pub(super) async fn resolve_sources(items: &[String]) -> BuiltinResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        for source in resolve_item(item).await? {
            let key = vfs::canonicalize_async(&source)
                .await
                .unwrap_or_else(|_| source.clone());
            if seen.insert(key) {
                out.push(source);
            }
        }
    }
    Ok(out)
}

async fn resolve_item(item: &str) -> BuiltinResult<Vec<PathBuf>> {
    if item_contains_wildcards(item) {
        return resolve_wildcard_item(item).await;
    }
    let expanded = expand_user_path(item, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|err| pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, err))?;
    if path_is_directory(&expanded)
        .await
        .map_err(|error| error.into_builtin_error(BUILTIN_NAME, &PCODE_ERROR_PATH_RESOLVE))?
    {
        return resolve_folder_item(&expanded).await;
    }
    if has_unsupported_extension(&expanded) {
        return Err(pcode_error_with_detail(
            &PCODE_ERROR_UNSUPPORTED_EXTENSION,
            item,
        ));
    }
    let Some(path) = find_file_with_extensions(item, SOURCE_EXTENSIONS, BUILTIN_NAME)
        .await
        .map_err(|error| error.into_builtin_error(BUILTIN_NAME, &PCODE_ERROR_PATH_RESOLVE))?
    else {
        return Err(pcode_error_with_detail(&PCODE_ERROR_FILE_NOT_FOUND, item));
    };
    if !is_m_file(&path) {
        return Err(pcode_error_with_detail(
            &PCODE_ERROR_UNSUPPORTED_EXTENSION,
            path.display().to_string(),
        ));
    }
    Ok(vec![path])
}

async fn resolve_folder_item(folder: &Path) -> BuiltinResult<Vec<PathBuf>> {
    let mut sources = Vec::new();
    let entries = vfs::read_dir_async(folder).await.map_err(|err| {
        pcode_error_with_detail(
            &PCODE_ERROR_PATH_RESOLVE,
            format!("{} ({err})", folder.display()),
        )
    })?;
    for entry in entries {
        if !entry.is_dir() && is_m_file(entry.path()) {
            sources.push(entry.path().to_path_buf());
        }
    }
    sources.sort();
    Ok(sources)
}

async fn resolve_wildcard_item(pattern_text: &str) -> BuiltinResult<Vec<PathBuf>> {
    let expanded = expand_user_path(pattern_text, BUILTIN_NAME)
        .map(PathBuf::from)
        .map_err(|err| pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, err))?;
    let parent = expanded.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = parent
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_pattern = expanded
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, pattern_text))?;
    let pattern = Pattern::new(file_pattern)
        .map_err(|err| pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, err.to_string()))?;
    let entries = vfs::read_dir_async(&dir).await.map_err(|err| {
        pcode_error_with_detail(
            &PCODE_ERROR_PATH_RESOLVE,
            format!("{} ({err})", dir.display()),
        )
    })?;
    let mut sources = Vec::new();
    for entry in entries {
        let Some(name) = entry.file_name().to_str() else {
            continue;
        };
        if !entry.is_dir() && is_m_file(entry.path()) && pattern.matches(name) {
            sources.push(entry.path().to_path_buf());
        }
    }
    sources.sort();
    if sources.is_empty() {
        return Err(pcode_error_with_detail(
            &PCODE_ERROR_FILE_NOT_FOUND,
            pattern_text,
        ));
    }
    Ok(sources)
}

pub(super) fn destination_for_source(source: &Path, inplace: bool) -> BuiltinResult<PathBuf> {
    let mut filename = source
        .file_name()
        .ok_or_else(|| {
            pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, source.display().to_string())
        })?
        .to_os_string();
    filename = PathBuf::from(filename).with_extension("p").into_os_string();

    if inplace {
        return Ok(source.with_file_name(filename));
    }

    let mut destination = vfs::current_dir()
        .map_err(|err| pcode_error_with_detail(&PCODE_ERROR_PATH_RESOLVE, err.to_string()))?;
    for component in package_or_class_suffix(source.parent()) {
        destination.push(component);
    }
    destination.push(filename);
    Ok(destination)
}

fn package_or_class_suffix(parent: Option<&Path>) -> Vec<String> {
    let Some(parent) = parent else {
        return Vec::new();
    };
    let mut current_run = Vec::new();
    for component in parent.components() {
        let Component::Normal(part) = component else {
            current_run.clear();
            continue;
        };
        let Some(part) = part.to_str() else {
            current_run.clear();
            continue;
        };
        if part.starts_with('+') || part.starts_with('@') {
            current_run.push(part.to_string());
        } else {
            current_run.clear();
        }
    }
    current_run
}

fn item_contains_wildcards(item: &str) -> bool {
    item.contains('*') || item.contains('?') || item.contains('[')
}

fn has_unsupported_extension(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    !extension.eq_ignore_ascii_case("m")
}

fn is_m_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("m"))
}
