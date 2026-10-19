use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ExportMode {
    Normal { transition: bool },
    CatalogPreview,
}

pub(super) struct ExportOptions {
    pub output: Option<PathBuf>,
    pub check_only: bool,
    pub mode: ExportMode,
    pub identities: BTreeSet<String>,
}

pub(super) fn parse_options(
    mut arguments: impl Iterator<Item = OsString>,
) -> Result<ExportOptions, Box<dyn Error>> {
    let mut output = None;
    let mut check_only = false;
    let mut transition = false;
    let mut catalog_preview = false;
    let mut identities = BTreeSet::new();
    while let Some(argument) = arguments.next() {
        if argument == "--check" {
            check_only = true;
        } else if argument == "--transition" {
            transition = true;
        } else if argument == "--catalog-preview" {
            catalog_preview = true;
        } else if argument == "--output" {
            let path = arguments.next().ok_or("--output requires a path")?;
            output = Some(PathBuf::from(path));
        } else if argument == "--identity" {
            let name = arguments.next().ok_or("--identity requires a name")?;
            let name = name.into_string().map_err(|_| "identity must be UTF-8")?;
            if !identities.insert(name.to_ascii_lowercase()) {
                return Err(format!("duplicate documentation identity: {name}").into());
            }
        } else {
            return Err(
                format!("unexpected argument: {}", PathBuf::from(argument).display()).into(),
            );
        }
    }

    if check_only && output.is_some() {
        return Err("--check and --output cannot be combined".into());
    }
    if catalog_preview && (check_only || transition || identities.is_empty()) {
        return Err(
            "--catalog-preview requires --identity and excludes --check/--transition".into(),
        );
    }

    Ok(ExportOptions {
        output,
        check_only,
        mode: if catalog_preview {
            ExportMode::CatalogPreview
        } else {
            ExportMode::Normal { transition }
        },
        identities,
    })
}
