use super::RunmatConfigDocumentError;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunmatConfigFormat {
    Toml,
    Json,
}

impl RunmatConfigFormat {
    pub fn from_path(path: &Path) -> Result<Self, RunmatConfigDocumentError> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("toml") => Ok(Self::Toml),
            Some("json") => Ok(Self::Json),
            Some(extension) => Err(RunmatConfigDocumentError::UnsupportedExtension(
                extension.to_string(),
            )),
            None => Err(RunmatConfigDocumentError::MissingExtension),
        }
    }
}
