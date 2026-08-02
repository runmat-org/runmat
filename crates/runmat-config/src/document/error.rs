use crate::desktop::schema::DesktopConfigValidationError;

#[derive(Debug, thiserror::Error)]
pub enum RunmatConfigDocumentError {
    #[error("unsupported RunMat config extension `{0}`; expected .toml or .json")]
    UnsupportedExtension(String),
    #[error("RunMat config path must end in .toml or .json")]
    MissingExtension,
    #[error("failed to parse RunMat TOML: {0}")]
    TomlParse(String),
    #[error("failed to parse RunMat JSON: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("failed to edit RunMat TOML: {0}")]
    TomlEdit(String),
    #[error("RunMat config must be a top-level object")]
    InvalidDocumentShape,
    #[error("{0}")]
    DesktopValidation(String),
    #[error("legacy Desktop configuration cannot be migrated automatically: {0}")]
    LegacyMigration(String),
}

impl From<DesktopConfigValidationError> for RunmatConfigDocumentError {
    fn from(value: DesktopConfigValidationError) -> Self {
        Self::DesktopValidation(value.message)
    }
}
