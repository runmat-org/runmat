use thiserror::Error;

#[derive(Error, Debug)]
pub enum RunMatFuncError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serde error: {0}")]
    SerdeJson(#[from] serde_json::Error),
    #[error("Serde YAML error: {0}")]
    SerdeYaml(#[from] serde_yaml::Error),
    #[error("Other error: {0}")]
    Other(String),
}
