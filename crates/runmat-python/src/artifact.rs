use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{
    PythonExecutionMode, PythonInstallation, PythonVersion, PYTHON_ADAPTER_ID,
    PYTHON_ADAPTER_VERSION,
};

pub const PYTHON_WHEEL_MEDIA_TYPE: &str = "application/vnd.python.wheel";
pub const PYTHON_ARTIFACT_BUNDLE_MEDIA_TYPE: &str = "application/vnd.runmat.python-artifact-bundle";
pub const PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION: u16 = 1;
const CANONICAL_MAGIC: &[u8; 16] = b"runmat-python-v1";
const CANONICAL_HEADER_BYTES: usize = CANONICAL_MAGIC.len() + std::mem::size_of::<u64>();
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 4_096;
const MAX_TOTAL_BYTES: usize = 1024 * 1024 * 1024;
const MAX_FILES: usize = 262_144;
const MAX_EXPANDED_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonEnvironmentIdentity {
    pub implementation: String,
    pub version: PythonVersion,
    pub abi_tag: String,
    pub platform_tag: String,
    pub execution_mode: PythonExecutionMode,
}

impl PythonEnvironmentIdentity {
    pub fn from_installation(
        installation: &PythonInstallation,
        execution_mode: PythonExecutionMode,
    ) -> Self {
        Self {
            implementation: installation.implementation.clone(),
            version: installation.version,
            abi_tag: installation.abi_tag.clone(),
            platform_tag: installation.platform_tag.clone(),
            execution_mode,
        }
    }

    pub fn artifact_identity(&self) -> String {
        format!(
            "{PYTHON_ADAPTER_ID}:v{PYTHON_ADAPTER_VERSION}:env:{}-{}:{}:{}:{}",
            self.implementation,
            self.version,
            self.abi_tag,
            self.platform_tag,
            match self.execution_mode {
                PythonExecutionMode::InProcess => "in-process",
                PythonExecutionMode::OutOfProcess => "out-of-process",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PythonArtifactIdentity(String);

impl PythonArtifactIdentity {
    pub fn for_wheel(logical_name: &str, module: &str, bytes: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"runmat-python-wheel-v1\0");
        digest.update(logical_name.as_bytes());
        digest.update(b"\0");
        digest.update(module.as_bytes());
        digest.update(b"\0");
        digest.update(bytes);
        Self(format!(
            "{PYTHON_ADAPTER_ID}:v{PYTHON_ADAPTER_VERSION}:wheel:sha256:{}",
            hex(digest.finalize())
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PythonArtifactIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonArtifactBundleEntry {
    pub logical_name: String,
    pub module: String,
    pub filename: String,
    pub identity: PythonArtifactIdentity,
    pub bytes: Vec<u8>,
}

impl PythonArtifactBundleEntry {
    pub fn wheel(
        logical_name: impl Into<String>,
        module: impl Into<String>,
        filename: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Self {
        let logical_name = logical_name.into();
        let module = module.into();
        Self {
            identity: PythonArtifactIdentity::for_wheel(&logical_name, &module, &bytes),
            logical_name,
            module,
            filename: filename.into(),
            bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonArtifactBundle {
    pub schema_version: u16,
    pub environment: Option<PythonEnvironmentIdentity>,
    pub artifacts: Vec<PythonArtifactBundleEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalBundleManifest {
    schema_version: u16,
    environment: Option<PythonEnvironmentIdentity>,
    artifacts: Vec<CanonicalArtifactManifest>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalArtifactManifest {
    logical_name: String,
    module: String,
    filename: String,
    identity: PythonArtifactIdentity,
    byte_length: u64,
}

impl PythonArtifactBundle {
    pub fn empty() -> Self {
        Self {
            schema_version: PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            environment: None,
            artifacts: Vec::new(),
        }
    }

    pub fn new(
        environment: PythonEnvironmentIdentity,
        mut artifacts: Vec<PythonArtifactBundleEntry>,
    ) -> Result<Self, PythonArtifactError> {
        artifacts.sort_by(|left, right| left.logical_name.cmp(&right.logical_name));
        let bundle = Self {
            schema_version: PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            environment: Some(environment),
            artifacts,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn artifacts_only(
        mut artifacts: Vec<PythonArtifactBundleEntry>,
    ) -> Result<Self, PythonArtifactError> {
        artifacts.sort_by(|left, right| left.logical_name.cmp(&right.logical_name));
        let bundle = Self {
            schema_version: PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            environment: None,
            artifacts,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn artifact_identities(&self) -> BTreeSet<String> {
        self.environment
            .iter()
            .map(PythonEnvironmentIdentity::artifact_identity)
            .chain(
                self.artifacts
                    .iter()
                    .map(|artifact| artifact.identity.to_string()),
            )
            .collect()
    }

    pub fn validate(&self) -> Result<(), PythonArtifactError> {
        if self.schema_version != PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION {
            return Err(PythonArtifactError::Invalid(format!(
                "unsupported Python artifact bundle schema {}",
                self.schema_version
            )));
        }
        if let Some(environment) = &self.environment {
            for (label, value) in [
                ("implementation", environment.implementation.as_str()),
                ("ABI tag", environment.abi_tag.as_str()),
                ("platform tag", environment.platform_tag.as_str()),
            ] {
                validate_token(label, value)?;
            }
            if environment.implementation != "cpython" || environment.version.major != 3 {
                return Err(PythonArtifactError::Invalid(
                    "Python artifact environment must identify CPython 3".into(),
                ));
            }
        }
        if self.artifacts.len() > MAX_ARTIFACTS {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle exceeds its entry limit".into(),
            ));
        }
        if self
            .artifacts
            .windows(2)
            .any(|pair| pair[0].logical_name >= pair[1].logical_name)
        {
            return Err(PythonArtifactError::Invalid(
                "Python artifacts must be sorted and unique by logical name".into(),
            ));
        }
        let mut identities = BTreeSet::new();
        let mut total = 0usize;
        for artifact in &self.artifacts {
            validate_logical_name(&artifact.logical_name)?;
            validate_module(&artifact.module)?;
            if Path::new(&artifact.filename)
                .file_name()
                .and_then(|name| name.to_str())
                != Some(artifact.filename.as_str())
                || !artifact.filename.ends_with(".whl")
            {
                return Err(PythonArtifactError::Invalid(format!(
                    "Python artifact `{}` must have a plain .whl filename",
                    artifact.logical_name
                )));
            }
            total = total.checked_add(artifact.bytes.len()).ok_or_else(|| {
                PythonArtifactError::Invalid("Python artifact byte total overflowed".into())
            })?;
            if total > MAX_TOTAL_BYTES {
                return Err(PythonArtifactError::Invalid(
                    "Python artifact bundle exceeds its byte limit".into(),
                ));
            }
            if artifact.identity
                != PythonArtifactIdentity::for_wheel(
                    &artifact.logical_name,
                    &artifact.module,
                    &artifact.bytes,
                )
            {
                return Err(PythonArtifactError::Invalid(format!(
                    "Python artifact `{}` does not match its identity",
                    artifact.logical_name
                )));
            }
            validate_wheel(artifact)?;
            if !identities.insert(&artifact.identity) {
                return Err(PythonArtifactError::Invalid(
                    "Python artifact identity appears more than once".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PythonArtifactError> {
        self.validate()?;
        let manifest = CanonicalBundleManifest {
            schema_version: self.schema_version,
            environment: self.environment.clone(),
            artifacts: self
                .artifacts
                .iter()
                .map(|artifact| {
                    Ok(CanonicalArtifactManifest {
                        logical_name: artifact.logical_name.clone(),
                        module: artifact.module.clone(),
                        filename: artifact.filename.clone(),
                        identity: artifact.identity.clone(),
                        byte_length: u64::try_from(artifact.bytes.len()).map_err(|_| {
                            PythonArtifactError::Encoding(
                                "Python artifact length exceeds u64".into(),
                            )
                        })?,
                    })
                })
                .collect::<Result<_, PythonArtifactError>>()?,
        };
        let manifest = serde_json::to_vec(&manifest)
            .map_err(|error| PythonArtifactError::Encoding(error.to_string()))?;
        if manifest.len() > MAX_MANIFEST_BYTES {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle manifest exceeds its byte limit".into(),
            ));
        }
        let artifact_bytes = self.artifacts.iter().try_fold(0usize, |total, artifact| {
            total.checked_add(artifact.bytes.len()).ok_or_else(|| {
                PythonArtifactError::Encoding("Python artifact byte total overflowed".into())
            })
        })?;
        let capacity = CANONICAL_HEADER_BYTES
            .checked_add(manifest.len())
            .and_then(|value| value.checked_add(artifact_bytes))
            .ok_or_else(|| {
                PythonArtifactError::Encoding("Python artifact bundle length overflowed".into())
            })?;
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(CANONICAL_MAGIC);
        output.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        output.extend_from_slice(&manifest);
        for artifact in &self.artifacts {
            output.extend_from_slice(&artifact.bytes);
        }
        Ok(output)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, PythonArtifactError> {
        if bytes.len() < CANONICAL_HEADER_BYTES
            || bytes.get(..CANONICAL_MAGIC.len()) != Some(CANONICAL_MAGIC)
        {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle has an invalid canonical header".into(),
            ));
        }
        let manifest_length = u64::from_le_bytes(
            bytes[CANONICAL_MAGIC.len()..CANONICAL_HEADER_BYTES]
                .try_into()
                .map_err(|_| {
                    PythonArtifactError::Invalid(
                        "Python artifact bundle has a truncated manifest length".into(),
                    )
                })?,
        );
        let manifest_length = usize::try_from(manifest_length).map_err(|_| {
            PythonArtifactError::Invalid(
                "Python artifact bundle manifest length exceeds this host".into(),
            )
        })?;
        if manifest_length > MAX_MANIFEST_BYTES {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle manifest exceeds its byte limit".into(),
            ));
        }
        let manifest_end = CANONICAL_HEADER_BYTES
            .checked_add(manifest_length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                PythonArtifactError::Invalid(
                    "Python artifact bundle has a truncated manifest".into(),
                )
            })?;
        let manifest: CanonicalBundleManifest =
            serde_json::from_slice(&bytes[CANONICAL_HEADER_BYTES..manifest_end])
                .map_err(|error| PythonArtifactError::Encoding(error.to_string()))?;
        if manifest.schema_version != PYTHON_ARTIFACT_BUNDLE_SCHEMA_VERSION
            || manifest.artifacts.len() > MAX_ARTIFACTS
        {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle manifest has unsupported bounds or schema".into(),
            ));
        }
        let declared_bytes = manifest.artifacts.iter().try_fold(
            0usize,
            |total, artifact| -> Result<usize, PythonArtifactError> {
                let byte_length = usize::try_from(artifact.byte_length).map_err(|_| {
                    PythonArtifactError::Invalid(format!(
                        "Python artifact `{}` length exceeds this host",
                        artifact.logical_name
                    ))
                })?;
                total.checked_add(byte_length).ok_or_else(|| {
                    PythonArtifactError::Invalid("Python artifact byte total overflowed".into())
                })
            },
        )?;
        let payload_end = manifest_end.checked_add(declared_bytes).ok_or_else(|| {
            PythonArtifactError::Invalid("Python artifact bundle length overflowed".into())
        })?;
        if declared_bytes > MAX_TOTAL_BYTES || bytes.len() != payload_end {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle payload length is invalid".into(),
            ));
        }
        let mut offset = manifest_end;
        let mut artifacts = Vec::with_capacity(manifest.artifacts.len());
        for artifact in manifest.artifacts {
            let byte_length = usize::try_from(artifact.byte_length).map_err(|_| {
                PythonArtifactError::Invalid(format!(
                    "Python artifact `{}` length exceeds this host",
                    artifact.logical_name
                ))
            })?;
            let end = offset
                .checked_add(byte_length)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| {
                    PythonArtifactError::Invalid(format!(
                        "Python artifact `{}` payload is truncated",
                        artifact.logical_name
                    ))
                })?;
            artifacts.push(PythonArtifactBundleEntry {
                logical_name: artifact.logical_name,
                module: artifact.module,
                filename: artifact.filename,
                identity: artifact.identity,
                bytes: bytes[offset..end].to_vec(),
            });
            offset = end;
        }
        if offset != bytes.len() {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle contains trailing bytes".into(),
            ));
        }
        let bundle = Self {
            schema_version: manifest.schema_version,
            environment: manifest.environment,
            artifacts,
        };
        bundle.validate()?;
        if bundle.canonical_bytes()? != bytes {
            return Err(PythonArtifactError::Invalid(
                "Python artifact bundle encoding is not canonical".into(),
            ));
        }
        Ok(bundle)
    }

    pub fn install(&self) -> Result<InstalledPythonArtifacts, PythonArtifactError> {
        self.validate()?;
        let root = tempfile::Builder::new()
            .prefix("runmat-python-")
            .tempdir()
            .map_err(|error| PythonArtifactError::Io {
                operation: "create Python artifact root",
                error: error.to_string(),
            })?;
        let module_paths = self.install_into(root.path())?;
        Ok(InstalledPythonArtifacts { root, module_paths })
    }

    fn install_into(&self, root: &Path) -> Result<Vec<PathBuf>, PythonArtifactError> {
        let mut module_paths = Vec::with_capacity(self.artifacts.len());
        let mut expanded = 0u64;
        let mut files = 0usize;
        for artifact in &self.artifacts {
            let destination = root.join(&artifact.logical_name);
            std::fs::create_dir(&destination).map_err(|error| PythonArtifactError::Io {
                operation: "create Python wheel directory",
                error: error.to_string(),
            })?;
            let mut archive = zip::ZipArchive::new(Cursor::new(&artifact.bytes))
                .map_err(|error| PythonArtifactError::Invalid(error.to_string()))?;
            for index in 0..archive.len() {
                files = files.checked_add(1).ok_or_else(|| {
                    PythonArtifactError::Invalid("Python wheel file count overflowed".into())
                })?;
                if files > MAX_FILES {
                    return Err(PythonArtifactError::Invalid(format!(
                        "Python artifact bundle exceeds its file limit while reading `{}`",
                        artifact.logical_name
                    )));
                }
                let mut file = archive
                    .by_index(index)
                    .map_err(|error| PythonArtifactError::Invalid(error.to_string()))?;
                expanded = expanded.checked_add(file.size()).ok_or_else(|| {
                    PythonArtifactError::Invalid("Python wheel expanded size overflowed".into())
                })?;
                if expanded > MAX_EXPANDED_BYTES {
                    return Err(PythonArtifactError::Invalid(format!(
                        "Python artifact bundle exceeds its expanded byte limit while reading `{}`",
                        artifact.logical_name
                    )));
                }
                let relative = wheel_install_path(file.name())?;
                let output = destination.join(relative);
                if file.is_dir() {
                    std::fs::create_dir_all(&output).map_err(io_error)?;
                    continue;
                }
                if let Some(parent) = output.parent() {
                    std::fs::create_dir_all(parent).map_err(io_error)?;
                }
                let mut target = std::fs::File::create(&output).map_err(io_error)?;
                std::io::copy(&mut file, &mut target).map_err(io_error)?;
            }
            module_paths.push(destination);
        }
        Ok(module_paths)
    }
}

impl Default for PythonArtifactBundle {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug)]
pub struct InstalledPythonArtifacts {
    root: tempfile::TempDir,
    module_paths: Vec<PathBuf>,
}

impl InstalledPythonArtifacts {
    pub fn module_paths(&self) -> &[PathBuf] {
        &self.module_paths
    }

    pub fn root(&self) -> &Path {
        self.root.path()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PythonArtifactError {
    #[error("invalid Python artifact: {0}")]
    Invalid(String),
    #[error("failed to encode Python artifact: {0}")]
    Encoding(String),
    #[error("{operation}: {error}")]
    Io {
        operation: &'static str,
        error: String,
    },
}

fn validate_token(label: &str, value: &str) -> Result<(), PythonArtifactError> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        Err(PythonArtifactError::Invalid(format!(
            "Python artifact {label} is invalid"
        )))
    } else {
        Ok(())
    }
}

fn validate_module(module: &str) -> Result<(), PythonArtifactError> {
    validate_token("module", module)?;
    if module.split('.').any(|part| !python_identifier(part)) {
        return Err(PythonArtifactError::Invalid(format!(
            "Python module `{module}` is invalid"
        )));
    }
    Ok(())
}

fn validate_logical_name(value: &str) -> Result<(), PythonArtifactError> {
    validate_token("logical name", value)?;
    if matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(PythonArtifactError::Invalid(format!(
            "Python artifact logical name `{value}` is invalid"
        )));
    }
    Ok(())
}

fn python_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_alphabetic())
        && characters.all(|character| character == '_' || character.is_alphanumeric())
}

fn validate_wheel(artifact: &PythonArtifactBundleEntry) -> Result<(), PythonArtifactError> {
    let mut archive = zip::ZipArchive::new(Cursor::new(&artifact.bytes))
        .map_err(|error| PythonArtifactError::Invalid(error.to_string()))?;
    let expected = artifact
        .module
        .split('.')
        .next()
        .unwrap_or(&artifact.module);
    let mut module_present = false;
    let mut metadata_present = false;
    for index in 0..archive.len() {
        let file = archive
            .by_index(index)
            .map_err(|error| PythonArtifactError::Invalid(error.to_string()))?;
        let path = wheel_install_path(file.name())?;
        let first = path.components().next();
        module_present |= matches!(
            first,
            Some(Component::Normal(name))
                if name == std::ffi::OsStr::new(expected)
                    || name == std::ffi::OsStr::new(&format!("{expected}.py"))
        );
        metadata_present |= file.name().ends_with(".dist-info/METADATA");
    }
    if !module_present || !metadata_present {
        return Err(PythonArtifactError::Invalid(format!(
            "Python artifact `{}` does not contain module `{}` and wheel metadata",
            artifact.logical_name, artifact.module
        )));
    }
    Ok(())
}

fn safe_archive_path(name: &str) -> Result<PathBuf, PythonArtifactError> {
    let path = Path::new(name);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(PythonArtifactError::Invalid(
            "Python wheel contains an unsafe path".into(),
        ));
    }
    Ok(path.to_path_buf())
}

fn wheel_install_path(name: &str) -> Result<PathBuf, PythonArtifactError> {
    let path = safe_archive_path(name)?;
    let mut components = path.components();
    let first = components.next();
    if matches!(first, Some(Component::Normal(value)) if value.to_string_lossy().ends_with(".data"))
    {
        let scheme = components.next();
        if matches!(scheme, Some(Component::Normal(value)) if value == "purelib" || value == "platlib")
        {
            let relocated = components.collect::<PathBuf>();
            if relocated.as_os_str().is_empty() {
                return Err(PythonArtifactError::Invalid(
                    "Python wheel contains an empty library relocation".into(),
                ));
            }
            return Ok(relocated);
        }
    }
    Ok(path)
}

fn io_error(error: std::io::Error) -> PythonArtifactError {
    PythonArtifactError::Io {
        operation: "materialize Python wheel",
        error: error.to_string(),
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
#[path = "artifact/tests.rs"]
mod tests;
