use runmat_execution::{Digest, ProgramEnvironment};
use runmat_native_codegen::{aot::NATIVE_OBJECT_SCHEMA_VERSION, NativeTarget};

use crate::{AotError, AotResult};

pub const RUNTIME_ARCHIVE_SCHEMA_VERSION: u16 = 4;
pub const MAX_RUNTIME_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_RUNTIME_PAYLOAD_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_LINK_TOKENS: usize = 256;
const MAX_RUNTIME_MANIFEST_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeArchiveEncoding {
    Raw,
    Zstd,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeArchiveCapabilities {
    pub static_program_calls: bool,
    pub runtime_builtins: bool,
    pub plot_core: bool,
    pub dynamic_source_loading: bool,
    pub closed_world_linking: bool,
}

impl RuntimeArchiveCapabilities {
    pub fn standalone_host() -> Self {
        Self {
            static_program_calls: true,
            runtime_builtins: true,
            plot_core: true,
            dynamic_source_loading: false,
            closed_world_linking: true,
        }
    }

    fn validate(&self) -> AotResult<()> {
        if !self.static_program_calls || !self.runtime_builtins {
            return Err(AotError::contract(
                "aot.archive.capabilities",
                "runtime archive lacks required standalone execution capabilities",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeArchiveManifest {
    pub schema_version: u16,
    pub runmat_version: String,
    pub target_triple: String,
    pub native_target: NativeTarget,
    pub native_ir_schema_version: u16,
    pub native_object_schema_version: u16,
    pub runtime_fingerprint: Digest,
    pub catalog_fingerprint: Digest,
    pub archive_digest: Digest,
    pub archive_bytes: u64,
    pub payload_encoding: RuntimeArchiveEncoding,
    pub payload_digest: Digest,
    pub payload_bytes: u64,
    pub capabilities: RuntimeArchiveCapabilities,
    pub native_link_tokens: Vec<String>,
}

#[derive(serde::Deserialize)]
struct RuntimeArchiveManifestAdmission {
    schema_version: u16,
    native_ir_schema_version: u16,
    native_object_schema_version: u16,
}

impl RuntimeArchiveManifest {
    /// Admit the revision header before decoding revision-sensitive manifest
    /// fields such as targets, capabilities, and native link tokens.
    pub fn from_json(bytes: &[u8]) -> AotResult<Self> {
        if bytes.len() > MAX_RUNTIME_MANIFEST_BYTES {
            return Err(AotError::contract(
                "aot.archive.manifest",
                format!(
                    "runtime archive manifest exceeds its {MAX_RUNTIME_MANIFEST_BYTES}-byte bound"
                ),
            ));
        }
        let admission: RuntimeArchiveManifestAdmission = serde_json::from_slice(bytes)
            .map_err(|error| AotError::contract("aot.archive.manifest", error.to_string()))?;
        validate_schema_revisions(
            admission.schema_version,
            admission.native_ir_schema_version,
            admission.native_object_schema_version,
        )?;
        serde_json::from_slice(bytes)
            .map_err(|error| AotError::contract("aot.archive.manifest", error.to_string()))
    }

    pub fn validate(&self) -> AotResult<()> {
        self.validate_schema_revisions()?;
        if self.runmat_version != env!("CARGO_PKG_VERSION") {
            return Err(AotError::contract(
                "aot.archive.runmat_version",
                format!(
                    "RunMat version actual {}, expected {}",
                    self.runmat_version,
                    env!("CARGO_PKG_VERSION")
                ),
            ));
        }
        self.native_target
            .validate()
            .map_err(|error| AotError::contract("aot.archive.target", error.to_string()))?;
        self.capabilities.validate()?;
        let triple = self
            .target_triple
            .parse::<target_lexicon::Triple>()
            .map_err(|_| {
                AotError::contract(
                    "aot.archive.target",
                    "runtime archive target triple is invalid",
                )
            })?;
        if triple != target_lexicon::HOST || self.native_target != NativeTarget::current() {
            return Err(AotError::contract(
                "aot.archive.target",
                "runtime archive does not match this RunMat host",
            ));
        }
        if self.archive_bytes == 0
            || self.archive_bytes > MAX_RUNTIME_ARCHIVE_BYTES
            || self.payload_bytes == 0
            || self.payload_bytes > MAX_RUNTIME_PAYLOAD_BYTES
            || self.native_link_tokens.len() > MAX_LINK_TOKENS
            || self
                .native_link_tokens
                .iter()
                .any(|token| !valid_link_token(token))
        {
            return Err(AotError::contract(
                "aot.archive.bounds",
                "runtime archive sizes or native-link tokens exceed their bounds",
            ));
        }
        Ok(())
    }

    fn validate_schema_revisions(&self) -> AotResult<()> {
        validate_schema_revisions(
            self.schema_version,
            self.native_ir_schema_version,
            self.native_object_schema_version,
        )
    }
}

fn validate_schema_revisions(
    schema_version: u16,
    native_ir_schema_version: u16,
    native_object_schema_version: u16,
) -> AotResult<()> {
    if schema_version != RUNTIME_ARCHIVE_SCHEMA_VERSION {
        return Err(AotError::contract(
            "aot.archive.schema",
            format!(
                "runtime archive schema actual {}, expected {}",
                schema_version, RUNTIME_ARCHIVE_SCHEMA_VERSION
            ),
        ));
    }
    if native_ir_schema_version != runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION {
        return Err(AotError::contract(
            "aot.archive.native_ir_schema",
            format!(
                "Native IR schema actual {}, expected {}",
                native_ir_schema_version,
                runmat_native_codegen::NATIVE_IR_SCHEMA_VERSION
            ),
        ));
    }
    if native_object_schema_version != NATIVE_OBJECT_SCHEMA_VERSION {
        return Err(AotError::contract(
            "aot.archive.native_object_schema",
            format!(
                "native object schema actual {}, expected {}",
                native_object_schema_version, NATIVE_OBJECT_SCHEMA_VERSION
            ),
        ));
    }
    Ok(())
}

impl RuntimeArchiveManifest {
    pub fn validate_environment(&self, environment: &ProgramEnvironment) -> AotResult<()> {
        if self.runtime_fingerprint != environment.runtime_fingerprint
            || self.catalog_fingerprint != environment.catalog_fingerprint
        {
            return Err(AotError::contract(
                "aot.archive.environment",
                "runtime archive does not match the program runtime and builtin catalog",
            ));
        }
        Ok(())
    }
}

fn valid_link_token(token: &str) -> bool {
    !token.is_empty()
        && token.len() <= 512
        && token.is_ascii()
        && !token.bytes().any(|byte| matches!(byte, 0 | b'\r' | b'\n'))
}
