use minicbor::{Decoder, Encoder};
use serde::{Deserialize, Serialize};

use super::Digest;
use crate::{schema::PROGRAM_REVISION_SCHEMA_V1, ContractError};

const MAX_CONTRIBUTIONS: usize = 32;
const MAX_CONTRIBUTION_NAME_BYTES: usize = 96;
const MAX_MODE_BYTES: usize = 32;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "DomainContributionWire")]
pub struct DomainContribution {
    name: String,
    digest: Digest,
}

impl DomainContribution {
    pub fn new(name: impl Into<String>, digest: Digest) -> Result<Self, ContractError> {
        let name = name.into();
        validate_token(
            "domain contribution name",
            &name,
            MAX_CONTRIBUTION_NAME_BYTES,
        )?;
        Ok(Self { name, digest })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn digest(&self) -> &Digest {
        &self.digest
    }
}

#[derive(Deserialize)]
struct DomainContributionWire {
    name: String,
    digest: Digest,
}

impl TryFrom<DomainContributionWire> for DomainContribution {
    type Error = ContractError;

    fn try_from(value: DomainContributionWire) -> Result<Self, Self::Error> {
        Self::new(value.name, value.digest)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "ProgramEnvironmentWire")]
pub struct ProgramEnvironment {
    pub semantic_schema: u32,
    pub compiler_schema: u32,
    pub runtime_fingerprint: Digest,
    pub catalog_fingerprint: Digest,
    pub compatibility_mode: String,
}

#[derive(Deserialize)]
struct ProgramEnvironmentWire {
    semantic_schema: u32,
    compiler_schema: u32,
    runtime_fingerprint: Digest,
    catalog_fingerprint: Digest,
    compatibility_mode: String,
}

impl TryFrom<ProgramEnvironmentWire> for ProgramEnvironment {
    type Error = ContractError;

    fn try_from(value: ProgramEnvironmentWire) -> Result<Self, Self::Error> {
        Self::new(
            value.semantic_schema,
            value.compiler_schema,
            value.runtime_fingerprint,
            value.catalog_fingerprint,
            value.compatibility_mode,
        )
    }
}

impl ProgramEnvironment {
    pub fn new(
        semantic_schema: u32,
        compiler_schema: u32,
        runtime_fingerprint: Digest,
        catalog_fingerprint: Digest,
        compatibility_mode: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let environment = Self {
            semantic_schema,
            compiler_schema,
            runtime_fingerprint,
            catalog_fingerprint,
            compatibility_mode: compatibility_mode.into(),
        };
        environment.validate()?;
        Ok(environment)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.semantic_schema == 0 || self.compiler_schema == 0 {
            return Err(ContractError::invalid(
                "program environment schemas",
                "semantic and compiler schemas must be non-zero",
            ));
        }
        validate_token(
            "compatibility mode",
            &self.compatibility_mode,
            MAX_MODE_BYTES,
        )
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "ProgramRevisionWire")]
pub struct ProgramRevision {
    schema_version: u16,
    graph_digest: Digest,
    source_digest: Digest,
    semantic_schema: u32,
    compiler_schema: u32,
    runtime_fingerprint: Digest,
    catalog_fingerprint: Digest,
    compatibility_mode: String,
    domain_contributions: Vec<DomainContribution>,
}

impl ProgramRevision {
    pub fn new(
        graph_digest: Digest,
        source_digest: Digest,
        environment: ProgramEnvironment,
    ) -> Result<Self, ContractError> {
        environment.validate()?;
        let revision = Self {
            schema_version: PROGRAM_REVISION_SCHEMA_V1,
            graph_digest,
            source_digest,
            semantic_schema: environment.semantic_schema,
            compiler_schema: environment.compiler_schema,
            runtime_fingerprint: environment.runtime_fingerprint,
            catalog_fingerprint: environment.catalog_fingerprint,
            compatibility_mode: environment.compatibility_mode,
            domain_contributions: Vec::new(),
        };
        revision.validate()?;
        Ok(revision)
    }

    pub fn with_domain_contribution(
        mut self,
        contribution: DomainContribution,
    ) -> Result<Self, ContractError> {
        match self
            .domain_contributions
            .binary_search_by(|candidate| candidate.name.cmp(&contribution.name))
        {
            Ok(_) => {
                return Err(ContractError::invalid(
                    "domain contributions",
                    format!("duplicate contribution `{}`", contribution.name),
                ))
            }
            Err(index) => self.domain_contributions.insert(index, contribution),
        }
        self.validate()?;
        Ok(self)
    }

    pub fn domain_contribution(&self, name: &str) -> Option<&Digest> {
        self.domain_contributions
            .binary_search_by(|candidate| candidate.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.domain_contributions[index].digest)
    }

    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn graph_digest(&self) -> &Digest {
        &self.graph_digest
    }

    pub fn source_digest(&self) -> &Digest {
        &self.source_digest
    }

    pub fn environment(&self) -> ProgramEnvironment {
        ProgramEnvironment {
            semantic_schema: self.semantic_schema,
            compiler_schema: self.compiler_schema,
            runtime_fingerprint: self.runtime_fingerprint,
            catalog_fingerprint: self.catalog_fingerprint,
            compatibility_mode: self.compatibility_mode.clone(),
        }
    }

    pub fn semantic_schema(&self) -> u32 {
        self.semantic_schema
    }

    pub fn compiler_schema(&self) -> u32 {
        self.compiler_schema
    }

    pub fn runtime_fingerprint(&self) -> &Digest {
        &self.runtime_fingerprint
    }

    pub fn catalog_fingerprint(&self) -> &Digest {
        &self.catalog_fingerprint
    }

    pub fn compatibility_mode(&self) -> &str {
        &self.compatibility_mode
    }

    pub fn domain_contributions(&self) -> &[DomainContribution] {
        &self.domain_contributions
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != PROGRAM_REVISION_SCHEMA_V1 {
            return Err(ContractError::UnsupportedSchema {
                actual: self.schema_version,
                supported: PROGRAM_REVISION_SCHEMA_V1,
            });
        }
        if self.semantic_schema == 0 || self.compiler_schema == 0 {
            return Err(ContractError::invalid(
                "program revision schemas",
                "semantic and compiler schemas must be non-zero",
            ));
        }
        validate_token(
            "compatibility mode",
            &self.compatibility_mode,
            MAX_MODE_BYTES,
        )?;
        if self.domain_contributions.len() > MAX_CONTRIBUTIONS {
            return Err(ContractError::Limit {
                field: "domain contributions",
                limit: MAX_CONTRIBUTIONS as u64,
            });
        }
        let mut previous: Option<&str> = None;
        for contribution in &self.domain_contributions {
            validate_token(
                "domain contribution name",
                &contribution.name,
                MAX_CONTRIBUTION_NAME_BYTES,
            )?;
            if previous.is_some_and(|name| name >= contribution.name.as_str()) {
                return Err(ContractError::invalid(
                    "domain contributions",
                    "contributions must be unique and sorted by name",
                ));
            }
            previous = Some(&contribution.name);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ContractError> {
        self.validate()?;
        let mut bytes = Vec::new();
        let mut encoder = Encoder::new(&mut bytes);
        encoder
            .map(9)
            .and_then(|encoder| encoder.u8(0))
            .and_then(|encoder| encoder.u16(self.schema_version))
            .and_then(|encoder| encoder.u8(1))
            .and_then(|encoder| encoder.bytes(self.graph_digest.bytes()))
            .and_then(|encoder| encoder.u8(2))
            .and_then(|encoder| encoder.bytes(self.source_digest.bytes()))
            .and_then(|encoder| encoder.u8(3))
            .and_then(|encoder| encoder.u32(self.semantic_schema))
            .and_then(|encoder| encoder.u8(4))
            .and_then(|encoder| encoder.u32(self.compiler_schema))
            .and_then(|encoder| encoder.u8(5))
            .and_then(|encoder| encoder.bytes(self.runtime_fingerprint.bytes()))
            .and_then(|encoder| encoder.u8(6))
            .and_then(|encoder| encoder.bytes(self.catalog_fingerprint.bytes()))
            .and_then(|encoder| encoder.u8(7))
            .and_then(|encoder| encoder.str(&self.compatibility_mode))
            .and_then(|encoder| encoder.u8(8))
            .and_then(|encoder| encoder.array(self.domain_contributions.len() as u64))
            .map_err(|error| ContractError::invalid("program revision", error.to_string()))?;
        for contribution in &self.domain_contributions {
            encoder
                .array(2)
                .and_then(|encoder| encoder.str(&contribution.name))
                .and_then(|encoder| encoder.bytes(contribution.digest.bytes()))
                .map_err(|error| ContractError::invalid("program revision", error.to_string()))?;
        }
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ContractError> {
        let mut decoder = Decoder::new(bytes);
        require_len(decoder.map(), 9, "program revision")?;
        require_key(&mut decoder, 0)?;
        let schema_version = decoder.u16().map_err(decode_error)?;
        require_key(&mut decoder, 1)?;
        let graph_digest = decode_digest(&mut decoder)?;
        require_key(&mut decoder, 2)?;
        let source_digest = decode_digest(&mut decoder)?;
        require_key(&mut decoder, 3)?;
        let semantic_schema = decoder.u32().map_err(decode_error)?;
        require_key(&mut decoder, 4)?;
        let compiler_schema = decoder.u32().map_err(decode_error)?;
        require_key(&mut decoder, 5)?;
        let runtime_fingerprint = decode_digest(&mut decoder)?;
        require_key(&mut decoder, 6)?;
        let catalog_fingerprint = decode_digest(&mut decoder)?;
        require_key(&mut decoder, 7)?;
        let compatibility_mode = decoder.str().map_err(decode_error)?.to_owned();
        require_key(&mut decoder, 8)?;
        let contribution_count =
            require_bounded_len(decoder.array(), MAX_CONTRIBUTIONS, "domain contributions")?;
        let mut domain_contributions = Vec::with_capacity(contribution_count);
        for _ in 0..contribution_count {
            require_len(decoder.array(), 2, "domain contribution")?;
            let name = decoder.str().map_err(decode_error)?.to_owned();
            let digest = decode_digest(&mut decoder)?;
            domain_contributions.push(DomainContribution::new(name, digest)?);
        }
        if decoder.position() != bytes.len() {
            return Err(ContractError::invalid(
                "program revision",
                "canonical encoding contains trailing data",
            ));
        }
        ProgramRevision::try_from(ProgramRevisionWire {
            schema_version,
            graph_digest,
            source_digest,
            semantic_schema,
            compiler_schema,
            runtime_fingerprint,
            catalog_fingerprint,
            compatibility_mode,
            domain_contributions,
        })
    }

    pub fn identity_digest(&self) -> Result<Digest, ContractError> {
        let mut framed = b"runmat-program-revision-v1\0".to_vec();
        framed.extend(self.canonical_bytes()?);
        Ok(Digest::sha256(framed))
    }

    pub fn canonical_identity(&self) -> String {
        self.identity_digest()
            .expect("validated ProgramRevision has a canonical identity")
            .to_string()
    }
}

fn require_key(decoder: &mut Decoder<'_>, expected: u8) -> Result<(), ContractError> {
    let actual = decoder.u8().map_err(decode_error)?;
    if actual != expected {
        return Err(ContractError::invalid(
            "program revision",
            format!("expected field key {expected}, found {actual}"),
        ));
    }
    Ok(())
}

fn require_len(
    length: Result<Option<u64>, minicbor::decode::Error>,
    expected: u64,
    field: &'static str,
) -> Result<(), ContractError> {
    match length.map_err(decode_error)? {
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(ContractError::invalid(
            field,
            format!("expected {expected} entries, found {actual}"),
        )),
        None => Err(ContractError::invalid(
            field,
            "indefinite-length CBOR is not canonical",
        )),
    }
}

fn require_bounded_len(
    length: Result<Option<u64>, minicbor::decode::Error>,
    maximum: usize,
    field: &'static str,
) -> Result<usize, ContractError> {
    match length.map_err(decode_error)? {
        Some(actual) if actual <= maximum as u64 => Ok(actual as usize),
        Some(_) => Err(ContractError::Limit {
            field,
            limit: maximum as u64,
        }),
        None => Err(ContractError::invalid(
            field,
            "indefinite-length CBOR is not canonical",
        )),
    }
}

fn decode_digest(decoder: &mut Decoder<'_>) -> Result<Digest, ContractError> {
    let bytes = decoder.bytes().map_err(decode_error)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| ContractError::invalid("digest", "expected exactly 32 bytes"))?;
    Ok(Digest::from_bytes(bytes))
}

fn decode_error(error: minicbor::decode::Error) -> ContractError {
    ContractError::invalid("program revision", error.to_string())
}

#[derive(Deserialize)]
struct ProgramRevisionWire {
    schema_version: u16,
    graph_digest: Digest,
    source_digest: Digest,
    semantic_schema: u32,
    compiler_schema: u32,
    runtime_fingerprint: Digest,
    catalog_fingerprint: Digest,
    compatibility_mode: String,
    domain_contributions: Vec<DomainContribution>,
}

impl TryFrom<ProgramRevisionWire> for ProgramRevision {
    type Error = ContractError;

    fn try_from(value: ProgramRevisionWire) -> Result<Self, Self::Error> {
        let revision = Self {
            schema_version: value.schema_version,
            graph_digest: value.graph_digest,
            source_digest: value.source_digest,
            semantic_schema: value.semantic_schema,
            compiler_schema: value.compiler_schema,
            runtime_fingerprint: value.runtime_fingerprint,
            catalog_fingerprint: value.catalog_fingerprint,
            compatibility_mode: value.compatibility_mode,
            domain_contributions: value.domain_contributions,
        };
        revision.validate()?;
        Ok(revision)
    }
}

fn validate_token(field: &'static str, value: &str, max_bytes: usize) -> Result<(), ContractError> {
    if value.is_empty()
        || value.len() > max_bytes
        || !value.is_ascii()
        || value.bytes().any(|byte| {
            !(byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'-' | b'_'))
        })
    {
        return Err(ContractError::invalid(
            field,
            format!(
                "must be 1..={max_bytes} bytes of lowercase ASCII letters, digits, `.`, `-`, or `_`"
            ),
        ));
    }
    Ok(())
}
