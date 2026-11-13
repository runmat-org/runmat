use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::identity::validate_identity;
use crate::{ContractError, Digest};

const MAX_OPTIONAL_SECTION_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionRequirement {
    Optional,
    Required,
}

/// Sections whose interpretation is part of RunMat's executable contract.
/// Hosts opt into these typed meanings rather than independently spelling a
/// required wire name and schema version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnownExecutableSection {
    StaticImports,
    LinkedNativeImports,
}

impl KnownExecutableSection {
    pub const fn name(self) -> &'static str {
        match self {
            Self::StaticImports => "runmat.static_imports",
            Self::LinkedNativeImports => "runmat.linked_native_imports",
        }
    }

    pub const fn schema_version(self) -> u16 {
        match self {
            Self::StaticImports => 1,
            Self::LinkedNativeImports => 1,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableOptionalSection {
    pub name: String,
    pub schema_version: u16,
    pub requirement: SectionRequirement,
    pub payload: Vec<u8>,
    pub payload_digest: Digest,
}

impl ExecutableOptionalSection {
    pub fn new(
        name: impl Into<String>,
        schema_version: u16,
        requirement: SectionRequirement,
        payload: Vec<u8>,
    ) -> Self {
        let payload_digest = Digest::sha256(&payload);
        Self {
            name: name.into(),
            schema_version,
            requirement,
            payload,
            payload_digest,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ContractError> {
        validate_identity("executable.optional_sections.name", &self.name, 128)?;
        if self.schema_version == 0 {
            return Err(ContractError::invalid(
                "executable.optional_sections.schema_version",
                "version must be non-zero",
            ));
        }
        if self.payload.len() > MAX_OPTIONAL_SECTION_BYTES {
            return Err(ContractError::Limit {
                field: "executable.optional_sections.payload",
                limit: MAX_OPTIONAL_SECTION_BYTES as u64,
            });
        }
        if self.payload_digest != Digest::sha256(&self.payload) {
            return Err(ContractError::invalid(
                "executable.optional_sections.payload_digest",
                "digest does not match payload",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutableSectionSupport(BTreeMap<String, u16>);

impl ExecutableSectionSupport {
    pub fn known(
        sections: impl IntoIterator<Item = KnownExecutableSection>,
    ) -> Result<Self, ContractError> {
        Self::new(
            sections
                .into_iter()
                .map(|section| (section.name().to_owned(), section.schema_version())),
        )
    }

    pub fn new(sections: impl IntoIterator<Item = (String, u16)>) -> Result<Self, ContractError> {
        let mut supported = BTreeMap::new();
        for (name, version) in sections {
            validate_identity("executable.section_support.name", &name, 128)?;
            if version == 0 {
                return Err(ContractError::invalid(
                    "executable.section_support.version",
                    "version must be non-zero",
                ));
            }
            if supported.insert(name, version).is_some() {
                return Err(ContractError::invalid(
                    "executable.section_support.name",
                    "section names must be unique",
                ));
            }
        }
        Ok(Self(supported))
    }

    pub(crate) fn validate_section(
        &self,
        section: &ExecutableOptionalSection,
    ) -> Result<(), ContractError> {
        let supported = self.0.get(&section.name).copied();
        if matches!(section.requirement, SectionRequirement::Required)
            && supported.is_none_or(|version| version < section.schema_version)
        {
            return Err(ContractError::invalid(
                "executable.optional_sections",
                format!(
                    "required section '{}' schema {} is unsupported",
                    section.name, section.schema_version
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_static_import_support_rejects_an_unknown_required_section() {
        let support =
            ExecutableSectionSupport::known([KnownExecutableSection::StaticImports]).unwrap();
        let known = ExecutableOptionalSection::new(
            KnownExecutableSection::StaticImports.name(),
            KnownExecutableSection::StaticImports.schema_version(),
            SectionRequirement::Required,
            Vec::new(),
        );
        support.validate_section(&known).expect("known section");
        let unknown = ExecutableOptionalSection::new(
            "runmat.unrecognized",
            1,
            SectionRequirement::Required,
            Vec::new(),
        );
        assert!(support.validate_section(&unknown).is_err());
    }
}
