//! A required canonical executable section for explicitly selected static
//! references outside this executable unit. This section declares source
//! meaning only; a host still has to admit the exact target product.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    ExecutableComponentDescriptor, ExecutableComponentPayload, ExecutableComponentRevisions,
    ExecutableIdentity, ExecutableOptionalSection, ExecutableUnitEnvelope, KnownExecutableSection,
    SectionRequirement,
};
use crate::value::GraphStaticReferenceSite;
use crate::{ContractError, FunctionBindingOrigin};
use runmat_types::{CapabilitySet, InteropManifest, ParallelManifest, RegionContract};

pub const STATIC_IMPORT_SECTION: &str = KnownExecutableSection::StaticImports.name();
pub const STATIC_IMPORT_SCHEMA: u16 = KnownExecutableSection::StaticImports.schema_version();

/// Identity of a selected unit's canonical code/source content with only its
/// outbound static-import links omitted. This nonrecursive anchor supports
/// valid cross-file mutual recursion; the enclosing graph separately checks
/// the final full-envelope digest for every unit-qualified target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SelectedUnitContentDigest(pub crate::Digest);

/// A function body's site-normalized semantics, including portable reference
/// origins but no recursively embedded target-product digests.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SelectedBodySemanticDigest(pub crate::Digest);

impl SelectedBodySemanticDigest {
    pub fn from_normalized_definition(fingerprint: crate::Digest) -> Self {
        let mut stream = crate::DigestStream::new();
        stream.update(b"runmat-selected-body-semantic-v1\0");
        stream.update(fingerprint.bytes());
        Self(stream.finish())
    }

    /// Existing VM semantic fingerprints have a canonical digest presentation.
    /// Parse only that digest encoding at the VM boundary; display names and
    /// session-local function IDs never participate in this identity.
    pub fn from_normalized_fingerprint(fingerprint: &str) -> Result<Self, ContractError> {
        let digest = crate::Digest::parse_canonical(fingerprint).ok_or_else(|| {
            ContractError::invalid("executable.static_imports", "invalid semantic digest")
        })?;
        Ok(Self::from_normalized_definition(digest))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "origin",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ExecutableStaticCaller {
    Script,
    Function(FunctionBindingOrigin),
}

/// Source-level provenance of a cross-product binding. The compiler supplies
/// this while partitioning its one-pass product; it is not inferred from the
/// imported registry's session-local numeric IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutableStaticImportKind {
    ExplicitExternal,
    PartitionedDefinition,
    /// Internal semantic-fingerprint projection of the separately versioned
    /// linked-native section. Never encoded in the VM static-import section.
    LinkedNative,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutableStaticImport {
    pub caller: ExecutableStaticCaller,
    pub site: GraphStaticReferenceSite,
    pub target: FunctionBindingOrigin,
    pub kind: ExecutableStaticImportKind,
    pub target_content: SelectedUnitContentDigest,
    pub target_body: SelectedBodySemanticDigest,
}

pub fn selected_unit_content_digest(
    envelope: &ExecutableUnitEnvelope,
) -> Result<SelectedUnitContentDigest, ContractError> {
    envelope.validate()?;
    let manifest = &envelope.manifest;
    // The borrowed projection has the exact field order of the canonical V3
    // envelope. Only the known outbound-link section is omitted. In
    // particular, code payloads and all other optional sections are borrowed
    // rather than duplicated to calculate this nonrecursive anchor.
    let content = ContentEnvelope {
        schema_version: envelope.schema_version,
        manifest: ContentManifest {
            schema_version: manifest.schema_version,
            identity: &manifest.identity,
            revisions: &manifest.revisions,
            components: &manifest.components,
            capabilities: &manifest.capabilities,
            regions: &manifest.regions,
            interop: &manifest.interop,
            parallel: &manifest.parallel,
            optional_sections: manifest
                .optional_sections
                .iter()
                .filter(|section| {
                    section.name != STATIC_IMPORT_SECTION
                        && section.name != super::linked_native_import::LINKED_NATIVE_IMPORT_SECTION
                })
                .collect(),
        },
        payloads: &envelope.payloads,
    };
    let bytes = serde_json::to_vec(&content)
        .map_err(|error| ContractError::invalid("executable.static_imports", error.to_string()))?;
    let mut stream = crate::DigestStream::new();
    stream.update(b"runmat-selected-unit-content-v1\0");
    stream.update(&(bytes.len() as u64).to_be_bytes());
    stream.update(&bytes);
    Ok(SelectedUnitContentDigest(stream.finish()))
}

#[derive(Serialize)]
struct ContentEnvelope<'a> {
    schema_version: u16,
    manifest: ContentManifest<'a>,
    payloads: &'a [ExecutableComponentPayload],
}

#[derive(Serialize)]
struct ContentManifest<'a> {
    schema_version: u16,
    identity: &'a ExecutableIdentity,
    revisions: &'a ExecutableComponentRevisions,
    components: &'a [ExecutableComponentDescriptor],
    capabilities: &'a CapabilitySet,
    regions: &'a [RegionContract],
    interop: &'a InteropManifest,
    parallel: &'a ParallelManifest,
    optional_sections: Vec<&'a ExecutableOptionalSection>,
}

pub fn static_import_section(
    mut imports: Vec<ExecutableStaticImport>,
) -> Result<ExecutableOptionalSection, ContractError> {
    imports.sort_by_key(import_key);
    validate_imports(&imports)?;
    let payload = serde_json::to_vec(&imports)
        .map_err(|error| ContractError::invalid("executable.static_imports", error.to_string()))?;
    Ok(ExecutableOptionalSection::new(
        STATIC_IMPORT_SECTION,
        STATIC_IMPORT_SCHEMA,
        SectionRequirement::Required,
        payload,
    ))
}

pub fn decode_static_imports(
    section: &ExecutableOptionalSection,
) -> Result<Vec<ExecutableStaticImport>, ContractError> {
    if section.name != STATIC_IMPORT_SECTION
        || section.schema_version != STATIC_IMPORT_SCHEMA
        || section.requirement != SectionRequirement::Required
    {
        return Err(ContractError::invalid(
            "executable.static_imports",
            "unsupported static-import section",
        ));
    }
    let imports: Vec<ExecutableStaticImport> = serde_json::from_slice(&section.payload)
        .map_err(|error| ContractError::invalid("executable.static_imports", error.to_string()))?;
    validate_imports(&imports)?;
    let canonical = serde_json::to_vec(&imports)
        .map_err(|error| ContractError::invalid("executable.static_imports", error.to_string()))?;
    if canonical != section.payload {
        return Err(ContractError::invalid(
            "executable.static_imports",
            "payload is not canonical",
        ));
    }
    Ok(imports)
}

fn import_key(
    import: &ExecutableStaticImport,
) -> (Option<(crate::Digest, u32)>, GraphStaticReferenceSite) {
    let caller = match import.caller {
        ExecutableStaticCaller::Script => None,
        ExecutableStaticCaller::Function(origin) => Some((origin.program, origin.function.0)),
    };
    (caller, import.site)
}

fn validate_imports(imports: &[ExecutableStaticImport]) -> Result<(), ContractError> {
    let mut sites = BTreeSet::new();
    for import in imports {
        if import.kind == ExecutableStaticImportKind::LinkedNative {
            return Err(ContractError::invalid(
                "executable.static_imports",
                "linked-native edges require their own typed section",
            ));
        }
        let key = import_key(import);
        if sites.last().is_some_and(|previous| previous >= &key) || !sites.insert(key) {
            return Err(ContractError::invalid(
                "executable.static_imports",
                "reference sites must be sorted and unique",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Digest;
    use runmat_types::ProgramFunctionId;

    #[test]
    fn required_section_rejects_repeated_site_and_noncanonical_order() {
        let target = FunctionBindingOrigin {
            program: Digest::sha256(b"parent"),
            function: ProgramFunctionId(2),
        };
        let first = ExecutableStaticImport {
            caller: ExecutableStaticCaller::Script,
            site: GraphStaticReferenceSite {
                instruction: 3,
                occurrence: 0,
            },
            target,
            kind: ExecutableStaticImportKind::ExplicitExternal,
            target_content: SelectedUnitContentDigest(Digest::sha256(b"target content")),
            target_body: SelectedBodySemanticDigest(Digest::sha256(b"target body")),
        };
        let second = ExecutableStaticImport {
            site: GraphStaticReferenceSite {
                instruction: 4,
                occurrence: 0,
            },
            ..first
        };
        let section = static_import_section(vec![second, first]).expect("sorted section");
        assert_eq!(
            decode_static_imports(&section).expect("canonical section"),
            vec![first, second]
        );
        assert!(static_import_section(vec![first, first]).is_err());
        let mut reordered = section;
        reordered.payload = serde_json::to_vec(&vec![second, first]).unwrap();
        reordered.payload_digest = Digest::sha256(&reordered.payload);
        assert!(decode_static_imports(&reordered).is_err());
    }
}
