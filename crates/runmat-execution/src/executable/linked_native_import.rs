//! A separately tagged selected edge from VM-generated source to an already
//! linked native function. It never pretends that Native IR is a VM V3 body.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{
    ExecutableOptionalSection, ExecutableStaticCaller, ExecutableStaticImport,
    ExecutableStaticImportKind, KnownExecutableSection, SectionRequirement,
    SelectedBodySemanticDigest, SelectedUnitContentDigest,
};
use crate::{value::GraphStaticReferenceSite, ContractError, Digest, FunctionBindingOrigin};
use runmat_types::ProgramSourceId;

pub const LINKED_NATIVE_IMPORT_SECTION: &str = KnownExecutableSection::LinkedNativeImports.name();
pub const LINKED_NATIVE_IMPORT_SCHEMA: u16 =
    KnownExecutableSection::LinkedNativeImports.schema_version();

/// The exact source/native-code selection declared by one compiler site.
/// These digests are declarations until the selected host pairs them with a
/// retained loaded-image lease and issued source at invocation admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkedNativeStaticImport {
    pub caller: ExecutableStaticCaller,
    pub site: GraphStaticReferenceSite,
    pub target: FunctionBindingOrigin,
    pub source: ProgramSourceId,
    pub selected_product: Digest,
    pub native_ir: Digest,
    pub native_function: Digest,
}

impl LinkedNativeStaticImport {
    /// Give the existing body-fingerprint normalizer an exact site/origin/code
    /// view. This is never accepted as a VM-product static-import section.
    pub fn fingerprint_projection(self) -> ExecutableStaticImport {
        let mut stream = crate::DigestStream::new();
        stream.update(b"runmat-linked-native-content-v1\0");
        stream.update(self.selected_product.bytes());
        stream.update(self.native_ir.bytes());
        ExecutableStaticImport {
            caller: self.caller,
            site: self.site,
            target: self.target,
            kind: ExecutableStaticImportKind::LinkedNative,
            target_content: SelectedUnitContentDigest(stream.finish()),
            target_body: SelectedBodySemanticDigest(self.native_function),
        }
    }
}

pub fn linked_native_import_section(
    mut imports: Vec<LinkedNativeStaticImport>,
) -> Result<ExecutableOptionalSection, ContractError> {
    imports.sort_by_key(import_key);
    validate(&imports)?;
    let payload = serde_json::to_vec(&imports).map_err(|error| {
        ContractError::invalid("executable.linked_native_imports", error.to_string())
    })?;
    Ok(ExecutableOptionalSection::new(
        LINKED_NATIVE_IMPORT_SECTION,
        LINKED_NATIVE_IMPORT_SCHEMA,
        SectionRequirement::Required,
        payload,
    ))
}

pub fn decode_linked_native_imports(
    section: &ExecutableOptionalSection,
) -> Result<Vec<LinkedNativeStaticImport>, ContractError> {
    if section.name != LINKED_NATIVE_IMPORT_SECTION
        || section.schema_version != LINKED_NATIVE_IMPORT_SCHEMA
        || section.requirement != SectionRequirement::Required
    {
        return Err(ContractError::invalid(
            "executable.linked_native_imports",
            "unsupported linked-native import section",
        ));
    }
    let imports: Vec<LinkedNativeStaticImport> =
        serde_json::from_slice(&section.payload).map_err(|error| {
            ContractError::invalid("executable.linked_native_imports", error.to_string())
        })?;
    validate(&imports)?;
    let canonical = serde_json::to_vec(&imports).map_err(|error| {
        ContractError::invalid("executable.linked_native_imports", error.to_string())
    })?;
    if canonical != section.payload {
        return Err(ContractError::invalid(
            "executable.linked_native_imports",
            "payload is not canonical",
        ));
    }
    Ok(imports)
}

fn import_key(
    import: &LinkedNativeStaticImport,
) -> (Option<(Digest, u32)>, GraphStaticReferenceSite) {
    let caller = match import.caller {
        ExecutableStaticCaller::Script => None,
        ExecutableStaticCaller::Function(origin) => Some((origin.program, origin.function.0)),
    };
    (caller, import.site)
}

fn validate(imports: &[LinkedNativeStaticImport]) -> Result<(), ContractError> {
    let mut sites = BTreeSet::new();
    for import in imports {
        let key = import_key(import);
        if sites.last().is_some_and(|previous| previous >= &key) || !sites.insert(key) {
            return Err(ContractError::invalid(
                "executable.linked_native_imports",
                "reference sites must be sorted and unique",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::ProgramFunctionId;

    #[test]
    fn linked_native_sites_are_canonical_and_unique() {
        let first = LinkedNativeStaticImport {
            caller: ExecutableStaticCaller::Script,
            site: GraphStaticReferenceSite {
                instruction: 1,
                occurrence: 0,
            },
            target: FunctionBindingOrigin {
                program: Digest::sha256(b"program"),
                function: ProgramFunctionId(2),
            },
            source: ProgramSourceId(0),
            selected_product: Digest::sha256(b"product"),
            native_ir: Digest::sha256(b"ir"),
            native_function: Digest::sha256(b"function"),
        };
        let second = LinkedNativeStaticImport {
            site: GraphStaticReferenceSite {
                instruction: 2,
                occurrence: 0,
            },
            ..first
        };
        let section = linked_native_import_section(vec![second, first]).expect("sorted");
        assert_eq!(
            decode_linked_native_imports(&section).expect("decoded"),
            vec![first, second]
        );
        assert!(linked_native_import_section(vec![first, first]).is_err());
    }
}
