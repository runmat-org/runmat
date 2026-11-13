use runmat_execution::{Digest, ExecutableSourceMapV2};

use super::super::RetainedCallableStageError;

/// The owner whose authenticated source namespace is being projected into a
/// retained-definition fingerprint.
#[derive(Clone, Copy)]
pub(super) enum DefinitionSourceDomain<'a> {
    Admitted(&'a ExecutableSourceMapV2),
    Runtime,
    Selected(&'a runmat_runtime::source_context::SelectedProgramSourceOwner),
}

/// Stable source identity for one retained definition. An admitted source
/// ordinal is deliberately absent: it is only a position in one envelope.
#[derive(serde::Serialize)]
pub(super) enum DefinitionSourceIdentity<'a> {
    Admitted {
        owner_identity: &'a str,
        relative_path: &'a str,
        content_revision: Digest,
    },
    Runtime(Digest),
}

pub(super) fn project_definition_source<'a>(
    source: Option<crate::BytecodeSourceRef>,
    domain: DefinitionSourceDomain<'a>,
) -> Result<Option<DefinitionSourceIdentity<'a>>, RetainedCallableStageError> {
    match (source, domain) {
        (None, _) => Ok(None),
        (Some(crate::BytecodeSourceRef::Runtime(source)), DefinitionSourceDomain::Runtime) => {
            Ok(Some(DefinitionSourceIdentity::Runtime(
                source.semantic_content_revision(),
            )))
        }
        (
            Some(crate::BytecodeSourceRef::Runtime(source)),
            DefinitionSourceDomain::Selected(owner),
        ) => {
            let ordinal = owner
                .source_ordinal_for(source)
                .ok_or(RetainedCallableStageError::MissingRuntimeSource)?;
            let record = owner
                .portable_map()
                .entries
                .get(
                    usize::try_from(ordinal.0)
                        .map_err(|_| RetainedCallableStageError::MissingPortableSource)?,
                )
                .filter(|record| record.source_id == ordinal)
                .ok_or(RetainedCallableStageError::MissingPortableSource)?;
            Ok(Some(DefinitionSourceIdentity::Admitted {
                owner_identity: &record.owner_identity,
                relative_path: &record.relative_path,
                content_revision: record.content_revision,
            }))
        }
        (
            Some(crate::BytecodeSourceRef::Program(source)),
            DefinitionSourceDomain::Admitted(map),
        ) => {
            let record = usize::try_from(source.0)
                .ok()
                .and_then(|ordinal| map.entries.get(ordinal))
                .filter(|record| record.source_id == source)
                .ok_or(RetainedCallableStageError::MissingPortableSource)?;
            Ok(Some(DefinitionSourceIdentity::Admitted {
                owner_identity: &record.owner_identity,
                relative_path: &record.relative_path,
                content_revision: record.content_revision,
            }))
        }
        (Some(crate::BytecodeSourceRef::Program(_)), DefinitionSourceDomain::Runtime) => {
            Err(RetainedCallableStageError::MissingRuntimeSource)
        }
        (Some(crate::BytecodeSourceRef::Program(_)), DefinitionSourceDomain::Selected(_)) => {
            Err(RetainedCallableStageError::MissingRuntimeSource)
        }
        (Some(crate::BytecodeSourceRef::Runtime(_)), DefinitionSourceDomain::Admitted(_)) => {
            Err(RetainedCallableStageError::MissingPortableSource)
        }
    }
}
