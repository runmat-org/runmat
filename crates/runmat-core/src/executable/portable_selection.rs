//! Publication role of the compiler's primary interactive source.
use super::ExecutableUnit;
use runmat_hir::{FunctionId, FunctionKind};
use runmat_types::{ClassIdentity, ClassMethodRole, ProgramSourceId};
use std::collections::{BTreeSet, HashSet};

pub(crate) enum SourceProductSelection {
    Script,
    Definitions,
    Constructor {
        class: ClassIdentity,
        source: ProgramSourceId,
    },
    Declaration {
        class: ClassIdentity,
        source: ProgramSourceId,
    },
}

impl SourceProductSelection {
    pub(crate) fn classify(
        unit: &ExecutableUnit,
        current: &HashSet<FunctionId>,
        primary: Option<ProgramSourceId>,
    ) -> Result<Self, String> {
        let source = primary.ok_or("interactive product has no primary source")?;
        if unit.primary_source_id()? != source {
            return Err("interactive product primary source differs from its source map".into());
        }
        // RegisterClass publication instructions do not constitute a script.
        // Inspect the typed entrypoint, including its kind, rather than the
        // nonempty bytecode or a source filename/display-name convention.
        if unit.mir().entrypoints.iter().any(|function| {
            unit.mir().functions.get(function).is_some_and(|metadata| {
                metadata.source == source && metadata.kind == FunctionKind::SyntheticEntrypoint
            })
        }) {
            return Ok(Self::Script);
        }
        if let Some(class) = unit.primary_declaration_only_class(source)? {
            return Ok(Self::Declaration { class, source });
        }
        let constructors = current
            .iter()
            .filter_map(|function| {
                let metadata = unit.mir().functions.get(function)?;
                let owner = metadata.class_method_owner.as_ref()?;
                (metadata.source == source
                    && !owner.is_static
                    && owner.role == ClassMethodRole::Constructor)
                    .then(|| owner.declaring_class.clone())
            })
            .collect::<BTreeSet<_>>();
        match constructors.len() {
            0 => Ok(Self::Definitions),
            1 => Ok(Self::Constructor {
                class: constructors.into_iter().next().unwrap(),
                source,
            }),
            _ => Err("primary source has ambiguous constructor classes".into()),
        }
    }

    pub(crate) fn declaration(&self) -> Option<&ClassIdentity> {
        match self {
            Self::Declaration { class, .. } => Some(class),
            _ => None,
        }
    }

    pub(crate) fn envelope(
        &self,
        unit: &ExecutableUnit,
        current: &HashSet<FunctionId>,
        primary: Option<ProgramSourceId>,
        interop: runmat_types::InteropManifest,
    ) -> Result<runmat_execution::ExecutableUnitEnvelope, String> {
        match self {
            Self::Script => {
                unit.portable_script_envelope_with_current_definitions(interop, current)
            }
            Self::Constructor { class, source } => {
                unit.portable_constructor_envelope_with_interop(class, *source, interop, current)
            }
            Self::Declaration { class, source } => {
                unit.portable_declaration_envelope_with_interop(class, *source, interop, current)
            }
            Self::Definitions => unit.portable_envelope_for_current_definition_with_interop(
                current,
                primary.ok_or("definition-only executable has no primary source")?,
                interop,
            ),
        }
    }
}
