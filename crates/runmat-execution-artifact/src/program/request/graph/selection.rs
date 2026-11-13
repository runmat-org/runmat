//! Borrowed exact selected-code admission shared by invocation and capture owners.

use super::*;

#[path = "selection/archives.rs"]
mod archives;
#[path = "selection/edges.rs"]
mod edges;

/// This view carries no job recipe, callback invoker or execution context.
/// A worker recipe may additionally authenticate its primary foreign policy.
pub(super) struct SelectedCode<'a> {
    pub primary_unit: &'a [u8],
    pub retained_units: &'a [Vec<u8>],
    pub graph: &'a PayloadGraph,
    pub class_dependencies: &'a [runmat_execution::value::GraphClassDependency],
    pub class_method_dependencies: &'a [GraphClassMethodDependency],
    pub static_dependencies: &'a [GraphStaticDependency],
    pub selected_bundle_archives: &'a [Option<Vec<u8>>],
    pub selected_foreign_policies: &'a [Option<runmat_execution::ForeignPolicyRef>],
    pub primary_policy: Option<&'a Option<crate::ObjectDescriptor>>,
}

impl SelectedCode<'_> {
    pub(super) fn selected_unit_host_requirements(
        &self,
    ) -> ArtifactResult<Vec<SelectedUnitHostRequirement>> {
        self.validate_selected_unit_bytes()?;
        std::iter::once(self.primary_unit)
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .map(|bytes| {
                let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                unit.manifest
                    .interop
                    .validate()
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                Ok(SelectedUnitHostRequirement {
                    environment: unit.manifest.identity.program.environment(),
                    interop: unit.manifest.interop.clone(),
                })
            })
            .collect()
    }

    pub(super) fn selected_source_map(&self) -> ArtifactResult<ExecutableSourceMapV2> {
        self.validate_selected_unit_bytes()?;
        // The ordered unit position is part of the selected source basis.
        // Two separately issued copies of identical V3 bytes may coexist in
        // one invocation; coalescing by program digest would make a returned
        // exception ordinal ambiguous to the client's exact result receipt.
        let mut selected = Vec::new();
        for bytes in
            std::iter::once(self.primary_unit).chain(self.retained_units.iter().map(Vec::as_slice))
        {
            let unit = ExecutableUnitEnvelope::from_canonical_bytes(bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            let component = unit
                .component(ExecutableComponentKind::SourceMap)
                .ok_or_else(|| ArtifactError::Invalid("selected V3 source map is absent".into()))?;
            let map = ExecutableSourceMapV2::decode(&component.bytes)
                .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
            for mut record in map.entries {
                record.source_id =
                    runmat_types::ProgramSourceId(u32::try_from(selected.len()).map_err(|_| {
                        ArtifactError::Limit("graph source ordinals exceed their ABI".into())
                    })?);
                selected.push(record);
            }
        }
        ExecutableSourceMapV2::new(selected)
            .map_err(|error| ArtifactError::Invalid(error.to_string()))
    }

    pub(super) fn validate_selected_unit_bytes(&self) -> ArtifactResult<()> {
        const MAX_SELECTED_UNITS: usize = runmat_execution::value::GRAPH_MAX_SELECTED_UNITS;
        const MAX_SELECTED_BYTES: usize = runmat_execution::EXECUTABLE_UNIT_ENVELOPE_MAX_BYTES;
        if self.retained_units.len() >= MAX_SELECTED_UNITS {
            return Err(ArtifactError::Limit(
                "graph request has too many selected units".into(),
            ));
        }
        let total =
            self.retained_units
                .iter()
                .try_fold(self.primary_unit.len(), |total, unit| {
                    total.checked_add(unit.len()).ok_or_else(|| {
                        ArtifactError::Limit("graph selected-unit bytes overflow".into())
                    })
                })?;
        if total > MAX_SELECTED_BYTES {
            return Err(ArtifactError::Limit(
                "graph selected-unit bytes exceed their shared bound".into(),
            ));
        }
        Ok(())
    }
}
