//! Exact selected archive admission, without merged foreign namespaces.

use super::*;

impl SelectedCode<'_> {
    pub(crate) fn selected_bundle_admission(&self) -> ArtifactResult<Vec<Option<ExecutionBundle>>> {
        self.validate_selected_unit_bytes()?;
        let selected_count = self.retained_units.len() + 1;
        if self.selected_bundle_archives.len() != selected_count {
            return Err(ArtifactError::Invalid(
                "graph bundle archive count differs from selected V3 units".into(),
            ));
        }
        super::super::policy::validate_refs(self, selected_count)?;
        let limits = ArchiveLimits::default();
        let mut total_archive_bytes = 0_u64;
        std::iter::once(self.primary_unit)
            .chain(self.retained_units.iter().map(Vec::as_slice))
            .zip(self.selected_bundle_archives)
            .enumerate()
            .map(|(index, (unit_bytes, archive))| {
                let unit = ExecutableUnitEnvelope::from_canonical_bytes(unit_bytes)
                    .map_err(|error| ArtifactError::Invalid(error.to_string()))?;
                let has_interop = !unit.manifest.interop.adapters.is_empty()
                    || !unit.manifest.interop.foreign_types.is_empty()
                    || !unit.manifest.interop.adapter_contracts.is_empty();
                let policy_ref = self.selected_foreign_policies.get(index).copied().flatten();
                match (policy_ref, archive) {
                    (None, None) if !has_interop => Ok(None),
                    (Some(expected_policy), Some(bytes)) => {
                        let length = u64::try_from(bytes.len()).map_err(|_| {
                            ArtifactError::Limit("graph bundle archive cannot fit its ABI".into())
                        })?;
                        total_archive_bytes =
                            total_archive_bytes.checked_add(length).ok_or_else(|| {
                                ArtifactError::Limit("graph bundle archive lengths overflow".into())
                            })?;
                        if length > limits.max_total_bytes
                            || total_archive_bytes > limits.max_total_bytes
                        {
                            return Err(ArtifactError::Limit(
                                "graph bundle archives exceed their aggregate bound".into(),
                            ));
                        }
                        let bundle = read_bundle(Cursor::new(bytes), limits)?;
                        if bundle.manifest.program_revision != unit.manifest.identity.program {
                            return Err(ArtifactError::Identity(
                                "graph bundle revision differs from selected V3".into(),
                            ));
                        }
                        if bundle.manifest.recipes.len() != 1
                            || bundle.manifest.artifacts.len() != 1
                        {
                            return Err(ArtifactError::Identity(
                                "graph bundle must contain one exact selected V3 recipe and artifact"
                                    .into(),
                            ));
                        }
                        let mut matching = bundle.manifest.artifacts.iter().filter(|artifact| {
                            artifact.form == ExecutableForm::ExecutableUnitV4
                                && artifact.executable_bytes.as_slice() == unit_bytes
                        });
                        let Some(matched) = matching.next() else {
                            return Err(ArtifactError::Identity(
                                "graph bundle has no exact selected V3 artifact".into(),
                            ));
                        };
                        if matching.next().is_some() {
                            return Err(ArtifactError::Identity(
                                "graph bundle does not contain one exact selected V3 artifact"
                                    .into(),
                            ));
                        }
                        let recipe = bundle
                            .manifest
                            .recipes
                            .iter()
                            .find(|recipe| recipe.id().ok() == Some(matched.recipe_id))
                            .ok_or_else(|| {
                                ArtifactError::Identity(
                                    "graph bundle selected V3 recipe is absent".into(),
                                )
                            })?;
                        if recipe.interop != unit.manifest.interop {
                            return Err(ArtifactError::Identity(
                                "graph bundle interop differs from selected V3".into(),
                            ));
                        }
                        super::super::policy::validate_archive(self, index, recipe, expected_policy)?;
                        if bundle
                            .manifest
                            .recipes
                            .iter()
                            .any(|recipe| recipe.interop != unit.manifest.interop)
                        {
                            return Err(ArtifactError::Identity(
                                "graph bundle contains foreign artifacts outside its selected V3 interop"
                                    .into(),
                            ));
                        }
                        Ok(Some(bundle))
                    }
                    _ => Err(ArtifactError::Invalid(
                        "graph bundle presence differs from its issued policy ref or selected V3 interop".into(),
                    )),
                }
            })
            .collect()
    }
}
