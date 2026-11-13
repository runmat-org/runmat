//! Canonical catalog integrity with counted coverage and streamed original histories.
use super::*;

impl ObservationPublicationCatalog {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.validate_metadata()
    }
    /// Metadata validation is a pure function of the catalog's content, which its identity
    /// seals: a catalog value exists only after `new` derived that identity from its content or
    /// decoding checked it (uncached), so a catalog whose identity already validated in this
    /// process is not validated again.
    pub(super) fn validate_metadata(&self) -> AnalysisExecutionResult<()> {
        let key = self.identity.digest();
        if seen(&VALIDATED_METADATA, key) {
            return Ok(());
        }
        self.validate_metadata_uncached()?;
        remember(&VALIDATED_METADATA, key);
        Ok(())
    }
    pub(super) fn validate_metadata_uncached(&self) -> AnalysisExecutionResult<()> {
        self.source.validate()?;
        if self.schema_version != 2
            || self.identity != self.derive_identity()?
            || self.declarations.len() != self.plan.observations.len()
        {
            return Err(invalid(
                "catalog schema, identity, or declaration coverage differs",
            ));
        }
        let mut expected = coverage::PublicationCoverage::derive(&self.declarations, &self.plan)?;
        self.waive_out_of_scope(&mut expected)?;
        let mut previous = None;
        for entry in &self.entries {
            match entry {
                ObservationPublicationEntry::Sample(value) => {
                    value.validate()?;
                    self.source.validate_canonical_selection(
                        &value.request.plan,
                        &value.request.selection,
                    )?;
                    match (self.source.source(), value.request.selection.source()) {
                        (
                            ObservationFieldCatalogSource::SampledRecovery { .. },
                            ObservationFieldSource::SampledRecovery { .. },
                        ) => self
                            .source
                            .validate_sampled_selection(&value.request.selection)?,
                        (
                            ObservationFieldCatalogSource::VolumeMeshRecovery { .. },
                            ObservationFieldSource::VolumeMeshRecovery { .. },
                        ) => self.source.validate_volume_mesh_selection_metadata(
                            &value.request.selection,
                            value.request.plan.topology_digest,
                        )?,
                        (
                            ObservationFieldCatalogSource::Structural { .. },
                            ObservationFieldSource::StructuralRecovery { .. },
                        ) => {
                            self.source
                                .validate_structural_selection(&value.request.selection)?;
                        }
                        (
                            ObservationFieldCatalogSource::Thermal { manifest },
                            ObservationFieldSource::ThermalResult {
                                workload,
                                result,
                                numerical_intent,
                                mesh_digest,
                                sample_ordinal,
                                ..
                            },
                        ) => {
                            let fields = manifest.fields.get(sample_ordinal).ok_or_else(|| {
                                invalid("sample ordinal absent from source catalog")
                            })?;
                            if workload != manifest.source_workload
                                || result != manifest.identity()
                                || numerical_intent != manifest.numerical_intent
                                || Digest::from_bytes(*mesh_digest.bytes())
                                    != manifest.topology_digest
                                || manifest.producer != self.producer
                                || value
                                    .request
                                    .selection
                                    .fields()
                                    .iter()
                                    .any(|field| !fields.contains(field))
                            {
                                return Err(invalid(
                                    "sample selection differs from source catalog",
                                ));
                            }
                        }
                        _ => return Err(invalid("sample source kind differs from catalog")),
                    }
                    if value.request.producer != self.producer || value.request.plan != self.plan {
                        return Err(invalid("sample producer or plan differs"));
                    }
                    let mut published_fields = 0;
                    for result in &value.manifest.results {
                        if expected.sample(result.observation_id.0.as_str())? {
                            published_fields += 1;
                        }
                    }
                    if published_fields == 0 {
                        return Err(invalid("sample cannot replace a complete history"));
                    }
                }
                ObservationPublicationEntry::History(value) => {
                    value.validate()?;
                    if value.request.producer != self.producer
                        || value.request.plan != self.plan
                        || value.request.source_catalog != self.source.identity()
                        || !expected.history(value.history.observation_id.0.as_str())?
                    {
                        return Err(invalid(
                            "history producer, source, plan, or declaration differs",
                        ));
                    }
                    let observation = self
                        .plan
                        .observations
                        .iter()
                        .find(|item| item.spec.observation_id == value.history.observation_id)
                        .ok_or_else(|| invalid("history observation absent"))?;
                    coordinates::validate_history_coordinates(
                        &self.source,
                        self.producer,
                        &self.plan,
                        observation,
                        &value.history,
                    )?;
                }
            }
            if previous.is_some_and(|id| id >= entry.first_id()) {
                return Err(invalid("publication entries are not canonically ordered"));
            }
            previous = Some(entry.first_id());
        }
        if !expected.complete() {
            return Err(invalid(
                "publication does not cover every authored observation",
            ));
        }
        self.visit_execution(|_, _, _, _| Ok(()))?;
        Ok(())
    }

    /// Mark the declarations another catalog of the same study discharges, so that this one
    /// can neither omit nor republish what its scope owns.
    fn waive_out_of_scope(
        &self,
        expected: &mut coverage::PublicationCoverage<'_>,
    ) -> AnalysisExecutionResult<()> {
        let entry_samples = match (self.scope, self.source.source()) {
            (ObservationPublicationScope::Complete, _) => return Ok(()),
            (
                ObservationPublicationScope::Entry,
                ObservationFieldCatalogSource::Structural { manifest, .. }
                | ObservationFieldCatalogSource::VolumeMeshRecovery { manifest, .. }
                | ObservationFieldCatalogSource::SampledRecovery { manifest, .. },
            ) => {
                let mut samples = Vec::new();
                for field in &manifest.fields {
                    if !samples.contains(&field.sample) {
                        samples.push(field.sample);
                    }
                }
                samples
            }
            (
                ObservationPublicationScope::EntryHistories,
                ObservationFieldCatalogSource::EntryHistory { .. },
            ) => Vec::new(),
            _ => {
                return Err(invalid(
                    "publication scope differs from its source catalog kind",
                ))
            }
        };
        for (declaration, observation) in self.declarations.iter().zip(&self.plan.observations) {
            let history = matches!(declaration, ObservationRequest::History(_));
            let owned = match self.scope {
                ObservationPublicationScope::Complete => true,
                ObservationPublicationScope::EntryHistories => history,
                ObservationPublicationScope::Entry => {
                    let mut scheduled = false;
                    for sample in &entry_samples {
                        scheduled |= resolved_observation_accepts_sample(
                            &self.plan,
                            observation,
                            *sample,
                            None,
                        )
                        .map_err(invalid)?;
                    }
                    !history && scheduled
                }
            };
            if !owned {
                expected.waive(observation.spec.observation_id.0.as_str())?;
            }
        }
        Ok(())
    }
}
