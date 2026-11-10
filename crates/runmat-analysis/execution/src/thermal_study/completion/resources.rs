//! Original retained completion and simultaneous replay/encoding workspace.
//! Shared structural codec estimates and owned transport capacities; not allocator RSS.
use super::*;
use runmat_canonical_codec::CanonicalLimits;
use runmat_execution_artifact::PublishedObjectCatalog;

fn limits(maximum: u64) -> AnalysisBridgeResult<CanonicalLimits> {
    Ok(CanonicalLimits::content_bounded(
        usize::try_from(maximum).map_err(|_| capacity("thermal workspace address space"))?,
        65_536,
        64,
    ))
}
fn typed<T: serde::Serialize + ?Sized>(value: &T) -> AnalysisBridgeResult<u64> {
    runmat_canonical_codec::typed_resident_bytes_estimate(value)
        .map_err(|_| capacity("thermal retained typed estimate overflow"))
}
fn capacity(message: &'static str) -> AnalysisBridgeError {
    runmat_analysis_fea::execution::AnalysisExecutionError::Limit(message).into()
}
fn add(total: &mut u64, bytes: u64) -> AnalysisBridgeResult<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| capacity("thermal retained storage overflow"))?;
    Ok(())
}

impl CompletedThermalStudy {
    /// Charge the complete original source independently of the caller's projected DTO.
    pub fn retained_storage_bytes(&self, maximum: u64) -> AnalysisBridgeResult<u64> {
        // Retained values are already resident typed Rust data: charge their typed storage,
        // not the JSON value-tree projection that typed code never builds.
        let _ = limits(maximum)?;
        let mut total = typed(&(
            &self.compiled,
            &self.workload,
            &self.replay.mesh,
            &self.closures,
            &self.replay.program_revision,
            &self.replay.artifact_access,
            &self.result.host.workload,
            &self.result.host.artifact_access,
            &self.result.manifest,
            &self.result.numerical_receipt,
            &self.result.state_publication_profiles,
            &self.result.response,
            &self.accuracy.artifact,
            &self.publication.result,
            &self.baseline_usage.evidence,
            &self.terminal_usage.evidence,
        ))?;
        add(&mut total, std::mem::size_of::<Self>() as u64)?;
        add(&mut total, typed(&self.replay.field_source)?)?;
        for objects in [
            &self.result.objects,
            &self.result.numerical_receipt_objects,
            &self.result.state_publication_profile_objects,
            &self.accuracy.objects,
            &self.publication.objects,
            &self.baseline_usage.objects,
            &self.terminal_usage.objects,
        ] {
            add(&mut total, objects.retained_storage_bytes()?)?;
        }
        add(
            &mut total,
            runmat_analysis_fea::execution::semantic_state_retained_storage_bytes(
                self.states.iter().map(|s| &s.state),
                limits(maximum)?,
            )?,
        )?;
        for state in &self.states {
            add(&mut total, state.objects.retained_storage_bytes()?)?;
            add(&mut total, typed(&state.response)?)?;
        }
        add(&mut total, typed(&self.observations.catalog)?)?;
        for sample in &self.observations.samples {
            add(&mut total, sample.catalog_objects.retained_storage_bytes()?)?;
            add(
                &mut total,
                typed(&(&sample.manifest, &sample.output, &sample.access))?,
            )?;
        }
        for history in &self.observations.histories {
            match &history.history.storage {
                crate::ImportedObservationHistoryStorage::Partitioned(objects) => {
                    add(&mut total, objects.retained_storage_bytes()?)?;
                }
                crate::ImportedObservationHistoryStorage::Whole { objects, history } => {
                    add(&mut total, objects.retained_storage_bytes()?)?;
                    add(&mut total, typed(history)?)?;
                }
            }
            if let Some(objects) = &history.series_objects {
                add(&mut total, objects.retained_storage_bytes()?)?;
            }
            add(
                &mut total,
                typed(&(
                    &history.publication,
                    &history.history.manifest,
                    &history.series_reduction,
                ))?,
            )?;
        }
        Ok(total)
    }

    /// Profile immutable original payloads with provider-enforced bounds before typed decode.
    /// Source, replay and root-wire construction can coexist; three complete inventories
    /// conservatively cover those lifetimes, although the root wire contains only metadata.
    pub fn publication_workspace_bytes(
        &self,
        source: &impl CacheImport,
        catalog: &PublishedObjectCatalog,
        maximum: u64,
    ) -> AnalysisBridgeResult<u64> {
        catalog
            .validate()
            .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
        let retained = self
            .retained_storage_bytes(maximum)?
            .checked_mul(3)
            .ok_or_else(|| capacity("thermal replay retained storage overflow"))?;
        let available = maximum
            .checked_sub(retained)
            .ok_or_else(|| capacity("thermal replay has no remaining codec workspace"))?;
        let decoded = crate::study_closure::profile_import_workspace_phases(
            source,
            &[&catalog.objects],
            &self.result.host.artifact_access,
            self.output_policy().inventory_limits(),
            available,
        )?;
        retained
            .checked_add(decoded)
            .ok_or_else(|| capacity("thermal replay workspace overflow"))
    }
}
