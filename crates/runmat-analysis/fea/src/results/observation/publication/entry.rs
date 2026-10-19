// Source-bound publication descriptors; numerical payloads stay in result artifacts.
use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesReductionPublicationContribution {
    pub identity: SeriesReductionContributionIdentity,
    pub ordinal: u32,
    pub sources: Vec<SeriesReductionSource>,
    pub value_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesReductionPublication {
    pub identity: SeriesReductionResultIdentity,
    pub source_history: ObservationHistoryIdentity,
    pub operation: SeriesReduction,
    pub extrema_ties: ExtremaTiePolicy,
    pub contributions: Vec<SeriesReductionPublicationContribution>,
    pub evidence_digest: Digest,
}
impl SeriesReductionPublication {
    pub fn new(
        result: &SeriesReductionResult,
        history: &ObservationHistory,
        plan: &ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<Self> {
        result.validate_source(plan, history).map_err(invalid)?;
        Self::from_result(result)
    }

    fn from_result(result: &SeriesReductionResult) -> AnalysisExecutionResult<Self> {
        result.validate().map_err(invalid)?;
        Ok(Self {
            identity: result.identity,
            source_history: result.source_history,
            operation: result.operation,
            extrema_ties: result.extrema_ties,
            contributions: result
                .evidence
                .contributions
                .iter()
                .map(|item| SeriesReductionPublicationContribution {
                    identity: item.identity,
                    ordinal: item.ordinal,
                    sources: item.sources.clone(),
                    value_count: item.values.len(),
                })
                .collect(),
            evidence_digest: digest("series-evidence", &result.evidence)?,
        })
    }

    /// Admit a saved reduction against the authenticated publication without reading field arrays.
    /// The execution/import owner has already admitted its complete source history.
    pub fn validate_published_result(
        &self,
        result: &SeriesReductionResult,
        history: &ObservationHistoryManifest,
        request: &ObservationHistoryFinalizationRequest,
    ) -> AnalysisExecutionResult<()> {
        self.validate(history, request)?;
        if result.observation_id != request.observation_id || *self != Self::from_result(result)? {
            return Err(invalid(
                "saved series reduction differs from publication authority",
            ));
        }
        Ok(())
    }
    pub fn validate_series(
        &self,
        result: &SeriesReductionResult,
        history: &ObservationHistory,
        plan: &ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<()> {
        if *self != Self::new(result, history, plan)? {
            return Err(invalid("series output differs from publication authority"));
        }
        Ok(())
    }
    pub(super) fn validate(
        &self,
        history: &ObservationHistoryManifest,
        request: &ObservationHistoryFinalizationRequest,
    ) -> AnalysisExecutionResult<()> {
        let observation = request
            .plan
            .observations
            .iter()
            .find(|item| item.spec.observation_id == request.observation_id)
            .ok_or_else(|| invalid("series observation absent"))?;
        if self.source_history != history.identity
            || Some(self.operation) != request.series_reduction
            || observation
                .spec
                .reduction
                .as_ref()
                .map(|item| item.extrema_ties)
                != Some(self.extrema_ties)
            || self.contributions.is_empty()
        {
            return Err(invalid("series metadata differs from authored history"));
        }
        let mut sources = Vec::new();
        for (ordinal, contribution) in self.contributions.iter().enumerate() {
            if contribution.ordinal as usize != ordinal
                || contribution.sources.is_empty()
                || contribution.value_count == 0
            {
                return Err(invalid("series contributions are empty or unordered"));
            }
            sources.extend(
                contribution
                    .sources
                    .iter()
                    .map(|item| (item.sample, item.coordinate)),
            );
        }
        if sources
            != history
                .samples
                .iter()
                .map(|item| (item.sample, item.coordinate))
                .collect::<Vec<_>>()
        {
            return Err(invalid(
                "series source descriptors differ from complete history",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationSamplePublication {
    pub request: ObservationEvaluationRequest,
    pub manifest: ObservationArtifactManifest,
    /// Execution provenance is separate from placement-neutral numerical result identities.
    pub partition_evidence: Vec<ObservationPartitionEvidence>,
}
impl ObservationSamplePublication {
    pub fn new(
        request: ObservationEvaluationRequest,
        artifact: &ObservationArtifact,
    ) -> AnalysisExecutionResult<Self> {
        ObservationAssemblyRequest::new(&request, artifact.results())?;
        let value = Self {
            request,
            manifest: artifact.manifest().map_err(invalid)?,
            partition_evidence: artifact
                .results()
                .iter()
                .map(|result| result.partition_evidence.clone())
                .collect(),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate_artifact(&self, artifact: &ObservationArtifact) -> AnalysisExecutionResult<()> {
        if *self != Self::new(self.request.clone(), artifact)? {
            return Err(invalid(
                "sample artifact differs from publication authority",
            ));
        }
        Ok(())
    }
    pub(super) fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        self.manifest.validate().map_err(invalid)?;
        if self.manifest.evaluation_request != self.request.identity()
            || self.manifest.producer != self.request.producer
            || self.manifest.source_selection != self.request.selection.identity()
            || self.manifest.source != self.request.selection.source()
            || self.manifest.plan_identity != self.request.plan.identity
        {
            return Err(invalid("sample artifact differs from evaluation request"));
        }
        let sample = self.manifest.results[0].sample;
        // Manifests carry no event proof; do not synthesize attribution from a coordinate.
        let expected = self.request.scheduled_observations(sample, None)?;
        if expected.len() != self.manifest.results.len()
            || self.partition_evidence.len() != self.manifest.results.len()
        {
            return Err(invalid("sample manifest omits scheduled observations"));
        }
        for ((observation, entry), evidence) in expected
            .into_iter()
            .zip(&self.manifest.results)
            .zip(&self.partition_evidence)
        {
            evidence.validate().map_err(invalid)?;
            let source = self
                .request
                .selection
                .fields()
                .iter()
                .find(|field| field.field == evidence.source_field)
                .ok_or_else(|| {
                    invalid("sample execution evidence source field is outside selection")
                })?;
            if evidence.identity != entry.partition_evidence
                || evidence.sample != entry.sample
                || evidence.event_attribution.is_some()
                || source.sample != entry.sample
                || source.definition.identity() != observation.source_field_definition_identity
                || evidence.source_partitions.iter().any(|partition| {
                    !source
                        .partitions
                        .iter()
                        .any(|descriptor| descriptor.partition == *partition)
                })
            {
                return Err(invalid(
                    "sample execution evidence differs from result or selected source descriptors",
                ));
            }
            if entry.observation_id != observation.spec.observation_id
                || entry.sample != sample
                || entry.quantity != observation.spec.quantity
                || entry.location != observation.spec.output_location
                || entry.value_shape != observation.result_value_shape
            {
                return Err(invalid("sample manifest does not preserve one accepted coordinate and resolved metadata"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationHistoryPublication {
    pub request: ObservationHistoryFinalizationRequest,
    pub checkpoint: ObservationHistoryCheckpointManifest,
    pub history: ObservationHistoryManifest,
    pub series: Option<SeriesReductionPublication>,
}
impl ObservationHistoryPublication {
    pub fn new(
        request: ObservationHistoryFinalizationRequest,
        checkpoint: &ObservationHistoryCheckpoint,
        catalog: &ObservationFieldCatalog,
        context: ObservationHistorySourceContext<'_>,
        history: &ObservationHistory,
        series: Option<&SeriesReductionResult>,
    ) -> AnalysisExecutionResult<Self> {
        request.validate_inputs(checkpoint, catalog, context)?;
        let expected = BoundedObservationHistory::resume(
            &request.plan,
            &request.observation_id,
            checkpoint.clone(),
        )
        .map_err(invalid)?
        .finish()
        .map_err(invalid)?;
        if &expected != history {
            return Err(invalid("history differs from terminal checkpoint"));
        }
        let value = Self {
            series: series
                .map(|result| SeriesReductionPublication::new(result, history, &request.plan))
                .transpose()?,
            request,
            checkpoint: checkpoint.manifest().clone(),
            history: history.manifest().clone(),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate_history(
        &self,
        checkpoint: &ObservationHistoryCheckpoint,
        catalog: &ObservationFieldCatalog,
        context: ObservationHistorySourceContext<'_>,
        history: &ObservationHistory,
        series: Option<&SeriesReductionResult>,
    ) -> AnalysisExecutionResult<()> {
        if *self
            != Self::new(
                self.request.clone(),
                checkpoint,
                catalog,
                context,
                history,
                series,
            )?
        {
            return Err(invalid("history output differs from publication authority"));
        }
        Ok(())
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate()?;
        self.checkpoint.validate().map_err(invalid)?;
        self.history.validate().map_err(invalid)?;
        let observation = self
            .request
            .plan
            .observations
            .iter()
            .find(|item| item.spec.observation_id == self.request.observation_id)
            .ok_or_else(|| invalid("history observation absent"))?;
        let expected_stream = ObservationHistoryStream::new(
            self.request.plan.identity,
            self.request.plan.topology_digest,
            self.request.plan.independent_domain.clone(),
            observation,
            observation.sample_bounds.maximum_produced_samples,
            observation.sample_bounds.maximum_retained_samples,
        )
        .map_err(invalid)?;
        if self.request.terminal_checkpoint != self.checkpoint.identity
            || self.checkpoint.stream != expected_stream
            || self.history.plan_identity != self.request.plan.identity
            || self.history.observation_id != self.request.observation_id
            || self.history.stream_identity != self.checkpoint.stream.identity
            || self.history.samples != self.checkpoint.samples
            || self.history.observed_sample_count != self.checkpoint.observed_sample_count
            || self.history.retained_prefix_digest != self.checkpoint.retained_prefix_digest
            || self.history.observed_prefix_digest != self.checkpoint.observed_prefix_digest
            || self.history.observed_sample_count != self.history.samples.len() as u64
            || self.series.is_some() != self.request.series_reduction.is_some()
        {
            return Err(invalid(
                "history metadata differs from complete terminal checkpoint",
            ));
        }
        if let Some(series) = &self.series {
            series.validate(&self.history, &self.request)?;
        }
        Ok(())
    }
}
