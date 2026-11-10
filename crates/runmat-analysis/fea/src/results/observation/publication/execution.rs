//! Compact execution provenance. Source-block groups are authenticated against exact topology
//! at source-context boundaries, independently of any supplied resource accounting.
use super::*;
#[cfg(test)]
mod tests;
mod workspace;
use crate::results::field::{
    select_canonical_field_partitions, RecoveredFieldArtifactEntry,
    RecoveredFieldPartitionIdentity, RecoveredFieldPartitionSelection,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationResultExecution {
    result: ObservationResultIdentity,
    source_partitions: Vec<Vec<RecoveredFieldPartitionIdentity>>,
}

impl ObservationPublicationCatalog {
    /// Visits exact source-bound task inputs without numerical arrays. Call `validate_for`
    /// at the source boundary to authenticate stored grouping against workload and topology.
    /// The callback runs once per published result, with canonical partition-order set IDs.
    pub(crate) fn visit_execution(
        &self,
        mut visit: impl FnMut(
            &ObservationEvaluationRequest,
            &ObservationPartitionPlan,
            &ObservationPartitionEvidence,
            &[RecoveredFieldPartitionSelection<'_>],
        ) -> AnalysisExecutionResult<()>,
    ) -> AnalysisExecutionResult<()> {
        let mut records = self.execution.iter();
        self.visit_results(|source, observation_id, result, evidence| {
            let record = records
                .next()
                .ok_or_else(|| invalid("execution omits a published result"))?;
            if record.result != result {
                return Err(invalid("execution result order differs from publication"));
            }
            let plan = self.execution_plan(source, observation_id, evidence)?;
            if record.source_partitions.len() != plan.partitions.len() {
                return Err(invalid("execution omits canonical partition groups"));
            }
            let field = source_field(source, &plan, evidence)?;
            let group_ids = record
                .source_partitions
                .iter()
                .try_fold(0usize, |count, group| {
                    count
                        .checked_add(group.len())
                        .ok_or_else(|| invalid("execution source group size overflow"))
                })?;
            let mut union = Vec::new();
            union.try_reserve_exact(group_ids).map_err(invalid)?;
            let mut sets = Vec::new();
            sets.try_reserve_exact(record.source_partitions.len())
                .map_err(invalid)?;
            for group in &record.source_partitions {
                // Resolve first indices through the authoritative descriptor inventory. The
                // shared selector proves canonical order/uniqueness and owns set identity.
                let mut indices = Vec::new();
                indices.try_reserve_exact(group.len()).map_err(invalid)?;
                for identity in group {
                    indices.push(
                        field
                            .partitions
                            .iter()
                            .find(|descriptor| descriptor.partition == *identity)
                            .map(|descriptor| descriptor.first_source_index)
                            .ok_or_else(|| {
                                invalid("execution storage block is outside source field")
                            })?,
                    );
                }
                let selected =
                    select_canonical_field_partitions(field, &indices).map_err(invalid)?;
                if !selected
                    .descriptors
                    .iter()
                    .map(|item| item.partition)
                    .eq(group.iter().copied())
                {
                    return Err(invalid("execution storage blocks are not canonical"));
                }
                union.extend(group.iter().copied());
                sets.push(selected);
            }
            union.sort_unstable();
            union.dedup();
            if !union
                .into_iter()
                .eq(evidence.source_partitions.iter().copied())
            {
                return Err(invalid(
                    "execution source block union differs from result evidence",
                ));
            }
            visit(source, &plan, evidence, &sets)
        })?;
        if records.next().is_some() {
            return Err(invalid("execution contains extra published results"));
        }
        Ok(())
    }

    pub(super) fn derive_execution(
        &self,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Vec<ObservationResultExecution>> {
        let mut records = Vec::new();
        self.visit_derived_execution(context, |record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    pub(super) fn validate_execution_for(
        &self,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<()> {
        if self.execution_policy != context.observation_execution() {
            return Err(invalid("execution policy differs from producing workload"));
        }
        let mut records = self.execution.iter();
        self.visit_derived_execution(context, |expected| {
            if records.next() != Some(&expected) {
                return Err(invalid(
                    "execution grouping differs from exact source topology",
                ));
            }
            Ok(())
        })?;
        if records.next().is_some() {
            return Err(invalid("execution contains extra result groups"));
        }
        Ok(())
    }

    fn visit_derived_execution(
        &self,
        context: ObservationHistorySourceContext<'_>,
        mut visit: impl FnMut(ObservationResultExecution) -> AnalysisExecutionResult<()>,
    ) -> AnalysisExecutionResult<()> {
        context.validate()?;
        // Every result of one catalog selects over the same mesh: derive its projection once,
        // and the observation topology once per face flag, not once per published result.
        let projection = std::cell::OnceCell::new();
        let topologies: [std::cell::OnceCell<crate::results::observation::ObservationTopology<'_>>;
            2] = Default::default();
        self.visit_results(|source, observation_id, result, evidence| {
            let plan = self.execution_plan(source, observation_id, evidence)?;
            let field = source_field(source, &plan, evidence)?;
            let mut groups = Vec::new();
            groups
                .try_reserve_exact(plan.partitions.len())
                .map_err(invalid)?;
            let volume_topology = match context {
                ObservationHistorySourceContext::Thermal { mesh, .. }
                | ObservationHistorySourceContext::VolumeMesh { mesh }
                    if !plan.partitions.is_empty() =>
                {
                    if projection.get().is_none() {
                        let derived =
                            crate::topology::AnalysisProjection::derive(mesh).map_err(invalid)?;
                        let _ = projection.set(derived);
                    }
                    let projection = projection.get().expect("derived projection");
                    let faces = crate::results::observation::distributed::source_selection::observation_topology_faces(
                        &source.plan,
                    );
                    let slot = &topologies[usize::from(faces)];
                    if slot.get().is_none() {
                        let derived =
                            crate::results::observation::ObservationTopology::from_projection_for(
                                projection, faces,
                            )
                            .map_err(invalid)?;
                        let _ = slot.set(derived);
                    }
                    slot.get()
                }
                _ => None,
            };
            for partition in &plan.partitions {
                let indices = match context {
                    ObservationHistorySourceContext::Magnetostatic { mesh, .. }
                    | ObservationHistorySourceContext::MaxwellHarmonic { mesh, .. } => {
                        let ObservationFieldCatalogSource::SampledRecovery { sampling, .. } =
                            self.source.source()
                        else {
                            return Err(invalid(
                                "sampled execution context requires original sampled catalog",
                            ));
                        };
                        sampled_observation_partition_source_indices(
                            &source.plan,
                            &plan,
                            partition,
                            sampling,
                            mesh,
                        )
                        .map_err(invalid)?
                    }
                    ObservationHistorySourceContext::Structural {
                        source: workload,
                        mesh,
                    } => structural_observation_partition_source_indices(
                        &source.plan,
                        &plan,
                        partition,
                        workload.model(),
                        mesh,
                    )
                    .map_err(invalid)?,
                    ObservationHistorySourceContext::Thermal { .. }
                    | ObservationHistorySourceContext::VolumeMesh { .. } => {
                        let topology = volume_topology
                            .ok_or_else(|| invalid("volume observation topology absent"))?;
                        crate::results::observation::distributed::source_selection::observation_partition_source_indices(
                            &source.plan,
                            &plan,
                            partition,
                            topology,
                        )
                        .map_err(invalid)?
                    }
                };
                let selected =
                    select_canonical_field_partitions(field, &indices).map_err(invalid)?;
                groups.push(
                    selected
                        .descriptors
                        .iter()
                        .map(|descriptor| descriptor.partition)
                        .collect(),
                );
            }
            visit(ObservationResultExecution {
                result,
                source_partitions: groups,
            })
        })
    }

    fn execution_plan(
        &self,
        source: &ObservationEvaluationRequest,
        observation: &runmat_analysis_core::ObservationId,
        evidence: &ObservationPartitionEvidence,
    ) -> AnalysisExecutionResult<ObservationPartitionPlan> {
        evidence.validate().map_err(invalid)?;
        // `visit_results` validated `source` (and so its plan identity) before visiting it.
        let plan =
            ObservationPartitionPlan::derive(&source.plan, observation, self.execution_policy)
                .map_err(invalid)?;
        if plan.identity != evidence.partition_plan
            || plan.partitions.len() != evidence.contributions.len()
        {
            return Err(invalid(
                "result evidence differs from canonical workload partition plan",
            ));
        }
        Ok(plan)
    }

    fn visit_results(
        &self,
        mut visit: impl FnMut(
            &ObservationEvaluationRequest,
            &runmat_analysis_core::ObservationId,
            ObservationResultIdentity,
            &ObservationPartitionEvidence,
        ) -> AnalysisExecutionResult<()>,
    ) -> AnalysisExecutionResult<()> {
        for entry in &self.entries {
            match entry {
                ObservationPublicationEntry::Sample(sample) => {
                    sample.validate()?;
                    for (result, evidence) in sample
                        .manifest
                        .results
                        .iter()
                        .zip(&sample.partition_evidence)
                    {
                        visit(
                            &sample.request,
                            &result.observation_id,
                            result.result,
                            evidence,
                        )?;
                    }
                }
                ObservationPublicationEntry::History(history) => {
                    history.validate()?;
                    for (sample, evidence) in history
                        .history
                        .samples
                        .iter()
                        .zip(&history.checkpoint.partition_evidence)
                    {
                        let mut source_catalog = &self.source;
                        let coordinate = match self.source.source() {
                            ObservationFieldCatalogSource::SampledRecovery { .. } => {
                                return Err(invalid("sampled native histories are not admitted"));
                            }
                            ObservationFieldCatalogSource::Structural { .. } => {
                                ObservationSelectionCoordinate::StructuralRecovery
                            }
                            ObservationFieldCatalogSource::VolumeMeshRecovery { .. } => {
                                ObservationSelectionCoordinate::VolumeMeshRecovery
                            }
                            ObservationFieldCatalogSource::EntryHistory {
                                samples,
                            } => {
                                source_catalog = samples.iter().find(|catalog| {
                                    catalog.entry_recovery().is_ok_and(|(_, manifest)| manifest.fields.iter().any(|field| field.field == evidence.source_field && field.sample == sample.coordinate))
                                }).ok_or_else(|| invalid("entry history execution source sample is absent"))?;
                                match source_catalog.source() {
                                    ObservationFieldCatalogSource::VolumeMeshRecovery { .. } => {
                                        ObservationSelectionCoordinate::VolumeMeshRecovery
                                    }
                                    _ => ObservationSelectionCoordinate::StructuralRecovery,
                                }
                            }
                            ObservationFieldCatalogSource::Thermal { manifest } => {
                                let ordinal = manifest
                                    .fields
                                    .iter()
                                    .position(|fields| {
                                        fields.iter().any(|field| {
                                            field.field == evidence.source_field
                                                && field.sample == sample.coordinate
                                        })
                                    })
                                    .ok_or_else(|| {
                                        invalid("history execution source sample is absent")
                                    })?;
                                ObservationSelectionCoordinate::ThermalSample { ordinal }
                            }
                        };
                        let selection =
                            source_catalog.canonical_selection(&self.plan, coordinate)?;
                        let request =
                            ObservationEvaluationRequest::new(selection, self.plan.clone())?;
                        visit(
                            &request,
                            &history.history.observation_id,
                            sample.result,
                            evidence,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn source_field<'a>(
    source: &'a ObservationEvaluationRequest,
    plan: &ObservationPartitionPlan,
    evidence: &ObservationPartitionEvidence,
) -> AnalysisExecutionResult<&'a RecoveredFieldArtifactEntry> {
    let observation = source
        .plan
        .observations
        .iter()
        .find(|item| item.spec.observation_id == plan.observation_id)
        .ok_or_else(|| invalid("execution observation is absent"))?;
    let field = source
        .selection
        .fields()
        .iter()
        .find(|field| field.field == evidence.source_field)
        .ok_or_else(|| invalid("execution field is outside canonical selection"))?;
    if field.definition.identity() != observation.source_field_definition_identity
        || field.sample != evidence.sample
    {
        return Err(invalid("execution source definition or sample differs"));
    }
    Ok(field)
}
