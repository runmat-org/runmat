//! Original thermal field authority, constructed once for a complete catalog.
//! This carrier cannot be decoded or supplied by a caller as a claimed projection.
use super::*;
pub(super) mod resources;

pub(in crate::results::observation) struct ThermalFieldSourceAuthority<'a> {
    workload: &'a ThermalWorkload,
    manifest: &'a ThermalSolverResultManifest,
    projection: AnalysisProjection,
    // Test instrumentation; production releases constructor facts after admission.
    #[cfg(test)]
    construction_resident_bytes: u64,
}

impl<'a> ThermalFieldSourceAuthority<'a> {
    pub(in crate::results::observation) fn new(
        workload: &'a ThermalWorkload,
        manifest: &'a ThermalSolverResultManifest,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<Self> {
        Self::new_admitted(workload, manifest, mesh, u64::MAX)
    }

    pub(in crate::results::observation) fn new_admitted(
        workload: &'a ThermalWorkload,
        manifest: &'a ThermalSolverResultManifest,
        mesh: &SolverMeshArtifact,
        maximum_resident_bytes: u64,
    ) -> AnalysisExecutionResult<Self> {
        let binding = workload.bind_source_identities_admitted(maximum_resident_bytes)?;
        let step = binding.step;
        let ceiling = maximum_resident_bytes.min(step.accuracy.resources.resident_byte_budget());
        let manifest_bytes = manifest.validate_admitted(ceiling)?;
        let projected_bytes =
            AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
        resources::require(projected_bytes, ceiling)?;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let retained = projection
            .clone_resident_bytes_estimate()
            .map_err(invalid)?;
        resources::require(retained, ceiling)?;
        let schedule = crate::execution::ThermalSolveSchedule::derive_admitted(
            &workload.model,
            step,
            workload.procedure,
            ceiling - retained,
        )
        .map_err(invalid)?;
        let construction_resident_bytes = retained
            .checked_add(schedule.construction_resident_bytes())
            .ok_or(AnalysisExecutionError::Limit(
                "thermal source construction workspace",
            ))?
            .max(projected_bytes)
            .max(manifest_bytes)
            .max(binding.resident_bytes);
        resources::require(construction_resident_bytes, ceiling)?;
        if manifest.source_workload != binding.workload
            || manifest.numerical_intent != binding.numerical_intent
            || manifest.producer != binding.producer
            || manifest.topology_digest != Digest::from_bytes(*projection.mesh_digest.bytes())
            || manifest.time_points_s != schedule.points_s()
            || !manifest
                .ordered_node_ids
                .iter()
                .copied()
                .eq(projection.nodes.iter().map(|node| node.node_id))
            || !manifest
                .ordered_element_ids
                .iter()
                .copied()
                .eq(projection.elements.iter().map(|element| element.element_id))
        {
            return Err(invalid(
                "thermal catalog differs from workload, mesh, producer, or schedule",
            ));
        }
        Ok(Self {
            workload,
            manifest,
            projection,
            #[cfg(test)]
            construction_resident_bytes,
        })
    }

    #[cfg(test)]
    pub(in crate::results::observation) fn construction_resident_bytes(&self) -> u64 {
        self.construction_resident_bytes
    }

    pub(in crate::results::observation) fn selection(
        &self,
        ordinal: usize,
        fields: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<ObservationFieldSelection> {
        ObservationFieldSelection::thermal_validated_metadata(
            self.manifest,
            self.projection.projection_digest,
            ordinal,
            fields,
        )
    }

    pub(in crate::results::observation) fn validate_selection(
        &self,
        selection: &ObservationFieldSelection,
    ) -> AnalysisExecutionResult<()> {
        let ObservationFieldSource::ThermalResult { sample_ordinal, .. } = selection.source else {
            return Err(invalid("selection is not thermal"));
        };
        let expected = self.selection(
            sample_ordinal,
            selection.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if selection != &expected {
            return Err(invalid("selection differs from thermal authority"));
        }
        Ok(())
    }

    pub(in crate::results::observation) fn validate_plan(
        &self,
        plan: &super::super::ResolvedObservationPlan,
    ) -> AnalysisExecutionResult<()> {
        let first = self
            .manifest
            .fields
            .first()
            .filter(|fields| !fields.is_empty())
            .ok_or_else(|| invalid("thermal source has no accepted sample"))?;
        let step = self
            .workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == self.workload.step_id)
            .ok_or_else(|| invalid("thermal authored step absent"))?;
        super::super::resolve::verify_authored_observations(step, plan).map_err(invalid)?;
        let definitions = first.iter().map(|entry| &entry.definition);
        super::super::resolve::verify_observation_plan(step, &self.projection, definitions, plan)
            .map_err(observation_error)
    }
}
