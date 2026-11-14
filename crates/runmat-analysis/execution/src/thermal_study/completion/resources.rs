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
    ///
    /// Publication authenticates the retained completion by replaying it from its original
    /// objects (`authenticate_original`). These owners coexist, in this order:
    /// 1. this retained completion, for the whole publication;
    /// 2. the replay, which grows to an equal completion while the original phase objects are
    ///    imported; the import's transient transport, verification and decode buffers are the
    ///    profiled `decoded` workspace (its typed contents also become the replay, so the
    ///    replay's storage is counted twice while it is being built);
    /// 3. after the import finishes and its transients are released, the root wire rebuilt from
    ///    the replay for comparison, which matches the caller's retained root wire
    ///    (`root_wire_bytes`), then one field partition or catalog object at a time, both
    ///    inside the profiled import envelope.
    ///
    /// The peak is therefore two completions plus the larger of the import workspace and the
    /// rebuilt root wire.
    pub fn publication_workspace_bytes(
        &self,
        source: &impl CacheImport,
        catalog: &PublishedObjectCatalog,
        root_wire_bytes: u64,
        maximum: u64,
    ) -> AnalysisBridgeResult<u64> {
        catalog
            .validate()
            .map_err(|e| AnalysisBridgeError::Invalid(e.to_string()))?;
        let retained = self
            .retained_storage_bytes(maximum)?
            .checked_mul(2)
            .ok_or_else(|| capacity("thermal replay retained storage overflow"))?;
        let available = maximum.checked_sub(retained).ok_or(
            runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "thermal publication original and replayed completion",
                required: retained,
                maximum,
            },
        )?;
        if root_wire_bytes > available {
            return Err(
                runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "thermal publication replayed root wire",
                    required: retained.saturating_add(root_wire_bytes),
                    maximum,
                }
                .into(),
            );
        }
        let decoded = crate::study_closure::profile_import_workspace_phases(
            source,
            &[&catalog.objects],
            &self.result.host.artifact_access,
            self.output_policy().inventory_limits(),
            available,
        )?;
        retained
            .checked_add(decoded.max(root_wire_bytes))
            .ok_or_else(|| capacity("thermal replay workspace overflow"))
    }
}

impl ThermalStudyCoordinator {
    /// Pre-solve upper bound on the memory publication needs, from the planned solve.
    ///
    /// Requires the prepared solve task: its plan's projected output transports bound the encoded
    /// result (`E`: every recovered field of every sample, the receipt and the state profiles). A decoded
    /// recovered-field partition holds, per entity, `k >= 1` f64 values and one index, which
    /// encode to at least `9k + 1` bytes and decode to `8k + 8`, so its typed storage is at most
    /// twice its encoding. Each declared state and each field observation is a copy of at most
    /// the result, encoded and decoded.
    ///
    /// The bulk owners that coexist during publication's source phase (the binding phase):
    /// - runtime staging: the public completion projection and its copy (definition plus decoded
    ///   fields and states), and the solver mesh;
    /// - the retained completion and its replay: definition, mesh, encoded result, states and
    ///   field observations, each twice (see `publication_workspace_bytes`);
    /// - the replay's import: three transport-sized buffers of every encoded object plus its
    ///   typed contents.
    ///
    /// Small metadata (closures, receipts, usage and accuracy evidence, scalar observations,
    /// the root wire) is charged at the codec's fixed per-contract allowance for every phase
    /// object.
    pub fn planned_publication_bytes(&self) -> AnalysisBridgeResult<u64> {
        let task = self
            .solve_task
            .as_ref()
            .ok_or_else(|| invalid("thermal publication planning requires the prepared solve"))?;
        let overflow = || capacity("thermal publication plan overflow");
        let step = self.preparation.selected_step();
        let definition = typed(&(
            &self.preparation.product.compiled,
            &self.preparation.product.workload,
        ))?;
        let mesh = typed(&self.preparation.mesh)?;
        let runmat_analysis_fea::planning::resource::FeaResourceDemand::ThermalSolve(demand) =
            &task.resource_plan.input.demand
        else {
            return Err(invalid("thermal solve plan carries another demand"));
        };
        let result = [
            &demand.result_output,
            &demand.receipt_output,
            &demand.state_profiles_output,
        ]
        .iter()
        .try_fold(0_u64, |total, output| {
            total.checked_add(output.transport_closure_bytes_upper_bound.0)
        })
        .ok_or_else(overflow)?;
        let copies = (step.state_outputs.len() as u64)
            .checked_add(
                step.observations
                    .iter()
                    .filter(|o| matches!(o, runmat_analysis_core::ObservationRequest::Field(_)))
                    .count() as u64,
            )
            .and_then(|n| n.checked_add(1))
            .ok_or_else(overflow)?;
        let encoded = result.checked_mul(copies).ok_or_else(overflow)?;
        let decoded = encoded.checked_mul(2).ok_or_else(overflow)?;
        // The completion keeps the result encoded; states and field observations also typed.
        let typed_copies = result
            .checked_mul(2)
            .and_then(|n| n.checked_mul(copies - 1))
            .ok_or_else(overflow)?;
        // Solve, observations, accuracy, states, baseline, terminal usage and publication, each
        // with its response, evidence, manifest and objects.
        let phases = (step.observations.len() as u64)
            .checked_add(step.state_outputs.len() as u64)
            .and_then(|n| n.checked_add(5))
            .ok_or_else(overflow)?;
        let metadata =
            runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
                0,
                phases.checked_mul(4).ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
        let sum = |terms: &[u64]| {
            terms
                .iter()
                .try_fold(0_u64, |total, term| total.checked_add(*term))
                .ok_or_else(overflow)
        };
        let projection = sum(&[definition, decoded, metadata])?;
        let completion = sum(&[definition, mesh, encoded, typed_copies, metadata])?;
        let import = sum(&[
            encoded.checked_mul(3).ok_or_else(overflow)?,
            decoded,
            metadata,
        ])?;
        sum(&[projection, projection, mesh, completion, completion, import])
    }

    /// Refuse, before the solve, a study whose publication cannot fit `maximum`, the memory
    /// publication is admitted to use. Returns the planned requirement.
    pub fn admit_publication(&self, maximum: u64) -> AnalysisBridgeResult<u64> {
        let required = self.planned_publication_bytes()?;
        if required > maximum {
            return Err(
                runmat_analysis_fea::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "thermal result publication",
                    required,
                    maximum,
                }
                .into(),
            );
        }
        Ok(required)
    }
}
