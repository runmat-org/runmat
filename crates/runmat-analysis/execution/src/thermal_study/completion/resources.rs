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
/// Encoded bound of `count` observation entity identities, each charged as the widest
/// single-identity entity.
fn entity_identity_bytes(count: usize) -> AnalysisBridgeResult<u64> {
    struct Repeated(usize);
    impl serde::Serialize for Repeated {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let widest =
                runmat_analysis_fea::results::observation::ObservationEntity::IntegrationPoint(
                    runmat_meshing_core::StableDigest::from_bytes([u8::MAX; 32]),
                );
            serializer.collect_seq(std::iter::repeat_n(&widest, self.0))
        }
    }
    runmat_canonical_codec::profile_serialized_shape(
        &Repeated(count),
        runmat_canonical_codec::CanonicalLimits::platform_content(65_536, 64),
    )
    .map(|profile| profile.encoded_bytes_upper_bound)
    .map_err(|_| capacity("thermal observation identity bound overflow"))
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
    /// Requires the prepared solve task. Its plan's projected output transports bound the encoded
    /// result, receipt and state profiles (`R`). The task also prices the decoded result as the
    /// shared codec estimators price it: typed storage `T` (each field partition from its shape,
    /// the header at the codec's per-byte typed decode bound) and the import workspace `M` (see
    /// `PreparedThermalPrimarySolve::result_import_workspace_bytes`).
    ///
    /// States and observations are copies of parts of the result. A thermal state holds the
    /// terminal temperature (and interface temperature) with the node IDs; a field observation
    /// holds one sample of one field over its selected entities, with an identity per entity.
    /// Their values and ID inventories are bounded by the result header plus one sample of every
    /// recovered field (`S`, the task's `sample_output_bytes`); a field observation's entity
    /// identities add at most one widest identity per mesh entity (`I`). A history observation
    /// holds one field over at most every sample: at most the result plus `I` per sample. These
    /// products (`P`, encoded) and the receipt and profiles are decoded at the codec's per-byte
    /// typed decode bound.
    ///
    /// The bulk owners that coexist during publication's source phase (the binding phase):
    /// - runtime staging: the public completion projection and its copy (definition plus decoded
    ///   fields, states and observations), and the solver mesh;
    /// - the retained completion and its replay: definition, mesh, encoded result and products,
    ///   and the products' typed values, each twice (see `publication_workspace_bytes`);
    /// - the replay's import: `M` for the result; three transport copies plus the typed decode
    ///   for every other object.
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
        let result = demand.result_output.transport_closure_bytes_upper_bound.0;
        let small = demand
            .receipt_output
            .transport_closure_bytes_upper_bound
            .0
            .checked_add(
                demand
                    .state_profiles_output
                    .transport_closure_bytes_upper_bound
                    .0,
            )
            .ok_or_else(overflow)?;
        let states = step.state_outputs.len() as u64;
        let count = |field: bool| {
            step.observations
                .iter()
                .filter(|o| match o {
                    runmat_analysis_core::ObservationRequest::Field(_) => field,
                    runmat_analysis_core::ObservationRequest::History(_) => !field,
                    _ => false,
                })
                .count() as u64
        };
        let (field_observations, histories) = (count(true), count(false));
        let topology = &self.preparation.mesh.topology;
        let identities =
            entity_identity_bytes(topology.nodes.len().max(topology.volume_elements.len()))?;
        let field_observation = task
            .sample_output_bytes
            .checked_add(identities)
            .ok_or_else(overflow)?;
        let history = identities
            .checked_mul(demand.sample_count.0)
            .and_then(|n| n.checked_add(result))
            .ok_or_else(overflow)?;
        let products = task
            .sample_output_bytes
            .checked_mul(states)
            .and_then(|n| n.checked_add(field_observation.checked_mul(field_observations)?))
            .and_then(|n| n.checked_add(history.checked_mul(histories)?))
            .ok_or_else(overflow)?;
        // Everything but the result, encoded, and its typed storage.
        let others = small.checked_add(products).ok_or_else(overflow)?;
        let others_typed =
            runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
                others, 0,
            )
            .ok_or_else(overflow)?;
        let products_typed =
            runmat_canonical_codec::projected_canonical_typed_decode_resident_bytes_estimate(
                products, 0,
            )
            .ok_or_else(overflow)?;
        let encoded = result.checked_add(others).ok_or_else(overflow)?;
        // Solve, observations, accuracy, states, baseline, terminal usage and publication, each
        // with its response, evidence, manifest and objects.
        let phases = (step.observations.len() as u64)
            .checked_add(states)
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
        let projection = sum(&[definition, task.result_typed_bytes, others_typed, metadata])?;
        let completion = sum(&[definition, mesh, encoded, products_typed, metadata])?;
        let import = sum(&[
            task.result_import_bytes,
            others.checked_mul(3).ok_or_else(overflow)?,
            others_typed,
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
