//! Original immutable run construction, field chunks and disposable tile preparation.
use super::*;
pub(super) fn prepare_run_artifact(
    run: &AnalysisRunResult,
    created_at: String,
) -> Result<PreparedRunArtifact, String> {
    prepare_run_artifact_with_admission(run, created_at, &mut |_| Ok(()), None)
}

/// Receives field chunks as they are produced so they need not stay resident in staging.
pub(super) trait FieldChunkSink {
    /// Admits one field's encoded payload and chunk copies while they are in flight.
    fn admit_in_flight(&mut self, bytes: u64) -> Result<(), String>;
    fn put(&mut self, digest: Digest, bytes: &[u8]) -> Result<(), String>;
}

/// Prepares the persisted run, charging `admit` before each phase allocates:
/// field payloads and their chunks, render tiles, history chunks, the typed
/// stripping copy, and the stripped run's JSON projection (the only JSON tree
/// built). Charges are cumulative requested estimates, not measured heap.
pub(super) fn prepare_run_artifact_with_admission(
    run: &AnalysisRunResult,
    created_at: String,
    admit: &mut dyn FnMut(u64) -> Result<(), String>,
    mut sink: Option<&mut dyn FieldChunkSink>,
) -> Result<PreparedRunArtifact, String> {
    let mut fields = BTreeMap::<String, PersistedFieldArtifact>::new();
    let mut chunks = HashMap::new();
    let mut seen_field_ids = HashSet::new();
    let mut preparation_error = None;
    crate::analysis::operations::visit_analysis_result_fields(run, |field| {
        if preparation_error.is_some() {
            return;
        }
        if let Err(error) = validate_field_id(&field.field_id) {
            preparation_error = Some(error);
            return;
        }
        if !seen_field_ids.insert(field.field_id.clone()) {
            preparation_error = Some(format!(
                "analysis field ID `{}` appears more than once in one run",
                field.field_id
            ));
            return;
        }
        let expected_count = match checked_element_count(&field.shape) {
            Ok(count) => count,
            Err(error) => {
                preparation_error = Some(error);
                return;
            }
        };
        match &field.values {
            AnalysisFieldValues::HostF64(values) => {
                if values.len() != expected_count {
                    preparation_error = Some(format!(
                        "analysis field `{}` shape requires {expected_count} values but contains {}",
                        field.field_id,
                        values.len()
                    ));
                    return;
                }
                if values.is_empty() {
                    return;
                }
                // The encoded payload and its chunk copies coexist while chunking.
                let payload_bytes = (values.len() as u64).saturating_mul(16);
                let admitted = match sink.as_deref_mut() {
                    Some(sink) => sink.admit_in_flight(payload_bytes),
                    None => admit(payload_bytes),
                };
                if let Err(error) = admitted {
                    preparation_error = Some(error);
                    return;
                }
                let payload = encode_field_values(values);
                let payload_digest = Digest::sha256(&payload);
                let logical_identity = field_logical_identity(field, payload_digest);
                let chunked = match chunk_numerical_payload(
                    AnalysisNumericalObjectKind::Field,
                    logical_identity,
                    &payload,
                    DEFAULT_ANALYSIS_CHUNK_BYTES,
                ) {
                    Ok(chunked) => chunked,
                    Err(error) => {
                        preparation_error = Some(format!(
                            "failed to chunk analysis field `{}`: {error}",
                            field.field_id
                        ));
                        return;
                    }
                };
                let artifact = PersistedFieldArtifact {
                    field_id: field.field_id.clone(),
                    shape: field.shape.clone(),
                    template: AnalysisField::host_f64(
                        &field.field_id,
                        field.layout.clone(),
                        field.shape.clone(),
                        Vec::new(),
                    ),
                    manifest: chunked.manifest.clone(),
                };
                for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
                    if let Some(sink) = sink.as_deref_mut() {
                        if let Err(error) = sink.put(descriptor.digest, &bytes) {
                            preparation_error = Some(error);
                            return;
                        }
                        continue;
                    }
                    match chunks.get(&descriptor.digest) {
                        Some(existing) if existing != &bytes => {
                            preparation_error =
                                Some("analysis field chunk digest collision".to_string());
                            return;
                        }
                        Some(_) => {}
                        None => {
                            chunks.insert(descriptor.digest, bytes);
                        }
                    }
                }
                fields.insert(field.field_id.clone(), artifact);
            }
            AnalysisFieldValues::DeviceRef(reference) => {
                if reference.element_count != expected_count {
                    preparation_error = Some(format!(
                        "device field `{}` shape requires {expected_count} values but its reference declares {}",
                        field.field_id, reference.element_count
                    ));
                } else {
                    preparation_error = Some(format!(
                        "device field `{}` is a process-local reference; materialize it through typed runtime value admission before persistence",
                        field.field_id
                    ));
                }
            }
        }
    });
    if let Some(error) = preparation_error {
        return Err(error);
    }

    let render_tiles = if let Some(topology) = run.render_topology.as_ref() {
        admit(typed_estimate(topology)?)?;
        let (mut index, objects) = render_tiles::partition(topology)?;
        index.field_summaries = summarize_render_fields(run, index.volume_element_count)?;
        index.validate()?;
        for (digest, bytes) in objects {
            match chunks.entry(digest) {
                std::collections::hash_map::Entry::Occupied(existing)
                    if existing.get() != &bytes =>
                {
                    return Err("render tile digest collision".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(bytes);
                }
            }
        }
        Some(index)
    } else {
        None
    };
    let mut observation_histories = Vec::with_capacity(run.run.observation_histories.len());
    for history in &run.run.observation_histories {
        // Canonical bytes plus their chunk copies.
        admit(typed_estimate(history)?.saturating_mul(2))?;
        let payload = history
            .canonical_bytes()
            .map_err(|error| error.to_string())?;
        let chunked = chunk_numerical_payload(
            AnalysisNumericalObjectKind::History,
            history.manifest().identity.digest(),
            &payload,
            DEFAULT_ANALYSIS_CHUNK_BYTES,
        )
        .map_err(|error| error.to_string())?;
        for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
            match chunks.entry(descriptor.digest) {
                std::collections::hash_map::Entry::Occupied(existing)
                    if existing.get() != &bytes =>
                {
                    return Err("observation history chunk digest collision".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(bytes);
                }
            }
        }
        observation_histories.push(PersistedObservationHistory {
            identity: history.manifest().identity,
            observation_id: history.manifest().observation_id.clone(),
            manifest: chunked.manifest,
        });
    }
    // Transient dynamic-balance vectors are numerical payload: chunk them like fields.
    let dynamic_balance = match run.transient_results.as_ref() {
        Some(transient)
            if transient
                .dynamic_balance_snapshots
                .iter()
                .any(|balance| !balance.applied_load.is_empty()) =>
        {
            let mut flat = Vec::new();
            let mut dof_counts = Vec::with_capacity(transient.dynamic_balance_snapshots.len());
            for balance in &transient.dynamic_balance_snapshots {
                dof_counts.push(balance.applied_load.len() as u64);
                for vector in [
                    &balance.applied_load,
                    &balance.inertia_force,
                    &balance.damping_force,
                    &balance.elastic_force,
                    &balance.equilibrium_residual,
                ] {
                    if vector.len() != balance.applied_load.len() {
                        return Err("transient dynamic balance vectors differ in length".into());
                    }
                    flat.extend_from_slice(vector);
                }
            }
            let payload_bytes = (flat.len() as u64).saturating_mul(16);
            match sink.as_deref_mut() {
                Some(sink) => sink.admit_in_flight(payload_bytes)?,
                None => admit(payload_bytes)?,
            }
            let payload = encode_field_values(&flat);
            drop(flat);
            let chunked = chunk_numerical_payload(
                AnalysisNumericalObjectKind::Field,
                Digest::sha256(&payload),
                &payload,
                DEFAULT_ANALYSIS_CHUNK_BYTES,
            )
            .map_err(|error| error.to_string())?;
            for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
                if let Some(sink) = sink.as_deref_mut() {
                    sink.put(descriptor.digest, &bytes)?;
                    continue;
                }
                match chunks.entry(descriptor.digest) {
                    std::collections::hash_map::Entry::Occupied(existing)
                        if existing.get() != &bytes =>
                    {
                        return Err("dynamic balance chunk digest collision".into());
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {}
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(bytes);
                    }
                }
            }
            Some(PersistedDynamicBalance {
                dof_counts,
                manifest: chunked.manifest,
            })
        }
        _ => None,
    };
    admit(typed_estimate(run)?)?;
    let mut stripped_run = run.clone();
    if dynamic_balance.is_some() {
        if let Some(transient) = stripped_run.transient_results.as_mut() {
            for balance in &mut transient.dynamic_balance_snapshots {
                *balance = runmat_analysis_fea::StructuralDynamicBalanceSnapshot {
                    applied_load: Vec::new(),
                    inertia_force: Vec::new(),
                    damping_force: Vec::new(),
                    elastic_force: Vec::new(),
                    equilibrium_residual: Vec::new(),
                };
            }
        }
    }
    stripped_run.observation_scalars =
        runmat_analysis_fea::results::assessment::scalar_observation_inventory(&run.run);
    stripped_run.render_topology = None;
    stripped_run.run.observation_histories = Vec::new();
    crate::analysis::operations::visit_analysis_result_fields_mut(&mut stripped_run, |field| {
        if let AnalysisFieldValues::HostF64(values) = &mut field.values {
            if !values.is_empty() {
                // Release the copied values rather than retaining their capacity.
                *values = Vec::new();
            }
        }
    });
    let field_inventory =
        crate::analysis::operations::collect_analysis_result_fields(&stripped_run);
    // Materialized observations and series reductions are chunked out so the typed metadata is
    // the record's only copy of the run content.
    let observation_payload = if stripped_run.run.observations.is_empty()
        && stripped_run.run.series_reductions.is_empty()
    {
        None
    } else {
        // Field values were stripped above; their shells stay in the inventory and rehydrate
        // from the field chunks when the run is materialized.
        let payload = serde_json::to_vec(&serde_json::json!({
            "observations": &stripped_run.run.observations,
            "series_reductions": &stripped_run.run.series_reductions,
        }))
        .map_err(|error| format!("failed to encode observation payload: {error}"))?;
        stripped_run.run.observations = Vec::new();
        stripped_run.run.series_reductions = Vec::new();
        match sink.as_deref_mut() {
            Some(sink) => sink.admit_in_flight(payload.len() as u64)?,
            None => admit(payload.len() as u64)?,
        }
        let chunked = chunk_numerical_payload(
            AnalysisNumericalObjectKind::History,
            Digest::sha256(&payload),
            &payload,
            DEFAULT_ANALYSIS_CHUNK_BYTES,
        )
        .map_err(|error| error.to_string())?;
        for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
            if let Some(sink) = sink.as_deref_mut() {
                sink.put(descriptor.digest, &bytes)?;
                continue;
            }
            match chunks.entry(descriptor.digest) {
                std::collections::hash_map::Entry::Occupied(existing)
                    if existing.get() != &bytes =>
                {
                    return Err("observation payload chunk digest collision".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(bytes);
                }
            }
        }
        Some(chunked.manifest)
    };
    admit(json_projection_estimate(&stripped_run)?)?;
    // The typed metadata is the single copy; the digest binds its JSON projection.
    let metadata = stripped_run;
    let run_value = serde_json::to_value(&metadata)
        .map_err(|error| format!("failed to encode stripped analysis run: {error}"))?;
    let materialized_field_ids = fields.keys().map(ToOwned::to_owned).collect::<Vec<_>>();
    let run_digest = persisted_run_digest(
        &run_value,
        &field_inventory,
        &materialized_field_ids,
        render_tiles.as_ref(),
        &observation_histories,
        None,
        dynamic_balance.as_ref(),
        observation_payload.as_ref(),
    )?;
    drop(run_value);
    Ok(PreparedRunArtifact {
        persisted: PersistedRunArtifact {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
            created_at,
            op_version: run_operation_version(run)?,
            run_digest,
            run: serde_json::Value::Null,
            metadata,
            field_inventory,
            materialized_field_ids,
            fields: fields.into_values().collect(),
            render_tiles,
            observation_histories,
            original_fields: None,
            dynamic_balance,
            observation_payload,
        },
        chunks,
    })
}

fn typed_estimate(value: &impl serde::Serialize) -> Result<u64, String> {
    runmat_canonical_codec::typed_resident_bytes_estimate(value).map_err(|error| error.to_string())
}

fn json_projection_estimate(value: &impl serde::Serialize) -> Result<u64, String> {
    runmat_canonical_codec::profile_serialized(
        value,
        runmat_canonical_codec::CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
    )
    .map_err(|error| error.to_string())?
    .json_projection_resident_bytes_estimate()
    .ok_or_else(|| "stripped run JSON projection estimate is invalid".to_string())
}

fn summarize_render_fields(
    run: &AnalysisRunResult,
    volume_element_count: usize,
) -> Result<Vec<RenderFieldSummary>, String> {
    let mut summaries = Vec::new();
    let mut failure = None;
    crate::analysis::operations::visit_analysis_result_fields(run, |field| {
        if failure.is_some() {
            return;
        }
        let descriptor = AnalysisFieldDescriptor::from_field(field);
        if !matches!(
            descriptor.location,
            crate::analysis::AnalysisFieldLocation::Node
                | crate::analysis::AnalysisFieldLocation::Element
        ) {
            return;
        }
        let Some(values) = field.as_host_f64() else {
            return;
        };
        let components = descriptor.component_count.unwrap_or(1);
        if components == 0 || values.is_empty() || values.len() % components != 0 {
            return;
        }
        let stride = if descriptor.location == crate::analysis::AnalysisFieldLocation::Element {
            let entities = values.len() / components;
            if volume_element_count == 0 || !entities.is_multiple_of(volume_element_count) {
                return;
            }
            entities / volume_element_count
        } else {
            1
        };
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for samples in values.chunks_exact(components * stride) {
            let value = samples
                .chunks_exact(components)
                .map(|components| {
                    if components.len() == 1 {
                        components[0]
                    } else {
                        components
                            .iter()
                            .map(|value| value * value)
                            .sum::<f64>()
                            .sqrt()
                    }
                })
                .sum::<f64>()
                / stride as f64;
            if !value.is_finite() {
                failure = Some(format!(
                    "field '{}' contains a non-finite plotted value",
                    field.field_id
                ));
                return;
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        if minimum.is_finite() {
            summaries.push(RenderFieldSummary {
                field_id: field.field_id.clone(),
                minimum,
                maximum,
            });
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(summaries)
}
