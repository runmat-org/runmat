//! Dataset write domain helpers.

use super::*;

pub(in crate::builtins::io::data) async fn write_array_full_on(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let mut manifest = crate::data::read_manifest_on(provider, root).await?;
    apply_write_to_manifest_on(
        provider,
        root,
        &mut manifest,
        array_name,
        slice_spec,
        value,
        budget,
    )
    .await?;
    manifest.updated_at = now_rfc3339();
    manifest.txn_sequence = manifest.txn_sequence.saturating_add(1);
    crate::data::write_manifest_on(provider, root, &manifest).await
}

pub(in crate::builtins::io::data) async fn apply_write_to_manifest_on(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    slice_spec: Option<&Value>,
    value: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let meta: DataArrayMeta = manifest
        .arrays
        .get(array_name)
        .cloned()
        .ok_or_else(|| data_error(format!("array '{array_name}' not found")))?;

    if let Some(slice_spec) = slice_spec {
        if apply_slice_write_chunked_on(
            provider, root, manifest, array_name, &meta, slice_spec, value, budget,
        )
        .await?
        {
            return Ok(());
        }
    }

    let payload =
        crate::data::read_array_payload_with_budget_on(provider, root, &meta, budget).await?;
    let next_payload = if let Some(slice_spec) = slice_spec {
        write_slice_payload(&payload, slice_spec, value, budget).await?
    } else {
        DataArrayPayload::from_value(payload.dtype.clone(), value)?
    };

    let (payload_path, chunk_index_path) = crate::data::write_array_payload_on(
        provider,
        root,
        array_name,
        &next_payload,
        &meta.chunk_shape,
        budget,
    )
    .await?;
    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = next_payload.shape.clone();
        updated.data_path = make_rel_data_path(root, &payload_path)?;
        updated.chunk_index_path = Some(make_rel_data_path(root, &chunk_index_path)?);
    }
    Ok(())
}

pub(in crate::builtins::io::data) async fn apply_slice_write_chunked_on(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    manifest: &mut DataManifest,
    array_name: &str,
    meta: &DataArrayMeta,
    slice_spec: &Value,
    value: &Value,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<bool> {
    crate::data::cooperate_data_work(budget, 0).await?;
    let Some(index_rel_path) = &meta.chunk_index_path else {
        return Ok(false);
    };
    let index_path = root.join(index_rel_path);
    if provider.metadata(&index_path).await.is_err() {
        return Ok(false);
    }
    let ranges = parse_slice_spec(slice_spec, &meta.shape)?;
    let rhs_shape: Vec<usize> = ranges
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect();
    let rhs = DataArrayPayload::from_value(meta.dtype.clone(), value)?;
    if rhs.shape != rhs_shape {
        return Err(data_error(format!(
            "SHAPE_MISMATCH: rhs shape {:?} must match target slice shape {:?}",
            rhs.shape, rhs_shape
        )));
    }

    let index_bytes = provider.read(&index_path).await.map_err(|err| {
        crate::data::data_error_with_source(
            format!(
                "failed to read chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;
    let mut chunk_index: DataChunkIndex = serde_json::from_slice(&index_bytes).map_err(|err| {
        crate::data::data_error_with_source(
            format!(
                "failed to parse chunk index '{}': {err}",
                index_path.display()
            ),
            err,
        )
    })?;

    let mut pos_by_key = HashMap::new();
    for (idx, entry) in chunk_index.chunks.iter().enumerate() {
        pos_by_key.insert(entry.key.clone(), idx);
    }
    let touched = touched_chunk_coords(&ranges, &meta.chunk_shape, &meta.shape);
    let mut upload_batch = Vec::<(DataChunkDescriptor, Vec<u8>)>::new();
    let revision = crate::data::create_array_revision(provider, root, array_name).await?;

    for coords in touched {
        crate::data::cooperate_data_work(budget, 0).await?;
        let key = chunk_key(&coords);
        let chunk_start = chunk_start_for_coords(&coords, &meta.chunk_shape);
        let chunk_extent = chunk_extent_for_start(&chunk_start, &meta.chunk_shape, &meta.shape);
        let intersection = chunk_intersection(&ranges, &chunk_start, &chunk_extent);
        if intersection.is_empty() {
            continue;
        }

        let (entry_index, existed, mut entry, mut chunk_payload) = load_or_init_chunk_on(
            provider,
            root,
            array_name,
            &key,
            &coords,
            &chunk_extent,
            &meta.dtype,
            &pos_by_key,
            &chunk_index,
            budget,
        )
        .await?;
        if chunk_payload.imaginary_values.is_none() && rhs.imaginary_values.is_some() {
            chunk_payload.imaginary_values = Some(
                crate::data::DataArrayValues::zeros_cooperative(
                    &meta.dtype,
                    chunk_payload.values.len(),
                    budget,
                )
                .await?,
            );
        } else if chunk_payload.imaginary_values.is_some() && rhs.imaginary_values.is_none() {
            return Err(data_error(
                "CLASS_MISMATCH: real and complex data-array slices cannot be assigned to one another",
            ));
        }

        let mut local = vec![0usize; intersection.len()];
        let intersection_shape: Vec<usize> = intersection
            .iter()
            .map(|r| r.end.saturating_sub(r.start))
            .collect();
        loop {
            crate::data::cooperate_data_work(budget, 1).await?;
            let mut global = Vec::with_capacity(intersection.len());
            for dim in 0..intersection.len() {
                global.push(intersection[dim].start + local[dim]);
            }
            let rhs_index: Vec<usize> = global
                .iter()
                .enumerate()
                .map(|(dim, g)| g.saturating_sub(ranges[dim].start))
                .collect();
            let chunk_index_local: Vec<usize> = global
                .iter()
                .enumerate()
                .map(|(dim, g)| g.saturating_sub(chunk_start[dim]))
                .collect();
            let rhs_linear = linear_index_column_major(&rhs_index, &rhs_shape)?;
            let chunk_linear = linear_index_column_major(&chunk_index_local, &chunk_extent)?;
            chunk_payload
                .values
                .set(chunk_linear, rhs.values.get(rhs_linear)?)?;
            if let (Some(target), Some(source)) =
                (&mut chunk_payload.imaginary_values, &rhs.imaginary_values)
            {
                target.set(chunk_linear, source.get(rhs_linear)?)?;
            }
            if !advance_index(&mut local, &intersection_shape) {
                break;
            }
        }

        let chunk_bytes = serde_json::to_vec(&chunk_payload).map_err(|err| {
            crate::data::data_error_with_source(
                format!("failed to encode chunk payload: {err}"),
                err,
            )
        })?;
        let identity = crate::data::ChunkStorageIdentity::for_payload(&chunk_bytes);
        entry.hash = identity.hash;
        entry.object_id = identity.object_id;
        let chunk_path = revision.join(identity.file_name);
        entry.data_path = make_rel_data_path(root, &chunk_path)?;
        provider
            .write(&chunk_path, &chunk_bytes)
            .await
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!(
                        "failed to write chunk payload '{}': {err}",
                        chunk_path.display()
                    ),
                    err,
                )
            })?;

        entry.coords = coords.clone();
        entry.shape = chunk_extent.clone();
        entry.bytes_raw = chunk_bytes.len() as u64;
        entry.bytes_stored = chunk_bytes.len() as u64;
        if existed {
            chunk_index.chunks[entry_index] = entry.clone();
        } else {
            chunk_index.chunks.push(entry.clone());
            pos_by_key.insert(key.clone(), chunk_index.chunks.len() - 1);
        }
        upload_batch.push((
            DataChunkDescriptor {
                key: key.clone(),
                object_id: entry.object_id.clone(),
                hash: entry.hash.clone(),
                bytes_raw: entry.bytes_raw,
                bytes_stored: entry.bytes_stored,
            },
            chunk_bytes,
        ));
    }

    maybe_upload_chunk_batch_on(provider, root, array_name, upload_batch, budget).await?;
    tracing::info!(
        target: "runmat.data",
        dataset = %root.display(),
        array = array_name,
        touched_chunks = chunk_index.chunks.len(),
        "chunked slice write committed"
    );
    let index_write = serde_json::to_vec(&chunk_index).map_err(|err| {
        crate::data::data_error_with_source(
            format!("failed to encode chunk index json: {err}"),
            err,
        )
    })?;
    let next_index_path = revision.join("index.json");
    provider
        .write(&next_index_path, &index_write)
        .await
        .map_err(|err| {
            crate::data::data_error_with_source(
                format!(
                    "failed to write chunk index '{}': {err}",
                    next_index_path.display()
                ),
                err,
            )
        })?;

    if let Some(updated) = manifest.arrays.get_mut(array_name) {
        updated.shape = meta.shape.clone();
        updated.chunk_index_path = Some(make_rel_data_path(root, &next_index_path)?);
    }
    Ok(true)
}

pub(in crate::builtins::io::data) async fn maybe_upload_chunk_batch_on(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    array_name: &str,
    batch: Vec<(DataChunkDescriptor, Vec<u8>)>,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<()> {
    if batch.is_empty() {
        return Ok(());
    }
    let request = DataChunkUploadRequest {
        dataset_path: root.to_string_lossy().to_string(),
        array: array_name.to_string(),
        chunks: batch.iter().map(|(d, _)| d.clone()).collect(),
    };
    let targets = match provider.data_chunk_upload_targets(&request).await {
        Ok(targets) => targets,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => return Ok(()),
        Err(err) => {
            return Err(crate::data::data_error_with_source(
                format!("failed to request data chunk upload targets: {err}"),
                err,
            ))
        }
    };
    for (descriptor, bytes) in batch {
        crate::data::cooperate_data_work(budget, bytes.len() as u64).await?;
        let target = targets
            .iter()
            .find(|t| t.key == descriptor.key)
            .ok_or_else(|| {
                data_error(format!(
                    "missing upload target for chunk '{}'",
                    descriptor.key
                ))
            })?;
        provider
            .data_upload_chunk(target, &bytes)
            .await
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!("failed to upload chunk '{}': {err}", descriptor.key),
                    err,
                )
            })?;
        tracing::info!(
            target: "runmat.data",
            dataset = %root.display(),
            array = array_name,
            chunk_key = descriptor.key,
            bytes = bytes.len(),
            "chunk upload completed"
        );
    }
    Ok(())
}

pub(in crate::builtins::io::data) fn chunk_rel_path(array_name: &str, object_id: &str) -> String {
    format!("arrays/{array_name}/chunks/{object_id}.json")
}

pub(in crate::builtins::io::data) async fn load_or_init_chunk_on(
    provider: &dyn runmat_filesystem::FsProvider,
    root: &std::path::Path,
    array_name: &str,
    key: &str,
    coords: &[usize],
    chunk_extent: &[usize],
    dtype: &str,
    pos_by_key: &HashMap<String, usize>,
    chunk_index: &DataChunkIndex,
    budget: &mut Option<crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<(usize, bool, DataChunkIndexEntry, DataArrayPayload)> {
    crate::data::cooperate_data_work(budget, 0).await?;
    if let Some(index) = pos_by_key.get(key).copied() {
        let entry = chunk_index
            .chunks
            .get(index)
            .cloned()
            .ok_or_else(|| data_error(format!("chunk index missing key '{key}'")))?;
        let bytes = provider
            .read(&root.join(&entry.data_path))
            .await
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!("failed to read chunk payload '{}': {err}", entry.data_path),
                    err,
                )
            })?;
        let payload: DataArrayPayload = serde_json::from_slice::<DataArrayPayload>(&bytes)
            .map_err(|err| {
                crate::data::data_error_with_source(
                    format!("failed to parse chunk payload '{}': {err}", entry.data_path),
                    err,
                )
            })?
            .normalize_for_dtype(dtype)?;
        return Ok((index, true, entry, payload));
    }

    let object_id = format!("obj_{}", key.replace('.', "_"));
    let entry = DataChunkIndexEntry {
        key: key.to_string(),
        object_id: object_id.clone(),
        hash: String::new(),
        bytes_raw: 0,
        bytes_stored: 0,
        coords: coords.to_vec(),
        shape: chunk_extent.to_vec(),
        data_path: chunk_rel_path(array_name, &object_id),
    };
    let payload =
        DataArrayPayload::zeros_cooperative(dtype.to_string(), chunk_extent.to_vec(), budget)
            .await?;
    Ok((chunk_index.chunks.len(), false, entry, payload))
}
