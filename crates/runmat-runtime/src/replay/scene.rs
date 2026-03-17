use chrono::Utc;
use futures::executor;
use serde::{Deserialize, Serialize};
#[cfg(feature = "plot-core")]
use serde_json::Value;

use crate::replay::limits::ReplayLimits;
use crate::runtime_error::{replay_error, replay_error_with_source, ReplayErrorKind};
use crate::RuntimeError;

const SCENE_SCHEMA_VERSION: u32 = 1;
const SCENE_KIND: &str = "figure-scene";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FigureScenePayload {
    schema_version: u32,
    kind: String,
    created_at: String,
    figure: runmat_plot::event::FigureScene,
}

#[cfg(feature = "plot-core")]
pub fn encode_figure_scene_payload(
    scene: &runmat_plot::event::FigureScene,
) -> Result<Vec<u8>, RuntimeError> {
    encode_figure_scene_payload_with_limits(scene, ReplayLimits::default())
}

#[cfg(feature = "plot-core")]
pub fn encode_figure_scene_payload_with_limits(
    scene: &runmat_plot::event::FigureScene,
    limits: ReplayLimits,
) -> Result<Vec<u8>, RuntimeError> {
    if scene.plots.len() > limits.max_scene_plots {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "figure scene contains {} plots, exceeding limit {}",
                scene.plots.len(),
                limits.max_scene_plots
            ),
        ));
    }
    let payload = FigureScenePayload {
        schema_version: SCENE_SCHEMA_VERSION,
        kind: SCENE_KIND.to_string(),
        created_at: Utc::now().to_rfc3339(),
        figure: scene.clone(),
    };
    let encoded = serde_json::to_vec(&payload).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to encode figure replay payload",
            err,
        )
    })?;
    if encoded.len() > limits.max_scene_payload_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "figure scene payload is {} bytes, exceeding limit {}",
                encoded.len(),
                limits.max_scene_payload_bytes
            ),
        ));
    }
    Ok(encoded)
}

#[cfg(feature = "plot-core")]
pub fn decode_figure_scene_payload(
    bytes: &[u8],
) -> Result<runmat_plot::event::FigureScene, RuntimeError> {
    decode_figure_scene_payload_with_limits(bytes, ReplayLimits::default())
}

#[cfg(feature = "plot-core")]
pub fn decode_figure_scene_payload_with_limits(
    bytes: &[u8],
    limits: ReplayLimits,
) -> Result<runmat_plot::event::FigureScene, RuntimeError> {
    executor::block_on(decode_figure_scene_payload_with_limits_async(bytes, limits))
}

#[cfg(feature = "plot-core")]
pub async fn decode_figure_scene_payload_async(
    bytes: &[u8],
) -> Result<runmat_plot::event::FigureScene, RuntimeError> {
    decode_figure_scene_payload_with_limits_async(bytes, ReplayLimits::default()).await
}

#[cfg(feature = "plot-core")]
pub async fn decode_figure_scene_payload_with_limits_async(
    bytes: &[u8],
    limits: ReplayLimits,
) -> Result<runmat_plot::event::FigureScene, RuntimeError> {
    if bytes.len() > limits.max_scene_payload_bytes {
        return Err(replay_error(
            ReplayErrorKind::PayloadTooLarge,
            format!(
                "figure scene payload is {} bytes, exceeding limit {}",
                bytes.len(),
                limits.max_scene_payload_bytes
            ),
        ));
    }
    let mut payload_json: Value = serde_json::from_slice(bytes).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to decode figure replay payload",
            err,
        )
    })?;
    hydrate_scene_data_refs_async(&mut payload_json).await?;
    let payload: FigureScenePayload = serde_json::from_value(payload_json).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            format!("failed to decode hydrated figure replay payload: {err}"),
            err,
        )
    })?;
    if payload.schema_version != SCENE_SCHEMA_VERSION {
        return Err(replay_error(
            ReplayErrorKind::UnsupportedSchema,
            format!(
                "unsupported figure replay schema version {}",
                payload.schema_version
            ),
        ));
    }
    if payload.kind != SCENE_KIND {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!("unexpected replay payload kind '{}'", payload.kind),
        ));
    }
    if payload.figure.plots.len() > limits.max_scene_plots {
        return Err(replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "figure scene contains {} plots, exceeding limit {}",
                payload.figure.plots.len(),
                limits.max_scene_plots
            ),
        ));
    }
    Ok(payload.figure)
}

#[cfg(feature = "plot-core")]
async fn hydrate_scene_data_refs_async(payload: &mut Value) -> Result<(), RuntimeError> {
    let Some(plots) = payload
        .get_mut("figure")
        .and_then(Value::as_object_mut)
        .and_then(|figure| figure.get_mut("plots"))
        .and_then(Value::as_array_mut)
    else {
        return Ok(());
    };
    for plot in plots.iter_mut() {
        let Some(kind) = plot.get("kind").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "surface" => {
                hydrate_plot_field_async(plot, "x").await?;
                hydrate_plot_field_async(plot, "y").await?;
                hydrate_plot_field_async(plot, "z").await?;
            }
            "scatter3" => {
                hydrate_plot_field_async(plot, "points").await?;
                hydrate_plot_field_async(plot, "colorsRgba").await?;
                hydrate_plot_field_async(plot, "pointSizes").await?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
async fn hydrate_plot_field_async(plot: &mut Value, field: &str) -> Result<(), RuntimeError> {
    let Some(obj) = plot.as_object_mut() else {
        return Ok(());
    };
    let Some(value) = obj.get(field).cloned() else {
        return Ok(());
    };
    let Some(data_ref) = parse_data_ref(&value) else {
        return Ok(());
    };
    let payload = read_scene_array_payload_async(&data_ref).await?;
    let target_shape = if data_ref.shape.is_empty() {
        payload.shape.as_slice()
    } else {
        data_ref.shape.as_slice()
    };
    let hydrated = shape_values_to_json(&payload.values, target_shape)?;
    obj.insert(field.to_string(), hydrated);
    Ok(())
}

async fn read_scene_array_payload_async(
    data_ref: &SceneDataRef,
) -> Result<crate::data::DataArrayPayload, RuntimeError> {
    let dataset_root = crate::data::dataset_root(&data_ref.dataset_path);
    match crate::data::read_manifest_async(&dataset_root).await {
        Ok(manifest) => {
            let meta = manifest.arrays.get(&data_ref.array).ok_or_else(|| {
                replay_error(
                    ReplayErrorKind::ImportRejected,
                    format!(
                        "scene dataset '{}' missing array '{}'",
                        data_ref.dataset_path, data_ref.array
                    ),
                )
            })?;
            crate::data::read_array_payload_async(&dataset_root, meta)
                .await
                .map_err(|err| {
                    replay_error(
                        ReplayErrorKind::ImportRejected,
                        format!(
                            "failed reading scene dataset array '{}.{}': {}",
                            data_ref.dataset_path, data_ref.array, err
                        ),
                    )
                })
        }
        Err(manifest_err) => {
            if data_ref.chunks.is_empty() {
                return Err(replay_error(
                    ReplayErrorKind::ImportRejected,
                    format!(
                        "failed to read scene dataset manifest '{}': {}",
                        data_ref.dataset_path, manifest_err
                    ),
                ));
            }
            let mut values = Vec::new();
            let chunk_payloads = read_scene_chunks_bytes_async(&data_ref.chunks, data_ref)
                .await
                .map_err(|err| {
                    replay_error(
                        ReplayErrorKind::ImportRejected,
                        format!("failed reading scene data chunks: {}", err),
                    )
                })?;
            for (chunk, bytes) in data_ref.chunks.iter().zip(chunk_payloads.into_iter()) {
                let payload: crate::data::DataArrayPayload = serde_json::from_slice(&bytes)
                    .map_err(|err| {
                        replay_error(
                            ReplayErrorKind::ImportRejected,
                            format!(
                                "failed decoding scene data chunk '{}': {}",
                                chunk
                                    .src
                                    .as_deref()
                                    .or(chunk.artifact_id.as_deref())
                                    .unwrap_or("<unknown>"),
                                err
                            ),
                        )
                    })?;
                values.extend(payload.values);
            }
            Ok(crate::data::DataArrayPayload {
                dtype: data_ref.dtype.clone().unwrap_or_else(|| "f64".to_string()),
                shape: vec![values.len()],
                values,
            })
        }
    }
}

#[cfg(feature = "plot-core")]
#[cfg(test)]
fn read_scene_chunk_bytes(
    chunk: &SceneDataChunkRef,
    data_ref: &SceneDataRef,
) -> std::io::Result<Vec<u8>> {
    let chunks = vec![chunk.clone()];
    let mut batch = executor::block_on(read_scene_chunks_bytes_async(&chunks, data_ref))?;
    batch
        .pop()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "scene chunk missing"))
}

async fn read_scene_chunks_bytes_async(
    chunks: &[SceneDataChunkRef],
    data_ref: &SceneDataRef,
) -> std::io::Result<Vec<Vec<u8>>> {
    let per_chunk_candidates = chunks
        .iter()
        .map(|chunk| build_scene_chunk_candidates(chunk, data_ref))
        .collect::<std::io::Result<Vec<_>>>()?;

    let mut unique_paths = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for candidates in &per_chunk_candidates {
        for candidate in candidates {
            if seen.insert(candidate.clone()) {
                unique_paths.push(candidate.clone());
            }
        }
    }

    let request_paths = unique_paths
        .iter()
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    let batch = runmat_filesystem::read_many_async(&request_paths).await?;
    let mut resolved = std::collections::HashMap::new();
    for (index, entry) in batch.into_iter().enumerate() {
        if let Some(path) = unique_paths.get(index) {
            resolved.insert(path.clone(), entry.into_bytes());
        }
    }

    let mut out = Vec::with_capacity(chunks.len());
    for (chunk, candidates) in chunks.iter().zip(per_chunk_candidates.into_iter()) {
        let mut found = None;
        for candidate in candidates {
            if let Some(Some(bytes)) = resolved.get(&candidate) {
                found = Some(bytes.clone());
                break;
            }
        }
        let bytes = found.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "unable to resolve scene chunk from refs {}",
                    chunk
                        .src
                        .as_deref()
                        .or(chunk.artifact_id.as_deref())
                        .unwrap_or("<unknown>")
                ),
            )
        })?;
        out.push(bytes);
    }
    Ok(out)
}

fn build_scene_chunk_candidates(
    chunk: &SceneDataChunkRef,
    data_ref: &SceneDataRef,
) -> std::io::Result<Vec<String>> {
    let mut base_paths: Vec<String> = Vec::new();
    if let Some(src) = &chunk.src {
        let normalized = src.trim();
        if !normalized.is_empty() {
            base_paths.push(normalized.to_string());
        }
    }
    if let Some(artifact_id) = &chunk.artifact_id {
        base_paths.extend(chunk_paths_from_artifact_id(
            artifact_id,
            data_ref.dtype.as_deref(),
        ));
    }
    if base_paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "empty scene chunk reference",
        ));
    }

    let mut candidates: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut push = |candidate: String| {
        if candidate.is_empty() {
            return;
        }
        if seen.insert(candidate.clone()) {
            candidates.push(candidate);
        }
    };

    for base in &base_paths {
        push(base.clone());
        let stripped = base.trim_start_matches("./");
        if !stripped.is_empty() {
            push(stripped.to_string());
            if !stripped.starts_with('/') {
                push(format!("/{stripped}"));
            }
        }
        if !base.starts_with('/') {
            push(format!("/{base}"));
        }
    }

    let cwd = runmat_filesystem::current_dir().ok();
    if let Some(cwd) = &cwd {
        for base in &base_paths {
            let stripped = base.trim_start_matches("./");
            push(cwd.join(base).to_string_lossy().to_string());
            if !stripped.is_empty() {
                push(cwd.join(stripped).to_string_lossy().to_string());
                let mut current = Some(cwd.as_path());
                while let Some(dir) = current {
                    push(dir.join(stripped).to_string_lossy().to_string());
                    current = dir.parent();
                }
            }
        }
    }

    Ok(candidates)
}

fn chunk_paths_from_artifact_id(artifact_id: &str, dtype: Option<&str>) -> Vec<String> {
    let Some(hash_hex) = artifact_id.strip_prefix("sha256:") else {
        return Vec::new();
    };
    if hash_hex.len() < 2 {
        return Vec::new();
    }
    let prefix = &hash_hex[..2];
    let mut suffixes = Vec::new();
    match dtype {
        Some("f32") => {
            suffixes.push("f32.chunk.json");
            suffixes.push("f64.chunk.json");
        }
        _ => {
            suffixes.push("f64.chunk.json");
            suffixes.push("f32.chunk.json");
        }
    }
    suffixes.push("chunk.json");
    suffixes.push("json");
    suffixes.push("bin");
    suffixes
        .into_iter()
        .map(|suffix| format!(".artifacts/objects/{prefix}/{hash_hex}.{suffix}"))
        .collect()
}

#[cfg(feature = "plot-core")]
fn shape_values_to_json(values: &[f64], shape: &[usize]) -> Result<Value, RuntimeError> {
    if shape.is_empty() {
        return Ok(Value::from(values.first().copied().unwrap_or(0.0)));
    }
    if shape.len() == 1 {
        return Ok(Value::Array(
            values.iter().copied().map(Value::from).collect(),
        ));
    }
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        if rows * cols != values.len() {
            return Err(replay_error(
                ReplayErrorKind::ImportRejected,
                format!(
                    "scene dataset shape mismatch: {:?} has {} values",
                    shape,
                    values.len()
                ),
            ));
        }
        let mut matrix = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                let idx = r + c * rows;
                row.push(Value::from(values[idx]));
            }
            matrix.push(Value::Array(row));
        }
        return Ok(Value::Array(matrix));
    }
    Err(replay_error(
        ReplayErrorKind::ImportRejected,
        format!("unsupported scene ref rank {}", shape.len()),
    ))
}

#[cfg(feature = "plot-core")]
struct SceneDataRef {
    dataset_path: String,
    array: String,
    shape: Vec<usize>,
    dtype: Option<String>,
    chunks: Vec<SceneDataChunkRef>,
}

#[derive(Clone, Debug)]
struct SceneDataChunkRef {
    src: Option<String>,
    artifact_id: Option<String>,
}

#[cfg(feature = "plot-core")]
fn parse_data_ref(value: &Value) -> Option<SceneDataRef> {
    let obj = value.as_object()?;
    let ref_kind = obj.get("refKind")?.as_str()?;
    if ref_kind != "runmat-data-array-v1" {
        return None;
    }
    let dataset_path = obj.get("datasetPath")?.as_str()?.to_string();
    let array = obj.get("array")?.as_str()?.to_string();
    let shape = obj
        .get("shape")
        .and_then(Value::as_array)
        .map(|dims| {
            dims.iter()
                .filter_map(Value::as_u64)
                .map(|dim| dim as usize)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let dtype = obj
        .get("dtype")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let chunks = obj
        .get("chunks")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    let obj = entry.as_object()?;
                    let src = obj
                        .get("src")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    let artifact_id = obj
                        .get("artifactId")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    if src.is_none() && artifact_id.is_none() {
                        return None;
                    }
                    Some(SceneDataChunkRef { src, artifact_id })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(SceneDataRef {
        dataset_path,
        array,
        shape,
        dtype,
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Instant;

    #[cfg(feature = "plot-core")]
    fn write_scene_dataset(
        root_path: &str,
        arrays: &[(&str, Vec<f64>)],
    ) -> Result<(), RuntimeError> {
        let root = crate::data::dataset_root(root_path);
        let mut manifest = crate::data::DataManifest {
            schema_version: 1,
            format: "runmat-data-scene-v1".to_string(),
            dataset_id: "scene_ds".to_string(),
            name: None,
            created_at: crate::data::now_rfc3339(),
            updated_at: crate::data::now_rfc3339(),
            arrays: BTreeMap::new(),
            attrs: BTreeMap::new(),
            txn_sequence: 0,
        };
        for (array_name, values) in arrays {
            let payload = crate::data::DataArrayPayload {
                dtype: "f64".to_string(),
                shape: vec![values.len()],
                values: values.clone(),
            };
            let chunk = vec![std::cmp::max(1usize, values.len())];
            let (payload_path, chunk_index_path) = futures::executor::block_on(
                crate::data::write_array_payload_async(&root, array_name, &payload, &chunk),
            )?;
            let data_path = payload_path
                .strip_prefix(&root)
                .map_err(|err| replay_error(ReplayErrorKind::ImportRejected, err.to_string()))?
                .to_string_lossy()
                .to_string();
            let chunk_index_rel = chunk_index_path
                .strip_prefix(&root)
                .map_err(|err| replay_error(ReplayErrorKind::ImportRejected, err.to_string()))?
                .to_string_lossy()
                .to_string();
            manifest.arrays.insert(
                (*array_name).to_string(),
                crate::data::DataArrayMeta {
                    dtype: "f64".to_string(),
                    shape: vec![values.len()],
                    chunk_shape: chunk,
                    order: "column_major".to_string(),
                    codec: "none".to_string(),
                    chunk_index_path: Some(chunk_index_rel),
                    data_path,
                },
            );
        }
        futures::executor::block_on(crate::data::write_manifest_async(&root, &manifest))?;
        Ok(())
    }

    #[cfg(feature = "plot-core")]
    fn make_surface_ref_payload(dataset_path: &str) -> Vec<u8> {
        let surface = runmat_plot::plots::SurfacePlot::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![vec![0.0, 1.0], vec![1.0, 2.0]],
        )
        .expect("surface");
        let mut figure = runmat_plot::plots::Figure::new();
        figure.add_surface_plot(surface);
        let scene = runmat_plot::event::FigureScene::capture(&figure);

        let mut payload = serde_json::to_value(FigureScenePayload {
            schema_version: SCENE_SCHEMA_VERSION,
            kind: SCENE_KIND.to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            figure: scene,
        })
        .expect("serialize payload");

        let plot = payload["figure"]["plots"]
            .as_array_mut()
            .and_then(|plots| plots.get_mut(0))
            .expect("first plot");
        plot["x"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": dataset_path,
            "array": "x",
            "shape": [2]
        });
        plot["y"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": dataset_path,
            "array": "y",
            "shape": [2]
        });
        plot["z"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": dataset_path,
            "array": "z",
            "shape": [2, 2]
        });
        serde_json::to_vec(&payload).expect("payload bytes")
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_schema_mismatch_rejects() {
        let scene = runmat_plot::event::FigureScene::capture(&runmat_plot::plots::Figure::new());
        let mut payload = serde_json::to_value(FigureScenePayload {
            schema_version: SCENE_SCHEMA_VERSION,
            kind: SCENE_KIND.to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            figure: scene,
        })
        .expect("serialize payload");
        payload["schemaVersion"] = serde_json::json!(99u32);
        let bytes = serde_json::to_vec(&payload).expect("serialize bytes");

        let err = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect_err("expected schema rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::UnsupportedSchema.identifier())
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_payload_too_large_rejects() {
        let scene = runmat_plot::event::FigureScene::capture(&runmat_plot::plots::Figure::new());
        let bytes = encode_figure_scene_payload_with_limits(
            &scene,
            ReplayLimits {
                max_scene_payload_bytes: 1024,
                ..ReplayLimits::default()
            },
        )
        .expect("encode scene");
        let err = decode_figure_scene_payload_with_limits(
            &bytes,
            ReplayLimits {
                max_scene_payload_bytes: 1,
                ..ReplayLimits::default()
            },
        )
        .expect_err("expected payload rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::PayloadTooLarge.identifier())
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_data_refs_hydrate_via_runtime_data_store() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_ref_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");
        let dataset_path = root.join("surface_values.data");
        let dataset_str = dataset_path.to_string_lossy().to_string();
        write_scene_dataset(
            &dataset_str,
            &[
                ("x", vec![0.0, 1.0]),
                ("y", vec![0.0, 1.0]),
                ("z", vec![0.0, 1.0, 1.0, 2.0]),
            ],
        )
        .expect("write scene dataset");

        let bytes = make_surface_ref_payload(&dataset_str);
        let hydrated = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect("decode with hydrated refs");
        match hydrated.plots.first() {
            Some(runmat_plot::event::ScenePlot::Surface { x, y, z, .. }) => {
                assert_eq!(x, &vec![0.0, 1.0]);
                assert_eq!(y, &vec![0.0, 1.0]);
                assert_eq!(z, &vec![vec![0.0, 1.0], vec![1.0, 2.0]]);
            }
            other => panic!("unexpected hydrated plot: {other:?}"),
        }
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_data_refs_missing_chunk_rejects_import() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_ref_missing_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");
        let dataset_path = root.join("surface_values.data");
        let dataset_str = dataset_path.to_string_lossy().to_string();
        write_scene_dataset(
            &dataset_str,
            &[
                ("x", vec![0.0, 1.0]),
                ("y", vec![0.0, 1.0]),
                ("z", vec![0.0, 1.0, 1.0, 2.0]),
            ],
        )
        .expect("write scene dataset");
        let missing_chunk = dataset_path.join("arrays/z/chunks/obj_0.json");
        futures::executor::block_on(runmat_filesystem::remove_file_async(&missing_chunk))
            .expect("remove z chunk");
        let bytes = make_surface_ref_payload(&dataset_str);
        let err = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect_err("expected import rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::ImportRejected.identifier())
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_data_refs_corrupt_chunk_rejects_import() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_ref_corrupt_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");
        let dataset_path = root.join("surface_values.data");
        let dataset_str = dataset_path.to_string_lossy().to_string();
        write_scene_dataset(
            &dataset_str,
            &[
                ("x", vec![0.0, 1.0]),
                ("y", vec![0.0, 1.0]),
                ("z", vec![0.0, 1.0, 1.0, 2.0]),
            ],
        )
        .expect("write scene dataset");
        let corrupt_chunk = dataset_path.join("arrays/z/chunks/obj_0.json");
        futures::executor::block_on(runmat_filesystem::write_async(&corrupt_chunk, b"not-json"))
            .expect("corrupt z chunk");
        let bytes = make_surface_ref_payload(&dataset_str);
        let err = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect_err("expected import rejection");
        assert_eq!(
            err.identifier(),
            Some(ReplayErrorKind::ImportRejected.identifier())
        );
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_data_refs_chunk_fallback_without_manifest_succeeds() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_ref_chunk_fallback_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");

        let chunk_path = root.join("z_chunk.json");
        let chunk_payload = serde_json::json!({
            "dtype": "f64",
            "shape": [4],
            "values": [0.0, 1.0, 1.0, 2.0]
        });
        futures::executor::block_on(runmat_filesystem::write_async(
            &chunk_path,
            serde_json::to_vec(&chunk_payload).expect("chunk json"),
        ))
        .expect("write chunk");

        let mut payload = serde_json::from_slice::<serde_json::Value>(&make_surface_ref_payload(
            ".artifacts/datasets/missing",
        ))
        .expect("payload json");
        let plot = payload["figure"]["plots"]
            .as_array_mut()
            .and_then(|plots| plots.get_mut(0))
            .expect("first plot");
        plot["z"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": ".artifacts/datasets/missing",
            "array": "z",
            "shape": [2, 2],
            "dtype": "f64",
            "chunks": [
                { "src": chunk_path.to_string_lossy() }
            ]
        });
        plot["x"] = serde_json::json!([0.0, 1.0]);
        plot["y"] = serde_json::json!([0.0, 1.0]);

        let bytes = serde_json::to_vec(&payload).expect("payload bytes");
        let hydrated = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect("decode with chunk fallback");
        match hydrated.plots.first() {
            Some(runmat_plot::event::ScenePlot::Surface { z, .. }) => {
                assert_eq!(z, &vec![vec![0.0, 1.0], vec![1.0, 2.0]]);
            }
            other => panic!("unexpected hydrated plot: {other:?}"),
        }
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_chunk_reader_resolves_provider_relative_paths() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_relpath_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        let chunk_rel = ".artifacts/objects/ab/test_chunk.f64.chunk.json";
        let chunk_path = root.join(chunk_rel);
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(
            chunk_path.parent().expect("chunk parent"),
        ))
        .expect("create chunk dir");
        futures::executor::block_on(runmat_filesystem::write_async(
            &chunk_path,
            b"{\"values\":[1,2,3]}",
        ))
        .expect("write chunk");

        let previous = runmat_filesystem::current_dir().expect("current dir");
        runmat_filesystem::set_current_dir(&root).expect("set cwd");
        let chunk = SceneDataChunkRef {
            src: Some(".artifacts/objects/ab/test_chunk.f64.chunk.json".to_string()),
            artifact_id: None,
        };
        let data_ref = SceneDataRef {
            dataset_path: ".artifacts/datasets/unused".to_string(),
            array: "values".to_string(),
            shape: vec![3],
            dtype: Some("f64".to_string()),
            chunks: vec![chunk.clone()],
        };
        let bytes =
            read_scene_chunk_bytes(&chunk, &data_ref).expect("read chunk via relative path");
        runmat_filesystem::set_current_dir(previous).expect("restore cwd");

        assert_eq!(bytes, b"{\"values\":[1,2,3]}");
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_chunk_paths_from_artifact_id_includes_expected_candidates() {
        let hash_hex = "7af8faff9e5fe6ba87fec8e4ce6d79dca7f29bbee9f9809a36119346b411ee36";
        let candidates = chunk_paths_from_artifact_id(&format!("sha256:{hash_hex}"), Some("f64"));
        assert!(candidates.iter().any(|path| {
            path == ".artifacts/objects/7a/7af8faff9e5fe6ba87fec8e4ce6d79dca7f29bbee9f9809a36119346b411ee36.f64.chunk.json"
        }));
    }

    #[cfg(feature = "plot-core")]
    #[test]
    fn scene_switch_decode_distinguishes_runs() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_switch_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");
        let dataset_a = root.join("surface_a.data");
        let dataset_b = root.join("surface_b.data");
        let dataset_a_str = dataset_a.to_string_lossy().to_string();
        let dataset_b_str = dataset_b.to_string_lossy().to_string();
        write_scene_dataset(
            &dataset_a_str,
            &[
                ("x", vec![0.0, 1.0]),
                ("y", vec![0.0, 1.0]),
                ("z", vec![0.0, 1.0, 1.0, 2.0]),
            ],
        )
        .expect("write dataset a");
        write_scene_dataset(
            &dataset_b_str,
            &[
                ("x", vec![0.0, 1.0]),
                ("y", vec![0.0, 1.0]),
                ("z", vec![2.0, 3.0, 3.0, 4.0]),
            ],
        )
        .expect("write dataset b");

        let bytes_a = make_surface_ref_payload(&dataset_a_str);
        let bytes_b = make_surface_ref_payload(&dataset_b_str);
        let scene_a = decode_figure_scene_payload_with_limits(&bytes_a, ReplayLimits::default())
            .expect("decode scene a");
        let scene_b = decode_figure_scene_payload_with_limits(&bytes_b, ReplayLimits::default())
            .expect("decode scene b");

        let z_a = match scene_a.plots.first() {
            Some(runmat_plot::event::ScenePlot::Surface { z, .. }) => z.clone(),
            other => panic!("unexpected scene a plot: {other:?}"),
        };
        let z_b = match scene_b.plots.first() {
            Some(runmat_plot::event::ScenePlot::Surface { z, .. }) => z.clone(),
            other => panic!("unexpected scene b plot: {other:?}"),
        };
        assert_ne!(z_a, z_b);
    }

    #[cfg(feature = "plot-core")]
    #[test]
    #[ignore = "benchmark sanity check"]
    fn bench_scene_ref_hydration_large_surface() {
        let root = std::env::temp_dir().join(format!(
            "runmat_scene_bench_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unix epoch")
                .as_nanos()
        ));
        futures::executor::block_on(runmat_filesystem::create_dir_all_async(&root))
            .expect("create temp root");
        let dataset_path = root.join("surface_bench.data");
        let dataset_str = dataset_path.to_string_lossy().to_string();
        let n = 512usize;
        let x = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let y = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let mut z = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                z.push((row as f64 * 0.01).sin() + (col as f64 * 0.01).cos());
            }
        }
        write_scene_dataset(&dataset_str, &[("x", x), ("y", y), ("z", z)])
            .expect("write bench dataset");
        let mut payload =
            serde_json::from_slice::<serde_json::Value>(&make_surface_ref_payload(&dataset_str))
                .expect("bench payload json");
        let plot = payload["figure"]["plots"]
            .as_array_mut()
            .and_then(|plots| plots.get_mut(0))
            .expect("bench plot");
        plot["x"]["shape"] = serde_json::json!([n]);
        plot["y"]["shape"] = serde_json::json!([n]);
        plot["z"]["shape"] = serde_json::json!([n, n]);
        let bytes = serde_json::to_vec(&payload).expect("bench payload bytes");
        let start = Instant::now();
        let _scene = decode_figure_scene_payload_with_limits(&bytes, ReplayLimits::default())
            .expect("decode bench scene");
        let elapsed = start.elapsed();
        eprintln!(
            "scene bench: {}x{} surface decode+hydrate took {} ms",
            n,
            n,
            elapsed.as_millis()
        );
    }
}
