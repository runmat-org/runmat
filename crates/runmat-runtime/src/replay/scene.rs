use chrono::Utc;
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
    hydrate_scene_data_refs(&mut payload_json)?;
    let payload: FigureScenePayload = serde_json::from_value(payload_json).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to decode hydrated figure replay payload",
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
fn hydrate_scene_data_refs(payload: &mut Value) -> Result<(), RuntimeError> {
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
                hydrate_plot_field(plot, "x")?;
                hydrate_plot_field(plot, "y")?;
                hydrate_plot_field(plot, "z")?;
            }
            "scatter3" => {
                hydrate_plot_field(plot, "points")?;
                hydrate_plot_field(plot, "colorsRgba")?;
                hydrate_plot_field(plot, "pointSizes")?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
fn hydrate_plot_field(plot: &mut Value, field: &str) -> Result<(), RuntimeError> {
    let Some(obj) = plot.as_object_mut() else {
        return Ok(());
    };
    let Some(value) = obj.get(field).cloned() else {
        return Ok(());
    };
    let Some(data_ref) = parse_data_ref(&value) else {
        return Ok(());
    };
    let dataset_root = crate::data::dataset_root(&data_ref.dataset_path);
    let manifest = crate::data::read_manifest(&dataset_root).map_err(|err| {
        replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "failed to read scene dataset manifest '{}': {}",
                data_ref.dataset_path, err
            ),
        )
    })?;
    let meta = manifest.arrays.get(&data_ref.array).ok_or_else(|| {
        replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "scene dataset '{}' missing array '{}'",
                data_ref.dataset_path, data_ref.array
            ),
        )
    })?;
    let payload = crate::data::read_array_payload(&dataset_root, meta).map_err(|err| {
        replay_error(
            ReplayErrorKind::ImportRejected,
            format!(
                "failed reading scene dataset array '{}.{}': {}",
                data_ref.dataset_path, data_ref.array, err
            ),
        )
    })?;
    let target_shape = if data_ref.shape.is_empty() {
        payload.shape.as_slice()
    } else {
        data_ref.shape.as_slice()
    };
    let hydrated = shape_values_to_json(&payload.values, target_shape)?;
    obj.insert(field.to_string(), hydrated);
    Ok(())
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
    Some(SceneDataRef {
        dataset_path,
        array,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

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
            let (payload_path, chunk_index_path) =
                crate::data::write_array_payload(&root, array_name, &payload, &chunk)?;
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
        crate::data::write_manifest(&root, &manifest)?;
        Ok(())
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
        std::fs::create_dir_all(&root).expect("create temp root");
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
            "datasetPath": dataset_str,
            "array": "x",
            "shape": [2]
        });
        plot["y"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": dataset_str,
            "array": "y",
            "shape": [2]
        });
        plot["z"] = serde_json::json!({
            "refKind": "runmat-data-array-v1",
            "datasetPath": dataset_str,
            "array": "z",
            "shape": [2, 2]
        });

        let bytes = serde_json::to_vec(&payload).expect("payload bytes");
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
}
