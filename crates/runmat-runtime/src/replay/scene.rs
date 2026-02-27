use chrono::Utc;
use serde::{Deserialize, Serialize};

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
    let payload: FigureScenePayload = serde_json::from_slice(bytes).map_err(|err| {
        replay_error_with_source(
            ReplayErrorKind::DecodeFailed,
            "failed to decode figure replay payload",
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
