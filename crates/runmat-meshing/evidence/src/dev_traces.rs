use serde::{Deserialize, Serialize};

use runmat_meshing_core::{
    contracts::AnalysisMeshArtifact, validation::AnalysisMeshValidationOptions,
};

use crate::{build_mesh_evidence_artifact, MeshEvidenceArtifact};

pub const MODULE_PURPOSE: &str = "feature-gated development traces only";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDebugEvidence {
    pub event_cap: usize,
    pub event_count: usize,
    pub emitted_event_count: usize,
    pub truncated_event_count: usize,
    pub events: Vec<MeshDebugEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDebugEvent {
    pub stage: String,
    pub severity: String,
    pub message: String,
}

impl MeshDebugEvent {
    pub fn new(
        stage: impl Into<String>,
        severity: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage: stage.into(),
            severity: severity.into(),
            message: message.into(),
        }
    }
}

pub fn build_mesh_evidence_artifact_with_debug(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    debug_events: Vec<MeshDebugEvent>,
    event_cap: usize,
) -> MeshEvidenceArtifact {
    let mut artifact = build_mesh_evidence_artifact(mesh, validation);
    artifact.debug = Some(debug_evidence(debug_events, event_cap));
    artifact
}

fn debug_evidence(mut events: Vec<MeshDebugEvent>, event_cap: usize) -> MeshDebugEvidence {
    let event_count = events.len();
    let cap = event_cap.min(event_count);
    events.truncate(cap);
    MeshDebugEvidence {
        event_cap,
        event_count,
        emitted_event_count: events.len(),
        truncated_event_count: event_count.saturating_sub(events.len()),
        events,
    }
}
