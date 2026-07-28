use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_meshing_core::contracts::AnalysisMeshArtifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSizingEvidence {
    pub global_target_size_m: Option<f64>,
    pub min_size_m: Option<f64>,
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub growth_rate: Option<f64>,
    pub sample_count: usize,
    #[serde(default)]
    pub generated_cad_sample_count: usize,
    #[serde(default)]
    pub anisotropic_sample_count: usize,
    #[serde(default)]
    pub valid_anisotropic_sample_count: usize,
    #[serde(default)]
    pub invalid_anisotropic_sample_count: usize,
    pub applied_sample_count: usize,
    pub rejected_sample_count: usize,
    pub inserted_breakpoint_count: usize,
    #[serde(default)]
    pub inserted_breakpoint_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub uninserted_sample_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_location_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_interpolated_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_exact_point_count: usize,
    #[serde(default)]
    pub rejected_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tetrahedron_refinement_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub dropped_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tetrahedron_refinement_dropped_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_acceptance_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_rejection_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_interpolated_ratio: Option<f64>,
    #[serde(default)]
    pub generated_cad_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub anisotropic_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub invalid_anisotropic_by_reason: BTreeMap<String, usize>,
    pub applied_by_reason: BTreeMap<String, usize>,
    pub rejected_by_status: BTreeMap<String, usize>,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

pub(crate) fn sizing_evidence(mesh: &AnalysisMeshArtifact) -> MeshSizingEvidence {
    let mut generated_cad_by_reason = BTreeMap::<String, usize>::new();
    for sample in &mesh.sizing.samples {
        if let Some(reason) = sample
            .reason
            .as_deref()
            .filter(|reason| reason.starts_with("cad."))
        {
            *generated_cad_by_reason
                .entry(reason.to_string())
                .or_default() += 1;
        }
    }

    let mut anisotropic_by_reason = BTreeMap::<String, usize>::new();
    let mut invalid_anisotropic_by_reason = BTreeMap::<String, usize>::new();
    for sample in &mesh.sizing.anisotropic_samples {
        let reason = sample
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *anisotropic_by_reason.entry(reason.clone()).or_default() += 1;
        if !sample.is_valid_metric() {
            *invalid_anisotropic_by_reason.entry(reason).or_default() += 1;
        }
    }
    let invalid_anisotropic_sample_count = invalid_anisotropic_by_reason.values().sum::<usize>();

    let mut applied_by_reason = BTreeMap::<String, usize>::new();
    let mut inserted_breakpoint_by_reason = BTreeMap::<String, usize>::new();
    let mut uninserted_sample_by_reason = BTreeMap::<String, usize>::new();
    let mut inserted_breakpoint_count = 0_usize;
    for application in &mesh.sizing.applied_samples {
        let reason = application
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *applied_by_reason.entry(reason.clone()).or_default() += 1;
        if application.inserted_breakpoint_count > 0 {
            *inserted_breakpoint_by_reason.entry(reason).or_default() +=
                application.inserted_breakpoint_count;
        } else {
            *uninserted_sample_by_reason.entry(reason).or_default() += 1;
        }
        inserted_breakpoint_count += application.inserted_breakpoint_count;
    }

    let mut rejected_by_status = BTreeMap::<String, usize>::new();
    let mut rejected_by_reason = BTreeMap::<String, usize>::new();
    for rejection in &mesh.sizing.rejected_samples {
        *rejected_by_status
            .entry(rejection.status.clone())
            .or_default() += 1;
        let reason = rejection
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *rejected_by_reason.entry(reason).or_default() += 1;
    }

    let accepted_requested_tetrahedron_refinement_point_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_point_count;
    let accepted_requested_tetrahedron_refinement_location_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_location_count;
    let accepted_requested_tetrahedron_refinement_interpolated_point_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_interpolated_point_count;

    MeshSizingEvidence {
        global_target_size_m: mesh.sizing.global_target_size_m,
        min_size_m: mesh.sizing.min_size_m,
        max_size_m: mesh.sizing.max_size_m,
        growth_rate: mesh.sizing.growth_rate,
        sample_count: mesh.sizing.samples.len(),
        generated_cad_sample_count: generated_cad_by_reason.values().sum(),
        anisotropic_sample_count: mesh.sizing.anisotropic_samples.len(),
        valid_anisotropic_sample_count: mesh
            .sizing
            .anisotropic_samples
            .len()
            .saturating_sub(invalid_anisotropic_sample_count),
        invalid_anisotropic_sample_count,
        applied_sample_count: mesh.sizing.applied_samples.len(),
        rejected_sample_count: mesh.sizing.rejected_samples.len(),
        inserted_breakpoint_count,
        inserted_breakpoint_by_reason,
        uninserted_sample_by_reason,
        requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_requested_refinement_point_count,
        accepted_requested_tetrahedron_refinement_location_count,
        accepted_requested_tetrahedron_refinement_point_count,
        accepted_requested_tetrahedron_refinement_interpolated_point_count,
        accepted_requested_tetrahedron_refinement_exact_point_count:
            accepted_requested_tetrahedron_refinement_point_count
                .saturating_sub(accepted_requested_tetrahedron_refinement_interpolated_point_count),
        rejected_requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_rejected_requested_refinement_point_count,
        requested_tetrahedron_refinement_rejected_by_reason: mesh
            .backend
            .tetrahedron_requested_refinement_rejected_by_reason
            .clone(),
        dropped_requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_dropped_requested_refinement_point_count,
        requested_tetrahedron_refinement_dropped_by_reason: mesh
            .backend
            .tetrahedron_requested_refinement_dropped_by_reason
            .clone(),
        requested_tetrahedron_refinement_acceptance_ratio: if mesh
            .backend
            .tetrahedron_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend
                    .tetrahedron_accepted_requested_refinement_point_count as f64
                    / mesh.backend.tetrahedron_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tetrahedron_refinement_rejection_ratio: if mesh
            .backend
            .tetrahedron_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend
                    .tetrahedron_rejected_requested_refinement_point_count as f64
                    / mesh.backend.tetrahedron_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tetrahedron_refinement_interpolated_ratio:
            if accepted_requested_tetrahedron_refinement_point_count > 0 {
                Some(
                    accepted_requested_tetrahedron_refinement_interpolated_point_count as f64
                        / accepted_requested_tetrahedron_refinement_point_count as f64,
                )
            } else {
                None
            },
        generated_cad_by_reason,
        anisotropic_by_reason,
        invalid_anisotropic_by_reason,
        applied_by_reason,
        rejected_by_status,
        rejected_by_reason,
    }
}
