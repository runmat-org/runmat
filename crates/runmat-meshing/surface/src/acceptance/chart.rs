use runmat_meshing_core::SurfaceQualityTargets;

use crate::{ExactFaceChartRefinedMesh, ExactFaceRefinementContext};

use super::{
    accept_exact_face_mesh, validate_exact_face_acceptance, ExactFaceAcceptanceError,
    ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions, ExactFaceChartAcceptanceReport,
};

pub fn accept_exact_face_chart_mesh(
    mesh: &ExactFaceChartRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<ExactFaceChartAcceptanceReport, ExactFaceAcceptanceError> {
    validate_chart_id(mesh)?;
    Ok(ExactFaceChartAcceptanceReport {
        chart_id: mesh.chart_id,
        acceptance: accept_exact_face_mesh(&mesh.mesh, context, quality, options)?,
    })
}

pub fn validate_exact_face_chart_acceptance(
    report: &ExactFaceChartAcceptanceReport,
    mesh: &ExactFaceChartRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<(), ExactFaceAcceptanceError> {
    validate_chart_id(mesh)?;
    if report.chart_id != mesh.chart_id {
        return Err(invalid(mesh, "chart acceptance identity is inconsistent"));
    }
    validate_exact_face_acceptance(&report.acceptance, &mesh.mesh, context, quality, options)
}

fn validate_chart_id(mesh: &ExactFaceChartRefinedMesh) -> Result<(), ExactFaceAcceptanceError> {
    if mesh.chart_id == runmat_meshing_core::StableDigest::ZERO {
        return Err(invalid(mesh, "chart acceptance identity must be nonzero"));
    }
    Ok(())
}

fn invalid(mesh: &ExactFaceChartRefinedMesh, reason: &str) -> ExactFaceAcceptanceError {
    ExactFaceAcceptanceError::new(
        ExactFaceAcceptanceErrorKind::InvalidInput,
        &mesh.mesh.geometry.source_face_id,
        reason,
    )
}
