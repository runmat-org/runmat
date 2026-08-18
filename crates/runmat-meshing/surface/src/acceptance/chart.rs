use runmat_meshing_core::SurfaceQualityTargets;

use crate::{ExactFaceChart, ExactFaceChartRefinedMesh, ExactFaceRefinementContext};

use super::{
    accept_exact_face_mesh_in_parameterization, validate_exact_face_acceptance_in_parameterization,
    ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceChartAcceptanceReport,
};

pub fn accept_exact_face_chart_mesh(
    chart: &ExactFaceChart,
    mesh: &ExactFaceChartRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<ExactFaceChartAcceptanceReport, ExactFaceAcceptanceError> {
    validate_chart_id(chart, mesh)?;
    Ok(ExactFaceChartAcceptanceReport {
        chart_id: mesh.chart_id,
        acceptance: accept_exact_face_mesh_in_parameterization(
            &mesh.mesh,
            context,
            quality,
            options,
            &chart.parameterization,
        )?,
    })
}

pub fn validate_exact_face_chart_acceptance(
    report: &ExactFaceChartAcceptanceReport,
    chart: &ExactFaceChart,
    mesh: &ExactFaceChartRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<(), ExactFaceAcceptanceError> {
    validate_chart_id(chart, mesh)?;
    if report.chart_id != mesh.chart_id {
        return Err(invalid(mesh, "chart acceptance identity is inconsistent"));
    }
    validate_exact_face_acceptance_in_parameterization(
        &report.acceptance,
        &mesh.mesh,
        context,
        quality,
        options,
        &chart.parameterization,
    )
}

fn validate_chart_id(
    chart: &ExactFaceChart,
    mesh: &ExactFaceChartRefinedMesh,
) -> Result<(), ExactFaceAcceptanceError> {
    if mesh.chart_id == runmat_meshing_core::StableDigest::ZERO
        || mesh.chart_id != chart.chart_id
        || mesh.mesh.geometry.source_face_id != chart.source_face_id
    {
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
