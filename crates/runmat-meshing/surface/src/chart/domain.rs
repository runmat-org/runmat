use crate::exact_cdt::{
    carve_validated_face_domain, recover_validated_face_segments,
    validate_face_constrained_topology, validate_face_delaunay_topology,
    validate_face_trimmed_topology,
};
use crate::{ExactFaceBoundary, ExactFaceDelaunayError, ExactFaceDelaunayOptions};

use super::{
    validate_exact_face_chart_delaunay, validate_exact_face_charts,
    ExactFaceChartConstrainedDomain, ExactFaceChartDelaunay, ExactFaceChartDelaunayContext,
    ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceCharts,
};

pub fn recover_exact_face_chart_domains(
    triangulations: &[ExactFaceChartDelaunay],
    charts: &ExactFaceCharts,
    source: &ExactFaceBoundary,
    context: ExactFaceChartDelaunayContext<'_>,
    chart_options: ExactFaceChartOptions,
    delaunay_options: ExactFaceDelaunayOptions,
) -> Result<Vec<ExactFaceChartConstrainedDomain>, ExactFaceChartError> {
    validate_exact_face_chart_delaunay(
        triangulations,
        charts,
        source,
        context,
        chart_options,
        delaunay_options,
    )?;
    let domains = triangulations
        .iter()
        .zip(&charts.charts)
        .map(|(triangulation, chart)| {
            let constrained = recover_validated_face_segments(
                &triangulation.triangulation,
                &chart.pslg,
                context.cancellation,
                delaunay_options,
            )
            .map_err(|failure| delaunay_error(source, failure))?;
            let trimmed = carve_validated_face_domain(
                &constrained,
                &chart.pslg,
                context.cancellation,
                delaunay_options,
            )
            .map_err(|failure| delaunay_error(source, failure))?;
            Ok(ExactFaceChartConstrainedDomain {
                chart_id: chart.chart_id,
                delaunay: triangulation.triangulation.clone(),
                constrained,
                trimmed,
            })
        })
        .collect::<Result<Vec<_>, ExactFaceChartError>>()?;
    validate_exact_face_chart_domains(
        &domains,
        charts,
        source,
        context,
        chart_options,
        delaunay_options,
    )?;
    Ok(domains)
}

pub fn validate_exact_face_chart_domains(
    domains: &[ExactFaceChartConstrainedDomain],
    charts: &ExactFaceCharts,
    source: &ExactFaceBoundary,
    context: ExactFaceChartDelaunayContext<'_>,
    chart_options: ExactFaceChartOptions,
    delaunay_options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceChartError> {
    validate_exact_face_charts(
        charts,
        source,
        context.topology,
        context.evaluator,
        context.geometry_control,
        chart_options,
    )?;
    if domains.len() != charts.charts.len() || domains.is_empty() {
        return Err(invalid(
            source,
            "face chart domain inventory is inconsistent",
        ));
    }
    for (domain, chart) in domains.iter().zip(&charts.charts) {
        if domain.chart_id != chart.chart_id {
            return Err(invalid(
                source,
                "face chart domain identity is inconsistent",
            ));
        }
        validate_face_delaunay_topology(
            &domain.delaunay,
            &chart.pslg,
            context.cancellation,
            delaunay_options,
        )
        .map_err(|failure| delaunay_error(source, failure))?;
        validate_face_constrained_topology(
            &domain.constrained,
            &chart.pslg,
            context.cancellation,
            delaunay_options,
        )
        .map_err(|failure| delaunay_error(source, failure))?;
        validate_face_trimmed_topology(
            &domain.trimmed,
            &domain.constrained,
            &chart.pslg,
            context.cancellation,
            delaunay_options,
        )
        .map_err(|failure| delaunay_error(source, failure))?;
    }
    Ok(())
}

fn delaunay_error(
    source: &ExactFaceBoundary,
    failure: ExactFaceDelaunayError,
) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::Delaunay(failure.kind),
        &source.source_face_id,
        failure.reason,
    )
}

fn invalid(source: &ExactFaceBoundary, reason: &str) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::InvalidInput,
        &source.source_face_id,
        reason,
    )
}
