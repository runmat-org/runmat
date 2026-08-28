use crate::exact_cdt::{triangulate_validated_face_pslg, validate_face_delaunay_topology};
use crate::{ExactFaceBoundary, ExactFaceDelaunayError, ExactFaceDelaunayOptions};

use super::{
    validate_exact_face_charts, ExactFaceChartDelaunay, ExactFaceChartDelaunayContext,
    ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceCharts,
};

pub fn triangulate_exact_face_charts(
    charts: &ExactFaceCharts,
    source: &ExactFaceBoundary,
    context: ExactFaceChartDelaunayContext<'_>,
    chart_options: ExactFaceChartOptions,
    delaunay_options: ExactFaceDelaunayOptions,
) -> Result<Vec<ExactFaceChartDelaunay>, ExactFaceChartError> {
    validate_exact_face_charts(
        charts,
        source,
        context.topology,
        context.evaluator,
        context.geometry_control,
        chart_options,
    )?;
    let result = charts
        .charts
        .iter()
        .map(|chart| {
            triangulate_validated_face_pslg(&chart.pslg, context.cancellation, delaunay_options)
                .map(|triangulation| ExactFaceChartDelaunay {
                    chart_id: chart.chart_id,
                    triangulation,
                })
                .map_err(|failure| delaunay_error(source, failure))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_exact_face_chart_delaunay(
        &result,
        charts,
        source,
        context,
        chart_options,
        delaunay_options,
    )?;
    Ok(result)
}

pub fn validate_exact_face_chart_delaunay(
    triangulations: &[ExactFaceChartDelaunay],
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
    if triangulations.len() != charts.charts.len() || triangulations.is_empty() {
        return Err(invalid(
            source,
            "face chart triangulation inventory is inconsistent",
        ));
    }
    for (triangulation, chart) in triangulations.iter().zip(&charts.charts) {
        if triangulation.chart_id != chart.chart_id {
            return Err(invalid(
                source,
                "face chart triangulation identity is inconsistent",
            ));
        }
        validate_face_delaunay_topology(
            &triangulation.triangulation,
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
