use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

use crate::{build_exact_face_pslg, ExactFaceBoundary, ExactFaceBoundaryLoop};

use super::{
    annulus::build_periodic_annulus_pslg, ExactFaceChart, ExactFaceChartError,
    ExactFaceChartErrorKind, ExactFaceChartOptions, ExactFaceCharts,
};

pub fn build_exact_face_charts(
    source: &ExactFaceBoundary,
    topology: &ExactBRepTopology,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceCharts, ExactFaceChartError> {
    let charts = build_without_validation(source, topology, evaluator, control, options)?;
    validate_exact_face_charts(&charts, source, topology, evaluator, control, options)?;
    Ok(charts)
}

pub fn validate_exact_face_charts(
    charts: &ExactFaceCharts,
    source: &ExactFaceBoundary,
    topology: &ExactBRepTopology,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<(), ExactFaceChartError> {
    let expected = build_without_validation(source, topology, evaluator, control, options)?;
    if charts != &expected {
        return Err(invalid(
            source,
            "face chart differs from canonical periodic reconstruction",
        ));
    }
    Ok(())
}

fn build_without_validation(
    source: &ExactFaceBoundary,
    topology: &ExactBRepTopology,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceCharts, ExactFaceChartError> {
    validate_options(source, options)?;
    control.checkpoint().map_err(|failure| {
        ExactFaceChartError::new(
            ExactFaceChartErrorKind::GeometryEvaluation(failure.kind),
            &source.source_face_id,
            failure.reason,
        )
    })?;
    let face = topology
        .faces
        .iter()
        .find(|face| face.id == source.source_face_id)
        .ok_or_else(|| invalid(source, "source face is absent from exact topology"))?;
    let periodicity = evaluator
        .periodicity(&face.surface_evaluator_id)
        .map_err(|failure| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::GeometryEvaluation(failure.kind),
                &source.source_face_id,
                failure.reason,
            )
        })?;
    if [face.periodic_u, face.periodic_v] != periodicity.map(|period| period.is_some())
        || periodicity
            .iter()
            .flatten()
            .any(|period| !period.is_finite() || *period <= 0.0)
    {
        return Err(invalid(
            source,
            "surface periodicity differs from exact topology",
        ));
    }
    let mut boundary = source.clone();
    let mut windings = vec![lift_loop(
        &mut boundary.outer_loop,
        periodicity,
        options,
        source,
    )?];
    for loop_boundary in &mut boundary.inner_loops {
        windings.push(lift_loop(loop_boundary, periodicity, options, source)?);
    }
    let chart_id = chart_id(&source.source_face_id, 0);
    let pslg = if windings.iter().all(|winding| *winding == [0, 0]) {
        build_exact_face_pslg(&boundary).map_err(|failure| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &source.source_face_id,
                failure.to_string(),
            )
        })?
    } else {
        build_periodic_annulus_pslg(&mut boundary, &windings, periodicity, chart_id, options)?
    };
    let chart = ExactFaceChart {
        chart_id,
        source_face_id: source.source_face_id.clone(),
        periodicity,
        boundary,
        pslg,
    };
    Ok(ExactFaceCharts {
        source_face_id: source.source_face_id.clone(),
        periodicity,
        charts: vec![chart],
    })
}

fn lift_loop(
    loop_boundary: &mut ExactFaceBoundaryLoop,
    periodicity: [Option<f64>; 2],
    options: ExactFaceChartOptions,
    source: &ExactFaceBoundary,
) -> Result<[i32; 2], ExactFaceChartError> {
    if loop_boundary.segments.is_empty() {
        return Err(invalid(source, "face chart loop is empty"));
    }
    let first_uv = loop_boundary.segments[0].node_uv[0];
    let mut previous_id = loop_boundary.segments[0].node_ids[0];
    let mut previous_uv = first_uv;
    for segment in &mut loop_boundary.segments {
        if segment.node_ids[0] != previous_id {
            return Err(invalid(
                source,
                "face chart loop node incidence is disconnected",
            ));
        }
        for axis in 0..2 {
            let context = CoordinateLift {
                period: periodicity[axis],
                options,
                source,
                wire_id: &loop_boundary.source_wire_id,
                axis,
            };
            segment.node_uv[0][axis] =
                context.lift(segment.node_uv[0][axis], previous_uv[axis], true)?;
            segment.node_uv[1][axis] =
                context.lift(segment.node_uv[1][axis], segment.node_uv[0][axis], false)?;
        }
        previous_id = segment.node_ids[1];
        previous_uv = segment.node_uv[1];
    }
    if previous_id != loop_boundary.segments[0].node_ids[0] {
        return Err(invalid(source, "face chart loop is not identity-closed"));
    }
    let mut winding = [0i32; 2];
    for axis in 0..2 {
        let residual = previous_uv[axis] - first_uv[axis];
        let tolerance = scaled_tolerance(previous_uv[axis], first_uv[axis], options);
        if residual.abs() <= tolerance {
            continue;
        }
        let Some(period) = periodicity[axis] else {
            return Err(invalid(
                source,
                "nonperiodic face chart loop does not close",
            ));
        };
        let periods = (residual / period).round();
        if !periods.is_finite()
            || periods.abs() > options.maximum_period_shifts as f64
            || (residual - periods * period).abs() > tolerance
        {
            return Err(ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &source.source_face_id,
                "periodic loop residual is not an integral bounded winding",
            )
            .with_witness(&loop_boundary.source_wire_id, axis, residual));
        }
        winding[axis] = periods as i32;
    }
    Ok(winding)
}

struct CoordinateLift<'a> {
    period: Option<f64>,
    options: ExactFaceChartOptions,
    source: &'a ExactFaceBoundary,
    wire_id: &'a runmat_geometry_core::PersistentEntityId,
    axis: usize,
}

impl CoordinateLift<'_> {
    fn lift(
        &self,
        value: f64,
        reference: f64,
        require_equivalent: bool,
    ) -> Result<f64, ExactFaceChartError> {
        if !value.is_finite() || !reference.is_finite() {
            return Err(invalid(
                self.source,
                "face chart contains nonfinite UV coordinates",
            ));
        }
        let Some(period) = self.period else {
            if !require_equivalent {
                return Ok(value);
            }
            let residual = value - reference;
            if residual.abs() > scaled_tolerance(value, reference, self.options) {
                return Err(invalid(
                    self.source,
                    "nonperiodic face chart endpoints disagree by identity",
                ));
            }
            return Ok(reference);
        };
        let shift = ((reference - value) / period).round();
        if !shift.is_finite() || shift.abs() > self.options.maximum_period_shifts as f64 {
            return Err(ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &self.source.source_face_id,
                "periodic chart shift exceeds its hard bound",
            )
            .with_witness(self.wire_id, self.axis, shift));
        }
        let lifted = value + shift * period;
        let residual = lifted - reference;
        let tolerance = scaled_tolerance(lifted, reference, self.options);
        if (require_equivalent && residual.abs() > tolerance)
            || (!require_equivalent && residual.abs() > period * 0.5 + tolerance)
        {
            return Err(ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &self.source.source_face_id,
                "periodic chart lift is not the nearest canonical image",
            )
            .with_witness(self.wire_id, self.axis, residual));
        }
        Ok(lifted)
    }
}

fn validate_options(
    source: &ExactFaceBoundary,
    options: ExactFaceChartOptions,
) -> Result<(), ExactFaceChartError> {
    if !options.maximum_periodic_residual.is_finite()
        || options.maximum_periodic_residual <= 0.0
        || options.maximum_period_shifts <= 0
        || options.maximum_charts_per_face == 0
    {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::InvalidOptions,
            &source.source_face_id,
            "chart residual, shift, and per-face chart bounds must be positive",
        ));
    }
    Ok(())
}

fn scaled_tolerance(left: f64, right: f64, options: ExactFaceChartOptions) -> f64 {
    options.maximum_periodic_residual * left.abs().max(right.abs()).max(1.0)
}

fn chart_id(face_id: &runmat_geometry_core::PersistentEntityId, chart_index: u32) -> StableDigest {
    let mut digest = Sha256::new();
    digest.update(b"runmat.exact-face-chart\0");
    digest.update(1u16.to_be_bytes());
    digest.update((face_id.source_topology_id.len() as u64).to_be_bytes());
    digest.update(face_id.source_topology_id.as_bytes());
    digest.update((face_id.assembly_path.len() as u64).to_be_bytes());
    for segment in &face_id.assembly_path {
        digest.update((segment.len() as u64).to_be_bytes());
        digest.update(segment.as_bytes());
    }
    digest.update(chart_index.to_be_bytes());
    StableDigest::from_bytes(digest.finalize().into())
}

fn invalid(source: &ExactFaceBoundary, reason: &str) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::InvalidInput,
        &source.source_face_id,
        reason,
    )
}
