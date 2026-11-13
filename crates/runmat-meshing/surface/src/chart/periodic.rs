use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, SurfaceEvaluatorId,
};
use runmat_meshing_core::StableDigest;
use sha2::{Digest, Sha256};

use crate::{build_exact_face_pslg, ExactFaceBoundary, ExactFaceBoundaryLoop};

use super::{
    annulus::build_periodic_annulus_pslg,
    parameterization::{build_singular_parameterization, is_singular},
    ExactFaceChart, ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions,
    ExactFaceChartParameterization, ExactFaceCharts,
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
    let lift_context = LoopLiftContext {
        periodicity,
        options,
        source,
        singular_surface: face.has_singularity.then_some(&face.surface_evaluator_id),
        evaluator,
        control,
    };
    let mut windings = vec![lift_loop(&mut boundary.outer_loop, &lift_context)?];
    for loop_boundary in &mut boundary.inner_loops {
        windings.push(lift_loop(loop_boundary, &lift_context)?);
    }
    if periodicity.into_iter().any(|period| period.is_some()) {
        // The admitted boundary keeps each OCCT coedge's seam image. Once a
        // periodic loop has been lifted into one canonical chart, that image is
        // represented by its lifted UV coordinates instead. Keeping the raw
        // per-coedge tag here would split otherwise identical endpoints where
        // a seam coedge meets an ordinary trim coedge.
        for loop_boundary in
            std::iter::once(&mut boundary.outer_loop).chain(&mut boundary.inner_loops)
        {
            for segment in &mut loop_boundary.segments {
                segment.seam_image = None;
            }
        }
    }
    let chart_id = chart_id(&source.source_face_id, 0);
    let zero_winding = windings.iter().all(|winding| *winding == [0, 0]);
    let (mut parameterization, mut pslg) = if face.has_singularity && zero_winding {
        let parameterization = build_singular_parameterization(
            &mut boundary,
            topology,
            &face.surface_evaluator_id,
            evaluator,
            control,
            options,
        )?;
        let pslg = build_exact_face_pslg(&boundary).map_err(|failure| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &source.source_face_id,
                failure.to_string(),
            )
        })?;
        (parameterization, pslg)
    } else if zero_winding {
        let pslg = build_exact_face_pslg(&boundary).map_err(|failure| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &source.source_face_id,
                failure.to_string(),
            )
        })?;
        (ExactFaceChartParameterization::EvaluatorParameters, pslg)
    } else {
        let pslg =
            build_periodic_annulus_pslg(&mut boundary, &windings, periodicity, chart_id, options)?;
        (ExactFaceChartParameterization::EvaluatorParameters, pslg)
    };
    if matches!(
        parameterization,
        ExactFaceChartParameterization::EvaluatorParameters
    ) {
        parameterization = scale_regular_chart(
            &mut boundary,
            &mut pslg,
            &face.surface_evaluator_id,
            evaluator,
            control,
            options,
        )?;
    }
    let chart = ExactFaceChart {
        chart_id,
        source_face_id: source.source_face_id.clone(),
        periodicity,
        parameterization,
        boundary,
        pslg,
    };
    Ok(ExactFaceCharts {
        source_face_id: source.source_face_id.clone(),
        periodicity,
        charts: vec![chart],
    })
}

fn scale_regular_chart(
    boundary: &mut ExactFaceBoundary,
    pslg: &mut crate::ExactFacePslg,
    evaluator_id: &SurfaceEvaluatorId,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    let origins: [f64; 2] = std::array::from_fn(|axis| {
        pslg.vertices
            .iter()
            .map(|vertex| vertex.uv[axis])
            .fold(f64::INFINITY, f64::min)
    });
    let ends: [f64; 2] = std::array::from_fn(|axis| {
        pslg.vertices
            .iter()
            .map(|vertex| vertex.uv[axis])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let spans: [f64; 2] = std::array::from_fn(|axis| ends[axis] - origins[axis]);
    if origins.iter().any(|value| !value.is_finite())
        || spans
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid(
            boundary,
            "regular face chart requires two finite nondegenerate parameter spans",
        ));
    }
    let reference_uv = std::array::from_fn(|axis| origins[axis] + 0.5 * spans[axis]);
    let derivatives = evaluator
        .derivatives(evaluator_id, reference_uv, control)
        .map_err(|error| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::GeometryEvaluation(error.kind),
                &boundary.source_face_id,
                error.reason,
            )
        })?;
    let physical_units_per_evaluator_unit = [
        derivatives
            .du_m
            .with_canonical_storage(|values, _axes| *values),
        derivatives
            .dv_m
            .with_canonical_storage(|values, _axes| *values),
    ]
    .map(|derivative| {
        derivative
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    });
    if physical_units_per_evaluator_unit
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(invalid(
            boundary,
            "regular face chart has a degenerate physical parameter direction at its reference point",
        ));
    }
    let evaluator_units_per_chart_unit =
        physical_units_per_evaluator_unit.map(|value| value.recip());
    let coordinate_quantum =
        canonical_coordinate_quantum(options.projection_tolerance_m.canonical_value());
    let scale = |uv: [f64; 2]| {
        let coordinates = [
            (uv[0] - origins[0]) / evaluator_units_per_chart_unit[0],
            (uv[1] - origins[1]) / evaluator_units_per_chart_unit[1],
        ];
        coordinates.map(|coordinate| canonicalize_coordinate(coordinate, coordinate_quantum))
    };
    for vertex in &mut pslg.vertices {
        vertex.uv = scale(vertex.uv);
    }
    for loop_boundary in std::iter::once(&mut boundary.outer_loop).chain(&mut boundary.inner_loops)
    {
        for segment in &mut loop_boundary.segments {
            segment.node_uv = segment.node_uv.map(scale);
        }
    }
    Ok(ExactFaceChartParameterization::AffineEvaluatorParameters {
        origins,
        evaluator_units_per_chart_unit,
    })
}

fn canonical_coordinate_quantum(projection_tolerance_m: f64) -> f64 {
    let upper_bound = projection_tolerance_m / 1_024.0;
    2.0_f64.powf(upper_bound.log2().floor())
}

fn canonicalize_coordinate(coordinate: f64, quantum: f64) -> f64 {
    let canonical = (coordinate / quantum).round() * quantum;
    if canonical == 0.0 {
        0.0
    } else {
        canonical
    }
}

struct LoopLiftContext<'a, Evaluator: ?Sized> {
    periodicity: [Option<f64>; 2],
    options: ExactFaceChartOptions,
    source: &'a ExactFaceBoundary,
    singular_surface: Option<&'a SurfaceEvaluatorId>,
    evaluator: &'a Evaluator,
    control: &'a dyn GeometryEvaluationControl,
}

fn lift_loop<Evaluator: ExactSurfaceEvaluator + ?Sized>(
    loop_boundary: &mut ExactFaceBoundaryLoop,
    lift_context: &LoopLiftContext<'_, Evaluator>,
) -> Result<[i32; 2], ExactFaceChartError> {
    if loop_boundary.segments.is_empty() {
        return Err(invalid(lift_context.source, "face chart loop is empty"));
    }
    let first_uv = loop_boundary.segments[0].node_uv[0];
    let mut previous_id = loop_boundary.segments[0].node_ids[0];
    let mut previous_uv = first_uv;
    let mut used_singular_transition = false;
    for segment in &mut loop_boundary.segments {
        if segment.node_ids[0] != previous_id {
            return Err(invalid(
                lift_context.source,
                "face chart loop node incidence is disconnected",
            ));
        }
        let singular_transition =
            is_singular_transition(previous_uv, segment.node_uv[0], lift_context)?;
        used_singular_transition |= singular_transition;
        for (axis, previous_coordinate) in previous_uv.into_iter().enumerate() {
            let context = CoordinateLift {
                period: lift_context.periodicity[axis],
                options: lift_context.options,
                source: lift_context.source,
                wire_id: &loop_boundary.source_wire_id,
                axis,
            };
            if !singular_transition {
                segment.node_uv[0][axis] =
                    context.lift(segment.node_uv[0][axis], previous_coordinate, true)?;
            }
            segment.node_uv[1][axis] =
                context.lift(segment.node_uv[1][axis], segment.node_uv[0][axis], false)?;
        }
        previous_id = segment.node_ids[1];
        previous_uv = segment.node_uv[1];
    }
    if previous_id != loop_boundary.segments[0].node_ids[0] {
        return Err(invalid(
            lift_context.source,
            "face chart loop is not identity-closed",
        ));
    }
    used_singular_transition |= is_singular_transition(previous_uv, first_uv, lift_context)?;
    if used_singular_transition {
        return Ok([0, 0]);
    }
    let mut winding = [0i32; 2];
    for axis in 0..2 {
        let residual = previous_uv[axis] - first_uv[axis];
        let tolerance = scaled_tolerance(previous_uv[axis], first_uv[axis], lift_context.options);
        if residual.abs() <= tolerance {
            loop_boundary.segments.last_mut().unwrap().node_uv[1][axis] = first_uv[axis];
            continue;
        }
        let Some(period) = lift_context.periodicity[axis] else {
            return Err(invalid(
                lift_context.source,
                "nonperiodic face chart loop does not close",
            ));
        };
        // The loop is already identity-closed: its last node is its first. What remains is
        // the topological winding, an integer, and the gap between the two pcurve images of that
        // one node (vertex tolerance, chained pcurve evaluations), which can exceed roundoff.
        // Any residual within a small fraction of a period of an integer winding is that gap.
        let periods = (residual / period).round();
        let winding_tolerance = tolerance.max(PERIODIC_WINDING_GAP_FRACTION * period);
        if !periods.is_finite()
            || periods.abs() > lift_context.options.maximum_period_shifts as f64
            || (residual - periods * period).abs() > winding_tolerance
        {
            return Err(ExactFaceChartError::new(
                ExactFaceChartErrorKind::InvalidInput,
                &lift_context.source.source_face_id,
                format!(
                    "periodic loop residual {residual} on axis {axis} (period {period}, tolerance {tolerance}) is not an integral bounded winding; loop starts at {first_uv:?} and ends at {previous_uv:?}"
                ),
            )
            .with_witness(&loop_boundary.source_wire_id, axis, residual));
        }
        if periods == 0.0 {
            loop_boundary.segments.last_mut().unwrap().node_uv[1][axis] = first_uv[axis];
        }
        winding[axis] = periods as i32;
    }
    Ok(winding)
}

fn is_singular_transition<Evaluator: ExactSurfaceEvaluator + ?Sized>(
    previous_uv: [f64; 2],
    current_uv: [f64; 2],
    context: &LoopLiftContext<'_, Evaluator>,
) -> Result<bool, ExactFaceChartError> {
    let Some(evaluator_id) = context.singular_surface else {
        return Ok(false);
    };
    if previous_uv == current_uv {
        return Ok(false);
    }
    let previous = context
        .evaluator
        .derivatives(evaluator_id, previous_uv, context.control)
        .map_err(|failure| geometry_failure(context.source, failure))?;
    let current = context
        .evaluator
        .derivatives(evaluator_id, current_uv, context.control)
        .map_err(|failure| geometry_failure(context.source, failure))?;
    if !is_singular(&previous) || !is_singular(&current) {
        return Ok(false);
    }
    let squared_distance = previous
        .point_m
        .canonical_values()
        .iter()
        .zip(current.point_m.canonical_values())
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>();
    Ok(squared_distance
        <= context.options.projection_tolerance_m.canonical_value()
            * context.options.projection_tolerance_m.canonical_value())
}

fn geometry_failure(
    source: &ExactFaceBoundary,
    failure: runmat_geometry_core::GeometryEvaluationError,
) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::GeometryEvaluation(failure.kind),
        &source.source_face_id,
        failure.reason,
    )
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
                format!(
                    "periodic chart lift is not the nearest canonical image: axis {} value {value} lifted to {lifted} against reference {reference} (period {period}, residual {residual}, tolerance {tolerance}, equivalent {require_equivalent})",
                    self.axis
                ),
            )
            .with_witness(self.wire_id, self.axis, residual));
        }
        if require_equivalent {
            Ok(reference)
        } else {
            Ok(lifted)
        }
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
        || !options.projection_tolerance_m.canonical_value().is_finite()
        || options.projection_tolerance_m.canonical_value() <= 0.0
    {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::InvalidOptions,
            &source.source_face_id,
            "chart residual, shift, projection tolerance, and per-face chart bounds must be positive",
        ));
    }
    Ok(())
}

/// Largest pcurve gap at a loop's closing node, as a fraction of the period, read as an integer
/// winding. A vertex gap is many orders smaller; a lifting error is a large fraction of a period.
const PERIODIC_WINDING_GAP_FRACTION: f64 = 1.0e-6;

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
