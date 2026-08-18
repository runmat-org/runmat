use runmat_meshing_core::SurfaceQualityTargets;

use crate::{
    validate_exact_face_geometry_in_parameterization, ExactFaceChartParameterization,
    ExactFaceGeometryContext, ExactFaceMetricError, ExactFaceRefinedMesh,
    ExactFaceRefinementContext, ResolvedFaceMetricField,
};

use super::{
    ExactFaceAcceptanceError, ExactFaceAcceptanceErrorKind, ExactFaceAcceptanceOptions,
    ExactFaceAcceptanceReport, ExactFaceTriangleAcceptance,
};

pub fn accept_exact_face_mesh(
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<ExactFaceAcceptanceReport, ExactFaceAcceptanceError> {
    let parameterization = ExactFaceChartParameterization::EvaluatorParameters;
    let report = sample(mesh, context, quality, options, &parameterization)?;
    validate_exact_face_acceptance_in_parameterization(
        &report,
        mesh,
        context,
        quality,
        options,
        &parameterization,
    )?;
    Ok(report)
}

pub fn validate_exact_face_acceptance(
    report: &ExactFaceAcceptanceReport,
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
) -> Result<(), ExactFaceAcceptanceError> {
    validate_exact_face_acceptance_in_parameterization(
        report,
        mesh,
        context,
        quality,
        options,
        &ExactFaceChartParameterization::EvaluatorParameters,
    )
}

pub(crate) fn accept_exact_face_mesh_in_parameterization(
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
    parameterization: &ExactFaceChartParameterization,
) -> Result<ExactFaceAcceptanceReport, ExactFaceAcceptanceError> {
    let report = sample(mesh, context, quality, options, parameterization)?;
    validate_exact_face_acceptance_in_parameterization(
        &report,
        mesh,
        context,
        quality,
        options,
        parameterization,
    )?;
    Ok(report)
}

pub(crate) fn validate_exact_face_acceptance_in_parameterization(
    report: &ExactFaceAcceptanceReport,
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
    parameterization: &ExactFaceChartParameterization,
) -> Result<(), ExactFaceAcceptanceError> {
    let expected = sample(mesh, context, quality, options, parameterization)?;
    if report != &expected {
        return Err(error(
            mesh,
            ExactFaceAcceptanceErrorKind::InvalidInput,
            "acceptance report differs from independent exact-surface resampling",
        ));
    }
    Ok(())
}

fn sample(
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
    parameterization: &ExactFaceChartParameterization,
) -> Result<ExactFaceAcceptanceReport, ExactFaceAcceptanceError> {
    validate_inputs(mesh, context, quality, options, parameterization)?;
    let field = ResolvedFaceMetricField::new(context.topology, context.metric_request)
        .map_err(|failure| map_metric(mesh, failure))?;
    let mut budget = SampleBudget::new(mesh, options.maximum_samples);
    let mut triangle_reports = Vec::with_capacity(mesh.geometry.triangles.len());
    let mut face_chordal = SampleMaximum::default();
    let mut face_normal = SampleMaximum::default();
    for evidence in &mesh.geometry.triangles {
        let corners = evidence
            .triangle
            .vertex_indices
            .map(|index| &mesh.geometry.vertices[index as usize]);
        let original_uv = corners.map(|corner| corner.evaluation.uv);
        let original_points = corners.map(|corner| corner.evaluation.point_m);
        let before = budget.consumed;
        let mut triangle_chordal = SampleMaximum::default();
        let mut triangle_normal = SampleMaximum::default();
        let mut cells = vec![(original_uv, 0u8)];
        while let Some((cell, depth)) = cells.pop() {
            let mut cell_chordal = 0.0_f64;
            let mut cell_normal = 0.0_f64;
            for uv in cell_samples(cell) {
                budget.consume()?;
                context.geometry_control.checkpoint().map_err(|failure| {
                    error(
                        mesh,
                        ExactFaceAcceptanceErrorKind::Metric(
                            crate::ExactFaceMetricErrorKind::GeometryEvaluation(failure.kind),
                        ),
                        failure.reason,
                    )
                })?;
                let evaluation = field
                    .evaluate_parameterized(
                        &mesh.geometry.source_face_id,
                        uv,
                        parameterization,
                        context.evaluator,
                        context.geometry_control,
                    )
                    .map_err(|failure| map_metric(mesh, failure))?;
                let weights = barycentric(original_uv, uv).ok_or_else(|| {
                    error(
                        mesh,
                        ExactFaceAcceptanceErrorKind::InvalidInput,
                        "acceptance triangle has singular parametric coordinates",
                    )
                })?;
                let chordal =
                    distance(evaluation.point_m, weighted_point(original_points, weights));
                let normal = angle(
                    evidence.unit_normal,
                    surface_normal(evaluation.derivative_u_m, evaluation.derivative_v_m)
                        .ok_or_else(|| {
                            error(
                                mesh,
                                ExactFaceAcceptanceErrorKind::InvalidInput,
                                "acceptance sample has a singular exact-surface normal",
                            )
                        })?,
                );
                if !chordal.is_finite() || !normal.is_finite() {
                    return Err(error(
                        mesh,
                        ExactFaceAcceptanceErrorKind::InvalidInput,
                        "acceptance sample evidence is not finite",
                    ));
                }
                cell_chordal = cell_chordal.max(chordal);
                cell_normal = cell_normal.max(normal);
                triangle_chordal.observe(chordal, uv);
                triangle_normal.observe(normal, uv);
                face_chordal.observe(chordal, uv);
                face_normal.observe(normal, uv);
            }
            if depth < options.minimum_subdivision_depth
                || (depth < options.maximum_subdivision_depth
                    && (cell_chordal
                        > quality.maximum_chordal_deviation_m * options.refinement_margin_ratio
                        || cell_normal
                            > quality.maximum_normal_deviation_degrees.to_radians()
                                * options.refinement_margin_ratio))
            {
                for child in subdivide(cell).into_iter().rev() {
                    cells.push((child, depth + 1));
                }
            }
        }
        enforce_quality(
            mesh,
            evidence.triangle,
            triangle_chordal,
            triangle_normal,
            quality,
        )?;
        triangle_reports.push(ExactFaceTriangleAcceptance {
            triangle: evidence.triangle,
            sample_count: budget.consumed - before,
            maximum_chordal_deviation_m: triangle_chordal.value,
            maximum_normal_deviation_rad: triangle_normal.value,
        });
    }
    Ok(ExactFaceAcceptanceReport {
        source_face_id: mesh.geometry.source_face_id.clone(),
        triangles: triangle_reports,
        sample_count: budget.consumed,
        maximum_chordal_deviation_m: face_chordal.value,
        maximum_chordal_deviation_uv: face_chordal.uv,
        maximum_normal_deviation_rad: face_normal.value,
        maximum_normal_deviation_uv: face_normal.uv,
    })
}

fn validate_inputs(
    mesh: &ExactFaceRefinedMesh,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    options: ExactFaceAcceptanceOptions,
    parameterization: &ExactFaceChartParameterization,
) -> Result<(), ExactFaceAcceptanceError> {
    quality.validate().map_err(|failure| {
        error(
            mesh,
            ExactFaceAcceptanceErrorKind::InvalidOptions,
            failure.to_string(),
        )
    })?;
    if options.minimum_subdivision_depth > options.maximum_subdivision_depth
        || options.maximum_subdivision_depth > 16
        || !options.refinement_margin_ratio.is_finite()
        || !(0.0..1.0).contains(&options.refinement_margin_ratio)
        || options.maximum_samples == 0
    {
        return Err(error(
            mesh,
            ExactFaceAcceptanceErrorKind::InvalidOptions,
            "acceptance depths, refinement margin, or sample bound are invalid",
        ));
    }
    validate_exact_face_geometry_in_parameterization(
        &mesh.geometry,
        &mesh.topology.trimmed,
        &mesh.topology.pslg,
        parameterization,
        ExactFaceGeometryContext::new(
            context.topology,
            context.metric_request,
            context.evaluator,
            context.geometry_control,
        ),
    )
    .map_err(|failure| {
        error(
            mesh,
            ExactFaceAcceptanceErrorKind::Geometry(failure.kind),
            failure.reason,
        )
    })
}

fn enforce_quality(
    mesh: &ExactFaceRefinedMesh,
    triangle: crate::ExactFaceDelaunayTriangle,
    chordal: SampleMaximum,
    normal: SampleMaximum,
    quality: SurfaceQualityTargets,
) -> Result<(), ExactFaceAcceptanceError> {
    if chordal.value > quality.maximum_chordal_deviation_m {
        return Err(error(
            mesh,
            ExactFaceAcceptanceErrorKind::UnsatisfiedQuality,
            "adaptive exact-surface samples exceed the chordal-deviation target",
        )
        .with_witness(triangle, chordal.uv));
    }
    if normal.value > quality.maximum_normal_deviation_degrees.to_radians() {
        return Err(error(
            mesh,
            ExactFaceAcceptanceErrorKind::UnsatisfiedQuality,
            "adaptive exact-surface samples exceed the normal-deviation target",
        )
        .with_witness(triangle, normal.uv));
    }
    Ok(())
}

fn cell_samples(cell: [[f64; 2]; 3]) -> [[f64; 2]; 4] {
    let midpoint = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    [
        midpoint(cell[0], cell[1]),
        midpoint(cell[1], cell[2]),
        midpoint(cell[2], cell[0]),
        [
            (cell[0][0] + cell[1][0] + cell[2][0]) / 3.0,
            (cell[0][1] + cell[1][1] + cell[2][1]) / 3.0,
        ],
    ]
}

fn subdivide(cell: [[f64; 2]; 3]) -> [[[f64; 2]; 3]; 4] {
    let [ab, bc, ca, _] = cell_samples(cell);
    [
        [cell[0], ab, ca],
        [ab, cell[1], bc],
        [ca, bc, cell[2]],
        [ab, bc, ca],
    ]
}

fn barycentric(triangle: [[f64; 2]; 3], point: [f64; 2]) -> Option<[f64; 3]> {
    let denominator = (triangle[1][1] - triangle[2][1]) * (triangle[0][0] - triangle[2][0])
        + (triangle[2][0] - triangle[1][0]) * (triangle[0][1] - triangle[2][1]);
    if !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let first = ((triangle[1][1] - triangle[2][1]) * (point[0] - triangle[2][0])
        + (triangle[2][0] - triangle[1][0]) * (point[1] - triangle[2][1]))
        / denominator;
    let second = ((triangle[2][1] - triangle[0][1]) * (point[0] - triangle[2][0])
        + (triangle[0][0] - triangle[2][0]) * (point[1] - triangle[2][1]))
        / denominator;
    Some([first, second, 1.0 - first - second])
}

fn weighted_point(points: [[f64; 3]; 3], weights: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        points[0][axis] * weights[0] + points[1][axis] * weights[1] + points[2][axis] * weights[2]
    })
}

fn surface_normal(first: [f64; 3], second: [f64; 3]) -> Option<[f64; 3]> {
    let cross = [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ];
    let length = cross
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    (length.is_finite() && length > 0.0).then(|| cross.map(|value| value / length))
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt()
}

fn angle(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
        .acos()
}

fn map_metric(
    mesh: &ExactFaceRefinedMesh,
    failure: ExactFaceMetricError,
) -> ExactFaceAcceptanceError {
    error(
        mesh,
        ExactFaceAcceptanceErrorKind::Metric(failure.kind),
        failure.reason,
    )
}

fn error(
    mesh: &ExactFaceRefinedMesh,
    kind: ExactFaceAcceptanceErrorKind,
    reason: impl Into<String>,
) -> ExactFaceAcceptanceError {
    ExactFaceAcceptanceError::new(kind, &mesh.geometry.source_face_id, reason)
}

#[derive(Clone, Copy, Default)]
struct SampleMaximum {
    value: f64,
    uv: [f64; 2],
}

impl SampleMaximum {
    fn observe(&mut self, value: f64, uv: [f64; 2]) {
        if value > self.value {
            self.value = value;
            self.uv = uv;
        }
    }
}

struct SampleBudget<'a> {
    mesh: &'a ExactFaceRefinedMesh,
    maximum: u64,
    consumed: u64,
}

impl<'a> SampleBudget<'a> {
    fn new(mesh: &'a ExactFaceRefinedMesh, maximum: u64) -> Self {
        Self {
            mesh,
            maximum,
            consumed: 0,
        }
    }

    fn consume(&mut self) -> Result<(), ExactFaceAcceptanceError> {
        self.consumed = self.consumed.checked_add(1).ok_or_else(|| {
            error(
                self.mesh,
                ExactFaceAcceptanceErrorKind::ResourceLimit,
                "acceptance sample count overflow",
            )
        })?;
        if self.consumed > self.maximum {
            return Err(error(
                self.mesh,
                ExactFaceAcceptanceErrorKind::ResourceLimit,
                "acceptance sample hard limit exceeded",
            ));
        }
        Ok(())
    }
}
