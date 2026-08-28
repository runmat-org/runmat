use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;

use crate::{
    validate_exact_face_metric_evaluation_in_parameterization, ExactFaceChartParameterization,
    ExactFaceMetricEvaluation, ExactFacePslg, ExactFaceTrimmedDelaunay, ParametricMetricTensor,
};

use super::{
    ExactFaceGeometry, ExactFaceGeometryContext, ExactFaceGeometryError, ExactFaceGeometryErrorKind,
};

pub fn validate_exact_face_geometry(
    geometry: &ExactFaceGeometry,
    trimmed: &ExactFaceTrimmedDelaunay,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &dyn ExactSurfaceEvaluator,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), ExactFaceGeometryError> {
    validate_exact_face_geometry_in_parameterization(
        geometry,
        trimmed,
        pslg,
        &ExactFaceChartParameterization::EvaluatorParameters,
        ExactFaceGeometryContext::new(topology, request, evaluator, control),
    )
}

pub fn validate_exact_face_geometry_in_parameterization(
    geometry: &ExactFaceGeometry,
    trimmed: &ExactFaceTrimmedDelaunay,
    pslg: &ExactFacePslg,
    parameterization: &ExactFaceChartParameterization,
    context: ExactFaceGeometryContext<'_>,
) -> Result<(), ExactFaceGeometryError> {
    if geometry.source_face_id != pslg.source_face_id
        || trimmed.source_face_id != pslg.source_face_id
        || geometry.vertices.len() != pslg.vertices.len()
        || geometry.triangles.len() != trimmed.triangles.len()
    {
        return Err(invalid(
            &pslg.source_face_id,
            "face geometry inventory differs from its inputs",
        ));
    }
    for (index, (actual, source)) in geometry.vertices.iter().zip(&pslg.vertices).enumerate() {
        checkpoint(context.control, &geometry.source_face_id)?;
        if actual.pslg_vertex_index != index as u32
            || actual.evaluation.source_face_id != geometry.source_face_id
            || actual.evaluation.uv != source.uv
        {
            return Err(invalid(
                &geometry.source_face_id,
                "face geometry vertex identity or UV is inconsistent",
            ));
        }
        validate_exact_face_metric_evaluation_in_parameterization(
            &actual.evaluation,
            context.topology,
            context.metric_request,
            parameterization,
            context.evaluator,
            context.control,
        )
        .map_err(|error| metric(&geometry.source_face_id, error))?;
        let expected_normal = independent_surface_normal(&actual.evaluation)?;
        if !same_vector(actual.unit_normal, expected_normal) {
            return Err(invalid(
                &geometry.source_face_id,
                "face geometry vertex normal is inconsistent",
            ));
        }
    }

    let mut maximum_metric_edge_length: f64 = 0.0;
    let mut minimum_metric_angle_rad = f64::INFINITY;
    let mut maximum_physical_aspect_ratio: f64 = 0.0;
    let mut maximum_chordal_deviation_m: f64 = 0.0;
    let mut maximum_normal_deviation_rad: f64 = 0.0;
    for (actual, triangle) in geometry.triangles.iter().zip(&trimmed.triangles) {
        checkpoint(context.control, &geometry.source_face_id)?;
        if actual.triangle != *triangle {
            return Err(invalid(
                &geometry.source_face_id,
                "face triangle geometry is not in trimmed-topology order",
            ));
        }
        let corners = triangle
            .vertex_indices
            .map(|index| geometry.vertices.get(index as usize))
            .map(|vertex| {
                vertex.ok_or_else(|| {
                    invalid(
                        &geometry.source_face_id,
                        "face triangle references an absent geometry vertex",
                    )
                })
            });
        let [first, second, third] = corners;
        let first = first?;
        let second = second?;
        let third = third?;
        let expected_uv = [
            (first.evaluation.uv[0] + second.evaluation.uv[0] + third.evaluation.uv[0]) / 3.0,
            (first.evaluation.uv[1] + second.evaluation.uv[1] + third.evaluation.uv[1]) / 3.0,
        ];
        if actual.centroid.source_face_id != geometry.source_face_id
            || actual.centroid.uv != expected_uv
        {
            return Err(invalid(
                &geometry.source_face_id,
                "face triangle centroid query is inconsistent",
            ));
        }
        validate_exact_face_metric_evaluation_in_parameterization(
            &actual.centroid,
            context.topology,
            context.metric_request,
            parameterization,
            context.evaluator,
            context.control,
        )
        .map_err(|error| metric(&geometry.source_face_id, error))?;
        let points = [
            first.evaluation.point_m,
            second.evaluation.point_m,
            third.evaluation.point_m,
        ];
        let area_vector = vector_product(
            difference(points[1], points[0]),
            difference(points[2], points[0]),
        );
        let double_area = magnitude(area_vector);
        if !double_area.is_finite() || double_area <= 0.0 {
            return Err(invalid(
                &geometry.source_face_id,
                "face triangle has invalid physical area",
            ));
        }
        let expected_normal = multiply(area_vector, double_area.recip());
        let physical_area_m2 = double_area * 0.5;
        let physical_edges = [
            magnitude(difference(points[1], points[0])),
            magnitude(difference(points[2], points[1])),
            magnitude(difference(points[0], points[2])),
        ];
        let longest = physical_edges.into_iter().fold(0.0, f64::max);
        let aspect = longest * physical_edges.into_iter().sum::<f64>()
            / (4.0 * 3.0_f64.sqrt() * physical_area_m2);
        let metric_edges = [
            independent_metric_edge(&first.evaluation, &second.evaluation)?,
            independent_metric_edge(&second.evaluation, &third.evaluation)?,
            independent_metric_edge(&third.evaluation, &first.evaluation)?,
        ];
        let vertices = [first, second, third];
        let minimum_angle = (0..3)
            .map(|corner| independent_metric_angle(vertices, corner))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let linear_centroid = multiply(sum(sum(points[0], points[1]), points[2]), 1.0 / 3.0);
        let chordal = magnitude(difference(actual.centroid.point_m, linear_centroid));
        let centroid_normal = independent_surface_normal(&actual.centroid)?;
        let normal_deviation = [
            normal_angle(expected_normal, first.unit_normal),
            normal_angle(expected_normal, second.unit_normal),
            normal_angle(expected_normal, third.unit_normal),
            normal_angle(expected_normal, centroid_normal),
        ]
        .into_iter()
        .fold(0.0, f64::max);
        if !same_vector(actual.unit_normal, expected_normal)
            || !same_measure(actual.physical_area_m2, physical_area_m2)
            || !same_vector(actual.metric_edge_lengths, metric_edges)
            || !same_measure(actual.minimum_metric_angle_rad, minimum_angle)
            || !same_measure(actual.physical_aspect_ratio, aspect)
            || !same_measure(actual.chordal_deviation_m, chordal)
            || !same_measure(actual.normal_deviation_rad, normal_deviation)
        {
            return Err(invalid(
                &geometry.source_face_id,
                "reported face triangle geometry is inconsistent",
            ));
        }
        maximum_metric_edge_length =
            maximum_metric_edge_length.max(metric_edges.into_iter().fold(0.0, f64::max));
        minimum_metric_angle_rad = minimum_metric_angle_rad.min(minimum_angle);
        maximum_physical_aspect_ratio = maximum_physical_aspect_ratio.max(aspect);
        maximum_chordal_deviation_m = maximum_chordal_deviation_m.max(chordal);
        maximum_normal_deviation_rad = maximum_normal_deviation_rad.max(normal_deviation);
    }
    if !same_measure(
        geometry.maximum_metric_edge_length,
        maximum_metric_edge_length,
    ) || !same_measure(geometry.minimum_metric_angle_rad, minimum_metric_angle_rad)
        || !same_measure(
            geometry.maximum_physical_aspect_ratio,
            maximum_physical_aspect_ratio,
        )
        || !same_measure(
            geometry.maximum_chordal_deviation_m,
            maximum_chordal_deviation_m,
        )
        || !same_measure(
            geometry.maximum_normal_deviation_rad,
            maximum_normal_deviation_rad,
        )
    {
        return Err(invalid(
            &geometry.source_face_id,
            "face geometry summary is inconsistent",
        ));
    }
    Ok(())
}

fn independent_metric_edge(
    left: &ExactFaceMetricEvaluation,
    right: &ExactFaceMetricEvaluation,
) -> Result<f64, ExactFaceGeometryError> {
    let tensor = ParametricMetricTensor {
        uu: 0.5 * (left.sizing_metric.uu + right.sizing_metric.uu),
        uv: 0.5 * (left.sizing_metric.uv + right.sizing_metric.uv),
        vv: 0.5 * (left.sizing_metric.vv + right.sizing_metric.vv),
    };
    let displacement = [right.uv[0] - left.uv[0], right.uv[1] - left.uv[1]];
    tensor
        .squared_length(displacement)
        .map(f64::sqrt)
        .map_err(|reason| invalid(&left.source_face_id, reason))
}

fn independent_metric_angle(
    vertices: [&crate::ExactFaceGeometryVertex; 3],
    corner: usize,
) -> Result<f64, ExactFaceGeometryError> {
    let origin = &vertices[corner].evaluation;
    let first_uv = vertices[(corner + 1) % 3].evaluation.uv;
    let second_uv = vertices[(corner + 2) % 3].evaluation.uv;
    let first = [first_uv[0] - origin.uv[0], first_uv[1] - origin.uv[1]];
    let second = [second_uv[0] - origin.uv[0], second_uv[1] - origin.uv[1]];
    let tensor = origin.sizing_metric;
    let first_norm = tensor
        .squared_length(first)
        .map_err(|reason| invalid(&origin.source_face_id, reason))?;
    let second_norm = tensor
        .squared_length(second)
        .map_err(|reason| invalid(&origin.source_face_id, reason))?;
    let inner = first[0] * (tensor.uu * second[0] + tensor.uv * second[1])
        + first[1] * (tensor.uv * second[0] + tensor.vv * second[1]);
    Ok((inner / (first_norm * second_norm).sqrt())
        .clamp(-1.0, 1.0)
        .acos())
}

fn independent_surface_normal(
    evaluation: &ExactFaceMetricEvaluation,
) -> Result<[f64; 3], ExactFaceGeometryError> {
    let normal = vector_product(evaluation.derivative_u_m, evaluation.derivative_v_m);
    let length = magnitude(normal);
    if !length.is_finite() || length <= 0.0 {
        Err(invalid(
            &evaluation.source_face_id,
            "surface chart is singular at an evaluated point",
        ))
    } else {
        Ok(multiply(normal, length.recip()))
    }
}

fn checkpoint(
    control: &dyn GeometryEvaluationControl,
    face_id: &runmat_geometry_core::PersistentEntityId,
) -> Result<(), ExactFaceGeometryError> {
    control.checkpoint().map_err(|error| {
        ExactFaceGeometryError::new(
            ExactFaceGeometryErrorKind::Metric(
                crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
            ),
            face_id,
            error.reason,
        )
    })
}

fn metric(
    face_id: &runmat_geometry_core::PersistentEntityId,
    error: crate::ExactFaceMetricError,
) -> ExactFaceGeometryError {
    ExactFaceGeometryError::new(
        ExactFaceGeometryErrorKind::Metric(error.kind),
        error.source_face_id.as_ref().unwrap_or(face_id),
        error.reason,
    )
}

fn invalid(
    face_id: &runmat_geometry_core::PersistentEntityId,
    reason: impl Into<String>,
) -> ExactFaceGeometryError {
    ExactFaceGeometryError::new(
        ExactFaceGeometryErrorKind::InvalidEvaluation,
        face_id,
        reason,
    )
}

fn sum(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn difference(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn multiply(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn vector_product(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn scalar_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn magnitude(vector: [f64; 3]) -> f64 {
    scalar_product(vector, vector).sqrt()
}

fn normal_angle(left: [f64; 3], right: [f64; 3]) -> f64 {
    let cosine = scalar_product(left, right).clamp(-1.0, 1.0);
    if 1.0 - cosine <= 64.0 * f64::EPSILON {
        0.0
    } else if cosine + 1.0 <= 64.0 * f64::EPSILON {
        std::f64::consts::PI
    } else {
        cosine.acos()
    }
}

fn same_vector<const N: usize>(left: [f64; N], right: [f64; N]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(actual, expected)| same_measure(actual, expected))
}

fn same_measure(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs()
            <= 64.0 * f64::EPSILON * actual.abs().max(expected.abs()).max(1.0)
}
