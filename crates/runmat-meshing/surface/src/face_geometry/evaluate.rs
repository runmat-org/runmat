use runmat_geometry_core::{ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl};
use runmat_meshing_core::MetricFieldRequest;

use crate::{
    ExactFaceMetricError, ExactFaceMetricEvaluation, ExactFacePslg, ExactFaceTrimmedDelaunay,
    ParametricMetricTensor, ResolvedFaceMetricField,
};

use super::{
    ExactFaceGeometry, ExactFaceGeometryError, ExactFaceGeometryErrorKind, ExactFaceGeometryVertex,
    ExactFaceTriangleGeometry,
};

pub fn evaluate_exact_face_geometry(
    trimmed: &ExactFaceTrimmedDelaunay,
    pslg: &ExactFacePslg,
    topology: &ExactBRepTopology,
    request: &MetricFieldRequest,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
) -> Result<ExactFaceGeometry, ExactFaceGeometryError> {
    validate_inventory(trimmed, pslg)?;
    let field = ResolvedFaceMetricField::new(topology, request)
        .map_err(|error| map_metric(&pslg.source_face_id, error))?;
    let mut vertices = Vec::with_capacity(pslg.vertices.len());
    for (index, vertex) in pslg.vertices.iter().enumerate() {
        control.checkpoint().map_err(|error| {
            ExactFaceGeometryError::new(
                ExactFaceGeometryErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                ),
                &pslg.source_face_id,
                error.reason,
            )
        })?;
        let evaluation = field
            .evaluate(&pslg.source_face_id, vertex.uv, evaluator, control)
            .map_err(|error| map_metric(&pslg.source_face_id, error))?;
        let unit_normal = surface_normal(&evaluation)?;
        vertices.push(ExactFaceGeometryVertex {
            pslg_vertex_index: index as u32,
            evaluation,
            unit_normal,
        });
    }

    let mut triangles = Vec::with_capacity(trimmed.triangles.len());
    for triangle in &trimmed.triangles {
        control.checkpoint().map_err(|error| {
            ExactFaceGeometryError::new(
                ExactFaceGeometryErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(error.kind),
                ),
                &pslg.source_face_id,
                error.reason,
            )
        })?;
        let corners = triangle
            .vertex_indices
            .map(|index| &vertices[index as usize]);
        let centroid_uv = [
            corners
                .iter()
                .map(|vertex| vertex.evaluation.uv[0])
                .sum::<f64>()
                / 3.0,
            corners
                .iter()
                .map(|vertex| vertex.evaluation.uv[1])
                .sum::<f64>()
                / 3.0,
        ];
        let centroid = field
            .evaluate(&pslg.source_face_id, centroid_uv, evaluator, control)
            .map_err(|error| map_metric(&pslg.source_face_id, error))?;
        triangles.push(evaluate_triangle(*triangle, corners, centroid)?);
    }
    let geometry = ExactFaceGeometry {
        source_face_id: pslg.source_face_id.clone(),
        maximum_metric_edge_length: triangles
            .iter()
            .flat_map(|triangle| triangle.metric_edge_lengths)
            .fold(0.0, f64::max),
        minimum_metric_angle_rad: triangles
            .iter()
            .map(|triangle| triangle.minimum_metric_angle_rad)
            .fold(f64::INFINITY, f64::min),
        maximum_physical_aspect_ratio: triangles
            .iter()
            .map(|triangle| triangle.physical_aspect_ratio)
            .fold(0.0, f64::max),
        maximum_chordal_deviation_m: triangles
            .iter()
            .map(|triangle| triangle.chordal_deviation_m)
            .fold(0.0, f64::max),
        maximum_normal_deviation_rad: triangles
            .iter()
            .map(|triangle| triangle.normal_deviation_rad)
            .fold(0.0, f64::max),
        vertices,
        triangles,
    };
    ensure_summary_is_finite(&geometry)?;
    Ok(geometry)
}

fn evaluate_triangle(
    triangle: crate::ExactFaceDelaunayTriangle,
    corners: [&ExactFaceGeometryVertex; 3],
    centroid: ExactFaceMetricEvaluation,
) -> Result<ExactFaceTriangleGeometry, ExactFaceGeometryError> {
    let face_id = centroid.source_face_id.clone();
    let points = corners.map(|vertex| vertex.evaluation.point_m);
    let first = subtract(points[1], points[0]);
    let second = subtract(points[2], points[0]);
    let area_vector = cross(first, second);
    let double_area = norm(area_vector);
    if !double_area.is_finite() || double_area <= 0.0 {
        return Err(invalid(
            &face_id,
            "triangle has zero or invalid physical area",
        ));
    }
    let unit_normal = scale(area_vector, double_area.recip());
    let edge_lengths = [
        norm(subtract(points[1], points[0])),
        norm(subtract(points[2], points[1])),
        norm(subtract(points[0], points[2])),
    ];
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= 0.0)
    {
        return Err(invalid(&face_id, "triangle has an invalid physical edge"));
    }
    let perimeter = edge_lengths.iter().sum::<f64>();
    let physical_area_m2 = double_area * 0.5;
    let physical_aspect_ratio = edge_lengths.iter().copied().fold(0.0, f64::max) * perimeter
        / (4.0 * 3.0_f64.sqrt() * physical_area_m2);

    let metric_edge_lengths = [
        metric_edge(&corners[0].evaluation, &corners[1].evaluation)?,
        metric_edge(&corners[1].evaluation, &corners[2].evaluation)?,
        metric_edge(&corners[2].evaluation, &corners[0].evaluation)?,
    ];
    let minimum_metric_angle_rad = (0..3)
        .map(|corner| metric_angle(corners, corner))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let planar_centroid = scale(add(add(points[0], points[1]), points[2]), 1.0 / 3.0);
    let chordal_deviation_m = norm(subtract(centroid.point_m, planar_centroid));
    let normal_deviation_rad = corners
        .iter()
        .map(|vertex| normal_angle(unit_normal, vertex.unit_normal))
        .chain(std::iter::once(normal_angle(
            unit_normal,
            surface_normal(&centroid)?,
        )))
        .fold(0.0, f64::max);
    let result = ExactFaceTriangleGeometry {
        triangle,
        centroid,
        unit_normal,
        physical_area_m2,
        metric_edge_lengths,
        minimum_metric_angle_rad,
        physical_aspect_ratio,
        chordal_deviation_m,
        normal_deviation_rad,
    };
    if [
        result.physical_area_m2,
        result.minimum_metric_angle_rad,
        result.physical_aspect_ratio,
        result.chordal_deviation_m,
        result.normal_deviation_rad,
    ]
    .into_iter()
    .chain(result.metric_edge_lengths)
    .any(|value| !value.is_finite())
    {
        return Err(invalid(
            &face_id,
            "triangle geometry evidence is not finite",
        ));
    }
    Ok(result)
}

fn metric_edge(
    left: &ExactFaceMetricEvaluation,
    right: &ExactFaceMetricEvaluation,
) -> Result<f64, ExactFaceGeometryError> {
    let average = ParametricMetricTensor {
        uu: (left.sizing_metric.uu + right.sizing_metric.uu) * 0.5,
        uv: (left.sizing_metric.uv + right.sizing_metric.uv) * 0.5,
        vv: (left.sizing_metric.vv + right.sizing_metric.vv) * 0.5,
    };
    let delta = [right.uv[0] - left.uv[0], right.uv[1] - left.uv[1]];
    average
        .squared_length(delta)
        .map(f64::sqrt)
        .map_err(|reason| invalid(&left.source_face_id, reason))
}

fn metric_angle(
    corners: [&ExactFaceGeometryVertex; 3],
    corner: usize,
) -> Result<f64, ExactFaceGeometryError> {
    let origin = &corners[corner].evaluation;
    let left = corners[(corner + 1) % 3].evaluation.uv;
    let right = corners[(corner + 2) % 3].evaluation.uv;
    let first = [left[0] - origin.uv[0], left[1] - origin.uv[1]];
    let second = [right[0] - origin.uv[0], right[1] - origin.uv[1]];
    let tensor = origin.sizing_metric;
    let first_squared = tensor
        .squared_length(first)
        .map_err(|reason| invalid(&origin.source_face_id, reason))?;
    let second_squared = tensor
        .squared_length(second)
        .map_err(|reason| invalid(&origin.source_face_id, reason))?;
    let product = tensor.uu * first[0] * second[0]
        + tensor.uv * (first[0] * second[1] + first[1] * second[0])
        + tensor.vv * first[1] * second[1];
    Ok((product / (first_squared * second_squared).sqrt())
        .clamp(-1.0, 1.0)
        .acos())
}

fn surface_normal(
    evaluation: &ExactFaceMetricEvaluation,
) -> Result<[f64; 3], ExactFaceGeometryError> {
    let vector = cross(evaluation.derivative_u_m, evaluation.derivative_v_m);
    let length = norm(vector);
    if !length.is_finite() || length <= 0.0 {
        return Err(invalid(
            &evaluation.source_face_id,
            "surface chart is singular at an evaluated point",
        ));
    }
    Ok(scale(vector, length.recip()))
}

fn normal_angle(left: [f64; 3], right: [f64; 3]) -> f64 {
    let cosine = dot(left, right).clamp(-1.0, 1.0);
    if 1.0 - cosine <= 64.0 * f64::EPSILON {
        0.0
    } else if cosine + 1.0 <= 64.0 * f64::EPSILON {
        std::f64::consts::PI
    } else {
        cosine.acos()
    }
}

fn validate_inventory(
    trimmed: &ExactFaceTrimmedDelaunay,
    pslg: &ExactFacePslg,
) -> Result<(), ExactFaceGeometryError> {
    if trimmed.source_face_id != pslg.source_face_id
        || trimmed.triangles.is_empty()
        || pslg.vertices.is_empty()
        || trimmed.triangles.iter().any(|triangle| {
            triangle
                .vertex_indices
                .iter()
                .any(|index| *index as usize >= pslg.vertices.len())
        })
    {
        return Err(ExactFaceGeometryError::new(
            ExactFaceGeometryErrorKind::InvalidInput,
            &pslg.source_face_id,
            "trimmed topology and face PSLG inventories are inconsistent",
        ));
    }
    Ok(())
}

fn ensure_summary_is_finite(geometry: &ExactFaceGeometry) -> Result<(), ExactFaceGeometryError> {
    if [
        geometry.maximum_metric_edge_length,
        geometry.minimum_metric_angle_rad,
        geometry.maximum_physical_aspect_ratio,
        geometry.maximum_chordal_deviation_m,
        geometry.maximum_normal_deviation_rad,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        Err(invalid(
            &geometry.source_face_id,
            "face geometry summary is not finite",
        ))
    } else {
        Ok(())
    }
}

fn map_metric(
    face_id: &runmat_geometry_core::PersistentEntityId,
    error: ExactFaceMetricError,
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

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
