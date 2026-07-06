use runmat_geometry_core::{CadCurveEvaluationSample, GeometryAsset};
use runmat_meshing_cad::{
    build_cad_evaluation_model, build_cad_topology, extract_source_topology, project_to_face,
    CadEvaluationModel,
};
use runmat_meshing_core::{
    quality::predicate::Point3, AnalysisMeshArtifact, MeshBackendKind, MeshSizingField,
    TopologyEntityId, VolumeMeshingOptions,
};
use runmat_meshing_curve::{
    build_curve_mesh_contract, discretize_cad_topology_curves_with_sizing_and_provider,
    CadCurveEvaluationRequest, CadCurveEvaluatorProvider, CurveValidationOptions,
};
use runmat_meshing_plc::build::build_protected_boundary_complex;
use runmat_meshing_surface::{
    build_surface_mesh_contract, discretize_cad_topology_surfaces_with_cad_curves,
    validate_cad_topology_surface_discretization, SurfaceValidationOptions,
};
use runmat_meshing_tetrahedron::{
    optimize::{
        optimize_recovered_tetrahedron_mesh, RecoveredTetrahedronMeshOptimizationOptions,
        TetrahedronBoundarySmoothingProjection, TetrahedronBoundarySmoothingProjector,
    },
    recover::recover_tetrahedron_mesh_from_plc,
    structured_grid,
};

mod artifact;
mod error;
mod options;
mod sizing;
mod stage_options;
mod tetrahedron_stage;

use artifact::{
    analysis_artifact_from_tetrahedron_mesh, backend_quality_evidence_from_tetrahedron_mesh,
};
pub use error::SolidMeshingError;
use options::validate_solid_options;
use sizing::sizing_with_curve_application_evidence;
use stage_options::{curve_discretization_options, surface_discretization_options};
use tetrahedron_stage::generate_solid_tetrahedron_mesh;

pub fn generate_analysis_mesh(
    geometry: &GeometryAsset,
    options: VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    match options.backend {
        MeshBackendKind::Auto | MeshBackendKind::Solid => generate_solid_analysis_mesh(
            geometry,
            &VolumeMeshingOptions {
                backend: MeshBackendKind::Solid,
                ..options
            },
        ),
        MeshBackendKind::StructuredGridTetrahedron => {
            structured_grid::generate_analysis_mesh(geometry, options)
                .map_err(SolidMeshingError::StructuredGrid)
        }
    }
}

pub fn generate_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    match options.backend {
        MeshBackendKind::Auto | MeshBackendKind::Solid => generate_solid_analysis_mesh_with_sizing(
            geometry,
            &VolumeMeshingOptions {
                backend: MeshBackendKind::Solid,
                ..options
            },
            sizing,
        ),
        MeshBackendKind::StructuredGridTetrahedron => {
            structured_grid::generate_analysis_mesh_with_sizing(geometry, options, sizing)
                .map_err(SolidMeshingError::StructuredGrid)
        }
    }
}

pub fn generate_solid_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    generate_solid_analysis_mesh_with_sizing(geometry, options, &MeshSizingField::default())
}

pub fn generate_solid_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    validate_solid_options(options)?;

    let topology = extract_source_topology(geometry).map_err(SolidMeshingError::SourceTopology)?;
    let cad_topology =
        build_cad_topology(geometry, &topology).map_err(SolidMeshingError::CadTopology)?;
    let cad_evaluation = build_cad_evaluation_model(&cad_topology, &topology)
        .map_err(SolidMeshingError::CadEvaluation)?;
    let curve_options = curve_discretization_options(options, geometry);
    let cad_curves = discretize_cad_topology_curves_with_sizing_and_provider(
        &topology,
        &cad_topology,
        curve_options,
        Some(sizing),
        &GeometryCadCurveEvaluatorProvider { geometry },
    )
    .map_err(SolidMeshingError::Curve)?;
    let artifact_sizing = sizing_with_curve_application_evidence(sizing, &topology, curve_options);
    let _curve_mesh_contract = build_curve_mesh_contract(
        "solid_curve_mesh",
        &topology,
        &cad_curves.curves,
        CurveValidationOptions::default(),
    )
    .map_err(SolidMeshingError::CurveValidation)?;
    let surface = discretize_cad_topology_surfaces_with_cad_curves(
        &cad_topology,
        &topology,
        &cad_evaluation,
        &cad_curves,
        surface_discretization_options(),
    )
    .map_err(SolidMeshingError::Surface)?;
    let surface_validation = validate_cad_topology_surface_discretization(
        &cad_topology,
        &topology,
        &surface,
        SurfaceValidationOptions::default(),
    )
    .map_err(SolidMeshingError::SurfaceValidation)?;
    let surface_mesh_contract =
        build_surface_mesh_contract("solid_surface_mesh", &surface, &surface_validation);
    let plc = build_protected_boundary_complex(&surface_mesh_contract)
        .map_err(SolidMeshingError::ProtectedBoundaryComplex)?;
    let tetrahedron_mesh = generate_solid_tetrahedron_mesh(&plc)?;
    let mut recovery = recover_tetrahedron_mesh_from_plc(&plc, tetrahedron_mesh)
        .map_err(SolidMeshingError::TetrahedronRecovery)?;
    let initial_backend_quality =
        backend_quality_evidence_from_tetrahedron_mesh(&recovery.tetrahedron_mesh);
    optimize_recovered_tetrahedron_mesh(
        &mut recovery.tetrahedron_mesh,
        &CadFaceBoundarySmoothingProjector {
            cad_evaluation: &cad_evaluation,
        },
        RecoveredTetrahedronMeshOptimizationOptions::default(),
    );

    Ok(analysis_artifact_from_tetrahedron_mesh(
        geometry,
        &artifact_sizing,
        &surface_mesh_contract,
        &recovery.recovery_queue,
        initial_backend_quality,
        recovery.tetrahedron_mesh,
    ))
}

struct GeometryCadCurveEvaluatorProvider<'a> {
    geometry: &'a GeometryAsset,
}

impl CadCurveEvaluatorProvider for GeometryCadCurveEvaluatorProvider<'_> {
    fn evaluate_curve(
        &self,
        request: &CadCurveEvaluationRequest<'_>,
    ) -> Vec<CadCurveEvaluationSample> {
        self.geometry
            .source_geometry
            .cad_evaluators
            .iter()
            .flat_map(|set| set.curves.iter())
            .filter(|curve| {
                request
                    .imported_curve_id
                    .is_some_and(|curve_id| curve.imported_curve_id == curve_id)
                    && request
                        .evaluator_id
                        .is_none_or(|evaluator_id| curve.evaluator_id == evaluator_id)
            })
            .flat_map(|curve| curve.evaluation_samples.iter())
            .filter(|sample| {
                request
                    .parameters
                    .iter()
                    .any(|parameter| (*parameter - sample.parameter).abs() <= 1.0e-12)
            })
            .cloned()
            .collect()
    }
}

struct CadFaceBoundarySmoothingProjector<'a> {
    cad_evaluation: &'a CadEvaluationModel,
}

impl TetrahedronBoundarySmoothingProjector for CadFaceBoundarySmoothingProjector<'_> {
    fn project_to_source_face(
        &self,
        source_face_id: &TopologyEntityId,
        point_m: Point3,
    ) -> Option<TetrahedronBoundarySmoothingProjection> {
        let source_face_id = source_face_id.id.parse::<u32>().ok()?;
        let frame = self
            .cad_evaluation
            .face_frames
            .iter()
            .find(|frame| frame.source_face_id == source_face_id)?;
        let projection = project_to_face(frame, point_m);
        Some(TetrahedronBoundarySmoothingProjection {
            point_m: projection.point_m,
            distance_m: projection.distance_m,
            in_bounds: projection.uv_in_bounds,
        })
    }
}

#[cfg(test)]
mod tests;
