//! Canonical geometry domain model for RunMat.

pub mod diagnostics;
pub mod model;
pub mod selection;

pub use diagnostics::{Diagnostic, DiagnosticSeverity};
pub use model::{
    admit_exact_geometry_closure, decode_exact_evaluators, decode_exact_topology,
    decode_geometry_document, decode_geometry_healing_report, encode_exact_evaluators,
    encode_exact_topology, encode_geometry_document, encode_geometry_healing_report,
    surface_principal_curvature, surface_unit_normal, AdmittedExactGeometry, AssemblyNode,
    BodyMassProperties, CadColorEvidence, CadCurveEvaluationSample, CadCurveEvaluationSampleSource,
    CadCurveEvaluator, CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource,
    CadFaceEvaluator, CadLabelRef, CadPhysicalMaterialEvidence, CadRegionOwnership,
    CadSemanticKind, CurveDerivatives, CurveEvaluatorId, CurveProjection, DisplayTessellationRef,
    EntityIdRange, ExactAssembly, ExactBRepModel, ExactBRepTopology, ExactBody, ExactCoedge,
    ExactContactPair, ExactCurveDefinition, ExactCurveEvaluator, ExactCurveEvaluatorRecord,
    ExactCurveImplementation, ExactEdge, ExactEvaluatorRegistry, ExactFace,
    ExactGeometryCapabilities, ExactGeometryManifest, ExactInstance, ExactLump,
    ExactMassPropertiesEvaluator, ExactMassPropertiesImplementation, ExactMassPropertiesRecord,
    ExactPcurveDefinition, ExactPcurveEvaluator, ExactPcurveEvaluatorRecord,
    ExactPcurveImplementation, ExactSharedInterface, ExactShell, ExactSolid,
    ExactSurfaceDefinition, ExactSurfaceEvaluator, ExactSurfaceEvaluatorRecord,
    ExactSurfaceImplementation, ExactTrimClassifier, ExactTrimClassifierImplementation,
    ExactTrimClassifierRecord, ExactVertex, ExactWire, FacetedSolidModel, GeometryAsset,
    GeometryContractError, GeometryDigest, GeometryDocument, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryHealingFailure,
    GeometryHealingOperation, GeometryHealingOperationKind, GeometryHealingPolicy,
    GeometryHealingReport, GeometryModel, GeometryObjectRef, GeometryRevisionConflict,
    GeometryRevisionConflictKind, GeometryRevisionIdentity, GeometryRevisionMap,
    GeometryRevisionMappingError, GeometryRevisionOperation, GeometryRevisionResolution,
    GeometrySource, GeometrySourceFormat, GeometrySourceIdentity, GeometryTolerancePolicy,
    GeometryTransform, KernelEvaluatorRef, MassPropertiesEvaluatorId, MaterialEvidence,
    MaterialEvidenceConfidence, MeshDescriptor, MeshKind, NurbsCurve2, NurbsCurve3, NurbsSurface3,
    OrientedEntityUse, ParameterRange, PcurveDerivatives, PcurveEvaluatorId, PersistentEntityId,
    PersistentEntityKind, PortableExactEvaluator, Region, RegionEntityMapping, SourceGeometry,
    SourceGeometryKind, SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceMesh,
    SurfaceProjection, TessellationProfile, TopologicalOrientation, TopologyValidity,
    TrimClassifierId, TrimDomainLocation, UnitSystem, DISPLAY_TESSELLATION_MEDIA_TYPE,
    EXACT_BREP_MEDIA_TYPE, EXACT_BREP_TOPOLOGY_SCHEMA_VERSION, EXACT_EVALUATOR_MEDIA_TYPE,
    EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION, EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION,
    EXACT_TOPOLOGY_MEDIA_TYPE, FACETED_SOLID_MEDIA_TYPE, GEOMETRY_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_HEALING_MEDIA_TYPE, GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION, GEOMETRY_REVISION_MAP_SCHEMA_VERSION,
    KERNEL_REPRESENTATION_MEDIA_TYPE,
};
pub use selection::{EntityKind, EntityRef};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_test".to_string(),
            source: GeometrySource {
                path: "/models/part.stl".to_string(),
                sha256: "abc123".to_string(),
                importer_version: "stl/v1".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: vec![],
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_1".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 3,
                element_count: 1,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "mesh_1",
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[0, 1, 2]],
            )],
            regions: vec![Region {
                region_id: "region_a".to_string(),
                name: "body".to_string(),
                tag: None,
                cad_ownership: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces("region_a", "mesh_1", 1)],
            diagnostics: vec![],
        }
    }

    #[test]
    fn entity_identity_stable_within_revision() {
        let first = EntityRef {
            geometry_id: "geo_test".to_string(),
            geometry_revision: 1,
            mesh_id: "mesh_1".to_string(),
            entity_kind: EntityKind::Face,
            entity_id: 42,
        };
        let second = EntityRef { ..first.clone() };
        assert_eq!(first, second);
    }

    #[test]
    fn geometry_asset_round_trips_via_json() {
        let asset = sample_asset();
        let json = serde_json::to_string(&asset).expect("serialize");
        let decoded: GeometryAsset = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, asset);
    }

    #[test]
    fn unit_metadata_must_be_present() {
        let mut asset = sample_asset();
        asset.units = UnitSystem::Unspecified;
        let error = asset
            .validate()
            .expect_err("expected unspecified units to fail");
        assert_eq!(error, "geometry units must be specified");
    }
}
