//! Canonical geometry domain model for RunMat.

pub mod diagnostics;
pub mod model;
pub mod selection;

pub use diagnostics::{Diagnostic, DiagnosticSeverity};
pub use model::{
    AssemblyNode, BodyMassPropertiesV2, CadColorEvidence, CadCurveEvaluationSample,
    CadCurveEvaluationSampleSource, CadCurveEvaluator, CadEvaluatorSet, CadFaceEvaluationSample,
    CadFaceEvaluationSampleSource, CadFaceEvaluator, CadLabelRef, CadPhysicalMaterialEvidence,
    CadRegionOwnership, CadSemanticKind, CurveDerivativesV2, CurveEvaluatorIdV2, CurveProjectionV2,
    DisplayTessellationRefV2, EntityIdRange, ExactAssemblyV2, ExactBRepModelV2,
    ExactBRepTopologyV2, ExactBodyV2, ExactCoedgeV2, ExactContactPairV2, ExactCurveDefinitionV2,
    ExactCurveEvaluatorRecordV2, ExactCurveEvaluatorV2, ExactCurveImplementationV2, ExactEdgeV2,
    ExactEvaluatorRegistryV2, ExactFaceV2, ExactGeometryCapabilitiesV2, ExactInstanceV2,
    ExactLumpV2, ExactMassPropertiesEvaluatorV2, ExactMassPropertiesImplementationV2,
    ExactMassPropertiesRecordV2, ExactPcurveDefinitionV2, ExactPcurveEvaluatorRecordV2,
    ExactPcurveEvaluatorV2, ExactPcurveImplementationV2, ExactSharedInterfaceV2, ExactShellV2,
    ExactSolidV2, ExactSurfaceDefinitionV2, ExactSurfaceEvaluatorRecordV2, ExactSurfaceEvaluatorV2,
    ExactSurfaceImplementationV2, ExactTrimClassifierImplementationV2, ExactTrimClassifierRecordV2,
    ExactTrimClassifierV2, ExactVertexV2, ExactWireV2, FacetedSolidModelV2, GeometryAsset,
    GeometryContractError, GeometryDigest, GeometryDocumentV2, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryHealingPolicyV2, GeometryModelV2,
    GeometryObjectRefV2, GeometryRevisionIdentityV2, GeometrySource, GeometrySourceFormatV2,
    GeometrySourceIdentityV2, GeometryTolerancePolicy, GeometryTransformV2, KernelEvaluatorRefV2,
    MassPropertiesEvaluatorIdV2, MaterialEvidence, MaterialEvidenceConfidence, MeshDescriptor,
    MeshKind, NurbsCurve2V2, NurbsCurve3V2, NurbsSurface3V2, OrientedEntityUseV2, ParameterRangeV2,
    PcurveDerivativesV2, PcurveEvaluatorIdV2, PersistentEntityId, PersistentEntityKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceCurvatureV2,
    SurfaceDerivativesV2, SurfaceEvaluatorIdV2, SurfaceMesh, SurfaceProjectionV2,
    TessellationProfile, TopologicalOrientationV2, TrimClassifierIdV2, TrimDomainLocationV2,
    UnitSystem, DISPLAY_TESSELLATION_MEDIA_TYPE_V2, EXACT_BREP_MEDIA_TYPE_V2,
    EXACT_BREP_TOPOLOGY_SCHEMA_VERSION, EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
    FACETED_SOLID_MEDIA_TYPE_V2, GEOMETRY_DOCUMENT_SCHEMA_VERSION,
    GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
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
