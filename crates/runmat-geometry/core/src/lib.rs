//! Canonical geometry domain model for RunMat.

pub use runmat_types::physical;
/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");

pub mod diagnostics;
pub mod model;
pub mod selection;

pub use diagnostics::{Diagnostic, DiagnosticSeverity};
pub use model::{
    admit_exact_geometry_closure, admit_faceted_solid, admit_faceted_solid_with_control,
    author_exact_contacts, build_exact_geometry_closure, build_faceted_solid_closure,
    decode_exact_evaluators, decode_exact_evaluators_with_control,
    decode_exact_geometry_components, decode_exact_geometry_components_with_control,
    decode_exact_topology, decode_exact_topology_with_control, decode_faceted_solid,
    decode_faceted_solid_with_control, decode_geometry_document, decode_geometry_healing_report,
    decode_geometry_healing_report_with_control, encode_exact_evaluators, encode_exact_topology,
    encode_faceted_solid, encode_geometry_document, encode_geometry_healing_report,
    surface_principal_curvature, surface_unit_normal, validate_exact_geometry_component_bindings,
    validate_exact_incidence, AdmittedExactGeometry, Ap242Association, Ap242AssociationTarget,
    Ap242Capability, Ap242CapabilityDisposition, Ap242CapabilityRecord, Ap242Change,
    Ap242ChangeSet, Ap242EngineeringProperty, Ap242EngineeringPropertyKind,
    Ap242EnvironmentCondition, Ap242ExchangeInventory, Ap242ExportDisposition,
    Ap242ExportItemReport, Ap242ExportReport, Ap242MaterialCandidate, Ap242MaterialClass,
    Ap242Measure, Ap242Parameter, Ap242ParameterAccess, Ap242PersistentId, Ap242PersistentIdRole,
    Ap242UnsupportedConcept, AssemblyNode, BodyMassProperties, CadColorEvidence,
    CadCurveEvaluationSample, CadCurveEvaluationSampleSource, CadCurveEvaluator, CadEvaluatorSet,
    CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator, CadLabelRef,
    CadParameterAccess, CadParameterDefinition, CadParameterPhysicalType, CadParameterSelector,
    CrossSectionDefinition, CrossSectionEnclosure, CrossSectionLayer, CrossSectionLayerRole,
    CrossSectionStackupSource, CrossSectionTrace, CrossSectionTraceBase, CrossSectionTraceRole,
    CadPhysicalMaterialEvidence, CadRegionOwnership, CadSemanticKind, CurveDerivatives,
    CurveEvaluatorId, CurveProjection, DecodedExactGeometryComponents, DerivedGeometryDefinition,
    DisplayTessellationRef, EncodedExactGeometryClosure, EncodedFacetedSolidClosure, EntityIdRange,
    ExactAssembly, ExactBRepModel, ExactBRepTopology, ExactBody, ExactCoedge,
    ExactContactDefinition, ExactContactPair, ExactCurveDefinition, ExactCurveEvaluator,
    ExactCurveEvaluatorRecord, ExactCurveImplementation, ExactEdge, ExactEvaluatorRegistry,
    ExactFace, ExactGeometryCapabilities, ExactGeometryComponentBytes, ExactGeometryManifest,
    ExactInstance, ExactLump, ExactMassPropertiesEvaluator, ExactMassPropertiesImplementation,
    ExactMassPropertiesRecord, ExactPcurveDefinition, ExactPcurveEvaluator,
    ExactPcurveEvaluatorRecord, ExactPcurveImplementation, ExactRegion, ExactSharedInterface,
    ExactShell, ExactSolid, ExactSurfaceDefinition, ExactSurfaceEvaluator,
    ExactSurfaceEvaluatorRecord, ExactSurfaceImplementation, ExactTrimClassifier,
    ExactTrimClassifierImplementation, ExactTrimClassifierRecord, ExactVertex, ExactWire,
    FacetedShell, FacetedShellOrientation, FacetedSolid, FacetedSolidModel, FacetedTriangle,
    FacetedVertex, GeneratedGeometryArgument, GeneratedGeometryDefinition,
    GeneratedGeometryProduct, GeneratedGeometryRegionOutput, GeometryAllocationReservation,
    GeometryAllocationScope, GeometryAsset, GeometryAuthoringCapabilities, GeometryAuthoringValue,
    GeometryCanonicalControl, GeometryContractError, GeometryDefinition, GeometryDefinitionRef,
    GeometryDefinitionSource, GeometryDerivative, GeometryDigest, GeometryDocument,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryFeatureRegionSelection, GeometryHealingFailure, GeometryHealingOperation,
    GeometryHealingOperationKind, GeometryHealingPolicy, GeometryHealingReport, GeometryModel,
    GeometryObjectRef, GeometryParameterAxis, GeometryParameterCoordinate,
    GeometryParameterDefinition, GeometryParameterDerivative, GeometryParameterFrame,
    GeometryParameterType, GeometryRegionCardinality, GeometryRegionDefinition, GeometryRegionRef,
    GeometryRegionRole, GeometryRegionSelector, GeometryRevisionConflict,
    GeometryRevisionConflictKind, GeometryRevisionIdentity, GeometryRevisionMap,
    GeometryRevisionMappingError, GeometryRevisionOperation, GeometryRevisionResolution,
    GeometryScalarExpression, GeometryScalarOperator, GeometrySource, GeometrySourceFormat,
    GeometrySourceIdentity, GeometryTolerancePolicy, GeometryTransform, ImportedGeometryDefinition,
    KernelEvaluatorRef, LengthUnit, MassPropertiesEvaluatorId, MaterialEvidence,
    MaterialEvidenceConfidence, MeshDescriptor, MeshKind, NurbsCurve2, NurbsCurve3, NurbsSurface3,
    OrientedEntityUse, ParameterRange, PcurveDerivatives, PcurveEvaluatorId, PersistentEntityId,
    PersistentEntityIdData, PersistentEntityKind, PortableExactEvaluator, QuantityKind, Region,
    RegionEntityMapping, ResolvedGeometryRegion, ResolvedGeometryRegions, SourceGeometry,
    SourceGeometryKind, SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceMesh,
    SurfaceProjection, TessellationProfile, TopologicalOrientation, TopologyValidity,
    TrimClassifierId, TrimDomainLocation, DISPLAY_TESSELLATION_MEDIA_TYPE, EXACT_BREP_MEDIA_TYPE,
    EXACT_BREP_TOPOLOGY_SCHEMA_VERSION, EXACT_CONTACT_PAIRING_SCHEMA_VERSION,
    EXACT_EVALUATOR_MEDIA_TYPE, EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION,
    EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION, EXACT_TOPOLOGY_MEDIA_TYPE, FACETED_SOLID_MEDIA_TYPE,
    FACETED_SOLID_SCHEMA_VERSION, GEOMETRY_DOCUMENT_SCHEMA_VERSION, GEOMETRY_HEALING_MEDIA_TYPE,
    GEOMETRY_HEALING_REPORT_SCHEMA_VERSION, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    GEOMETRY_REVISION_MAP_SCHEMA_VERSION, KERNEL_REPRESENTATION_MEDIA_TYPE,
};
pub use selection::{EntityKind, EntityRef};

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_test".to_string(),
            source: GeometrySource {
                unit_evidence: Default::default(),
                path: "/models/part.stl".to_string(),
                sha256: "abc123".to_string(),
                importer_version: "stl/v1".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: vec![],
                exchange: None,
                cad_evaluators: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            coordinate_units: None,
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
                name: "body".into(),
                tag: None,
                cad_ownership: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces("region_a", "mesh_1", 1)],
            diagnostics: vec![],
        }
    }

    #[test]
    fn preview_scale_cannot_create_missing_file_unit_evidence() {
        let mut asset = sample_asset();
        assert!(asset.validate().is_ok());
        asset.coordinate_units = Some(physical::CoordinateUnits::new(LengthUnit::Meter));
        assert!(asset.validate().is_err());
        asset.source.unit_evidence = physical::LengthUnitEvidence::explicit_fallback(
            LengthUnit::Meter,
            "fixture source interpretation",
        );
        assert!(asset.validate().is_ok());
    }

    #[test]
    fn legacy_unit_label_assets_require_reimport() {
        let mut wire = serde_json::to_value(sample_asset()).unwrap();
        let object = wire.as_object_mut().unwrap();
        object.remove("coordinate_units");
        object.insert(
            "units".into(),
            serde_json::to_value(LengthUnit::Meter).unwrap(),
        );
        assert!(serde_json::from_value::<GeometryAsset>(wire).is_err());
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
    fn invalid_preview_coordinate_conversion_is_rejected() {
        let mut value = serde_json::to_value(sample_asset()).unwrap();
        value["coordinate_units"] = serde_json::json!({"dimension": {"length":1,"mass":0,"time":0,"current":0,"temperature":0,"amount":0,"luminous_intensity":0,"information":0},"scale":-1.0,"offset":0.0});
        assert!(serde_json::from_value::<GeometryAsset>(value).is_err());
    }
}
