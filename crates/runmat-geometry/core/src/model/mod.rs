mod analysis_identity;
mod assembly;
mod authoring;
mod authoring_feature_validation;
#[cfg(test)]
mod authoring_tests;
mod authoring_value;
mod canonical;
pub use canonical::GeometryCanonicalControl;
mod document;
mod exact_artifact;
mod exact_contacts;
mod exact_evaluator;
mod exact_topology;
mod exact_topology_assembly_validation;
mod exact_topology_solid_validation;
mod exact_topology_validation;
mod exact_topology_validation_support;
mod exact_transform;
mod exchange;
mod faceted_artifact;
mod field;
mod geometry;
mod healing;
mod material_evidence;
mod mesh;
mod occurrence_transforms;
mod regions;
mod revision_mapping;
mod source_geometry;
mod tessellation_profile;
mod topology;
mod units;
mod validation_storage;

pub use analysis_identity::{
    GeometryContractError, GeometryTolerancePolicy, PersistentEntityId, PersistentEntityIdData,
    PersistentEntityKind,
};
pub use assembly::AssemblyNode;
pub use authoring::*;
pub use authoring_value::*;
pub use document::*;
pub use exact_artifact::*;
pub use exact_contacts::*;
pub use exact_evaluator::*;
pub use exact_topology::*;
pub use exchange::*;
pub use faceted_artifact::*;
pub use field::{FieldLocation, FieldValueKind};
pub use geometry::{GeometryAsset, GeometrySource};
pub use healing::*;
pub use material_evidence::{MaterialEvidence, MaterialEvidenceConfidence};
pub use mesh::{MeshDescriptor, MeshKind, SurfaceMesh};
pub use occurrence_transforms::PreparedOccurrenceTransforms;
pub use regions::{
    CadColorEvidence, CadLabelRef, CadPhysicalMaterialEvidence, CadRegionOwnership,
    CadSemanticKind, EntityIdRange, Region, RegionEntityMapping,
};
pub use revision_mapping::*;
pub use source_geometry::{
    CadCurveEvaluationSample, CadCurveEvaluationSampleSource, CadCurveEvaluator, CadEvaluatorSet,
    CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator, SourceGeometry,
    SourceGeometryKind,
};
pub use tessellation_profile::{HealingMode, TessellationProfile};
pub use topology::ElementKind;
pub use units::LengthUnit;

#[cfg(test)]
mod exact_topology_tests;
