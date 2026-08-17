mod analysis_identity;
mod assembly;
mod document;
mod exact_artifact;
mod exact_evaluator;
mod exact_topology;
mod exact_topology_assembly_validation;
mod exact_topology_validation;
mod exact_topology_validation_support;
mod field;
mod geometry;
mod healing;
mod material_evidence;
mod mesh;
mod regions;
mod revision_mapping;
mod source_geometry;
mod tessellation_profile;
mod topology;
mod units;

pub use analysis_identity::{
    GeometryContractError, GeometryTolerancePolicy, PersistentEntityId, PersistentEntityKind,
};
pub use assembly::AssemblyNode;
pub use document::*;
pub use exact_artifact::*;
pub use exact_evaluator::*;
pub use exact_topology::*;
pub use field::{FieldLocation, FieldValueKind};
pub use geometry::{GeometryAsset, GeometrySource};
pub use healing::*;
pub use material_evidence::{MaterialEvidence, MaterialEvidenceConfidence};
pub use mesh::{MeshDescriptor, MeshKind, SurfaceMesh};
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
pub use units::UnitSystem;

#[cfg(test)]
mod exact_topology_tests;
