mod codec;
mod construction;
mod validation;

use serde::{Deserialize, Serialize};

use super::{
    ExactBRepTopology, ExactEvaluatorRegistry, GeometryDigest, GeometryHealingReport,
    GeometryObjectRef, GeometryRevisionIdentity,
};

pub use codec::{
    decode_exact_evaluators, decode_exact_topology, decode_geometry_healing_report,
    encode_exact_evaluators, encode_exact_topology, encode_geometry_healing_report,
};
pub use construction::build_exact_geometry_closure;
pub use validation::admit_exact_geometry_closure;

pub const EXACT_GEOMETRY_MANIFEST_SCHEMA_VERSION: u16 = 2;
pub const EXACT_TOPOLOGY_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry.exact-topology.v2+cbor";
pub const EXACT_EVALUATOR_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry.exact-evaluators.v2+cbor";
pub const KERNEL_REPRESENTATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry.kernel-representation.v2+brep";
pub const GEOMETRY_HEALING_MEDIA_TYPE: &str =
    "application/vnd.runmat.geometry.healing-report.v2+cbor";

/// Root of an exact-geometry closure. Component bytes remain separate immutable objects so large
/// CAD models use shared artifact transport rather than scheduler frames.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactGeometryManifest {
    pub schema_version: u16,
    pub source_digest: GeometryDigest,
    pub revision: GeometryRevisionIdentity,
    pub kernel_abi: String,
    pub topology: GeometryObjectRef,
    pub evaluators: GeometryObjectRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_representation: Option<GeometryObjectRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub healing_report: Option<GeometryObjectRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedExactGeometry {
    pub manifest: ExactGeometryManifest,
    pub topology: ExactBRepTopology,
    pub evaluators: ExactEvaluatorRegistry,
    pub kernel_representation: Option<Vec<u8>>,
    pub healing_report: Option<GeometryHealingReport>,
}

/// Canonical byte closure produced by geometry before any storage or transport projection.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedExactGeometryClosure {
    pub document: crate::GeometryDocument,
    pub manifest: ExactGeometryManifest,
    pub manifest_bytes: Vec<u8>,
    pub topology_bytes: Vec<u8>,
    pub evaluator_bytes: Vec<u8>,
    pub healing_bytes: Option<Vec<u8>>,
}

#[cfg(test)]
mod tests;
