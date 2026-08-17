mod codec;
mod identity;
mod types;
mod validation;

pub use codec::{decode_shared_curve_mesh, encode_shared_curve_mesh};
pub use identity::shared_curve_node_id;
pub use types::{
    CurveResolutionEvidence, CurveResolutionPolicy, SharedCurve, SharedCurveFaceUse,
    SharedCurveMesh, SharedCurveNode, SHARED_CURVE_MESH_SCHEMA_VERSION,
};
pub use validation::SharedCurveValidationError;

#[cfg(test)]
mod tests;
