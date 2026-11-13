//! Governed cold reads share raw domains and unchanged semantic validators.
use super::*;
use crate::{model::canonical::decode_with_control, GeometryCanonicalControl};
impl ExactGeometryManifest {
    pub fn canonical_decode_with_control(
        bytes: &[u8],
        control: &dyn GeometryCanonicalControl,
    ) -> Result<Self, GeometryContractError> {
        let value: Self = decode_with_control(MANIFEST_DOMAIN, bytes, MANIFEST_LIMITS, control)?;
        value.validate()?;
        Ok(value)
    }
}
pub fn decode_exact_topology_with_control(
    bytes: &[u8],
    model: &ExactBRepModel,
    control: &dyn GeometryCanonicalControl,
) -> Result<ExactBRepTopology, GeometryContractError> {
    let value: ExactBRepTopology =
        decode_with_control(TOPOLOGY_DOMAIN, bytes, COMPONENT_LIMITS, control)?;
    value.validate_against_with_control(model, control)?;
    Ok(value)
}
pub fn decode_exact_evaluators_with_control(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
    control: &dyn GeometryCanonicalControl,
) -> Result<ExactEvaluatorRegistry, GeometryContractError> {
    let value: ExactEvaluatorRegistry =
        decode_with_control(EVALUATORS_DOMAIN, bytes, COMPONENT_LIMITS, control)?;
    value.validate_against_with_control(topology, model, control)?;
    Ok(value)
}
pub fn decode_geometry_healing_report_with_control(
    bytes: &[u8],
    control: &dyn GeometryCanonicalControl,
) -> Result<GeometryHealingReport, GeometryContractError> {
    let value: GeometryHealingReport =
        decode_with_control(HEALING_DOMAIN, bytes, COMPONENT_LIMITS, control)?;
    value.validate_with_control(control)?;
    Ok(value)
}
