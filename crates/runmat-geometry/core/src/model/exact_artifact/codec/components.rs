//! Fused geometry-owned component reads validate the same immutable topology once.
//! Returned data remains ordinary mutable data, never serialized validation authority.
use super::*;
use crate::{model::validation_storage::ValidationWork, GeometryCanonicalControl};
pub struct ExactGeometryComponentBytes<'a> {
    pub topology: &'a [u8],
    pub evaluators: &'a [u8],
    pub healing: Option<&'a [u8]>,
}
pub struct DecodedExactGeometryComponents {
    pub topology: ExactBRepTopology,
    pub evaluators: ExactEvaluatorRegistry,
    pub healing_report: Option<GeometryHealingReport>,
}
pub fn decode_exact_geometry_components(
    bytes: ExactGeometryComponentBytes<'_>,
    model: &ExactBRepModel,
) -> Result<DecodedExactGeometryComponents, GeometryContractError> {
    decode(bytes, model, None)
}
pub fn decode_exact_geometry_components_with_control(
    bytes: ExactGeometryComponentBytes<'_>,
    model: &ExactBRepModel,
    control: &dyn GeometryCanonicalControl,
) -> Result<DecodedExactGeometryComponents, GeometryContractError> {
    decode(bytes, model, Some(control))
}
fn decode(
    bytes: ExactGeometryComponentBytes<'_>,
    model: &ExactBRepModel,
    control: Option<&dyn GeometryCanonicalControl>,
) -> Result<DecodedExactGeometryComponents, GeometryContractError> {
    let topology = match control {
        Some(control) => super::decode_exact_topology_with_control(bytes.topology, model, control)?,
        None => super::decode_exact_topology(bytes.topology, model)?,
    };
    let evaluators: ExactEvaluatorRegistry = match control {
        Some(control) => crate::model::canonical::decode_with_control(
            EVALUATORS_DOMAIN,
            bytes.evaluators,
            COMPONENT_LIMITS,
            control,
        )?,
        None => {
            crate::model::canonical::decode(EVALUATORS_DOMAIN, bytes.evaluators, COMPONENT_LIMITS)?
        }
    };
    evaluators.validate_for_validated_topology(
        &topology,
        model,
        ValidationWork(control.map(|control| control as &dyn crate::GeometryEvaluationControl)),
    )?;
    let healing_report = bytes
        .healing
        .map(|bytes| match control {
            Some(control) => super::decode_geometry_healing_report_with_control(bytes, control),
            None => super::decode_geometry_healing_report(bytes),
        })
        .transpose()?;
    Ok(DecodedExactGeometryComponents {
        topology,
        evaluators,
        healing_report,
    })
}
pub(crate) fn encode_evaluators_after_topology_validation(
    evaluators: &ExactEvaluatorRegistry,
    topology: &ExactBRepTopology,
    model: &ExactBRepModel,
) -> Result<Vec<u8>, GeometryContractError> {
    evaluators.validate_for_validated_topology(topology, model, ValidationWork(None))?;
    crate::model::canonical::encode(EVALUATORS_DOMAIN, evaluators, COMPONENT_LIMITS)
}
