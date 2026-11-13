//! Shared bindings for already semantically validated typed components. Fresh
//! construction uses these checks without decoding/copying its own new buffers;
//! raw admission independently verifies and decodes bytes before reaching here.
use super::validation::{invalid, topology_contains, verify_object};
use super::*;
use crate::{
    ExactBRepTopology, ExactEvaluatorRegistry, GeometryContractError, GeometryDocument,
    GeometryHealingReport,
};
/// Check cross-component identities after each component has been independently
/// decoded and semantically validated. This does not validate component contents
/// and does not create construction or import authority by itself.
pub fn validate_exact_geometry_component_bindings(
    document: &GeometryDocument,
    manifest: &ExactGeometryManifest,
    topology: &ExactBRepTopology,
    evaluators: &ExactEvaluatorRegistry,
    representation: Option<&[u8]>,
    healing: Option<&GeometryHealingReport>,
) -> Result<(), GeometryContractError> {
    manifest.validate_against_document(document)?;
    match (
        &manifest.kernel_representation,
        representation,
        evaluators.kernel_representation_digest()?,
    ) {
        (Some(reference), Some(bytes), Some(required)) => {
            verify_object(reference, bytes)?;
            if reference.digest.bytes() != &required {
                return Err(invalid(
                    "kernel representation binding",
                    "every kernel evaluator must bind the inventoried representation object",
                ));
            }
        }
        (None, None, None) => {}
        (Some(_), Some(_), None) => {
            return Err(invalid(
                "kernel representation component",
                "portable evaluator closures cannot carry an unreferenced kernel representation",
            ))
        }
        _ => {
            return Err(invalid(
                "kernel representation component",
                "manifest reference, supplied bytes, and kernel evaluator references must agree",
            ))
        }
    }
    match (&manifest.healing_report, healing) {
        (Some(_), Some(report)) => {
            if report.healed_topology_digest != manifest.topology.digest
                || report.revision_map.target_revision != manifest.revision
            {
                return Err(invalid(
                    "exact geometry healing binding",
                    "healing must produce this topology and target revision",
                ));
            }
            for target in report
                .revision_map
                .operations
                .iter()
                .flat_map(|operation| operation.targets())
            {
                if !topology_contains(topology, target) {
                    return Err(invalid(
                        "exact geometry healed topology mapping",
                        "every mapped target must exist in the admitted healed topology",
                    ));
                }
            }
        }
        (None, None) => {}
        _ => {
            return Err(invalid(
                "exact geometry healing component",
                "manifest reference and supplied healing bytes must agree",
            ))
        }
    }
    Ok(())
}
