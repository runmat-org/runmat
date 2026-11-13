//! Failure-only borrowed component profiles; no original numerical values are logged.
use super::StructuralStudyResult;

pub(super) fn describe(result: &StructuralStudyResult, maximum: u64) -> String {
    // These are structural codec projections, not typed heap/RSS measurements.
    // Streaming traversal does not materialize the complete result to diagnose it.
    fn projection(value: &impl serde::Serialize, maximum: u64) -> String {
        crate::analysis::storage::publication::resident(value, maximum)
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|_| "exceeds_profile_limits".into())
    }
    let modal = match result.numerical() {
        crate::analysis::StructuralStudyNumericalResult::StructuralModal(modal) => {
            let basis = modal.result.basis();
            let maximum_residual = basis
                .modes()
                .iter()
                .map(|mode| mode.residual_norm())
                .fold(0.0, f64::max);
            format!(
                "retained_modes={}, maximum_basis_residual={maximum_residual:e}",
                basis.modes().len()
            )
        }
        crate::analysis::StructuralStudyNumericalResult::ThermalPrestress {
            thermal,
            structural,
            ..
        } => format!(
            "thermal_prefix_projection_bytes={}, prestressed_structural_projection_bytes={}",
            projection(thermal.as_ref(), maximum),
            projection(structural.as_ref(), maximum),
        ),
        _ => "modal_basis=not_applicable".into(),
    };
    let mesh_storage = result
        .mesh()
        .retained_storage_bytes(maximum, &mut || {
            if crate::interrupt::is_cancelled() {
                Err(runmat_meshing_core::MeshingContractError::invalid(
                    "publication storage profiling",
                    "cancelled",
                ))
            } else {
                Ok(())
            }
        })
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|_| "exceeds_storage_controls".into());
    format!(
        "typed structural preparation facts: {modal}, nodes={}, elements={}, mesh_retained_storage_bytes={mesh_storage}, compiled_projection_bytes={}, mesh_projection_bytes={}, meshing_evidence_projection_bytes={}, workload_projection_bytes={}, numerical_projection_bytes={}",
        result.mesh().topology.nodes.len(),
        result.mesh().topology.volume_elements.len(),
        projection(result.compiled(), maximum),
        projection(result.mesh(), maximum),
        projection(result.meshing_evidence(), maximum),
        projection(result.workloads(), maximum),
        projection(result.numerical(), maximum),
    )
}
