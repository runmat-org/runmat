//! Canonical element-density validation and conservative volume-source integration.
use super::*;
use crate::excitation::RealizedExcitation;

pub(super) fn validate_fields(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    projection: &AnalysisProjection,
    excitations: &RealizedExcitations,
) -> Result<(), ThermalAssemblyError> {
    // Authenticate the complete canonical field before allocating/integrating operators.
    // Region selection never changes canonical row numbering or permits a subset tensor.
    for excitation in excitations.iter() {
        if let RealizedExcitationKind::HeatGenerationField(ref field) = excitation.kind {
            field
                .validate_mesh(
                    runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
                    projection.elements.len(),
                )
                .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?;
            if excitation.policy.frame != runmat_analysis_core::CoordinateFrame::Global
                || excitation.policy.distribution
                    != runmat_analysis_core::ExcitationDistribution::TopologyField
                || excitation.policy.kinematics
                    != runmat_analysis_core::ExcitationKinematics::FixedFrame
            {
                return Err(ThermalAssemblyError::InvalidProjection(
                    "thermal element density requires global fixed-frame topology-field distribution".into(),
                ));
            }
            if !field.is_real() {
                return Err(ThermalAssemblyError::ComplexSource {
                    excitation_id: excitation.excitation_id.0.clone(),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn integrate(
    projection: &AnalysisProjection,
    excitation: &RealizedExcitation,
    real: f64,
    applied_source: &mut [f64],
    evidence: &mut ScalarProjectionAccumulator,
) -> Result<(), ThermalAssemblyError> {
    for (canonical_row, element) in projection
        .elements
        .iter()
        .enumerate()
        .filter(|(_, element)| volume_matches(element, &excitation.target))
    {
        let density = match &excitation.kind {
            RealizedExcitationKind::HeatGenerationField(field) => {
                field
                    .value_at(canonical_row)
                    .map_err(|error| ThermalAssemblyError::InvalidProjection(error.to_string()))?
                    .real
            }
            _ => real,
        };
        projection
            .visit_element_quadrature(element, |point| {
                let integrated = density * point.volume_weight_m3;
                evidence.add_requested(integrated, 0.0);
                for (&node, &shape) in element.node_indices.iter().zip(point.shape_values()) {
                    let nodal = integrated * shape;
                    applied_source[node] += nodal;
                    evidence.add_projected(node, nodal, 0.0);
                }
            })
            .map_err(ThermalAssemblyError::from)?;
    }
    Ok(())
}
