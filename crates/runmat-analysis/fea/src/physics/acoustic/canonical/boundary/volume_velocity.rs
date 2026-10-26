//! Total outward boundary volume flow. This is not an interior monopole.
use super::*;
use crate::excitation::{ComplexScalar, RealizedExcitation};

#[cfg(test)]
mod tests;

impl AcousticAssembly {
    pub(super) fn add_volume_velocity(
        &mut self,
        load: &RealizedExcitation,
        flow: ComplexScalar,
        faces: Vec<ProjectedBoundaryFace>,
    ) -> Result<(), AcousticHarmonicError> {
        let mut source_error = None;
        if load.policy.distribution != ExcitationDistribution::PreserveResultant {
            return Err(AcousticHarmonicError::Unsupported(
                "volume velocity requires total-resultant distribution".into(),
            ));
        }
        let mut identities = BTreeSet::new();
        let mut area = 0.0;
        for face in &faces {
            if !identities.insert(face.face_id)
                || face.role != BoundaryFaceRole::Exterior
                || face.adjacent_element_indices.len() != 1
                || face.adjacent_element_indices[0] >= self.projection.elements.len()
            {
                return Err(AcousticHarmonicError::Boundary(
                    "volume velocity requires unique exterior faces with one physical owner".into(),
                ));
            }
            self.projection
                .visit_boundary_quadrature(face, |point| {
                    area += point.area_weight_m2;
                })
                .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
        }
        if !area.is_finite() || area <= 0.0 {
            return Err(AcousticHarmonicError::Boundary(
                "volume velocity patch area must be finite and positive".into(),
            ));
        }
        let velocity = [flow.real / area, flow.imaginary / area];
        if velocity.iter().any(|value| !value.is_finite())
            || (flow.real != 0.0 && velocity[0] == 0.0)
            || (flow.imaginary != 0.0 && velocity[1] == 0.0)
        {
            return Err(AcousticHarmonicError::Boundary(
                "volume velocity cannot be represented on this patch area".into(),
            ));
        }
        let mut evidence = ScalarProjectionAccumulator::new(
            load,
            &self.projection,
            SpatialIntegrationRule::BoundaryDunavantSeven,
            faces.len(),
        )
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        // The independently authored total is requested ONCE, not reconstructed
        // from quadrature or repeated for each face in the selected patch.
        evidence.add_requested(flow.real, flow.imaginary);
        for face in faces {
            self.projection
                .visit_boundary_quadrature(&face, |point| {
                    for (i, &node) in face.node_indices.iter().enumerate() {
                        // Preserve signed higher-order weights.
                        let weight = point.area_weight_m2 * point.shape_values()[i];
                        let real = velocity[0] * weight;
                        let imaginary = velocity[1] * weight;
                        evidence.add_projected(node, real, imaginary);
                        // exp(+i omega t): b=-i omega integral(N_i v_n).
                        if let Err(error) =
                            crate::physics::acoustic::canonical::quantity_layout::accumulate_source(
                                &mut self.source,
                                node,
                                self.derivative_rate.canonical_value() * imaginary,
                                -(self.derivative_rate.canonical_value() * real),
                            )
                        {
                            source_error = Some(error);
                        }
                    }
                })
                .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
        }
        if let Some(error) = source_error {
            return Err(error);
        }
        let evidence = evidence.finish();
        evidence
            .validate()
            .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
        if !evidence.accepted {
            return Err(AcousticHarmonicError::Boundary(
                "volume velocity projection conservation".into(),
            ));
        }
        self.source_evidence.push(evidence);
        Ok(())
    }
}
