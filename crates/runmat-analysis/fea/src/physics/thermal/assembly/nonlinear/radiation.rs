use super::*;
use crate::{
    physics::thermal::radiation::{integrate_face, RadiationFaceForm},
    topology::ProjectedBoundaryFace,
};

impl ThermalOperatorAssembly {
    /// Re-sample the original ambient on the exact authored signal approach.
    pub(in crate::physics::thermal::assembly) fn visit_radiation(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference: &[f64],
        departure: &[f64],
        visit: impl FnMut(&ProjectedBoundaryFace, &RadiationFaceForm),
    ) -> Result<(), ThermalAssemblyError> {
        self.validate_nonlinear_state(reference, departure, None, 0.0)?;
        self.visit_radiation_reference(model, sample, |node| reference[node], departure, visit)
    }

    pub(in crate::physics::thermal::assembly) fn visit_radiation_scalar_reference(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference: f64,
        departure: &[f64],
        visit: impl FnMut(&ProjectedBoundaryFace, &RadiationFaceForm),
    ) -> Result<(), ThermalAssemblyError> {
        if !reference.is_finite()
            || departure.len() != self.projection.nodes.len()
            || departure
                .iter()
                .any(|d| !d.is_finite() || !(reference + d).is_finite() || reference + d < 0.0)
        {
            return Err(ThermalAssemblyError::Radiation(
                crate::physics::thermal::radiation::RadiationError::InvalidState,
            ));
        }
        self.visit_radiation_reference(model, sample, |_| reference, departure, visit)
    }

    fn visit_radiation_reference(
        &self,
        model: &AnalysisModel,
        sample: ExcitationSamplePoint<'_>,
        reference: impl Fn(usize) -> f64,
        departure: &[f64],
        mut visit: impl FnMut(&ProjectedBoundaryFace, &RadiationFaceForm),
    ) -> Result<(), ThermalAssemblyError> {
        for boundary in &model.boundaries {
            let BoundaryOperator::ThermalRadiation {
                ambient_temperature,
                emissivity,
            } = &boundary.operator
            else {
                continue;
            };
            let ambient = sample_ambient_temperature(
                model,
                &boundary.boundary_id,
                ambient_temperature,
                sample,
            )?;
            let mut faces = self
                .projection
                .boundary_faces_for_target(&boundary.target)
                .peekable();
            if faces.peek().is_none() {
                return Err(ThermalAssemblyError::UnmappedBoundary {
                    operator_id: boundary.boundary_id.clone(),
                });
            }
            for face in faces {
                let mut temperatures = [0.0; 6];
                for (i, &node) in face.node_indices.iter().enumerate() {
                    temperatures[i] = reference(node) + departure[node];
                }
                let form = integrate_face(
                    &self.projection,
                    face,
                    &temperatures[..face.node_indices.len()],
                    ambient,
                    *emissivity,
                )
                .map_err(ThermalAssemblyError::Radiation)?;
                visit(face, &form);
            }
        }
        Ok(())
    }
}
