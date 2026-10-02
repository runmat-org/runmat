//! Uniform normally incoming plane-wave pressure through the shared excitation contract.
use super::*;
use crate::{
    algebra::RealSparseAssembler,
    excitation::{ComplexScalar, RealizedExcitation},
};

#[cfg(test)]
mod tests;

impl AcousticAssembly {
    pub(super) fn add_incident_pressure(
        &mut self,
        load: &RealizedExcitation,
        pressure: ComplexScalar,
        faces: Vec<ProjectedBoundaryFace>,
        impedance: &mut RealSparseAssembler,
        owned: &mut BTreeSet<u64>,
    ) -> Result<(), AcousticHarmonicError> {
        if load.policy.distribution != ExcitationDistribution::UniformKinematic {
            return Err(AcousticHarmonicError::Unsupported(
                "nonuniform incident pressure".into(),
            ));
        }
        self.validate_incident_plane(&faces)?;
        self.incident_faces
            .try_reserve_exact(faces.len())
            .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        let mut evidence = ScalarProjectionAccumulator::new(
            load,
            &self.projection,
            SpatialIntegrationRule::BoundaryDunavantSeven,
            faces.len(),
        )
        .map_err(|error| AcousticHarmonicError::Operator(error.to_string()))?;
        for face in faces {
            let admittance = self.radiation_admittance(&face)?;
            owned.insert(face.face_id);
            self.projection
                .visit_boundary_quadrature(&face, |point| {
                    let real = pressure.real * point.area_weight_m2;
                    let imaginary = pressure.imaginary * point.area_weight_m2;
                    evidence.add_requested(real, imaginary);
                    for (i, &row) in face.node_indices.iter().enumerate() {
                        let shape = point.shape_values()[i];
                        evidence.add_projected(row, shape * real, shape * imaginary);
                        // exp(+i omega t): v_inc=-Y p_inc; b=i omega(Y p_inc-v_inc).
                        self.source[row].real -= 2.0 * self.omega * admittance * shape * imaginary;
                        self.source[row].imaginary += 2.0 * self.omega * admittance * shape * real;
                        for (j, &column) in face.node_indices.iter().enumerate() {
                            impedance.add(
                                row,
                                column,
                                admittance * shape * point.shape_values()[j] * point.area_weight_m2,
                            );
                        }
                    }
                })
                .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
            self.incident_faces.push((face, admittance));
        }
        let evidence = evidence.finish();
        evidence
            .validate()
            .map_err(|error| AcousticHarmonicError::Boundary(error.to_string()))?;
        if !evidence.accepted {
            return Err(AcousticHarmonicError::Boundary(
                "incident pressure projection conservation".into(),
            ));
        }
        self.source_evidence.push(evidence);
        Ok(())
    }

    fn validate_incident_plane(
        &self,
        faces: &[ProjectedBoundaryFace],
    ) -> Result<(), AcousticHarmonicError> {
        let first = faces
            .first()
            .ok_or_else(|| AcousticHarmonicError::Boundary("empty incident plane".into()))?;
        let norm = |value: [f64; 3]| value.into_iter().map(|x| x * x).sum::<f64>().sqrt();
        let normal = first
            .oriented_area_vector_m2
            .map(|x| x / norm(first.oriented_area_vector_m2));
        let origin = self.projection.nodes[first.node_indices[0]].coordinates_m;
        let admittance = self.radiation_admittance(first)?;
        let first_owner = first.adjacent_element_indices[0];
        let density = self.density[first_owner];
        let compressibility = self.compressibility[first_owner];
        for face in faces {
            let y = self.radiation_admittance(face)?;
            let owner = face.adjacent_element_indices[0];
            let direction = face
                .oriented_area_vector_m2
                .map(|x| x / norm(face.oriented_area_vector_m2));
            let scale = face
                .node_indices
                .iter()
                .map(|&node| {
                    norm(std::array::from_fn(|axis| {
                        self.projection.nodes[node].coordinates_m[axis] - origin[axis]
                    }))
                })
                .fold(0.0_f64, f64::max);
            let mut planar = scale > 0.0
                && (y - admittance).abs() <= 1e-12 * admittance
                && (self.density[owner] - density).abs() <= 1e-12 * density
                && (self.compressibility[owner] - compressibility).abs() <= 1e-12 * compressibility
                && norm(std::array::from_fn(|axis| direction[axis] - normal[axis])) <= 1e-10;
            self.projection
                .visit_boundary_quadrature(face, |point| {
                    let signed_distance = (0..3)
                        .map(|axis| normal[axis] * (point.position_m[axis] - origin[axis]))
                        .sum::<f64>();
                    let point_normal = point
                        .area_vector_weight_m2
                        .map(|x| x / point.area_weight_m2);
                    planar &= signed_distance.abs() <= 1e-10 * scale
                        && norm(std::array::from_fn(|axis| {
                            point_normal[axis] - normal[axis]
                        })) <= 1e-10;
                })
                .map_err(|error| AcousticHarmonicError::Topology(error.to_string()))?;
            if !planar {
                return Err(AcousticHarmonicError::Unsupported(
                    "incident pressure requires coplanar equally oriented exterior faces in one homogeneous medium".into(),
                ));
            }
        }
        Ok(())
    }
}
