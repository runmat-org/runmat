use super::{
    assembly::{assembler, finish, AcousticAssembly},
    AcousticHarmonicError,
};
use crate::{
    algebra::OperatorRole,
    excitation::{
        sample_prescribed_pressure, ExcitationSamplePoint, ProjectionScalar,
        RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations,
        ScalarProjectionAccumulator, SpatialIntegrationRule,
    },
    topology::ProjectedBoundaryFace,
};
use runmat_analysis_core::{
    AcousticConstraintKind, AnalysisConstraint, AnalysisModel, BoundaryOperator,
    ExcitationDistribution, ExcitationKinematics, ProcedureStep, RegionTarget, StepObjectRef,
};
use runmat_meshing_core::BoundaryFaceRole;
use std::collections::BTreeSet;

impl AcousticAssembly {
    pub(super) fn with_boundaries(
        mut self,
        model: &AnalysisModel,
        step: &ProcedureStep,
        excitations: &RealizedExcitations,
        sample: ExcitationSamplePoint<'_>,
    ) -> Result<Self, AcousticHarmonicError> {
        let mut impedance = assembler(
            self.projection.nodes.len(),
            OperatorRole::Damping,
            "boundary_admittance",
        );
        let mut owned = BTreeSet::new();
        let mut pressure_faces = BTreeSet::new();
        let mut velocity_faces = BTreeSet::new();
        for boundary in &model.boundaries {
            if !step.object_is_active(&StepObjectRef::Boundary(boundary.boundary_id.clone())) {
                continue;
            }
            let admittance = match boundary.operator {
                BoundaryOperator::AcousticRigidWall => 0.0,
                BoundaryOperator::AcousticImpedance { impedance } => {
                    if !impedance.pascal_seconds_per_meter.is_finite()
                        || impedance.pascal_seconds_per_meter <= 0.0
                    {
                        return Err(AcousticHarmonicError::Boundary(
                            "specific impedance must be positive finite".into(),
                        ));
                    }
                    impedance.pascal_seconds_per_meter.recip()
                }
                BoundaryOperator::AcousticRadiation => {
                    return Err(AcousticHarmonicError::Unsupported(
                        "radiation boundary".into(),
                    ))
                }
                _ => {
                    return Err(AcousticHarmonicError::Unsupported(
                        "non-acoustic boundary in acoustic model".into(),
                    ))
                }
            };
            let faces = self.faces(&boundary.target)?;
            for face in faces {
                if !owned.insert(face.face_id) {
                    return Err(AcousticHarmonicError::Boundary(
                        "overlapping acoustic boundaries".into(),
                    ));
                }
                self.projection
                    .visit_boundary_quadrature(&face, |point| {
                        for (i, &row) in face.node_indices.iter().enumerate() {
                            for (j, &column) in face.node_indices.iter().enumerate() {
                                impedance.add(
                                    row,
                                    column,
                                    admittance
                                        * point.shape_values()[i]
                                        * point.shape_values()[j]
                                        * point.area_weight_m2,
                                );
                            }
                        }
                    })
                    .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
            }
        }
        self.impedance = finish(impedance)?;
        for constraint in &model.constraints {
            if !step.object_is_active(&StepObjectRef::Constraint(
                constraint.constraint_id().to_string(),
            )) {
                continue;
            }
            let AnalysisConstraint::Acoustic(constraint) = constraint else {
                return Err(AcousticHarmonicError::Unsupported(
                    "non-acoustic constraint in acoustic model".into(),
                ));
            };
            let AcousticConstraintKind::PrescribedPressure { target, value } = &constraint.kind
            else {
                return Err(AcousticHarmonicError::Unsupported(
                    "periodic or Bloch pressure constraints".into(),
                ));
            };
            let value = sample_prescribed_pressure(value, &model.signals, sample)
                .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))?;
            for face in self.faces(target)? {
                if owned.contains(&face.face_id) {
                    return Err(AcousticHarmonicError::Boundary(
                        "pressure overlaps rigid or impedance boundary".into(),
                    ));
                }
                pressure_faces.insert(face.face_id);
            }
            self.prescribe(
                target,
                ProjectionScalar {
                    real: value.real,
                    imaginary: value.imaginary,
                },
            )?;
        }
        for load in excitations.iter() {
            if load.policy.kinematics != ExcitationKinematics::FixedFrame {
                return Err(AcousticHarmonicError::Unsupported(
                    "follower acoustic source".into(),
                ));
            }
            let RealizedExcitationTarget::Region(target) = &load.target else {
                return Err(AcousticHarmonicError::Unsupported(
                    "non-region acoustic source".into(),
                ));
            };
            match load.kind {
                RealizedExcitationKind::AcousticPressure(value) => {
                    if load.policy.distribution != ExcitationDistribution::UniformKinematic {
                        return Err(AcousticHarmonicError::Unsupported(
                            "nonuniform pressure distribution".into(),
                        ));
                    }
                    for face in self.faces(target)? {
                        if owned.contains(&face.face_id) || velocity_faces.contains(&face.face_id) {
                            return Err(AcousticHarmonicError::Boundary(
                                "pressure overlaps rigid or impedance boundary".into(),
                            ));
                        }
                        pressure_faces.insert(face.face_id);
                    }
                    self.prescribe(
                        target,
                        ProjectionScalar {
                            real: value.real,
                            imaginary: value.imaginary,
                        },
                    )?;
                }
                RealizedExcitationKind::NormalVelocity(value) => {
                    if load.policy.distribution != ExcitationDistribution::UniformKinematic {
                        return Err(AcousticHarmonicError::Unsupported(
                            "nonuniform normal velocity distribution".into(),
                        ));
                    }
                    let faces = self.faces(target)?;
                    let mut evidence = ScalarProjectionAccumulator::new(
                        load,
                        &self.projection,
                        SpatialIntegrationRule::BoundaryDunavantSeven,
                        faces.len(),
                    )
                    .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
                    for face in faces {
                        if owned.contains(&face.face_id) || pressure_faces.contains(&face.face_id) {
                            return Err(AcousticHarmonicError::Boundary(
                                "normal velocity overlaps pressure, rigid or impedance boundary"
                                    .into(),
                            ));
                        }
                        velocity_faces.insert(face.face_id);
                        self.projection
                            .visit_boundary_quadrature(&face, |point| {
                                let real = value.real * point.area_weight_m2;
                                let imaginary = value.imaginary * point.area_weight_m2;
                                evidence.add_requested(real, imaginary);
                                for (i, &node) in face.node_indices.iter().enumerate() {
                                    let real = real * point.shape_values()[i];
                                    let imaginary = imaginary * point.shape_values()[i];
                                    evidence.add_projected(node, real, imaginary);
                                    self.source[node].real += self.omega * imaginary;
                                    self.source[node].imaginary -= self.omega * real;
                                }
                            })
                            .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
                    }
                    let evidence = evidence.finish();
                    evidence
                        .validate()
                        .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))?;
                    if !evidence.accepted {
                        return Err(AcousticHarmonicError::Boundary(
                            "normal velocity projection conservation".into(),
                        ));
                    }
                    self.source_evidence.push(evidence);
                }
                _ => {
                    return Err(AcousticHarmonicError::Unsupported(
                        "non-acoustic excitation".into(),
                    ))
                }
            }
        }
        if self
            .source
            .iter()
            .any(|value| !value.real.is_finite() || !value.imaginary.is_finite())
        {
            return Err(AcousticHarmonicError::Boundary(
                "nonfinite integrated source".into(),
            ));
        }
        Ok(self)
    }

    fn prescribe(
        &mut self,
        target: &RegionTarget,
        value: ProjectionScalar,
    ) -> Result<(), AcousticHarmonicError> {
        if !value.real.is_finite() || !value.imaginary.is_finite() {
            return Err(AcousticHarmonicError::Boundary(
                "nonfinite prescribed pressure".into(),
            ));
        }
        for face in self.faces(target)? {
            for node in face.node_indices {
                if self.prescribed[node].is_some_and(|old| old != value) {
                    return Err(AcousticHarmonicError::Boundary(
                        "conflicting prescribed pressure".into(),
                    ));
                }
                self.prescribed[node] = Some(value);
            }
        }
        Ok(())
    }

    fn faces(
        &self,
        target: &RegionTarget,
    ) -> Result<Vec<ProjectedBoundaryFace>, AcousticHarmonicError> {
        let faces = self
            .projection
            .boundary_faces_for_target(target)
            .cloned()
            .collect::<Vec<_>>();
        if faces.is_empty()
            || faces
                .iter()
                .any(|face| face.role != BoundaryFaceRole::Exterior)
        {
            return Err(AcousticHarmonicError::Boundary(format!(
                "{} must resolve exterior faces",
                target.region_id
            )));
        }
        Ok(faces)
    }
}
