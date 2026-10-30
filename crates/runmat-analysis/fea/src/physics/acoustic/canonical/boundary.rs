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

mod incident;
mod volume_velocity;

#[cfg(test)]
mod radiation_tests;

impl AcousticAssembly {
    pub(super) fn with_boundaries(
        mut self,
        model: &AnalysisModel,
        step: &ProcedureStep,
        excitations: &RealizedExcitations,
        sample: ExcitationSamplePoint<'_>,
    ) -> Result<Self, AcousticHarmonicError> {
        let mut source_error = None;
        let mut impedance = assembler(
            self.projection.nodes.len(),
            OperatorRole::Damping,
            "boundary_admittance",
        );
        // Spherical-radiation curvature is positive semidefinite; a reactive impedance law can
        // make the boundary stiffness indefinite (mass-like walls).
        let mut boundary_stiffness = crate::algebra::RealSparseAssembler::square(
            self.projection.nodes.len(),
            OperatorRole::Stiffness,
            crate::algebra::MatrixProperties::SYMMETRIC_INDEFINITE,
        );
        boundary_stiffness.add_contribution(crate::algebra::ContributionProvenance {
            domain: crate::algebra::ContributionDomain::Acoustic,
            contribution_id: crate::algebra::ContributionId("boundary_reactance".into()),
        });
        let mut owned = BTreeSet::new();
        let mut pressure_faces = BTreeSet::new();
        let mut velocity_faces = BTreeSet::new();
        let mut volume_faces = BTreeSet::new();
        for boundary in &model.boundaries {
            if !step.object_is_active(&StepObjectRef::Boundary(boundary.boundary_id.clone())) {
                continue;
            }
            let admittance = match boundary.operator {
                BoundaryOperator::AcousticRigidWall => 0.0,
                BoundaryOperator::AcousticImpedance { impedance } => {
                    if !impedance.canonical_value().is_finite()
                        || impedance.canonical_value() <= 0.0
                    {
                        return Err(AcousticHarmonicError::Boundary(
                            "specific impedance must be positive finite".into(),
                        ));
                    }
                    impedance.canonical_value().recip()
                }
                // Radiation is face-local: its medium is resolved below from the unique
                // exterior owner, not from the first material in the model.
                BoundaryOperator::AcousticRadiation
                | BoundaryOperator::AcousticSphericalRadiation { .. }
                | BoundaryOperator::AcousticImpedanceLaw { .. } => 0.0,
                _ => {
                    return Err(AcousticHarmonicError::Unsupported(
                        "non-acoustic boundary in acoustic model".into(),
                    ))
                }
            };
            let faces = self.faces(&boundary.target)?;
            for face in faces {
                // A complex law Z = R + iX gives the admittance Y = G + iS = 1/Z in the facing
                // medium; G enters the loss term iωB, and iω(iS) = −ωS a boundary stiffness.
                let mut susceptance = 0.0;
                let admittance = if let BoundaryOperator::AcousticImpedanceLaw { law } =
                    &boundary.operator
                {
                    let [owner] = face.adjacent_element_indices.as_slice() else {
                        return Err(AcousticHarmonicError::Boundary(
                            "impedance law requires exactly one exterior owning element".into(),
                        ));
                    };
                    let rho = self.density.canonical_values()[*owner];
                    let c = (1.0 / (rho * self.compressibility.canonical_values()[*owner])).sqrt();
                    let (r, x) = law
                        .evaluate(
                            self.derivative_rate.canonical_value() / std::f64::consts::TAU,
                            rho,
                            c,
                        )
                        .ok_or_else(|| {
                            AcousticHarmonicError::Boundary(
                                "impedance law has no finite passive value at this frequency"
                                    .into(),
                            )
                        })?;
                    let magnitude = r * r + x * x;
                    susceptance = -x / magnitude;
                    r / magnitude
                } else if matches!(
                    boundary.operator,
                    BoundaryOperator::AcousticRadiation
                        | BoundaryOperator::AcousticSphericalRadiation { .. }
                ) {
                    let admittance = self.radiation_admittance(&face)?;
                    self.port_faces.push(super::assembly::AcousticPortFace {
                        face: face.clone(),
                        admittance,
                        incident: None,
                    });
                    admittance
                } else {
                    admittance
                };
                // Spherical outgoing wave: dp/dn = -(ik + 1/r) p. The 1/r part is a real,
                // frequency-independent boundary stiffness integral(N_i N_j / (rho r)).
                if let BoundaryOperator::AcousticSphericalRadiation { center_m } = boundary.operator
                {
                    let [owner] = face.adjacent_element_indices.as_slice() else {
                        return Err(AcousticHarmonicError::Boundary(
                            "radiation requires exactly one exterior owning element".into(),
                        ));
                    };
                    let rho = self.density.canonical_values()[*owner];
                    if !center_m.iter().all(|value| value.is_finite()) {
                        return Err(AcousticHarmonicError::Boundary(
                            "spherical radiation center must be finite".into(),
                        ));
                    }
                    let mut failure = None;
                    self.projection
                        .visit_boundary_quadrature(&face, |point| {
                            let radius = point
                                .position_m
                                .iter()
                                .zip(center_m)
                                .map(|(x, c)| (x - c).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            if !(radius.is_finite() && radius > 0.0) {
                                failure = Some(());
                                return;
                            }
                            for (i, &row) in face.node_indices.iter().enumerate() {
                                for (j, &column) in face.node_indices.iter().enumerate() {
                                    boundary_stiffness.add(
                                        row,
                                        column,
                                        point.shape_values()[i]
                                            * point.shape_values()[j]
                                            * point.area_weight_m2
                                            / (rho * radius),
                                    );
                                }
                            }
                        })
                        .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
                    if failure.is_some() {
                        return Err(AcousticHarmonicError::Boundary(
                            "spherical radiation surface passes through its center".into(),
                        ));
                    }
                }
                if !owned.insert(face.face_id) {
                    return Err(AcousticHarmonicError::Boundary(
                        "overlapping acoustic boundaries".into(),
                    ));
                }
                let reactive = -self.derivative_rate.canonical_value() * susceptance;
                self.projection
                    .visit_boundary_quadrature(&face, |point| {
                        for (i, &row) in face.node_indices.iter().enumerate() {
                            for (j, &column) in face.node_indices.iter().enumerate() {
                                let mass = point.shape_values()[i]
                                    * point.shape_values()[j]
                                    * point.area_weight_m2;
                                impedance.add(row, column, admittance * mass);
                                if reactive != 0.0 {
                                    boundary_stiffness.add(row, column, reactive * mass);
                                }
                            }
                        }
                    })
                    .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
            }
        }
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
                // Paired relations are compiled against the sealed mesh after source
                // realization, so prescribed values and natural-boundary conflicts are known.
                continue;
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
                RealizedExcitationKind::AcousticIncidentPressure(value) => {
                    let faces = self.faces(target)?;
                    if faces.iter().any(|face| {
                        owned.contains(&face.face_id)
                            || pressure_faces.contains(&face.face_id)
                            || velocity_faces.contains(&face.face_id)
                    }) {
                        return Err(AcousticHarmonicError::Boundary(
                            "incident pressure overlaps another acoustic source or boundary".into(),
                        ));
                    }
                    self.add_incident_pressure(load, value, faces, &mut impedance, &mut owned)?;
                }
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
                RealizedExcitationKind::VolumeVelocity(value) => {
                    let faces = self.faces(target)?;
                    if faces.iter().any(|face| {
                        owned.contains(&face.face_id)
                            || pressure_faces.contains(&face.face_id)
                            || velocity_faces.contains(&face.face_id)
                    }) {
                        return Err(AcousticHarmonicError::Boundary(
                            "volume velocity overlaps another acoustic source or boundary".into(),
                        ));
                    }
                    for face in &faces {
                        velocity_faces.insert(face.face_id);
                        volume_faces.insert(face.face_id);
                    }
                    self.add_volume_velocity(load, value, faces)?;
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
                        if owned.contains(&face.face_id)
                            || pressure_faces.contains(&face.face_id)
                            || volume_faces.contains(&face.face_id)
                        {
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
                                    if let Err(error) = crate::physics::acoustic::canonical::quantity_layout::accumulate_source(&mut self.source, node, self.derivative_rate.canonical_value() * imaginary, -(self.derivative_rate.canonical_value() * real)) { source_error = Some(error); }
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
        let boundary_admittance = finish(impedance)?;
        self.impedance = super::quantity_layout::pressure_operator(
            self.volume_admittance
                .with_canonical_storage(|volume| {
                    crate::algebra::RealSparseMatrix::scaled_sum(
                        &[(1.0, volume), (1.0, &boundary_admittance)],
                        OperatorRole::Damping,
                        crate::algebra::MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
                    )
                })
                .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?,
            1,
        )?;
        let boundary_stiffness = finish(boundary_stiffness)?;
        self.stiffness = super::quantity_layout::pressure_operator(
            self.volume_stiffness
                .with_canonical_storage(|volume| {
                    crate::algebra::RealSparseMatrix::scaled_sum(
                        &[(1.0, volume), (1.0, &boundary_stiffness)],
                        OperatorRole::Stiffness,
                        crate::algebra::MatrixProperties::SYMMETRIC_INDEFINITE,
                    )
                })
                .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?,
            2,
        )?;
        if self
            .source
            .canonical_values()
            .iter()
            .any(|value| !value.real.is_finite() || !value.imaginary.is_finite())
        {
            return Err(AcousticHarmonicError::Boundary(
                "nonfinite integrated source".into(),
            ));
        }
        Ok(self)
    }

    /// First-order local outgoing condition for exp(+i omega t): v_n=p/(rho*c).
    /// beta=1/(rho*c*c) is already authenticated from the owning compiled medium.
    fn radiation_admittance(
        &self,
        face: &ProjectedBoundaryFace,
    ) -> Result<f64, AcousticHarmonicError> {
        if face.role != BoundaryFaceRole::Exterior {
            return Err(AcousticHarmonicError::Boundary(
                "radiation requires an exterior face".into(),
            ));
        }
        let [owner] = face.adjacent_element_indices.as_slice() else {
            return Err(AcousticHarmonicError::Boundary(
                "radiation requires exactly one exterior owning element".into(),
            ));
        };
        let (Some(&rho), Some(&beta)) = (
            self.density.canonical_values().get(*owner),
            self.compressibility.canonical_values().get(*owner),
        ) else {
            return Err(AcousticHarmonicError::Boundary(
                "radiation owning medium is missing".into(),
            ));
        };
        // Separate square roots avoid overflowing rho*c or underflowing beta/rho.
        let admittance = beta.sqrt() / rho.sqrt();
        if !rho.is_finite()
            || rho <= 0.0
            || !beta.is_finite()
            || beta <= 0.0
            || !admittance.is_finite()
            || admittance <= 0.0
        {
            return Err(AcousticHarmonicError::Boundary(
                "radiation owning medium admittance must be positive finite".into(),
            ));
        }
        Ok(admittance)
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
                if self.prescribed.canonical_values()[node].is_some_and(|old| old != value) {
                    return Err(AcousticHarmonicError::Boundary(
                        "conflicting prescribed pressure".into(),
                    ));
                }
                self.prescribed
                    .try_replace(node, runmat_types::physical::Quantity::new(Some(value)))
                    .map_err(|e| AcousticHarmonicError::Boundary(e.to_string()))?;
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
