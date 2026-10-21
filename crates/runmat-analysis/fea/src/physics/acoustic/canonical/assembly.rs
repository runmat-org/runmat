use super::AcousticHarmonicError;
use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, no_visit, ElementAssemblyError, ElementOutputShape,
            ElementOutputs,
        },
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix,
    },
    excitation::{ProjectionScalar, SpatialProjectionEvidence},
    properties::CompiledPropertyMap,
    topology::AnalysisProjection,
};

pub(super) struct AcousticAssembly {
    pub projection: AnalysisProjection,
    /// Volume stiffness plus the reactive boundary stiffness of spherical radiation.
    pub stiffness: RealSparseMatrix,
    /// Volume stiffness only; boundary resampling rebuilds `stiffness` from it.
    pub volume_stiffness: RealSparseMatrix,
    pub mass: RealSparseMatrix,
    pub impedance: RealSparseMatrix,
    pub omega: f64,
    pub density: Vec<f64>,
    pub compressibility: Vec<f64>,
    pub source: Vec<ProjectionScalar>,
    pub prescribed: Vec<Option<ProjectionScalar>>,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    pub incident_faces: Vec<(crate::topology::ProjectedBoundaryFace, f64)>,
    pub reduction: Option<crate::constraints::AffineConstraintReduction>,
}

impl AcousticAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        omega: f64,
    ) -> Result<Self, AcousticHarmonicError> {
        let size = projection.nodes.len();
        let mut density = Vec::with_capacity(projection.elements.len());
        let mut compressibility = Vec::with_capacity(projection.elements.len());
        let mut sound_speed = Vec::with_capacity(projection.elements.len());
        for element in &projection.elements {
            let material = &properties
                .solid_material(element.stable_identity)
                .ok_or_else(|| AcousticHarmonicError::Material("unassigned fluid element".into()))?
                .material;
            let acoustic = material
                .require_acoustic()
                .map_err(|e| AcousticHarmonicError::Material(e.to_string()))?;
            let rho = material
                .require_mass_density()
                .map_err(|e| AcousticHarmonicError::Material(e.to_string()))?;
            let c = acoustic.speed_of_sound_m_per_s;
            if !rho.is_finite() || rho <= 0.0 || !c.is_finite() || c <= 0.0 {
                return Err(AcousticHarmonicError::Material(
                    "density and sound speed must be positive finite".into(),
                ));
            }
            if acoustic.damping_ratio != 0.0 {
                return Err(AcousticHarmonicError::Unsupported(
                    "material attenuation law".into(),
                ));
            }
            density.push(rho);
            let beta = 1.0 / rho / c / c;
            if !beta.is_finite() || beta <= 0.0 {
                return Err(AcousticHarmonicError::Material(
                    "fluid compressibility is not representable as a positive finite value".into(),
                ));
            }
            compressibility.push(beta);
            sound_speed.push(c);
        }
        // Element kernels run on the admitted CPU pool; scatter order is canonical.
        let map = crate::topology::projection_element_map(&projection)
            .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        let plan = crate::planning::resource::LocalExecutionPlan::serial();
        let pattern = map
            .symbolic_pattern(plan)
            .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
        let assembled = assemble_elements(
            &map,
            &pattern,
            ElementOutputShape {
                matrices: 2,
                ..Default::default()
            },
            plan,
            |index, _, out: &mut ElementOutputs<'_, f64>| {
                let element = &projection.elements[index];
                let (rho, c) = (density[index], sound_speed[index]);
                let n = element.node_indices.len();
                let mut local_stiffness = vec![0.0; n * n];
                let mut local_mass = vec![0.0; n * n];
                projection
                    .visit_element_quadrature(element, |point| {
                        let gradients = point.physical_shape_gradients();
                        let shapes = point.shape_values();
                        for i in 0..n {
                            for j in 0..n {
                                let gradient = gradients[i]
                                    .iter()
                                    .zip(gradients[j])
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>();
                                local_stiffness[i * n + j] +=
                                    gradient * point.volume_weight_m3 / rho;
                                local_mass[i * n + j] +=
                                    shapes[i] * shapes[j] * point.volume_weight_m3 / rho / c / c;
                            }
                        }
                    })
                    .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
                out.matrix(0).copy_from_slice(&local_stiffness);
                out.matrix(1).copy_from_slice(&local_mass);
                Ok(())
            },
            no_visit,
        )
        .map_err(|error| match error {
            ElementAssemblyError::Element { error, .. } => error,
            other => AcousticHarmonicError::Operator(other.to_string()),
        })?;
        let mut matrices = assembled.matrices.into_iter();
        let stiffness = planned(
            &pattern,
            matrices.next().unwrap_or_default(),
            OperatorRole::Stiffness,
            "pressure_gradient",
        )?;
        let mass = planned(
            &pattern,
            matrices.next().unwrap_or_default(),
            OperatorRole::Mass,
            "pressure_compressibility",
        )?;
        let stiffness = finish(stiffness)?;
        Ok(Self {
            projection,
            volume_stiffness: stiffness.clone(),
            stiffness,
            mass: finish(mass)?,
            impedance: finish(assembler(
                size,
                OperatorRole::Damping,
                "boundary_admittance",
            ))?,
            omega,
            density,
            compressibility,
            source: vec![
                ProjectionScalar {
                    real: 0.0,
                    imaginary: 0.0
                };
                size
            ],
            prescribed: vec![None; size],
            source_evidence: Vec::new(),
            incident_faces: Vec::new(),
            reduction: None,
        })
    }

    /// Reset sources and prescribed values before resampling boundaries at another time.
    /// Operators and projection are retained; the boundary admittance is rebuilt.
    pub(super) fn with_cleared_sources(mut self) -> Self {
        self.source.iter_mut().for_each(|value| {
            *value = ProjectionScalar {
                real: 0.0,
                imaginary: 0.0,
            }
        });
        self.prescribed.iter_mut().for_each(|value| *value = None);
        self.source_evidence.clear();
        self.incident_faces.clear();
        self
    }

    pub(super) fn apply(&self, real: &[f64], imaginary: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let kr = spmv(&self.stiffness, real).expect("validated pressure shape");
        let ki = spmv(&self.stiffness, imaginary).expect("validated pressure shape");
        let mr = spmv(&self.mass, real).expect("validated pressure shape");
        let mi = spmv(&self.mass, imaginary).expect("validated pressure shape");
        let br = spmv(&self.impedance, real).expect("validated pressure shape");
        let bi = spmv(&self.impedance, imaginary).expect("validated pressure shape");
        (
            (0..real.len())
                .map(|i| kr[i] - self.omega * self.omega * mr[i] - self.omega * bi[i])
                .collect(),
            (0..real.len())
                .map(|i| ki[i] - self.omega * self.omega * mi[i] + self.omega * br[i])
                .collect(),
        )
    }
}

pub(super) fn assembler(size: usize, role: OperatorRole, id: &str) -> RealSparseAssembler {
    let mut result = RealSparseAssembler::square(
        size,
        role,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    result.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Acoustic,
        contribution_id: ContributionId(id.into()),
    });
    result
}

fn planned(
    pattern: &crate::algebra::SparsePattern,
    values: Vec<f64>,
    role: OperatorRole,
    id: &str,
) -> Result<RealSparseAssembler, AcousticHarmonicError> {
    let mut result = RealSparseAssembler::from_planned_values(
        pattern.clone(),
        values,
        role,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))?;
    result.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Acoustic,
        contribution_id: ContributionId(id.into()),
    });
    Ok(result)
}

pub(super) fn finish(
    value: RealSparseAssembler,
) -> Result<RealSparseMatrix, AcousticHarmonicError> {
    value
        .finish()
        .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))
}
