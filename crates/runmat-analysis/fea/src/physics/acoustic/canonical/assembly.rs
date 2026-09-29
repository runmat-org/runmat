use super::AcousticHarmonicError;
use crate::{
    algebra::{
        spmv, ContributionDomain, ContributionId, ContributionProvenance, MatrixProperties,
        OperatorRole, RealSparseAssembler, RealSparseMatrix,
    },
    excitation::{ProjectionScalar, SpatialProjectionEvidence},
    properties::CompiledPropertyMap,
    topology::AnalysisProjection,
};

pub(super) struct AcousticAssembly {
    pub projection: AnalysisProjection,
    pub stiffness: RealSparseMatrix,
    pub mass: RealSparseMatrix,
    pub impedance: RealSparseMatrix,
    pub omega: f64,
    pub density: Vec<f64>,
    pub compressibility: Vec<f64>,
    pub source: Vec<ProjectionScalar>,
    pub prescribed: Vec<Option<ProjectionScalar>>,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
}

impl AcousticAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        omega: f64,
    ) -> Result<Self, AcousticHarmonicError> {
        let size = projection.nodes.len();
        let mut stiffness = assembler(size, OperatorRole::Stiffness, "pressure_gradient");
        let mut mass = assembler(size, OperatorRole::Mass, "pressure_compressibility");
        let mut density = Vec::with_capacity(projection.elements.len());
        let mut compressibility = Vec::with_capacity(projection.elements.len());
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
            projection
                .visit_element_quadrature(element, |point| {
                    for (i, &row) in element.node_indices.iter().enumerate() {
                        for (j, &column) in element.node_indices.iter().enumerate() {
                            let gradient = point.physical_shape_gradients()[i]
                                .iter()
                                .zip(point.physical_shape_gradients()[j])
                                .map(|(a, b)| a * b)
                                .sum::<f64>();
                            stiffness.add(row, column, gradient * point.volume_weight_m3 / rho);
                            mass.add(
                                row,
                                column,
                                point.shape_values()[i]
                                    * point.shape_values()[j]
                                    * point.volume_weight_m3
                                    / rho
                                    / c
                                    / c,
                            );
                        }
                    }
                })
                .map_err(|e| AcousticHarmonicError::Topology(e.to_string()))?;
        }
        Ok(Self {
            projection,
            stiffness: finish(stiffness)?,
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
        })
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

pub(super) fn finish(
    value: RealSparseAssembler,
) -> Result<RealSparseMatrix, AcousticHarmonicError> {
    value
        .finish()
        .map_err(|e| AcousticHarmonicError::Operator(e.to_string()))
}
