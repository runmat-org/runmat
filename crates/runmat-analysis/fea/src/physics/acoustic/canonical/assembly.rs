use super::quantity_layout::{pressure_operator, PressureOperator};
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
    pub stiffness: PressureOperator,
    pub mass: PressureOperator,
    pub impedance: PressureOperator,
    pub derivative_rate: runmat_types::physical::TimeDerivativeRate,
    pub density:
        runmat_types::physical::Quantity<runmat_types::physical::kind::MassDensity, Vec<f64>>,
    pub compressibility:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Compressibility, Vec<f64>>,
    pub source: runmat_types::physical::Quantity<
        runmat_types::physical::DynamicDimension,
        Vec<ProjectionScalar>,
    >,
    pub prescribed: runmat_types::physical::Quantity<
        runmat_types::physical::kind::Pressure,
        Vec<Option<ProjectionScalar>>,
    >,
    pub source_evidence: Vec<SpatialProjectionEvidence>,
    pub incident_faces: Vec<(crate::topology::ProjectedBoundaryFace, f64)>,
    pub reduction: Option<crate::constraints::AffineConstraintReduction>,
}

impl AcousticAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        omega: runmat_types::physical::AngularFrequency,
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
                .map_err(|e| AcousticHarmonicError::Material(e.to_string()))?
                .canonical_value();
            let c = acoustic.speed_of_sound_m_per_s.canonical_value();
            if !rho.is_finite() || rho <= 0.0 || !c.is_finite() || c <= 0.0 {
                return Err(AcousticHarmonicError::Material(
                    "density and sound speed must be positive finite".into(),
                ));
            }
            if acoustic.damping_ratio.canonical_value() != 0.0 {
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
            stiffness: pressure_operator(finish(stiffness)?, 2)?,
            mass: pressure_operator(finish(mass)?, 0)?,
            impedance: pressure_operator(
                finish(assembler(
                    size,
                    OperatorRole::Damping,
                    "boundary_admittance",
                ))?,
                1,
            )?,
            derivative_rate: omega.phasor_derivative_rate(),
            density: runmat_types::physical::Quantity::new(density),
            compressibility: runmat_types::physical::Quantity::new(compressibility),
            source: runmat_types::physical::Quantity::try_from_descriptor(
                super::quantity_layout::equation_descriptor(2),
                vec![
                    ProjectionScalar {
                        real: 0.0,
                        imaginary: 0.0
                    };
                    size
                ],
            )
            .expect("finite zero acoustic weak source"),
            prescribed: runmat_types::physical::Quantity::new(vec![None; size]),
            source_evidence: Vec::new(),
            incident_faces: Vec::new(),
            reduction: None,
        })
    }

    pub(super) fn apply(&self, real: &[f64], imaginary: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let kr = self
            .stiffness
            .with_canonical_storage(|matrix| spmv(matrix, real))
            .expect("validated pressure shape");
        let ki = self
            .stiffness
            .with_canonical_storage(|matrix| spmv(matrix, imaginary))
            .expect("validated pressure shape");
        let mr = self
            .mass
            .with_canonical_storage(|matrix| spmv(matrix, real))
            .expect("validated pressure shape");
        let mi = self
            .mass
            .with_canonical_storage(|matrix| spmv(matrix, imaginary))
            .expect("validated pressure shape");
        let br = self
            .impedance
            .with_canonical_storage(|matrix| spmv(matrix, real))
            .expect("validated pressure shape");
        let bi = self
            .impedance
            .with_canonical_storage(|matrix| spmv(matrix, imaginary))
            .expect("validated pressure shape");
        (
            (0..real.len())
                .map(|i| {
                    kr[i]
                        - self.derivative_rate.canonical_value()
                            * self.derivative_rate.canonical_value()
                            * mr[i]
                        - self.derivative_rate.canonical_value() * bi[i]
                })
                .collect(),
            (0..real.len())
                .map(|i| {
                    ki[i]
                        - self.derivative_rate.canonical_value()
                            * self.derivative_rate.canonical_value()
                            * mi[i]
                        + self.derivative_rate.canonical_value() * br[i]
                })
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
