use super::quantity_layout::{pressure_operator, PressureOperator};
use super::AcousticHarmonicError;
use crate::{
    algebra::{
        element_assembly::{
            assemble_elements, no_visit, ElementAssemblyError, ElementOutputShape, ElementOutputs,
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
    pub stiffness: PressureOperator,
    /// Volume stiffness only; boundary resampling rebuilds `stiffness` from it.
    pub volume_stiffness: PressureOperator,
    pub mass: PressureOperator,
    /// Total admittance B of the harmonic operator K − ω²M + iωB: the volume loss of
    /// damped regions plus the boundary admittance.
    pub impedance: PressureOperator,
    /// Volume loss alone; boundary resampling rebuilds `impedance` from it.
    pub volume_admittance: PressureOperator,
    /// Whether any region carries material loss (a complex bulk modulus).
    pub lossy: bool,
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
    /// Port faces of the plane-wave power decomposition: incident-pressure faces with their
    /// authored incident wave, and anechoic (radiation) faces.
    pub port_faces: Vec<AcousticPortFace>,
    pub reduction: Option<crate::constraints::AffineConstraintReduction>,
}

/// One face of an acoustic port: `incident` is the authored incident pressure phasor on an
/// incident-pressure face, `None` on an anechoic face. `admittance` is the real plane-wave
/// admittance 1/(ρc) of the owning medium.
#[derive(Debug, Clone)]
pub(super) struct AcousticPortFace {
    pub face: crate::topology::ProjectedBoundaryFace,
    pub admittance: f64,
    pub incident: Option<crate::excitation::ComplexScalar>,
}

impl AcousticAssembly {
    pub(super) fn assemble(
        projection: AnalysisProjection,
        properties: &CompiledPropertyMap,
        omega: runmat_types::physical::AngularFrequency,
    ) -> Result<Self, AcousticHarmonicError> {
        let derivative_rate = omega.phasor_derivative_rate();
        let omega = omega.canonical_value();
        let size = projection.nodes.len();
        let mut density = Vec::with_capacity(projection.elements.len());
        let mut compressibility = Vec::with_capacity(projection.elements.len());
        let mut sound_speed = Vec::with_capacity(projection.elements.len());
        // Loss factor η = 2ζ of each element: complex bulk modulus K(1 + iη), so the
        // compressibility term −ω²M/(1 + iη) splits into −ω²M/(1 + η²) and +iω·ωηM/(1 + η²).
        let mut loss = Vec::with_capacity(projection.elements.len());
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
            let zeta = acoustic.damping_ratio.canonical_value();
            if !zeta.is_finite() || !(0.0..1.0).contains(&zeta) {
                return Err(AcousticHarmonicError::Material(
                    "acoustic damping ratio must be finite in [0, 1)".into(),
                ));
            }
            loss.push(2.0 * zeta);
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
                matrices: 3,
                ..Default::default()
            },
            plan,
            |index, _, out: &mut ElementOutputs<'_, f64>| {
                let element = &projection.elements[index];
                let (rho, c) = (density[index], sound_speed[index]);
                let eta = loss[index];
                let mass_scale = 1.0 / (1.0 + eta * eta);
                let loss_scale = omega * eta * mass_scale;
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
                for (target, value) in out.matrix(1).iter_mut().zip(&local_mass) {
                    *target = value * mass_scale;
                }
                for (target, value) in out.matrix(2).iter_mut().zip(&local_mass) {
                    *target = value * loss_scale;
                }
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
        let volume_admittance = finish(planned(
            &pattern,
            matrices.next().unwrap_or_default(),
            OperatorRole::Damping,
            "volume_loss",
        )?)?;
        let volume_admittance = pressure_operator(volume_admittance, 1)?;
        let stiffness = pressure_operator(finish(stiffness)?, 2)?;
        Ok(Self {
            projection,
            volume_stiffness: stiffness.clone(),
            stiffness,
            mass: pressure_operator(finish(mass)?, 0)?,
            impedance: volume_admittance.clone(),
            volume_admittance,
            lossy: loss.iter().any(|eta| *eta > 0.0),
            derivative_rate,
            density: runmat_types::physical::Quantity::new(density),
            compressibility: runmat_types::physical::Quantity::new(compressibility),
            source: zero_source(size),
            prescribed: runmat_types::physical::Quantity::new(vec![None; size]),
            source_evidence: Vec::new(),
            incident_faces: Vec::new(),
            port_faces: Vec::new(),
            reduction: None,
        })
    }

    /// Reset sources and prescribed values before resampling boundaries at another time.
    /// Operators and projection are retained; the boundary admittance is rebuilt.
    pub(super) fn with_cleared_sources(mut self) -> Self {
        let size = self.projection.nodes.len();
        self.source = zero_source(size);
        self.prescribed = runmat_types::physical::Quantity::new(vec![None; size]);
        self.source_evidence.clear();
        self.incident_faces.clear();
        self.port_faces.clear();
        self
    }

    pub(super) fn apply(&self, real: &[f64], imaginary: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let product = |operator: &PressureOperator, values: &[f64]| {
            operator
                .with_canonical_storage(|matrix| spmv(matrix, values))
                .expect("validated pressure shape")
        };
        let (kr, ki) = (
            product(&self.stiffness, real),
            product(&self.stiffness, imaginary),
        );
        let (mr, mi) = (product(&self.mass, real), product(&self.mass, imaginary));
        let (br, bi) = (
            product(&self.impedance, real),
            product(&self.impedance, imaginary),
        );
        let omega = self.derivative_rate.canonical_value();
        (
            (0..real.len())
                .map(|i| kr[i] - omega * omega * mr[i] - omega * bi[i])
                .collect(),
            (0..real.len())
                .map(|i| ki[i] - omega * omega * mi[i] + omega * br[i])
                .collect(),
        )
    }
}

/// Zero weak source in the acoustic equation's (second time order) units.
fn zero_source(
    size: usize,
) -> runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, Vec<ProjectionScalar>>
{
    runmat_types::physical::Quantity::try_from_descriptor(
        super::quantity_layout::equation_descriptor(2),
        vec![
            ProjectionScalar {
                real: 0.0,
                imaginary: 0.0
            };
            size
        ],
    )
    .expect("finite zero acoustic weak source")
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
