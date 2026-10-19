use runmat_analysis_core::{
    AnalysisModel, BoundaryTransferImplementation, SourceImplementation, StructuralConstraintKind,
};

use crate::excitation::{RealizedExcitationKind, RealizedExcitationTarget, RealizedExcitations};

use super::StructuralAssemblyError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralLoadKind {
    Force,
    Moment,
    Traction,
    Pressure,
    BodyForce,
    HeatGeneration,
    HeatFlux,
    TemperatureField,
    AmbientTemperature,
    TerminalVoltage,
    ElectricCurrent,
    ElectricCharge,
    ElectricChargeDensity,
    CurrentDensity,
    MagneticVolumeCurrentDensity,
    MagneticVectorPotential,
    AcousticPressure,
    NormalVelocity,
    VolumeVelocity,
    InletVelocity,
    MassFlow,
    WallMotion,
}

impl StructuralLoadKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Force => "force",
            Self::Moment => "moment",
            Self::Traction => "traction",
            Self::Pressure => "pressure",
            Self::BodyForce => "body_force",
            Self::HeatGeneration => "heat_generation",
            Self::HeatFlux => "heat_flux",
            Self::TemperatureField => "temperature_field",
            Self::AmbientTemperature => "ambient_temperature",
            Self::TerminalVoltage => "terminal_voltage",
            Self::ElectricCurrent => "electric_current",
            Self::ElectricCharge => "electric_charge",
            Self::ElectricChargeDensity => "electric_charge_density",
            Self::CurrentDensity => "current_density",
            Self::MagneticVolumeCurrentDensity => "magnetic_volume_current_density",
            Self::MagneticVectorPotential => "magnetic_vector_potential",
            Self::AcousticPressure => "acoustic_pressure",
            Self::NormalVelocity => "normal_velocity",
            Self::VolumeVelocity => "volume_velocity",
            Self::InletVelocity => "inlet_velocity",
            Self::MassFlow => "mass_flow",
            Self::WallMotion => "wall_motion",
        }
    }
}

impl std::fmt::Display for StructuralLoadKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StructuralProjectionBasis {
    CanonicalSolid,
    ExplicitConnectivity,
    Synthetic,
}

pub(super) fn apply_synthetic_structural_rhs(excitations: &RealizedExcitations, rhs: &mut [f64]) {
    for (index, load) in excitations.iter().enumerate() {
        let base = (index * 3) % rhs.len();
        match &load.kind {
            RealizedExcitationKind::Force(value) | RealizedExcitationKind::BodyForce(value) => {
                let [x, y, z] = value.real;
                rhs[base] += x;
                if base + 1 < rhs.len() {
                    rhs[base + 1] += y;
                }
                if base + 2 < rhs.len() {
                    rhs[base + 2] += z;
                }
            }
            RealizedExcitationKind::Pressure(value) => {
                rhs[base] += value.real * 1.0e-3;
                if base + 1 < rhs.len() {
                    rhs[base + 1] -= value.real * 1.0e-3;
                }
            }
            RealizedExcitationKind::TemperatureField(_)
            | RealizedExcitationKind::TerminalVoltage(_) => {}
            _ => unreachable!("structural input validation must reject unsupported loads"),
        }
    }
}

pub(super) fn requested_moment_norm(excitations: &RealizedExcitations) -> f64 {
    let moment = excitations.iter().fold([0.0_f64; 3], |mut total, entry| {
        if let RealizedExcitationKind::Moment(value) = entry.kind {
            for (total, value) in total.iter_mut().zip(value.real) {
                *total += value;
            }
        }
        total
    });
    moment
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

pub(super) fn structural_source_implementations(
    excitations: &RealizedExcitations,
    basis: StructuralProjectionBasis,
) -> Vec<SourceImplementation> {
    let mut implementations = Vec::new();
    for excitation in excitations.iter() {
        let implementation = if basis == StructuralProjectionBasis::CanonicalSolid
            && matches!(
                excitation.target,
                RealizedExcitationTarget::ReferencePoint { .. }
            )
            && matches!(
                excitation.kind,
                RealizedExcitationKind::Force(_) | RealizedExcitationKind::Moment(_)
            ) {
            SourceImplementation::ReferencePointGeneralizedLoad
        } else {
            match (&excitation.kind, basis) {
                (RealizedExcitationKind::Force(_), StructuralProjectionBasis::CanonicalSolid) => {
                    SourceImplementation::CanonicalBoundaryProjection
                }
                (
                    RealizedExcitationKind::Pressure(_),
                    StructuralProjectionBasis::CanonicalSolid,
                ) => SourceImplementation::CanonicalBoundaryPressureIntegration,
                (
                    RealizedExcitationKind::Traction(_),
                    StructuralProjectionBasis::CanonicalSolid,
                ) => SourceImplementation::CanonicalBoundaryProjection,
                (RealizedExcitationKind::Moment(_), StructuralProjectionBasis::CanonicalSolid) => {
                    SourceImplementation::BoundaryResultantMomentProjection
                }
                (
                    RealizedExcitationKind::BodyForce(_)
                    | RealizedExcitationKind::BodyForceField(_),
                    StructuralProjectionBasis::CanonicalSolid,
                ) => SourceImplementation::CanonicalVolumeProjection,
                (
                    RealizedExcitationKind::Force(_)
                    | RealizedExcitationKind::Moment(_)
                    | RealizedExcitationKind::Pressure(_)
                    | RealizedExcitationKind::Traction(_)
                    | RealizedExcitationKind::BodyForce(_),
                    StructuralProjectionBasis::ExplicitConnectivity,
                ) => SourceImplementation::ExplicitRegionNodalDistribution,
                (
                    RealizedExcitationKind::Force(_)
                    | RealizedExcitationKind::Pressure(_)
                    | RealizedExcitationKind::BodyForce(_),
                    StructuralProjectionBasis::Synthetic,
                ) => SourceImplementation::OrdinalDistribution,
                _ => continue,
            }
        };
        push_unique(&mut implementations, implementation);
    }
    if implementations.is_empty() {
        implementations.push(SourceImplementation::NotApplicable);
    }
    implementations
}

pub(super) fn structural_boundary_implementations(
    model: &AnalysisModel,
    basis: StructuralProjectionBasis,
) -> Vec<BoundaryTransferImplementation> {
    if !model
        .constraints
        .iter()
        .any(|constraint| constraint.structural().is_some())
    {
        return vec![BoundaryTransferImplementation::NotApplicable];
    }
    vec![match basis {
        StructuralProjectionBasis::CanonicalSolid => {
            BoundaryTransferImplementation::CanonicalBoundaryProjection
        }
        StructuralProjectionBasis::ExplicitConnectivity => {
            BoundaryTransferImplementation::ExplicitRegionConnectivity
        }
        StructuralProjectionBasis::Synthetic => BoundaryTransferImplementation::GlobalCountProfile,
    }]
}

pub(super) fn validate_structural_inputs(
    model: &AnalysisModel,
    excitations: &RealizedExcitations,
    basis: StructuralProjectionBasis,
) -> Result<(), StructuralAssemblyError> {
    for excitation in excitations.iter() {
        if matches!(
            excitation.kind,
            RealizedExcitationKind::TemperatureField(_)
                | RealizedExcitationKind::TerminalVoltage(_)
        ) {
            continue;
        }
        let (kind, supported) = match (&excitation.kind, basis) {
            (RealizedExcitationKind::Force(_), _) => (StructuralLoadKind::Force, true),
            (
                RealizedExcitationKind::Moment(_),
                StructuralProjectionBasis::CanonicalSolid
                | StructuralProjectionBasis::ExplicitConnectivity,
            ) => (StructuralLoadKind::Moment, true),
            (RealizedExcitationKind::Pressure(_), StructuralProjectionBasis::CanonicalSolid)
            | (RealizedExcitationKind::Pressure(_), StructuralProjectionBasis::Synthetic) => {
                (StructuralLoadKind::Pressure, true)
            }
            (RealizedExcitationKind::Traction(_), StructuralProjectionBasis::CanonicalSolid) => {
                (StructuralLoadKind::Traction, true)
            }
            (
                RealizedExcitationKind::Pressure(_) | RealizedExcitationKind::Traction(_),
                StructuralProjectionBasis::ExplicitConnectivity,
            ) => (
                match excitation.kind {
                    RealizedExcitationKind::Pressure(_) => StructuralLoadKind::Pressure,
                    _ => StructuralLoadKind::Traction,
                },
                true,
            ),
            (RealizedExcitationKind::Traction(_), StructuralProjectionBasis::Synthetic) => {
                (StructuralLoadKind::Traction, false)
            }
            (RealizedExcitationKind::BodyForceField(_), basis) => (
                StructuralLoadKind::BodyForce,
                basis == StructuralProjectionBasis::CanonicalSolid,
            ),
            (RealizedExcitationKind::BodyForce(_), StructuralProjectionBasis::CanonicalSolid)
            | (
                RealizedExcitationKind::BodyForce(_),
                StructuralProjectionBasis::ExplicitConnectivity,
            )
            | (RealizedExcitationKind::BodyForce(_), StructuralProjectionBasis::Synthetic) => {
                (StructuralLoadKind::BodyForce, true)
            }
            (RealizedExcitationKind::Moment(_), _) => (StructuralLoadKind::Moment, false),
            (
                RealizedExcitationKind::HeatGeneration(_)
                | RealizedExcitationKind::HeatGenerationField(_),
                _,
            ) => (StructuralLoadKind::HeatGeneration, false),
            (RealizedExcitationKind::HeatFlux(_), _) => (StructuralLoadKind::HeatFlux, false),
            (RealizedExcitationKind::TemperatureField(_), _) => {
                unreachable!("temperature fields are coupling inputs, not structural loads")
            }
            (RealizedExcitationKind::AmbientTemperature(_), _) => {
                (StructuralLoadKind::AmbientTemperature, false)
            }
            (RealizedExcitationKind::TerminalVoltage(_), _) => {
                unreachable!("terminal voltage is a coupling input, not a structural load")
            }
            (RealizedExcitationKind::ElectricCurrent(_), _) => {
                (StructuralLoadKind::ElectricCurrent, false)
            }
            (RealizedExcitationKind::ElectricCharge(_), _) => {
                (StructuralLoadKind::ElectricCharge, false)
            }
            (RealizedExcitationKind::ElectricChargeDensity(_), _) => {
                (StructuralLoadKind::ElectricChargeDensity, false)
            }
            (RealizedExcitationKind::CurrentDensity(_), _) => {
                (StructuralLoadKind::CurrentDensity, false)
            }
            (RealizedExcitationKind::MagneticVolumeCurrentDensity(_), _) => {
                (StructuralLoadKind::MagneticVolumeCurrentDensity, false)
            }
            (RealizedExcitationKind::MagneticVectorPotential(_), _) => {
                (StructuralLoadKind::MagneticVectorPotential, false)
            }
            (
                RealizedExcitationKind::AcousticPressure(_)
                | RealizedExcitationKind::AcousticIncidentPressure(_),
                _,
            ) => (StructuralLoadKind::AcousticPressure, false),
            (RealizedExcitationKind::NormalVelocity(_), _) => {
                (StructuralLoadKind::NormalVelocity, false)
            }
            (RealizedExcitationKind::VolumeVelocity(_), _) => {
                (StructuralLoadKind::VolumeVelocity, false)
            }
            (RealizedExcitationKind::InletVelocity(_), _) => {
                (StructuralLoadKind::InletVelocity, false)
            }
            (RealizedExcitationKind::MassFlow(_), _) => (StructuralLoadKind::MassFlow, false),
            (RealizedExcitationKind::WallMotion(_), _) => (StructuralLoadKind::WallMotion, false),
        };
        if !supported {
            return Err(StructuralAssemblyError::UnsupportedStructuralLoad {
                load_id: excitation.excitation_id.0.clone(),
                load_kind: kind,
            });
        }
        if basis != StructuralProjectionBasis::Synthetic
            && kind != StructuralLoadKind::BodyForce
            && excitation.target.solver_region_id().is_none()
        {
            return Err(StructuralAssemblyError::UnmappedStructuralLoad {
                load_id: excitation.excitation_id.0.clone(),
                region_id: "<non-region target>".to_string(),
                load_kind: kind,
            });
        }
    }

    for constraint in model
        .constraints
        .iter()
        .filter_map(|value| value.structural())
    {
        let supported = match basis {
            StructuralProjectionBasis::CanonicalSolid => matches!(
                constraint.kind,
                StructuralConstraintKind::Fixed { .. }
                    | StructuralConstraintKind::Prescribed(_)
                    | StructuralConstraintKind::PrescribedMotion(_)
                    | StructuralConstraintKind::NormalDisplacement { .. }
                    | StructuralConstraintKind::Symmetry { .. }
                    | StructuralConstraintKind::Antisymmetry { .. }
                    | StructuralConstraintKind::RigidCoupling(_)
                    | StructuralConstraintKind::DistributingCoupling(_)
                    | StructuralConstraintKind::Tie(_)
                    | StructuralConstraintKind::Periodic(_)
                    | StructuralConstraintKind::Equation(_)
            ),
            StructuralProjectionBasis::ExplicitConnectivity => matches!(
                constraint.kind,
                StructuralConstraintKind::Fixed { .. }
                    | StructuralConstraintKind::Prescribed(_)
                    | StructuralConstraintKind::PrescribedMotion(_)
                    | StructuralConstraintKind::NormalDisplacement { .. }
                    | StructuralConstraintKind::Symmetry { .. }
                    | StructuralConstraintKind::Antisymmetry { .. }
                    | StructuralConstraintKind::RigidCoupling(_)
                    | StructuralConstraintKind::DistributingCoupling(_)
                    | StructuralConstraintKind::Tie(_)
                    | StructuralConstraintKind::Periodic(_)
                    | StructuralConstraintKind::Equation(_)
            ),
            StructuralProjectionBasis::Synthetic => true,
        };
        if !supported {
            return Err(StructuralAssemblyError::UnsupportedStructuralConstraint {
                constraint_id: constraint.constraint_id.clone(),
                constraint_kind: structural_constraint_kind_name(&constraint.kind),
            });
        }
        if let StructuralConstraintKind::Prescribed(values) = &constraint.kind {
            if values
                .translations
                .iter()
                .any(|value| value.value.study_constant(&model.signals).is_none())
                || values
                    .rotations
                    .iter()
                    .any(|value| value.value.study_constant(&model.signals).is_none())
            {
                return Err(StructuralAssemblyError::UnsupportedScheduledConstraint {
                    constraint_id: constraint.constraint_id.clone(),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn structural_constraint_kind_name(kind: &StructuralConstraintKind) -> &'static str {
    match kind {
        StructuralConstraintKind::Fixed { .. } => "fixed",
        StructuralConstraintKind::Prescribed(_) => "prescribed",
        StructuralConstraintKind::PrescribedMotion(_) => "prescribed_motion",
        StructuralConstraintKind::NormalDisplacement { .. } => "normal_displacement",
        StructuralConstraintKind::Symmetry { .. } => "symmetry",
        StructuralConstraintKind::Antisymmetry { .. } => "antisymmetry",
        StructuralConstraintKind::RigidCoupling(_) => "rigid_coupling",
        StructuralConstraintKind::DistributingCoupling(_) => "distributing_coupling",
        StructuralConstraintKind::Tie(_) => "tie",
        StructuralConstraintKind::Periodic(_) => "periodic",
        StructuralConstraintKind::Equation(_) => "equation",
    }
}

pub(super) fn explicit_load_kind(kind: &RealizedExcitationKind) -> StructuralLoadKind {
    match kind {
        RealizedExcitationKind::Force(_) => StructuralLoadKind::Force,
        RealizedExcitationKind::Moment(_) => StructuralLoadKind::Moment,
        RealizedExcitationKind::Traction(_) => StructuralLoadKind::Traction,
        RealizedExcitationKind::Pressure(_) => StructuralLoadKind::Pressure,
        RealizedExcitationKind::BodyForce(_) | RealizedExcitationKind::BodyForceField(_) => {
            StructuralLoadKind::BodyForce
        }
        RealizedExcitationKind::HeatGeneration(_)
        | RealizedExcitationKind::HeatGenerationField(_) => StructuralLoadKind::HeatGeneration,
        RealizedExcitationKind::HeatFlux(_) => StructuralLoadKind::HeatFlux,
        RealizedExcitationKind::TemperatureField(_) => StructuralLoadKind::TemperatureField,
        RealizedExcitationKind::AmbientTemperature(_) => StructuralLoadKind::AmbientTemperature,
        RealizedExcitationKind::TerminalVoltage(_) => StructuralLoadKind::TerminalVoltage,
        RealizedExcitationKind::ElectricCurrent(_) => StructuralLoadKind::ElectricCurrent,
        RealizedExcitationKind::ElectricCharge(_) => StructuralLoadKind::ElectricCharge,
        RealizedExcitationKind::ElectricChargeDensity(_) => {
            StructuralLoadKind::ElectricChargeDensity
        }
        RealizedExcitationKind::CurrentDensity(_) => StructuralLoadKind::CurrentDensity,
        RealizedExcitationKind::MagneticVolumeCurrentDensity(_) => {
            StructuralLoadKind::MagneticVolumeCurrentDensity
        }
        RealizedExcitationKind::MagneticVectorPotential(_) => {
            StructuralLoadKind::MagneticVectorPotential
        }
        RealizedExcitationKind::AcousticPressure(_)
        | RealizedExcitationKind::AcousticIncidentPressure(_) => {
            StructuralLoadKind::AcousticPressure
        }
        RealizedExcitationKind::NormalVelocity(_) => StructuralLoadKind::NormalVelocity,
        RealizedExcitationKind::VolumeVelocity(_) => StructuralLoadKind::VolumeVelocity,
        RealizedExcitationKind::InletVelocity(_) => StructuralLoadKind::InletVelocity,
        RealizedExcitationKind::MassFlow(_) => StructuralLoadKind::MassFlow,
        RealizedExcitationKind::WallMotion(_) => StructuralLoadKind::WallMotion,
    }
}

fn push_unique<T: PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}
