use crate::{
    AcousticConstraint, AcousticConstraintKind, AnalysisConstraint, AuthoredConstraint,
    AuthoredConstraintKind, AuthoredConstraintTarget, AuthoredPairedConstraint,
    AuthoredPairingPolicy, AuthoredPeriodicConstraint, AuthoredPeriodicTransform,
    AuthoredPhaseShiftedPeriodicConstraint, CoordinateFrame, ElectricalConstraint,
    ElectricalConstraintKind, FluidConstraint, FluidConstraintKind, MagneticConstraint,
    MagneticConstraintKind, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
    PhaseShiftedPeriodicConstraint, PrescribedStructuralDofs, PrescribedStructuralMotion,
    ReferencePointCoupling, RotationDofValue, StructuralConstraint, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, TargetCardinality,
    ThermalConstraint, ThermalConstraintKind, TranslationDofValue,
};

use super::{targets::TargetResolver, AnalysisModelRealizationError};

#[cfg(test)]
mod fluid_tests;

pub(super) fn extend_constraints(
    target: &mut Vec<AnalysisConstraint>,
    source: &[AuthoredConstraint],
    targets: &TargetResolver<'_>,
) -> Result<(), AnalysisModelRealizationError> {
    target.extend(
        source
            .iter()
            .map(|value| realize_constraint(value, targets))
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(())
}

fn realize_constraint(
    value: &AuthoredConstraint,
    targets: &TargetResolver<'_>,
) -> Result<AnalysisConstraint, AnalysisModelRealizationError> {
    use AuthoredConstraintKind as Authored;
    let id = value.constraint_id.clone();
    Ok(match &value.kind {
        Authored::StructuralFixed { target } => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Fixed {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::StructuralSymmetry { target } => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Symmetry {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::StructuralAntisymmetry { target } => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Antisymmetry {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::StructuralPrescribed(dofs) => structural(
            id,
            dofs.frame.clone(),
            StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: constraint_target(&dofs.target, targets)?,
                translations: dofs
                    .translations
                    .iter()
                    .map(|entry| TranslationDofValue {
                        dof: entry.dof,
                        value: entry.value.clone(),
                    })
                    .collect(),
                rotations: dofs
                    .rotations
                    .iter()
                    .map(|entry| RotationDofValue {
                        dof: entry.dof,
                        value: entry.value.clone(),
                    })
                    .collect(),
            }),
        ),
        Authored::StructuralPrescribedMotion(motion) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
                target: constraint_target(&motion.target, targets)?,
                translations: motion.translations.clone(),
            }),
        ),
        Authored::StructuralNormalDisplacement { target, value } => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::NormalDisplacement {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::StructuralRigidCoupling(value) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
                target: constraint_target(&value.target, targets)?,
                reference_point: value.reference_point.clone(),
            }),
        ),
        Authored::StructuralDistributingCoupling(value) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                target: constraint_target(&value.target, targets)?,
                reference_point: value.reference_point.clone(),
            }),
        ),
        Authored::StructuralTie(value) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Tie(pair(value, targets)?),
        ),
        Authored::StructuralPeriodic(value) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Periodic(periodic(value, targets)?),
        ),
        Authored::StructuralEquation(value) => structural(
            id,
            CoordinateFrame::Global,
            StructuralConstraintKind::Equation(StructuralConstraintEquation {
                terms: value
                    .terms
                    .iter()
                    .map(|term| {
                        Ok(StructuralConstraintEquationTerm {
                            target: constraint_target(&term.target, targets)?,
                            dof: term.dof,
                            coefficient: term.coefficient,
                        })
                    })
                    .collect::<Result<_, AnalysisModelRealizationError>>()?,
                right_hand_side: value.right_hand_side,
            }),
        ),
        Authored::ThermalTemperature { target, value } => {
            AnalysisConstraint::Thermal(ThermalConstraint {
                constraint_id: id,
                kind: ThermalConstraintKind::Prescribed {
                    target: constraint_target(target, targets)?,
                    value: value.clone(),
                },
            })
        }
        Authored::ThermalPeriodic(value) => thermal(
            id,
            ThermalConstraintKind::Periodic(periodic(value, targets)?),
        ),
        Authored::ThermalTie(value) => {
            thermal(id, ThermalConstraintKind::Tie(pair(value, targets)?))
        }
        Authored::ThermalAffineJump(value) => thermal(
            id,
            ThermalConstraintKind::AffineJump {
                periodic: periodic(&value.periodic, targets)?,
                temperature_jump: value.temperature_jump.clone(),
            },
        ),
        Authored::ElectricalPotential { target, value } => electrical(
            id,
            ElectricalConstraintKind::PrescribedPotential {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::ElectricPerfectConductor { target } => electrical(
            id,
            ElectricalConstraintKind::ElectricPerfectConductor {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::ElectricalEquipotential { target } => electrical(
            id,
            ElectricalConstraintKind::Equipotential {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::ElectricalFloatingConductor { target } => electrical(
            id,
            ElectricalConstraintKind::FloatingConductor {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::ElectricalIntegralCurrent { target, value } => electrical(
            id,
            ElectricalConstraintKind::IntegralCurrent {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::ElectricalPotentialReference { target } => electrical(
            id,
            ElectricalConstraintKind::PotentialReference {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::ElectricalPeriodic(value) => electrical(
            id,
            ElectricalConstraintKind::Periodic(periodic(value, targets)?),
        ),
        Authored::MagneticVectorPotential { target, value } => magnetic(
            id,
            MagneticConstraintKind::PrescribedVectorPotential {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::MagneticVectorPotentialGround { target } => magnetic(
            id,
            MagneticConstraintKind::VectorPotentialGround {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::MagneticCoulombGauge { target } => magnetic(
            id,
            MagneticConstraintKind::CoulombGauge {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::MagneticPeriodic(value) => magnetic(
            id,
            MagneticConstraintKind::Periodic(periodic(value, targets)?),
        ),
        Authored::MagneticPhaseShiftedPeriodic(value) => magnetic(
            id,
            MagneticConstraintKind::PhaseShiftedPeriodic(phase_periodic(value, targets)?),
        ),
        Authored::AcousticPressure { target, value } => acoustic(
            id,
            AcousticConstraintKind::PrescribedPressure {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::AcousticPeriodic(value) => acoustic(
            id,
            AcousticConstraintKind::Periodic(periodic(value, targets)?),
        ),
        Authored::AcousticPhaseShiftedPeriodic(value) => acoustic(
            id,
            AcousticConstraintKind::PhaseShiftedPeriodic(phase_periodic(value, targets)?),
        ),
        Authored::FluidVelocity { target, value } => fluid(
            id,
            FluidConstraintKind::PrescribedVelocity {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::FluidMovingWall { target, velocity } => fluid(
            id,
            FluidConstraintKind::MovingWall {
                target: constraint_target(target, targets)?,
                velocity: velocity.clone(),
            },
        ),
        Authored::FluidNoSlipWall { target } => fluid(
            id,
            FluidConstraintKind::NoSlipWall {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::FluidSlipWall { target } => fluid(
            id,
            FluidConstraintKind::SlipWall {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::FluidSymmetry { target } => fluid(
            id,
            FluidConstraintKind::Symmetry {
                target: constraint_target(target, targets)?,
            },
        ),
        Authored::FluidPressureReference { target, value } => fluid(
            id,
            FluidConstraintKind::PressureReference {
                target: constraint_target(target, targets)?,
                value: value.clone(),
            },
        ),
        Authored::FluidPeriodic(value) => {
            fluid(id, FluidConstraintKind::Periodic(periodic(value, targets)?))
        }
    })
}

fn constraint_target(
    value: &AuthoredConstraintTarget,
    targets: &TargetResolver<'_>,
) -> Result<crate::RegionTarget, AnalysisModelRealizationError> {
    let cardinality = match value.cardinality {
        crate::AuthoredTargetCardinality::One => TargetCardinality::One,
        crate::AuthoredTargetCardinality::OneOrMore => TargetCardinality::OneOrMore,
    };
    targets.region_target(&value.region, cardinality)
}

fn pair(
    value: &AuthoredPairedConstraint,
    targets: &TargetResolver<'_>,
) -> Result<PairedConstraint, AnalysisModelRealizationError> {
    Ok(PairedConstraint {
        primary: constraint_target(&value.primary, targets)?,
        secondary: constraint_target(&value.secondary, targets)?,
        pairing: match value.pairing {
            AuthoredPairingPolicy::Matching => PairingPolicy::Matching,
            AuthoredPairingPolicy::GeometricTolerance(value) => {
                PairingPolicy::GeometricTolerance(value)
            }
        },
    })
}

fn periodic(
    value: &AuthoredPeriodicConstraint,
    targets: &TargetResolver<'_>,
) -> Result<PeriodicConstraint, AnalysisModelRealizationError> {
    Ok(PeriodicConstraint {
        pair: pair(&value.pair, targets)?,
        transform: match value.transform {
            AuthoredPeriodicTransform::Identity => PeriodicTransform::Identity,
            AuthoredPeriodicTransform::Translation(value) => PeriodicTransform::Translation(value),
            AuthoredPeriodicTransform::Rotation {
                origin,
                axis,
                angle,
            } => PeriodicTransform::Rotation {
                origin,
                axis,
                angle,
            },
        },
    })
}

fn phase_periodic(
    value: &AuthoredPhaseShiftedPeriodicConstraint,
    targets: &TargetResolver<'_>,
) -> Result<PhaseShiftedPeriodicConstraint, AnalysisModelRealizationError> {
    Ok(PhaseShiftedPeriodicConstraint {
        periodic: periodic(&value.periodic, targets)?,
        phase: value.phase,
    })
}

fn structural(
    id: String,
    frame: CoordinateFrame,
    kind: StructuralConstraintKind,
) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: id,
        frame,
        kind,
    })
}
fn thermal(id: String, kind: ThermalConstraintKind) -> AnalysisConstraint {
    AnalysisConstraint::Thermal(ThermalConstraint {
        constraint_id: id,
        kind,
    })
}
fn electrical(id: String, kind: ElectricalConstraintKind) -> AnalysisConstraint {
    AnalysisConstraint::Electrical(ElectricalConstraint {
        constraint_id: id,
        kind,
    })
}
fn magnetic(id: String, kind: MagneticConstraintKind) -> AnalysisConstraint {
    AnalysisConstraint::Magnetic(MagneticConstraint {
        constraint_id: id,
        kind,
    })
}
fn acoustic(id: String, kind: AcousticConstraintKind) -> AnalysisConstraint {
    AnalysisConstraint::Acoustic(AcousticConstraint {
        constraint_id: id,
        kind,
    })
}
fn fluid(id: String, kind: FluidConstraintKind) -> AnalysisConstraint {
    AnalysisConstraint::Fluid(FluidConstraint {
        constraint_id: id,
        kind,
    })
}
