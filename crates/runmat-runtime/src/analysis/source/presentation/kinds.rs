use runmat_analysis_core::{
    AuthoredBoundaryKind, AuthoredConnectorKind, AuthoredConstraintKind,
    AuthoredInitialConditionKind, AuthoredInterfaceKind,
};

use super::{
    FeaBoundaryPresentationKind, FeaConnectorPresentationKind, FeaConstraintPresentationKind,
    FeaInitialConditionPresentationKind, FeaInterfacePresentationKind,
};

pub(super) fn constraint_kind(kind: &AuthoredConstraintKind) -> FeaConstraintPresentationKind {
    use AuthoredConstraintKind as Source;
    use FeaConstraintPresentationKind as Target;
    match kind {
        Source::StructuralFixed { .. } => Target::StructuralFixed,
        Source::StructuralSymmetry { .. } => Target::StructuralSymmetry,
        Source::StructuralAntisymmetry { .. } => Target::StructuralAntisymmetry,
        Source::StructuralPrescribed(_) => Target::StructuralPrescribed,
        Source::StructuralPrescribedMotion(_) => Target::StructuralPrescribedMotion,
        Source::StructuralNormalDisplacement { .. } => Target::StructuralNormalDisplacement,
        Source::StructuralRigidCoupling(_) => Target::StructuralRigidCoupling,
        Source::StructuralDistributingCoupling(_) => Target::StructuralDistributingCoupling,
        Source::StructuralTie(_) => Target::StructuralTie,
        Source::StructuralPeriodic(_) => Target::StructuralPeriodic,
        Source::StructuralEquation(_) => Target::StructuralEquation,
        Source::ThermalTemperature { .. } => Target::ThermalTemperature,
        Source::ThermalPeriodic(_) => Target::ThermalPeriodic,
        Source::ThermalTie(_) => Target::ThermalTie,
        Source::ThermalAffineJump(_) => Target::ThermalAffineJump,
        Source::ElectricalPotential { .. } => Target::ElectricalPotential,
        Source::ElectricalEquipotential { .. } => Target::ElectricalEquipotential,
        Source::ElectricalFloatingConductor { .. } => Target::ElectricalFloatingConductor,
        Source::ElectricalIntegralCurrent { .. } => Target::ElectricalIntegralCurrent,
        Source::ElectricalPotentialReference { .. } => Target::ElectricalPotentialReference,
        Source::ElectricalPeriodic(_) => Target::ElectricalPeriodic,
        Source::MagneticVectorPotential { .. } => Target::MagneticVectorPotential,
        Source::MagneticVectorPotentialGround { .. } => Target::MagneticVectorPotentialGround,
        Source::MagneticCoulombGauge { .. } => Target::MagneticCoulombGauge,
        Source::MagneticPeriodic(_) => Target::MagneticPeriodic,
        Source::MagneticPhaseShiftedPeriodic(_) => Target::MagneticPhaseShiftedPeriodic,
        Source::AcousticPressure { .. } => Target::AcousticPressure,
        Source::AcousticPeriodic(_) => Target::AcousticPeriodic,
        Source::AcousticPhaseShiftedPeriodic(_) => Target::AcousticPhaseShiftedPeriodic,
        Source::FluidVelocity { .. } => Target::FluidVelocity,
        Source::FluidMovingWall { .. } => Target::FluidMovingWall,
        Source::FluidNoSlipWall { .. } => Target::FluidNoSlipWall,
        Source::FluidSlipWall { .. } => Target::FluidSlipWall,
        Source::FluidSymmetry { .. } => Target::FluidSymmetry,
        Source::FluidPressureReference { .. } => Target::FluidPressureReference,
        Source::FluidPeriodic(_) => Target::FluidPeriodic,
    }
}

pub(super) fn boundary_kind(kind: &AuthoredBoundaryKind) -> FeaBoundaryPresentationKind {
    match kind {
        AuthoredBoundaryKind::StructuralPressure { .. } => {
            FeaBoundaryPresentationKind::StructuralPressure
        }
        AuthoredBoundaryKind::ThermalHeatFlux { .. } => {
            FeaBoundaryPresentationKind::ThermalHeatFlux
        }
        AuthoredBoundaryKind::ThermalConvection { .. } => {
            FeaBoundaryPresentationKind::ThermalConvection
        }
        AuthoredBoundaryKind::MagneticInsulation => FeaBoundaryPresentationKind::MagneticInsulation,
        AuthoredBoundaryKind::AcousticRigidWall => FeaBoundaryPresentationKind::AcousticRigidWall,
        AuthoredBoundaryKind::AcousticRadiation => FeaBoundaryPresentationKind::AcousticRadiation,
        AuthoredBoundaryKind::AcousticImpedance { .. } => {
            FeaBoundaryPresentationKind::AcousticImpedance
        }
        AuthoredBoundaryKind::FluidOutletPressure { .. } => {
            FeaBoundaryPresentationKind::FluidOutletPressure
        }
    }
}

pub(super) fn connector_kind(kind: &AuthoredConnectorKind) -> FeaConnectorPresentationKind {
    match kind {
        AuthoredConnectorKind::AxialSpring { .. } => FeaConnectorPresentationKind::AxialSpring,
        AuthoredConnectorKind::AxialDashpot { .. } => FeaConnectorPresentationKind::AxialDashpot,
        AuthoredConnectorKind::IsotropicGroundSpring { .. } => {
            FeaConnectorPresentationKind::IsotropicGroundSpring
        }
        AuthoredConnectorKind::SurfaceNormalFoundation { .. } => {
            FeaConnectorPresentationKind::SurfaceNormalFoundation
        }
    }
}

pub(super) fn interface_kind(kind: &AuthoredInterfaceKind) -> FeaInterfacePresentationKind {
    match kind {
        AuthoredInterfaceKind::ContactFrictionless => {
            FeaInterfacePresentationKind::ContactFrictionless
        }
        AuthoredInterfaceKind::ContactCoulomb { .. } => {
            FeaInterfacePresentationKind::ContactCoulomb
        }
        AuthoredInterfaceKind::FluidStructureNoSlip => {
            FeaInterfacePresentationKind::FluidStructureNoSlip
        }
        AuthoredInterfaceKind::ConjugateHeatTransferPerfect => {
            FeaInterfacePresentationKind::ConjugateHeatTransferPerfect
        }
        AuthoredInterfaceKind::ConjugateHeatTransferConductance { .. } => {
            FeaInterfacePresentationKind::ConjugateHeatTransferConductance
        }
        AuthoredInterfaceKind::ConjugateHeatTransferResistance { .. } => {
            FeaInterfacePresentationKind::ConjugateHeatTransferResistance
        }
    }
}

pub(super) fn initial_condition_kind(
    kind: &AuthoredInitialConditionKind,
) -> FeaInitialConditionPresentationKind {
    match kind {
        AuthoredInitialConditionKind::Displacement(_) => {
            FeaInitialConditionPresentationKind::Displacement
        }
        AuthoredInitialConditionKind::Velocity(_) => FeaInitialConditionPresentationKind::Velocity,
        AuthoredInitialConditionKind::AngularVelocity(_) => {
            FeaInitialConditionPresentationKind::AngularVelocity
        }
        AuthoredInitialConditionKind::Temperature(_) => {
            FeaInitialConditionPresentationKind::Temperature
        }
        AuthoredInitialConditionKind::ElectricPotential(_) => {
            FeaInitialConditionPresentationKind::ElectricPotential
        }
        AuthoredInitialConditionKind::MagneticVectorPotential(_) => {
            FeaInitialConditionPresentationKind::MagneticVectorPotential
        }
        AuthoredInitialConditionKind::AcousticPressure(_) => {
            FeaInitialConditionPresentationKind::AcousticPressure
        }
        AuthoredInitialConditionKind::FluidPressure(_) => {
            FeaInitialConditionPresentationKind::FluidPressure
        }
        AuthoredInitialConditionKind::FluidVelocity(_) => {
            FeaInitialConditionPresentationKind::FluidVelocity
        }
    }
}
