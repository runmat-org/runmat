use crate::{
    AnalysisBoundary, AnalysisConnector, AnalysisExcitation, AnalysisInterface,
    AnalysisInterfaceKind, AnalysisModel, AuthoredBoundaryKind, AuthoredConnectorKind,
    AuthoredExcitation, AuthoredExcitationKind, AuthoredExcitationTarget, AuthoredInitialCondition,
    AuthoredInitialConditionKind, AuthoredInitialConditionTarget, AuthoredInterfaceKind,
    AuthoredModel, BoundaryOperator, ConjugateHeatTransferInterfaceModel, ConnectorKind,
    ContactFriction, ContactInterfaceModel, ExcitationAction, ExcitationCase, ExcitationCaseId,
    ExcitationId, ExcitationTarget, FluidStructureInterfaceModel, FluidStructureKinematics,
    InitialCondition, InitialConditionTarget, InitialConditionValue, TargetCardinality,
    ThermalInterfaceTransfer,
};

use super::{targets::TargetResolver, AnalysisModelRealizationError};

pub(super) fn extend_model_physics(
    target: &mut AnalysisModel,
    source: &AuthoredModel,
    targets: &TargetResolver<'_>,
) -> Result<(), AnalysisModelRealizationError> {
    target.signals.extend(source.signals.clone());
    target
        .response_spectra
        .extend(source.response_spectra.clone());
    target
        .random_excitations
        .extend(source.random_excitations.clone());
    target.boundaries.extend(
        source
            .boundaries
            .iter()
            .map(|value| {
                Ok(AnalysisBoundary {
                    boundary_id: value.boundary_id.clone(),
                    target: targets.region_target(&value.target, TargetCardinality::OneOrMore)?,
                    operator: boundary_operator(&value.kind),
                })
            })
            .collect::<Result<Vec<_>, AnalysisModelRealizationError>>()?,
    );
    target.connectors.extend(
        source
            .connectors
            .iter()
            .map(|value| {
                Ok(AnalysisConnector {
                    connector_id: value.connector_id.clone(),
                    primary: targets.region_target(&value.primary, TargetCardinality::OneOrMore)?,
                    secondary: value
                        .secondary
                        .as_ref()
                        .map(|source| targets.region_target(source, TargetCardinality::OneOrMore))
                        .transpose()?,
                    kind: match value.kind {
                        AuthoredConnectorKind::TranslationalSpring { stiffness } => {
                            ConnectorKind::TranslationalSpring { stiffness }
                        }
                        AuthoredConnectorKind::TranslationalDashpot { damping } => {
                            ConnectorKind::TranslationalDashpot { damping }
                        }
                        AuthoredConnectorKind::ElasticFoundation { stiffness } => {
                            ConnectorKind::ElasticFoundation { stiffness }
                        }
                    },
                })
            })
            .collect::<Result<Vec<_>, AnalysisModelRealizationError>>()?,
    );
    target.interfaces.extend(
        source
            .interfaces
            .iter()
            .map(|value| {
                Ok(AnalysisInterface {
                    interface_id: value.interface_id.clone(),
                    primary: targets.region_target(&value.primary, TargetCardinality::OneOrMore)?,
                    secondary: targets
                        .region_target(&value.secondary, TargetCardinality::OneOrMore)?,
                    kind: interface_kind(&value.kind),
                })
            })
            .collect::<Result<Vec<_>, AnalysisModelRealizationError>>()?,
    );
    target
        .excitation_cases
        .extend(source.excitation_cases.iter().map(|value| {
            ExcitationCase {
                case_id: ExcitationCaseId(value.case_id.clone()),
                action_ids: value
                    .excitation_ids
                    .iter()
                    .cloned()
                    .map(ExcitationId)
                    .collect(),
                activation: value.activation.clone(),
            }
        }));
    target.excitations.extend(
        source
            .excitations
            .iter()
            .map(|value| realize_excitation(value, targets))
            .collect::<Result<Vec<_>, _>>()?,
    );
    target.initial_conditions.extend(
        source
            .initial_conditions
            .iter()
            .map(|value| realize_initial_condition(value, target, targets))
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(())
}

fn boundary_operator(value: &AuthoredBoundaryKind) -> BoundaryOperator {
    match value {
        AuthoredBoundaryKind::StructuralPressure { value } => {
            BoundaryOperator::StructuralPressure {
                value: value.clone(),
            }
        }
        AuthoredBoundaryKind::ThermalHeatFlux { value } => BoundaryOperator::ThermalHeatFlux {
            value: value.clone(),
        },
        AuthoredBoundaryKind::ThermalConvection {
            ambient_temperature,
            coefficient,
        } => BoundaryOperator::ThermalConvection {
            ambient_temperature: ambient_temperature.clone(),
            coefficient: *coefficient,
        },
        AuthoredBoundaryKind::MagneticInsulation => BoundaryOperator::MagneticInsulation,
        AuthoredBoundaryKind::AcousticRigidWall => BoundaryOperator::AcousticRigidWall,
        AuthoredBoundaryKind::AcousticRadiation => BoundaryOperator::AcousticRadiation,
        AuthoredBoundaryKind::AcousticImpedance { impedance } => {
            BoundaryOperator::AcousticImpedance {
                impedance: *impedance,
            }
        }
        AuthoredBoundaryKind::FluidOutletPressure { value } => {
            BoundaryOperator::FluidOutletPressure {
                value: value.clone(),
            }
        }
    }
}

fn interface_kind(value: &AuthoredInterfaceKind) -> AnalysisInterfaceKind {
    match value {
        AuthoredInterfaceKind::ContactFrictionless => {
            AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                friction: ContactFriction::Frictionless,
            })
        }
        AuthoredInterfaceKind::ContactCoulomb { coefficient } => {
            AnalysisInterfaceKind::Contact(ContactInterfaceModel {
                friction: ContactFriction::Coulomb {
                    coefficient: *coefficient,
                },
            })
        }
        AuthoredInterfaceKind::FluidStructureNoSlip => {
            AnalysisInterfaceKind::FluidStructure(FluidStructureInterfaceModel {
                kinematics: FluidStructureKinematics::NoSlip,
            })
        }
        AuthoredInterfaceKind::ConjugateHeatTransferPerfect => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
                transfer: ThermalInterfaceTransfer::Perfect,
            })
        }
        AuthoredInterfaceKind::ConjugateHeatTransferConductance { coefficient } => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
                transfer: ThermalInterfaceTransfer::Conductance(*coefficient),
            })
        }
        AuthoredInterfaceKind::ConjugateHeatTransferResistance { resistance } => {
            AnalysisInterfaceKind::ConjugateHeatTransfer(ConjugateHeatTransferInterfaceModel {
                transfer: ThermalInterfaceTransfer::Resistance(*resistance),
            })
        }
    }
}

fn realize_excitation(
    value: &AuthoredExcitation,
    targets: &TargetResolver<'_>,
) -> Result<ExcitationAction, AnalysisModelRealizationError> {
    let target = match &value.target {
        AuthoredExcitationTarget::Region(source) => {
            ExcitationTarget::Region(targets.region_target(source, TargetCardinality::OneOrMore)?)
        }
        AuthoredExcitationTarget::WholeModel => ExcitationTarget::WholeModel,
    };
    let excitation = AnalysisExcitation {
        excitation_id: ExcitationId(value.excitation_id.clone()),
        target,
        frame: value.frame.clone(),
        direction: value.direction,
        distribution: value.distribution,
        kinematics: value.kinematics,
        signal_id: value.signal_id.clone(),
        active_step_ids: value.active_step_ids.clone(),
        combination_group: value.combination_group.clone(),
        projection: value.projection,
    };
    macro_rules! action {
        ($variant:ident) => {
            ExcitationAction::$variant(excitation)
        };
    }
    Ok(match value.kind {
        AuthoredExcitationKind::Force => action!(Force),
        AuthoredExcitationKind::Moment => action!(Moment),
        AuthoredExcitationKind::Traction => action!(Traction),
        AuthoredExcitationKind::Pressure => action!(Pressure),
        AuthoredExcitationKind::BodyForce => action!(BodyForce),
        AuthoredExcitationKind::HeatGeneration => action!(HeatGeneration),
        AuthoredExcitationKind::HeatFlux => action!(HeatFlux),
        AuthoredExcitationKind::TemperatureField => action!(TemperatureField),
        AuthoredExcitationKind::AmbientTemperature => action!(AmbientTemperature),
        AuthoredExcitationKind::TerminalVoltage => action!(TerminalVoltage),
        AuthoredExcitationKind::ElectricCurrent => action!(ElectricCurrent),
        AuthoredExcitationKind::ElectricCharge => action!(ElectricCharge),
        AuthoredExcitationKind::CurrentDensity => action!(CurrentDensity),
        AuthoredExcitationKind::MagneticVectorPotential => action!(MagneticVectorPotential),
        AuthoredExcitationKind::AcousticPressure => action!(AcousticPressure),
        AuthoredExcitationKind::NormalVelocity => action!(NormalVelocity),
        AuthoredExcitationKind::InletVelocity => action!(InletVelocity),
        AuthoredExcitationKind::MassFlow => action!(MassFlow),
        AuthoredExcitationKind::WallMotion => action!(WallMotion),
    })
}

fn realize_initial_condition(
    value: &AuthoredInitialCondition,
    model: &AnalysisModel,
    targets: &TargetResolver<'_>,
) -> Result<InitialCondition, AnalysisModelRealizationError> {
    let activation = model
        .sequence
        .steps
        .iter()
        .filter(|step| {
            step.object_is_active(&crate::StepObjectRef::InitialCondition(
                value.initial_condition_id.clone(),
            ))
        })
        .map(|step| step.step_id.0.as_str())
        .collect::<Vec<_>>();
    let [step_id] = activation.as_slice() else {
        return Err(AnalysisModelRealizationError::InitialConditionActivation(
            value.initial_condition_id.clone(),
        ));
    };
    let target = match &value.target {
        AuthoredInitialConditionTarget::Region(source) => InitialConditionTarget::Region(
            targets.region_target(source, TargetCardinality::OneOrMore)?,
        ),
        AuthoredInitialConditionTarget::WholeModel => InitialConditionTarget::WholeModel,
    };
    macro_rules! condition {
        ($variant:ident, $initial:expr) => {
            InitialCondition::$variant(InitialConditionValue {
                initial_condition_id: value.initial_condition_id.clone(),
                target,
                frame: value.frame.clone(),
                value: $initial.clone(),
                applies_before_step_id: (*step_id).to_string(),
            })
        };
    }
    Ok(match &value.kind {
        AuthoredInitialConditionKind::Displacement(initial) => condition!(Displacement, initial),
        AuthoredInitialConditionKind::Velocity(initial) => condition!(Velocity, initial),
        AuthoredInitialConditionKind::AngularVelocity(initial) => {
            condition!(AngularVelocity, initial)
        }
        AuthoredInitialConditionKind::Temperature(initial) => condition!(Temperature, initial),
        AuthoredInitialConditionKind::ElectricPotential(initial) => {
            condition!(ElectricPotential, initial)
        }
        AuthoredInitialConditionKind::MagneticVectorPotential(initial) => {
            condition!(MagneticVectorPotential, initial)
        }
        AuthoredInitialConditionKind::AcousticPressure(initial) => {
            condition!(AcousticPressure, initial)
        }
        AuthoredInitialConditionKind::FluidPressure(initial) => condition!(FluidPressure, initial),
        AuthoredInitialConditionKind::FluidVelocity(initial) => condition!(FluidVelocity, initial),
    })
}
