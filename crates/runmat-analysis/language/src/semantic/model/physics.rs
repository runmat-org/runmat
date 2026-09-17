use super::{diagnostic, properties::*, region_source};
use crate::{
    BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, PhysicalDimension,
};
use runmat_analysis_core::{
    AuthoredBoundary, AuthoredBoundaryKind, AuthoredConnector, AuthoredConnectorKind,
    AuthoredInterface, AuthoredInterfaceKind, Damping, FrictionCoefficient, HeatFlux,
    HeatTransferCoefficient, PrescribedValue, Pressure, SpecificAcousticImpedance, Stiffness,
    Temperature, ThermalContactResistance,
};
use runmat_parser::fea::FeaModelDefinitionKind;

pub(super) struct LoweredModelPhysics {
    pub boundaries: Vec<AuthoredBoundary>,
    pub connectors: Vec<AuthoredConnector>,
    pub interfaces: Vec<AuthoredInterface>,
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> LoweredModelPhysics {
    let mut boundaries = Vec::new();
    let mut connectors = Vec::new();
    let mut interfaces = Vec::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::ModelDefinition {
            kind,
            name,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        match kind {
            FeaModelDefinitionKind::Load => {
                if let Some(value) = boundary(declaration, bound, name, statements, diagnostics) {
                    boundaries.push(value);
                }
            }
            FeaModelDefinitionKind::Connector => {
                if let Some(value) = connector(declaration, bound, name, statements, diagnostics) {
                    connectors.push(value);
                }
            }
            FeaModelDefinitionKind::Interface => {
                if let Some(value) = interface(declaration, bound, name, statements, diagnostics) {
                    interfaces.push(value);
                }
            }
            _ => {}
        }
    }
    LoweredModelPhysics {
        boundaries,
        connectors,
        interfaces,
    }
}

fn boundary(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredBoundary> {
    let (kind, body) = sole_kind_block(declaration, statements, "load", diagnostics)?;
    let target = region_source(
        bound,
        declaration,
        required_assignment(declaration, body, "target", diagnostics)?,
        diagnostics,
    )?;
    let supported = match kind {
        "structural.pressure" | "thermal.heat_flux" | "fluid.outlet_pressure" => {
            &["target", "value"][..]
        }
        "thermal.convection" => &["target", "ambient_temperature", "coefficient"][..],
        "acoustic.impedance" => &["target", "impedance"][..],
        "magnetic.insulation" | "acoustic.rigid_wall" | "acoustic.radiation" => &["target"][..],
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported natural load or boundary `{kind}`"),
                diagnostics,
            );
            return None;
        }
    };
    validate_assignments(declaration, body, supported, diagnostics);
    let kind = match kind {
        "structural.pressure" => AuthoredBoundaryKind::StructuralPressure {
            value: PrescribedValue::Constant(Pressure::new(required_quantity(
                declaration,
                body,
                "value",
                PhysicalDimension::PRESSURE,
                diagnostics,
            )?)),
        },
        "thermal.heat_flux" => AuthoredBoundaryKind::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(required_quantity(
                declaration,
                body,
                "value",
                PhysicalDimension::HEAT_FLUX,
                diagnostics,
            )?)),
        },
        "thermal.convection" => AuthoredBoundaryKind::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(required_quantity(
                declaration,
                body,
                "ambient_temperature",
                PhysicalDimension::TEMPERATURE,
                diagnostics,
            )?)),
            coefficient: HeatTransferCoefficient::new(required_quantity(
                declaration,
                body,
                "coefficient",
                heat_transfer_coefficient()?,
                diagnostics,
            )?),
        },
        "magnetic.insulation" => AuthoredBoundaryKind::MagneticInsulation,
        "acoustic.rigid_wall" => AuthoredBoundaryKind::AcousticRigidWall,
        "acoustic.radiation" => AuthoredBoundaryKind::AcousticRadiation,
        "acoustic.impedance" => AuthoredBoundaryKind::AcousticImpedance {
            impedance: SpecificAcousticImpedance::new(required_quantity(
                declaration,
                body,
                "impedance",
                specific_acoustic_impedance()?,
                diagnostics,
            )?),
        },
        "fluid.outlet_pressure" => AuthoredBoundaryKind::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(required_quantity(
                declaration,
                body,
                "value",
                PhysicalDimension::PRESSURE,
                diagnostics,
            )?)),
        },
        _ => return None,
    };
    let value = AuthoredBoundary {
        boundary_id: id.into(),
        target,
        kind,
    };
    validate_physics(declaration, value.validate(), diagnostics)?;
    Some(value)
}

fn connector(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConnector> {
    let (kind, body) = sole_kind_block(declaration, statements, "connector", diagnostics)?;
    let primary = region_source(
        bound,
        declaration,
        required_assignment(declaration, body, "primary", diagnostics)?,
        diagnostics,
    )?;
    let secondary = match optional_assignment(body, "secondary") {
        Some(value) => Some(region_source(bound, declaration, value, diagnostics)?),
        None => None,
    };
    let (supported, kind) = match kind {
        "translational_spring" => (
            &["primary", "secondary", "stiffness"][..],
            AuthoredConnectorKind::TranslationalSpring {
                stiffness: Stiffness::new(required_quantity(
                    declaration,
                    body,
                    "stiffness",
                    PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
                    diagnostics,
                )?),
            },
        ),
        "translational_dashpot" => (
            &["primary", "secondary", "damping"][..],
            AuthoredConnectorKind::TranslationalDashpot {
                damping: Damping::new(required_quantity(
                    declaration,
                    body,
                    "damping",
                    PhysicalDimension::FORCE
                        .multiply(PhysicalDimension::TIME)?
                        .divide(PhysicalDimension::LENGTH)?,
                    diagnostics,
                )?),
            },
        ),
        "elastic_foundation" => (
            &["primary", "stiffness"][..],
            AuthoredConnectorKind::ElasticFoundation {
                stiffness: Stiffness::new(required_quantity(
                    declaration,
                    body,
                    "stiffness",
                    PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
                    diagnostics,
                )?),
            },
        ),
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported connector formulation `{kind}`"),
                diagnostics,
            );
            return None;
        }
    };
    validate_assignments(declaration, body, supported, diagnostics);
    let value = AuthoredConnector {
        connector_id: id.into(),
        primary,
        secondary,
        kind,
    };
    validate_physics(declaration, value.validate(), diagnostics)?;
    Some(value)
}

fn interface(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredInterface> {
    let (kind, body) = sole_kind_block(declaration, statements, "interface", diagnostics)?;
    let primary = region_source(
        bound,
        declaration,
        required_assignment(declaration, body, "primary", diagnostics)?,
        diagnostics,
    )?;
    let secondary = region_source(
        bound,
        declaration,
        required_assignment(declaration, body, "secondary", diagnostics)?,
        diagnostics,
    )?;
    let (supported, kind) = match kind {
        "contact.frictionless" => (
            &["primary", "secondary"][..],
            AuthoredInterfaceKind::ContactFrictionless,
        ),
        "contact.coulomb" => (
            &["primary", "secondary", "coefficient"][..],
            AuthoredInterfaceKind::ContactCoulomb {
                coefficient: FrictionCoefficient::new(required_scalar(
                    declaration,
                    body,
                    "coefficient",
                    diagnostics,
                )?),
            },
        ),
        "fluid_structure.no_slip" => (
            &["primary", "secondary"][..],
            AuthoredInterfaceKind::FluidStructureNoSlip,
        ),
        "conjugate_heat_transfer.perfect" => (
            &["primary", "secondary"][..],
            AuthoredInterfaceKind::ConjugateHeatTransferPerfect,
        ),
        "conjugate_heat_transfer.conductance" => (
            &["primary", "secondary", "coefficient"][..],
            AuthoredInterfaceKind::ConjugateHeatTransferConductance {
                coefficient: HeatTransferCoefficient::new(required_quantity(
                    declaration,
                    body,
                    "coefficient",
                    heat_transfer_coefficient()?,
                    diagnostics,
                )?),
            },
        ),
        "conjugate_heat_transfer.resistance" => (
            &["primary", "secondary", "resistance"][..],
            AuthoredInterfaceKind::ConjugateHeatTransferResistance {
                resistance: ThermalContactResistance::new(required_quantity(
                    declaration,
                    body,
                    "resistance",
                    thermal_contact_resistance()?,
                    diagnostics,
                )?),
            },
        ),
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                format!("unsupported interface formulation `{kind}`"),
                diagnostics,
            );
            return None;
        }
    };
    validate_assignments(declaration, body, supported, diagnostics);
    let value = AuthoredInterface {
        interface_id: id.into(),
        primary,
        secondary,
        kind,
    };
    validate_physics(declaration, value.validate(), diagnostics)?;
    Some(value)
}

fn optional_assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a crate::CheckedExpression> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

fn validate_physics<E: std::fmt::Display>(
    declaration: &CheckedFeaDeclaration,
    result: Result<(), E>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    if let Err(error) = result {
        diagnostic(
            declaration,
            declaration.span,
            error.to_string(),
            diagnostics,
        );
        None
    } else {
        Some(())
    }
}

fn heat_transfer_coefficient() -> Option<PhysicalDimension> {
    PhysicalDimension::POWER
        .divide(PhysicalDimension::AREA)?
        .divide(PhysicalDimension::TEMPERATURE)
}

fn specific_acoustic_impedance() -> Option<PhysicalDimension> {
    PhysicalDimension::PRESSURE
        .multiply(PhysicalDimension::TIME)?
        .divide(PhysicalDimension::LENGTH)
}

fn thermal_contact_resistance() -> Option<PhysicalDimension> {
    PhysicalDimension::AREA
        .multiply(PhysicalDimension::TEMPERATURE)?
        .divide(PhysicalDimension::POWER)
}
