use super::{diagnostic, properties::*, region_source, signals};
use crate::{
    BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_analysis_core::{
    AuthoredBoundary, AuthoredBoundaryCoefficient, AuthoredBoundaryKind, AuthoredConnector,
    AuthoredConnectorKind, AuthoredInterface, AuthoredInterfaceKind, CanonicalQuantity, Damping,
    FoundationModulus, FrictionCoefficient, HeatFlux, HeatTransferCoefficient, PrescribedValue,
    Pressure, SpecificAcousticImpedance, Stiffness, SurfaceEmissivity, Temperature,
    ThermalContactResistance,
};
use runmat_design_syntax::FeaModelDefinitionKind;

pub(super) struct LoweredModelPhysics {
    pub boundaries: Vec<AuthoredBoundary>,
    pub connectors: Vec<AuthoredConnector>,
    pub interfaces: Vec<AuthoredInterface>,
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    tables: &super::tables::ImpedanceTables,
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
                if !is_applied_excitation(statements) {
                    if let Some(value) =
                        boundary(declaration, bound, tables, name, statements, diagnostics)
                    {
                        boundaries.push(value);
                    }
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

fn is_applied_excitation(statements: &[CheckedFeaStatement]) -> bool {
    matches!(
        statements,
        [CheckedFeaStatement {
            kind: CheckedFeaStatementKind::AppliedLoad { .. },
            ..
        }]
    )
}

fn boundary(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    tables: &super::tables::ImpedanceTables,
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
        "thermal.radiation" => &["target", "ambient_temperature", "emissivity"][..],
        "acoustic.impedance" => &["target", "impedance"][..],
        "acoustic.radiation" => &["target", "center"][..],
        "magnetic.insulation" | "acoustic.rigid_wall" => &["target"][..],
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
            coefficient: boundary_coefficient::<HeatTransferCoefficient>(
                declaration,
                body,
                "coefficient",
                &[FeaType::Quantity(
                    PhysicalDimension::HEAT_TRANSFER_COEFFICIENT.into(),
                )],
                |diagnostics| {
                    required_quantity(
                        declaration,
                        body,
                        "coefficient",
                        PhysicalDimension::HEAT_TRANSFER_COEFFICIENT,
                        diagnostics,
                    )
                },
                diagnostics,
            )?,
        },
        "thermal.radiation" => AuthoredBoundaryKind::ThermalRadiation {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(required_quantity(
                declaration,
                body,
                "ambient_temperature",
                PhysicalDimension::TEMPERATURE,
                diagnostics,
            )?)),
            emissivity: boundary_coefficient::<SurfaceEmissivity>(
                declaration,
                body,
                "emissivity",
                &[
                    FeaType::Real,
                    FeaType::Quantity(PhysicalDimension::DIMENSIONLESS.into()),
                ],
                |diagnostics| required_scalar(declaration, body, "emissivity", diagnostics),
                diagnostics,
            )?,
        },
        "magnetic.insulation" => AuthoredBoundaryKind::MagneticInsulation,
        "acoustic.rigid_wall" => AuthoredBoundaryKind::AcousticRigidWall,
        // Without `center` the outgoing wave is planar (normal incidence); with a center it
        // is spherical about that point.
        "acoustic.radiation" if optional_property_assignment(body, "center").is_some() => {
            AuthoredBoundaryKind::AcousticSphericalRadiation {
                center_m: required_vector(
                    declaration,
                    body,
                    "center",
                    PhysicalDimension::LENGTH,
                    diagnostics,
                )?,
            }
        }
        "acoustic.radiation" => AuthoredBoundaryKind::AcousticRadiation,
        // `impedance = table.<name>` takes a complex, frequency-dependent law; a quantity is
        // a constant real impedance.
        "acoustic.impedance"
            if optional_property_assignment(body, "impedance").is_some_and(|value| {
                matches!(&value.kind, crate::semantic::expression::CheckedExpressionKind::Reference(path) if path.first().map(String::as_str) == Some("table"))
            }) =>
        {
            let value = optional_property_assignment(body, "impedance")?;
            let crate::semantic::expression::CheckedExpressionKind::Reference(path) = &value.kind
            else {
                return None;
            };
            let Some(law) = path
                .get(1)
                .and_then(|name| tables.get(&(declaration.module.clone(), name.clone())))
            else {
                diagnostic(
                    declaration,
                    value.span,
                    "impedance table is unknown or invalid",
                    diagnostics,
                );
                return None;
            };
            AuthoredBoundaryKind::AcousticImpedanceLaw { law: law.clone() }
        }
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
    let (supported, kind) = match kind {
        "axial_spring" => (
            &["primary", "secondary", "stiffness"][..],
            AuthoredConnectorKind::AxialSpring {
                primary: required_connector_target(
                    declaration,
                    bound,
                    body,
                    "primary",
                    diagnostics,
                )?,
                secondary: required_connector_target(
                    declaration,
                    bound,
                    body,
                    "secondary",
                    diagnostics,
                )?,
                stiffness: Stiffness::new(required_quantity(
                    declaration,
                    body,
                    "stiffness",
                    PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
                    diagnostics,
                )?),
            },
        ),
        "axial_dashpot" => (
            &["primary", "secondary", "damping"][..],
            AuthoredConnectorKind::AxialDashpot {
                primary: required_connector_target(
                    declaration,
                    bound,
                    body,
                    "primary",
                    diagnostics,
                )?,
                secondary: required_connector_target(
                    declaration,
                    bound,
                    body,
                    "secondary",
                    diagnostics,
                )?,
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
        "isotropic_ground_spring" => (
            &["target", "stiffness"][..],
            AuthoredConnectorKind::IsotropicGroundSpring {
                target: required_connector_target(declaration, bound, body, "target", diagnostics)?,
                stiffness: Stiffness::new(required_quantity(
                    declaration,
                    body,
                    "stiffness",
                    PhysicalDimension::FORCE.divide(PhysicalDimension::LENGTH)?,
                    diagnostics,
                )?),
            },
        ),
        "surface_normal_foundation" => (
            &["target", "modulus"][..],
            AuthoredConnectorKind::SurfaceNormalFoundation {
                target: required_connector_target(declaration, bound, body, "target", diagnostics)?,
                modulus: FoundationModulus::new(required_quantity(
                    declaration,
                    body,
                    "modulus",
                    PhysicalDimension::FORCE.divide(PhysicalDimension::VOLUME)?,
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
        kind,
    };
    validate_physics(declaration, value.validate(), diagnostics)?;
    Some(value)
}

fn required_connector_target(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    body: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<runmat_analysis_core::AuthoredRegionSource> {
    region_source(
        bound,
        declaration,
        required_assignment(declaration, body, name, diagnostics)?,
        diagnostics,
    )
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
                    PhysicalDimension::HEAT_TRANSFER_COEFFICIENT,
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

/// A boundary coefficient is a constant, or pure arithmetic over numeric model inputs that study
/// realization evaluates for each bound input set.
fn boundary_coefficient<Q: CanonicalQuantity>(
    declaration: &CheckedFeaDeclaration,
    body: &[CheckedFeaStatement],
    name: &str,
    input_types: &[FeaType],
    constant: impl FnOnce(&mut Vec<FeaDiagnostic>) -> Option<f64>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredBoundaryCoefficient<Q>> {
    let value = required_assignment(declaration, body, name, diagnostics)?;
    if !signals::depends_on_model_input(value) {
        return constant(diagnostics)
            .map(|value| AuthoredBoundaryCoefficient::Constant(Q::from_canonical_value(value)));
    }
    if !input_types.contains(&value.value_type) {
        diagnostic(
            declaration,
            value.span,
            format!("definition member `{name}` requires a quantity of the required dimension"),
            diagnostics,
        );
        return None;
    }
    signals::model_input_expression(declaration, value, diagnostics)
        .map(AuthoredBoundaryCoefficient::ModelInput)
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
