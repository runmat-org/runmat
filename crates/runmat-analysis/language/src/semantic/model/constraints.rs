use super::{diagnostic, properties::*, region_source};
use crate::{
    BoundFeaProgram, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, PhysicalDimension,
};
use runmat_analysis_core::{
    AuthoredConstraint, AuthoredConstraintKind, AuthoredConstraintTarget,
    AuthoredTargetCardinality, ElectricCurrent, ElectricPotential, MagneticVectorPotential,
    PhysicalQuantityKind, PhysicalSignal, Pressure, Temperature, Velocity,
};
use runmat_fea_syntax::FeaModelDefinitionKind;

mod motion;
mod relational;
mod values;

use values::{prescribed_value, prescribed_vector};

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredConstraint> {
    declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::ModelDefinition {
                kind: FeaModelDefinitionKind::Constraint,
                name,
                statements,
            } = &statement.kind
            else {
                return None;
            };
            constraint(declaration, bound, signals, name, statements, diagnostics)
        })
        .collect()
}

fn constraint(
    declaration: &CheckedFeaDeclaration,
    bound: &BoundFeaProgram,
    signals: &[PhysicalSignal],
    id: &str,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraint> {
    let (kind, body) = sole_kind_block(declaration, statements, "constraint", diagnostics)?;
    if relational::recognizes(kind) {
        let kind = relational::lower(declaration, bound, signals, kind, body, diagnostics)?;
        return finish(declaration, id, kind, diagnostics);
    }
    let target_expression = required_assignment(declaration, body, "target", diagnostics)?;
    let target = AuthoredConstraintTarget {
        region: region_source(bound, declaration, target_expression, diagnostics)?,
        cardinality: AuthoredTargetCardinality::OneOrMore,
    };
    let supported = if requires_value(kind) {
        &["target", "value"][..]
    } else {
        &["target"][..]
    };
    validate_assignments(declaration, body, supported, diagnostics);
    let kind = match kind {
        "structural.fixed" => AuthoredConstraintKind::StructuralFixed { target },
        "structural.symmetry" => AuthoredConstraintKind::StructuralSymmetry { target },
        "structural.antisymmetry" => AuthoredConstraintKind::StructuralAntisymmetry { target },
        "thermal.temperature" => AuthoredConstraintKind::ThermalTemperature {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (
                    PhysicalDimension::TEMPERATURE,
                    PhysicalQuantityKind::Temperature,
                ),
                signals,
                Temperature::new,
                diagnostics,
            )?,
        },
        "electrical.potential" => AuthoredConstraintKind::ElectricalPotential {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (
                    PhysicalDimension::ELECTRIC_POTENTIAL,
                    PhysicalQuantityKind::ElectricPotential,
                ),
                signals,
                ElectricPotential::new,
                diagnostics,
            )?,
        },
        "electrical.perfect_conductor" => {
            AuthoredConstraintKind::ElectricPerfectConductor { target }
        }
        "electrical.equipotential" => AuthoredConstraintKind::ElectricalEquipotential { target },
        "electrical.floating_conductor" => {
            AuthoredConstraintKind::ElectricalFloatingConductor { target }
        }
        "electrical.integral_current" => AuthoredConstraintKind::ElectricalIntegralCurrent {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (
                    PhysicalDimension::ELECTRIC_CURRENT,
                    PhysicalQuantityKind::ElectricCurrent,
                ),
                signals,
                ElectricCurrent::new,
                diagnostics,
            )?,
        },
        "electrical.potential_reference" => {
            AuthoredConstraintKind::ElectricalPotentialReference { target }
        }
        "magnetic.vector_potential_ground" => {
            AuthoredConstraintKind::MagneticVectorPotentialGround { target }
        }
        "magnetic.coulomb_gauge" => AuthoredConstraintKind::MagneticCoulombGauge { target },
        "acoustic.pressure" => AuthoredConstraintKind::AcousticPressure {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (PhysicalDimension::PRESSURE, PhysicalQuantityKind::Pressure),
                signals,
                Pressure::new,
                diagnostics,
            )?,
        },
        "fluid.velocity" => AuthoredConstraintKind::FluidVelocity {
            target,
            value: prescribed_vector(
                declaration,
                body,
                "value",
                (PhysicalDimension::VELOCITY, PhysicalQuantityKind::Velocity),
                signals,
                Velocity::new,
                diagnostics,
            )?,
        },
        "fluid.moving_wall" => AuthoredConstraintKind::FluidMovingWall {
            target,
            velocity: prescribed_vector(
                declaration,
                body,
                "value",
                (PhysicalDimension::VELOCITY, PhysicalQuantityKind::Velocity),
                signals,
                Velocity::new,
                diagnostics,
            )?,
        },
        "fluid.no_slip_wall" => AuthoredConstraintKind::FluidNoSlipWall { target },
        "fluid.slip_wall" => AuthoredConstraintKind::FluidSlipWall { target },
        "fluid.symmetry" => AuthoredConstraintKind::FluidSymmetry { target },
        "fluid.pressure_reference" => AuthoredConstraintKind::FluidPressureReference {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (PhysicalDimension::PRESSURE, PhysicalQuantityKind::Pressure),
                signals,
                Pressure::new,
                diagnostics,
            )?,
        },
        "magnetic.vector_potential" => AuthoredConstraintKind::MagneticVectorPotential {
            target,
            value: prescribed_value(
                declaration,
                body,
                "value",
                (
                    PhysicalDimension::MAGNETIC_FLUX_DENSITY.multiply(PhysicalDimension::LENGTH)?,
                    PhysicalQuantityKind::MagneticVectorPotential,
                ),
                signals,
                MagneticVectorPotential::new,
                diagnostics,
            )?,
        },
        _ => {
            diagnostic(
                declaration,
                declaration.span,
                crate::semantic::pending::with_pending(
                    format!("unsupported essential constraint `{kind}`"),
                    crate::semantic::pending::constraint(&kind),
                ),
                diagnostics,
            );
            return None;
        }
    };
    finish(declaration, id, kind, diagnostics)
}

fn finish(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    kind: AuthoredConstraintKind,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredConstraint> {
    let constraint = AuthoredConstraint {
        constraint_id: id.into(),
        kind,
    };
    if let Err(error) = constraint.validate() {
        diagnostic(
            declaration,
            declaration.span,
            error.to_string(),
            diagnostics,
        );
        return None;
    }
    Some(constraint)
}

fn requires_value(kind: &str) -> bool {
    matches!(
        kind,
        "thermal.temperature"
            | "electrical.potential"
            | "electrical.integral_current"
            | "magnetic.vector_potential"
            | "acoustic.pressure"
            | "fluid.velocity"
            | "fluid.moving_wall"
            | "fluid.pressure_reference"
    )
}
