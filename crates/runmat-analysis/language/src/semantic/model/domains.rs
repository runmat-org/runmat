//! Explicit model-domain constants; no material or procedure-derived defaults.
use super::{diagnostic, properties};
use crate::{CheckedFeaDeclaration, CheckedFeaStatementKind, FeaDiagnostic, PhysicalDimension};
use runmat_analysis_core::{
    ElectromagneticDomain, Frequency, MagnetostaticFormulation, StaticElectricalEquation,
    Temperature, ThermalDomain, ThermoMechanicalDomain,
};
use std::collections::BTreeSet;

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> (
    Option<ThermalDomain>,
    Option<ThermoMechanicalDomain>,
    Option<ElectromagneticDomain>,
) {
    let mut thermal = None;
    let mut thermo_mechanical = None;
    let mut electromagnetic = None;
    let mut seen = BTreeSet::new();
    for statement in &declaration.statements {
        let CheckedFeaStatementKind::Block {
            header,
            header_span,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        let Some(family @ ("thermal" | "thermomechanical" | "electromagnetic")) =
            header.split_whitespace().next()
        else {
            continue;
        };
        if header.trim() != family {
            diagnostic(
                declaration,
                *header_span,
                "model domains require an unqualified domain block",
                diagnostics,
            );
            continue;
        }
        if !seen.insert(family) {
            diagnostic(
                declaration,
                *header_span,
                format!("duplicate model domain `{family}`"),
                diagnostics,
            );
            continue;
        }
        if family == "electromagnetic" {
            let Some(value) =
                properties::required_enum(declaration, statements, "equation", diagnostics)
            else {
                continue;
            };
            let assignments: &[&str] = if value == "harmonic_maxwell" {
                &["equation", "reference_frequency"]
            } else {
                &["equation"]
            };
            properties::validate_assignments(declaration, statements, assignments, diagnostics);
            let domain = match value {
                "harmonic_maxwell" => {
                    let Some(hertz) = properties::required_quantity(
                        declaration,
                        statements,
                        "reference_frequency",
                        PhysicalDimension::FREQUENCY,
                        diagnostics,
                    ) else {
                        continue;
                    };
                    if !hertz.is_finite() || hertz <= 0.0 {
                        diagnostic(
                            declaration,
                            *header_span,
                            "harmonic Maxwell reference frequency must be finite and positive",
                            diagnostics,
                        );
                        continue;
                    }
                    ElectromagneticDomain::Harmonic {
                        reference_frequency: Frequency::new(hertz),
                    }
                }
                "electrostatics" => ElectromagneticDomain::Static {
                    equation: StaticElectricalEquation::Electrostatics,
                },
                "steady_conduction" => ElectromagneticDomain::Static {
                    equation: StaticElectricalEquation::SteadyConduction,
                },
                "magnetostatics" => ElectromagneticDomain::Magnetostatic {
                    formulation: MagnetostaticFormulation::MagneticVectorPotential,
                },
                _ => {
                    diagnostic(
                        declaration,
                        *header_span,
                        "electromagnetic equation must be electrostatics, steady_conduction, magnetostatics, or harmonic_maxwell",
                        diagnostics,
                    );
                    continue;
                }
            };
            electromagnetic = Some(domain);
            continue;
        }
        properties::validate_assignments(
            declaration,
            statements,
            &["reference_temperature"],
            diagnostics,
        );
        let Some(reference) = properties::required_quantity(
            declaration,
            statements,
            "reference_temperature",
            PhysicalDimension::TEMPERATURE,
            diagnostics,
        ) else {
            continue;
        };
        if !reference.is_finite() || reference < 0.0 {
            diagnostic(
                declaration,
                *header_span,
                "reference_temperature must be a finite nonnegative absolute temperature",
                diagnostics,
            );
            continue;
        }
        let reference_temperature = Temperature::new(reference);
        match family {
            "thermal" => {
                thermal = Some(ThermalDomain {
                    reference_temperature,
                })
            }
            "thermomechanical" => {
                thermo_mechanical = Some(ThermoMechanicalDomain {
                    reference_temperature,
                })
            }
            _ => unreachable!(),
        }
    }
    (thermal, thermo_mechanical, electromagnetic)
}
