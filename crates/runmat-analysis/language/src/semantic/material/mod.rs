mod imported;
mod values;

use super::{CheckedFeaDeclaration, CheckedGeometryInspection, PhysicalDimension};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, LogicalSourcePath,
};
use runmat_analysis_core::{
    validate_material_definitions, MaterialAcousticModel, MaterialElectricalModel,
    MaterialMechanicalModel, MaterialModel, MaterialPlasticModel, MaterialThermalModel,
    MechanicalDampingModel, RayleighMassCoefficient, RayleighStiffnessCoefficient,
};
use std::collections::{BTreeMap, BTreeSet};

pub fn lower_materials(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    inspections: &BTreeMap<(LogicalSourcePath, String), CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), MaterialModel> {
    let mut result = BTreeMap::new();
    for (identity, declaration) in declarations {
        if identity.1.kind != FeaSymbolKind::Material {
            continue;
        }
        let initial_errors = diagnostics.len();
        let material = if declaration.material_source.is_some() {
            imported::lower(bound, declaration, inspections, diagnostics)
        } else {
            Some(lower_material(declaration, diagnostics))
        };
        if diagnostics.len() == initial_errors {
            let Some(material) = material else {
                continue;
            };
            if let Err(validation) = validate_material_definitions(std::slice::from_ref(&material))
            {
                error(declaration, validation.to_string(), diagnostics);
            } else {
                result.insert(identity.clone(), material);
            }
        }
    }
    result
}

fn lower_material(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> MaterialModel {
    validate_authored_shape(declaration, diagnostics);
    let name = values::string_assignment(&declaration.statements, "name")
        .unwrap_or_else(|| declaration.key.name.clone());
    let mass_density_kg_per_m3 = values::optional_quantity(
        declaration,
        &declaration.statements,
        "mass_density",
        PhysicalDimension::DENSITY,
        diagnostics,
    );
    let mechanical = values::block(&declaration.statements, "elastic").and_then(|statements| {
        Some(MaterialMechanicalModel {
            youngs_modulus_pa: values::required_quantity(
                declaration,
                statements,
                "youngs_modulus",
                PhysicalDimension::PRESSURE,
                diagnostics,
            )?,
            poisson_ratio: values::required_ratio(
                declaration,
                statements,
                "poisson_ratio",
                diagnostics,
            )?,
        })
    });
    let damping =
        values::block(&declaration.statements, "rayleigh_damping").and_then(|statements| {
            Some(MechanicalDampingModel::Rayleigh {
                mass_coefficient: RayleighMassCoefficient::new(values::required_quantity(
                    declaration,
                    statements,
                    "mass_coefficient",
                    PhysicalDimension::FREQUENCY,
                    diagnostics,
                )?),
                stiffness_coefficient: RayleighStiffnessCoefficient::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "stiffness_coefficient",
                        PhysicalDimension::TIME,
                        diagnostics,
                    )?,
                ),
            })
        });
    let thermal = values::block(&declaration.statements, "thermal").and_then(|statements| {
        Some(MaterialThermalModel {
            reference_temperature_k: values::required_quantity(
                declaration,
                statements,
                "reference_temperature",
                PhysicalDimension::TEMPERATURE,
                diagnostics,
            )?,
            modulus_temp_coeff_per_k: values::required_quantity(
                declaration,
                statements,
                "modulus_temperature_coefficient",
                inverse_temperature(),
                diagnostics,
            )?,
            conductivity_w_per_mk: values::required_quantity(
                declaration,
                statements,
                "conductivity",
                PhysicalDimension::THERMAL_CONDUCTIVITY,
                diagnostics,
            )?,
            specific_heat_j_per_kgk: values::required_quantity(
                declaration,
                statements,
                "specific_heat",
                PhysicalDimension::SPECIFIC_HEAT,
                diagnostics,
            )?,
            expansion_coefficient_per_k: values::required_quantity(
                declaration,
                statements,
                "expansion_coefficient",
                inverse_temperature(),
                diagnostics,
            )?,
        })
    });
    let acoustic = values::block(&declaration.statements, "acoustic").and_then(|statements| {
        Some(MaterialAcousticModel {
            speed_of_sound_m_per_s: values::required_quantity(
                declaration,
                statements,
                "speed_of_sound",
                PhysicalDimension::VELOCITY,
                diagnostics,
            )?,
            damping_ratio: values::required_ratio(
                declaration,
                statements,
                "damping_ratio",
                diagnostics,
            )?,
        })
    });
    let electrical = values::block(&declaration.statements, "electrical").and_then(|statements| {
        Some(MaterialElectricalModel {
            reference_temperature_k: values::required_quantity(
                declaration,
                statements,
                "reference_temperature",
                PhysicalDimension::TEMPERATURE,
                diagnostics,
            )?,
            conductivity_s_per_m: values::required_quantity(
                declaration,
                statements,
                "conductivity",
                PhysicalDimension::ELECTRICAL_CONDUCTIVITY,
                diagnostics,
            )?,
            resistive_heating_coefficient: values::required_quantity(
                declaration,
                statements,
                "resistive_heating_coefficient",
                inverse_temperature(),
                diagnostics,
            )?,
            relative_permittivity: values::required_ratio(
                declaration,
                statements,
                "relative_permittivity",
                diagnostics,
            )?,
            relative_permeability: values::required_ratio(
                declaration,
                statements,
                "relative_permeability",
                diagnostics,
            )?,
            conductivity_frequency_response: Vec::new(),
        })
    });
    let plastic = values::block(&declaration.statements, "plastic").and_then(|statements| {
        Some(MaterialPlasticModel {
            yield_strain: values::required_ratio(
                declaration,
                statements,
                "yield_strain",
                diagnostics,
            )?,
            hardening_modulus_ratio: values::required_ratio(
                declaration,
                statements,
                "hardening_modulus_ratio",
                diagnostics,
            )?,
            saturation_exponent: values::required_ratio(
                declaration,
                statements,
                "saturation_exponent",
                diagnostics,
            )?,
        })
    });
    MaterialModel {
        material_id: declaration.key.name.clone(),
        name,
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3,
        mechanical,
        damping,
        thermal,
        acoustic,
        electrical,
        plastic,
    }
}

fn validate_authored_shape(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    const ASSIGNMENTS: &[&str] = &["name", "mass_density"];
    const BLOCKS: &[(&str, &[&str])] = &[
        ("elastic", &["youngs_modulus", "poisson_ratio"]),
        (
            "rayleigh_damping",
            &["mass_coefficient", "stiffness_coefficient"],
        ),
        (
            "thermal",
            &[
                "reference_temperature",
                "modulus_temperature_coefficient",
                "conductivity",
                "specific_heat",
                "expansion_coefficient",
            ],
        ),
        ("acoustic", &["speed_of_sound", "damping_ratio"]),
        (
            "electrical",
            &[
                "reference_temperature",
                "conductivity",
                "resistive_heating_coefficient",
                "relative_permittivity",
                "relative_permeability",
            ],
        ),
        (
            "plastic",
            &[
                "yield_strain",
                "hardening_modulus_ratio",
                "saturation_exponent",
            ],
        ),
    ];
    let mut seen = BTreeSet::new();
    for statement in &declaration.statements {
        let member = match &statement.kind {
            crate::CheckedFeaStatementKind::Assignment { target, .. }
                if target.len() == 1 && ASSIGNMENTS.contains(&target[0].as_str()) =>
            {
                Some(target[0].as_str())
            }
            crate::CheckedFeaStatementKind::Block {
                header, statements, ..
            } => BLOCKS
                .iter()
                .find(|(name, _)| *name == header.trim())
                .map(|(name, members)| {
                    validate_capability_shape(declaration, name, members, statements, diagnostics);
                    *name
                }),
            _ => None,
        };
        let Some(member) = member else {
            material_shape_error(
                declaration,
                statement.span,
                "authored material contains an unknown or misplaced member",
                diagnostics,
            );
            continue;
        };
        if !seen.insert(member) {
            material_shape_error(
                declaration,
                statement.span,
                format!("authored material member `{member}` is declared more than once"),
                diagnostics,
            );
        }
    }
}

fn validate_capability_shape(
    declaration: &CheckedFeaDeclaration,
    capability: &str,
    allowed: &[&str],
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut seen = BTreeSet::new();
    for statement in statements {
        let member = match &statement.kind {
            crate::CheckedFeaStatementKind::Assignment { target, .. }
                if target.len() == 1 && allowed.contains(&target[0].as_str()) =>
            {
                Some(target[0].as_str())
            }
            _ => None,
        };
        let Some(member) = member else {
            material_shape_error(
                declaration,
                statement.span,
                format!("material capability `{capability}` contains an unknown member"),
                diagnostics,
            );
            continue;
        };
        if !seen.insert(member) {
            material_shape_error(
                declaration,
                statement.span,
                format!(
                    "material capability `{capability}` member `{member}` is declared more than once"
                ),
                diagnostics,
            );
        }
    }
}

fn material_shape_error(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.material_member",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}

fn inverse_temperature() -> PhysicalDimension {
    PhysicalDimension::DIMENSIONLESS
        .divide(PhysicalDimension::TEMPERATURE)
        .expect("bounded SI exponent")
}

fn error(
    declaration: &CheckedFeaDeclaration,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.material",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span: declaration.span,
        }),
    ));
}
