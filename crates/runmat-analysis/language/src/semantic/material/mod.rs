mod imported;
mod values;

use super::{CheckedFeaDeclaration, CheckedGeometryInspection, PhysicalDimension};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, LogicalSourcePath,
};
use runmat_analysis_core::{
    validate_material_definitions, MaterialAcousticModel, MaterialElasticityModel,
    MaterialElectricalModel, MaterialFluidModel, MaterialMechanicalModel, MaterialModel,
    MaterialOrthotropicLaminaModel, MaterialPlasticModel, MaterialThermalModel,
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
    let fluid = match (
        values::block(&declaration.statements, "fluid"),
        values::block(&declaration.statements, "porous"),
    ) {
        (Some(_), Some(_)) => {
            error(
                declaration,
                "a material is either a fluid or a porous medium the fluid flows through, not both",
                diagnostics,
            );
            None
        }
        (Some(statements), None) => fluid_law(declaration, statements, diagnostics),
        (None, Some(statements)) => porous_law(declaration, statements, diagnostics),
        (None, None) => None,
    };
    let mass_density_kg_per_m3 = values::optional_quantity(
        declaration,
        &declaration.statements,
        "mass_density",
        PhysicalDimension::DENSITY,
        diagnostics,
    );
    let isotropic = values::block(&declaration.statements, "elastic").and_then(|statements| {
        Some(MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "youngs_modulus",
                    PhysicalDimension::PRESSURE,
                    diagnostics,
                )?,
            ),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(values::required_ratio(
                declaration,
                statements,
                "poisson_ratio",
                diagnostics,
            )?),
        })
    });
    let orthotropic =
        values::block(&declaration.statements, "orthotropic_lamina").and_then(|statements| {
            Some(MaterialOrthotropicLaminaModel {
                longitudinal_modulus_pa: runmat_types::physical::YoungsModulus::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "longitudinal_modulus",
                        PhysicalDimension::PRESSURE,
                        diagnostics,
                    )?,
                ),
                transverse_modulus_pa: runmat_types::physical::YoungsModulus::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "transverse_modulus",
                        PhysicalDimension::PRESSURE,
                        diagnostics,
                    )?,
                ),
                major_poisson_ratio: runmat_types::physical::PoissonRatio::new(
                    values::required_ratio(
                        declaration,
                        statements,
                        "major_poisson_ratio",
                        diagnostics,
                    )?,
                ),
                in_plane_shear_modulus_pa: runmat_types::physical::ShearModulus::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "in_plane_shear_modulus",
                        PhysicalDimension::PRESSURE,
                        diagnostics,
                    )?,
                ),
                transverse_shear_modulus_13_pa: runmat_types::physical::ShearModulus::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "transverse_shear_modulus_13",
                        PhysicalDimension::PRESSURE,
                        diagnostics,
                    )?,
                ),
                transverse_shear_modulus_23_pa: runmat_types::physical::ShearModulus::new(
                    values::required_quantity(
                        declaration,
                        statements,
                        "transverse_shear_modulus_23",
                        PhysicalDimension::PRESSURE,
                        diagnostics,
                    )?,
                ),
            })
        });
    let elasticity = match (isotropic, orthotropic) {
        (Some(value), None) => Some(MaterialElasticityModel::IsotropicLinear(value)),
        (None, Some(value)) => Some(MaterialElasticityModel::OrthotropicLamina(value)),
        (None, None) => None,
        (Some(_), Some(_)) => {
            error(
                declaration,
                "material must declare at most one elasticity law",
                diagnostics,
            );
            None
        }
    };
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
            reference_temperature_k: runmat_types::physical::Temperature::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "reference_temperature",
                    PhysicalDimension::TEMPERATURE,
                    diagnostics,
                )?,
            ),
            modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "modulus_temperature_coefficient",
                    inverse_temperature(),
                    diagnostics,
                )?,
            ),
            conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "conductivity",
                    PhysicalDimension::THERMAL_CONDUCTIVITY,
                    diagnostics,
                )?,
            ),
            specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "specific_heat",
                    PhysicalDimension::SPECIFIC_HEAT,
                    diagnostics,
                )?,
            ),
            expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "expansion_coefficient",
                    inverse_temperature(),
                    diagnostics,
                )?,
            ),
        })
    });
    let acoustic = values::block(&declaration.statements, "acoustic").and_then(|statements| {
        Some(MaterialAcousticModel {
            speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "speed_of_sound",
                    PhysicalDimension::VELOCITY,
                    diagnostics,
                )?,
            ),
            damping_ratio: runmat_types::physical::DampingRatio::new(values::required_ratio(
                declaration,
                statements,
                "damping_ratio",
                diagnostics,
            )?),
        })
    });
    let electrical = values::block(&declaration.statements, "electrical").and_then(|statements| {
        Some(MaterialElectricalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "reference_temperature",
                    PhysicalDimension::TEMPERATURE,
                    diagnostics,
                )?,
            ),
            conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "conductivity",
                    PhysicalDimension::ELECTRICAL_CONDUCTIVITY,
                    diagnostics,
                )?,
            ),
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "resistive_heating_coefficient",
                    inverse_temperature(),
                    diagnostics,
                )?,
            ),
            relative_permittivity: runmat_types::physical::RelativePermittivity::new(
                values::required_ratio(
                    declaration,
                    statements,
                    "relative_permittivity",
                    diagnostics,
                )?,
            ),
            relative_permeability: runmat_types::physical::RelativePermeability::new(
                values::required_ratio(
                    declaration,
                    statements,
                    "relative_permeability",
                    diagnostics,
                )?,
            ),
            conductivity_frequency_response: Vec::new(),
        })
    });
    let plastic = values::block(&declaration.statements, "plastic").and_then(|statements| {
        values::require_closed_law(
            declaration,
            statements,
            "j2_isotropic_linear_hardening",
            diagnostics,
        )?;
        Some(MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(values::required_quantity(
                declaration,
                statements,
                "yield_stress",
                PhysicalDimension::PRESSURE,
                diagnostics,
            )?),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(
                values::required_quantity(
                    declaration,
                    statements,
                    "isotropic_hardening_modulus",
                    PhysicalDimension::PRESSURE,
                    diagnostics,
                )?,
            ),
        })
    });
    MaterialModel {
        material_id: declaration.key.name.clone(),
        name,
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: (mass_density_kg_per_m3)
            .map(runmat_types::physical::MassDensity::new),
        elasticity,
        damping,
        thermal,
        acoustic,
        fluid,
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
            "orthotropic_lamina",
            &[
                "longitudinal_modulus",
                "transverse_modulus",
                "major_poisson_ratio",
                "in_plane_shear_modulus",
                "transverse_shear_modulus_13",
                "transverse_shear_modulus_23",
            ],
        ),
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
            "porous",
            &["law", "permeability", "inertial_coefficient", "porosity"],
        ),
        (
            "fluid",
            &[
                "law",
                "dynamic_viscosity",
                "consistency",
                "power_index",
                "minimum_shear_rate",
                "zero_shear_viscosity",
                "infinite_shear_viscosity",
                "relaxation_time",
                "transition_exponent",
                "time_constant",
                "rate_index",
                "yield_stress",
                "plastic_viscosity",
                "regularization",
            ],
        ),
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
            &["law", "yield_stress", "isotropic_hardening_modulus"],
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
            let pending = match &statement.kind {
                crate::CheckedFeaStatementKind::Block { header, .. } => header
                    .split_whitespace()
                    .next()
                    .and_then(crate::semantic::pending::material_member),
                _ => None,
            };
            material_shape_error(
                declaration,
                statement.span,
                crate::semantic::pending::with_pending(
                    "authored material contains an unknown or misplaced member",
                    pending,
                ),
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

/// The fluid law: `incompressible_newtonian`, or a generalized-Newtonian (shear-rate dependent)
/// viscosity law. Consistency is in SI (Pa·s^n), written as a plain number or, for n = 1, in Pa·s.
fn fluid_law(
    declaration: &CheckedFeaDeclaration,
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MaterialFluidModel> {
    use runmat_analysis_core::GeneralizedNewtonianLaw as Law;
    let viscosity_dimension = PhysicalDimension::new([-1, 1, -1, 0, 0, 0, 0, 0]);
    let rate_dimension = PhysicalDimension::FREQUENCY;
    let law = match values::assignment(statements, "law").map(|value| &value.kind) {
        Some(crate::CheckedExpressionKind::Enum(name)) => name.clone(),
        _ => {
            error(
                declaration,
                "fluid law must be explicit: incompressible_newtonian, power_law, carreau, \
                 carreau_yasuda, cross, bingham, herschel_bulkley or casson",
                diagnostics,
            );
            return None;
        }
    };
    let quantity =
        |name: &str, dimension: PhysicalDimension, diagnostics: &mut Vec<FeaDiagnostic>| {
            values::required_quantity(declaration, statements, name, dimension, diagnostics)
        };
    let ratio = |name: &str, diagnostics: &mut Vec<FeaDiagnostic>| {
        values::required_ratio(declaration, statements, name, diagnostics)
    };
    let optional_rate = |name: &str, default: f64, diagnostics: &mut Vec<FeaDiagnostic>| {
        values::optional_quantity(declaration, statements, name, rate_dimension, diagnostics)
            .unwrap_or(default)
    };
    let regularization = |diagnostics: &mut Vec<FeaDiagnostic>| {
        values::optional_quantity(
            declaration,
            statements,
            "regularization",
            PhysicalDimension::TIME,
            diagnostics,
        )
        .unwrap_or(DEFAULT_YIELD_REGULARIZATION_S)
    };
    let consistency = |index: f64, diagnostics: &mut Vec<FeaDiagnostic>| {
        let value = values::assignment(statements, "consistency");
        match value.map(|value| (&value.value_type, value)) {
            Some((crate::FeaType::Integer | crate::FeaType::Real, _)) => {
                values::required_ratio(declaration, statements, "consistency", diagnostics)
            }
            Some((_, value)) if index == 1.0 => values::typed_quantity(value, viscosity_dimension)
                .or_else(|| {
                    error(
                        declaration,
                        "consistency must be in Pa*s^n (SI)",
                        diagnostics,
                    );
                    None
                }),
            _ => {
                error(
                    declaration,
                    crate::semantic::pending::with_pending(
                        "power-law consistency is required in SI units (Pa*s^n) as a number",
                        Some(crate::semantic::pending::FLUID_NON_NEWTONIAN),
                    ),
                    diagnostics,
                );
                None
            }
        }
    };
    let law = match law.as_str() {
        "incompressible_newtonian" => {
            return Some(MaterialFluidModel::IncompressibleNewtonian {
                dynamic_viscosity: runmat_analysis_core::DynamicViscosity::new(quantity(
                    "dynamic_viscosity",
                    viscosity_dimension,
                    diagnostics,
                )?),
            })
        }
        "power_law" => {
            let power_index = ratio("power_index", diagnostics)?;
            Law::PowerLaw {
                consistency_pa_s_n: consistency(power_index, diagnostics)?,
                power_index,
                minimum_shear_rate_per_s: optional_rate(
                    "minimum_shear_rate",
                    DEFAULT_MINIMUM_SHEAR_RATE_PER_S,
                    diagnostics,
                ),
            }
        }
        "carreau" | "carreau_yasuda" => Law::CarreauYasuda {
            zero_shear_viscosity_pa_s: quantity(
                "zero_shear_viscosity",
                viscosity_dimension,
                diagnostics,
            )?,
            infinite_shear_viscosity_pa_s: quantity(
                "infinite_shear_viscosity",
                viscosity_dimension,
                diagnostics,
            )?,
            relaxation_time_s: quantity("relaxation_time", PhysicalDimension::TIME, diagnostics)?,
            power_index: ratio("power_index", diagnostics)?,
            transition_exponent: if law == "carreau" {
                2.0
            } else {
                ratio("transition_exponent", diagnostics)?
            },
        },
        "cross" => Law::Cross {
            zero_shear_viscosity_pa_s: quantity(
                "zero_shear_viscosity",
                viscosity_dimension,
                diagnostics,
            )?,
            infinite_shear_viscosity_pa_s: quantity(
                "infinite_shear_viscosity",
                viscosity_dimension,
                diagnostics,
            )?,
            time_constant_s: quantity("time_constant", PhysicalDimension::TIME, diagnostics)?,
            rate_index: ratio("rate_index", diagnostics)?,
        },
        "bingham" => Law::HerschelBulkley {
            yield_stress_pa: quantity("yield_stress", PhysicalDimension::PRESSURE, diagnostics)?,
            consistency_pa_s_n: quantity("plastic_viscosity", viscosity_dimension, diagnostics)?,
            power_index: 1.0,
            regularization_s: regularization(diagnostics),
            minimum_shear_rate_per_s: optional_rate(
                "minimum_shear_rate",
                DEFAULT_MINIMUM_SHEAR_RATE_PER_S,
                diagnostics,
            ),
        },
        "herschel_bulkley" => {
            let power_index = ratio("power_index", diagnostics)?;
            Law::HerschelBulkley {
                yield_stress_pa: quantity(
                    "yield_stress",
                    PhysicalDimension::PRESSURE,
                    diagnostics,
                )?,
                consistency_pa_s_n: consistency(power_index, diagnostics)?,
                power_index,
                regularization_s: regularization(diagnostics),
                minimum_shear_rate_per_s: optional_rate(
                    "minimum_shear_rate",
                    DEFAULT_MINIMUM_SHEAR_RATE_PER_S,
                    diagnostics,
                ),
            }
        }
        "casson" => Law::Casson {
            yield_stress_pa: quantity("yield_stress", PhysicalDimension::PRESSURE, diagnostics)?,
            plastic_viscosity_pa_s: quantity(
                "plastic_viscosity",
                viscosity_dimension,
                diagnostics,
            )?,
            regularization_s: regularization(diagnostics),
            minimum_shear_rate_per_s: optional_rate(
                "minimum_shear_rate",
                DEFAULT_MINIMUM_SHEAR_RATE_PER_S,
                diagnostics,
            ),
        },
        other => {
            error(
                declaration,
                format!("unknown fluid law `{other}`"),
                diagnostics,
            );
            return None;
        }
    };
    if !law.is_valid() {
        error(
            declaration,
            "fluid viscosity law parameters are out of range",
            diagnostics,
        );
        return None;
    }
    Some(MaterialFluidModel::GeneralizedNewtonian { law })
}

/// A porous zone: `darcy` (viscous resistance μ/K) or `darcy_forchheimer` (adds the inertial
/// resistance ½ρC₂|u|). The flowing fluid is the one assigned to the surrounding fluid regions.
fn porous_law(
    declaration: &CheckedFeaDeclaration,
    statements: &[crate::CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MaterialFluidModel> {
    let law = match values::assignment(statements, "law").map(|value| &value.kind) {
        Some(crate::CheckedExpressionKind::Enum(name)) => name.clone(),
        _ => {
            error(
                declaration,
                "porous law must be explicit: darcy or darcy_forchheimer",
                diagnostics,
            );
            return None;
        }
    };
    let permeability_m2 = values::required_quantity(
        declaration,
        statements,
        "permeability",
        PhysicalDimension::AREA,
        diagnostics,
    )?;
    let inertial_coefficient_per_m = match law.as_str() {
        "darcy" => {
            if values::assignment(statements, "inertial_coefficient").is_some() {
                error(
                    declaration,
                    "a darcy porous law has no inertial coefficient; use darcy_forchheimer",
                    diagnostics,
                );
                return None;
            }
            0.0
        }
        "darcy_forchheimer" => values::required_quantity(
            declaration,
            statements,
            "inertial_coefficient",
            PhysicalDimension::DIMENSIONLESS
                .divide(PhysicalDimension::LENGTH)
                .expect("bounded SI exponent"),
            diagnostics,
        )?,
        other => {
            error(
                declaration,
                format!("unknown porous law `{other}`"),
                diagnostics,
            );
            return None;
        }
    };
    let porosity = values::required_ratio(declaration, statements, "porosity", diagnostics)?;
    Some(MaterialFluidModel::PorousMedium {
        law: runmat_analysis_core::PorousMediumLaw {
            permeability_m2,
            inertial_coefficient_per_m,
            porosity,
        },
    })
}

/// Shear rate below which a shear-thinning or yield-stress law holds its viscosity constant.
const DEFAULT_MINIMUM_SHEAR_RATE_PER_S: f64 = 1e-3;
/// Papanastasiou regularization time of a yield-stress law.
const DEFAULT_YIELD_REGULARIZATION_S: f64 = 1000.0;
