use runmat_types::PhysicalDimension;
use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    ImportedCadEnvironmentCondition, ImportedCadMaterialAssociation,
    ImportedCadMaterialAssociationTarget, ImportedCadMaterialProperty,
    ImportedCadMaterialPropertyKind, ImportedCadMaterialProvenance, MaterialMechanicalModel,
    MaterialModel, MaterialProvenance,
};
use runmat_geometry_core::{
    Ap242AssociationTarget, Ap242EngineeringProperty, Ap242EngineeringPropertyKind,
    Ap242ExchangeInventory, Ap242MaterialClass, QuantityKind,
};

use super::{error, CheckedFeaDeclaration, CheckedGeometryInspection};
use crate::{
    BoundFeaProgram, CheckedExpressionKind, CheckedFeaStatementKind, FeaDiagnostic, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin, LogicalSourcePath,
};

pub(super) fn lower(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    inspections: &BTreeMap<(LogicalSourcePath, String), CheckedGeometryInspection>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<MaterialModel> {
    let source = declaration.material_source.as_ref()?;
    let (geometry_module, geometry_id) = resolve_geometry(bound, declaration, &source.geometry)
        .or_else(|| {
            error(
                declaration,
                format!("unknown geometry source `{}`", source.geometry),
                diagnostics,
            );
            None
        })?;
    let inventory = inspections
        .get(&(geometry_module.clone(), geometry_id.clone()))
        .and_then(|inspection| inspection.facts.exchange.as_ref())
        .or_else(|| {
            error(
                declaration,
                "CAD material adoption requires a normalized AP242 exchange inventory",
                diagnostics,
            );
            None
        })?;
    let candidates = inventory
        .materials
        .iter()
        .filter(|candidate| {
            candidate.material_id == source.material_id
                || candidate.persistent_id.as_deref() == Some(source.material_id.as_str())
        })
        .collect::<Vec<_>>();
    let [candidate] = candidates.as_slice() else {
        error(
            declaration,
            if candidates.is_empty() {
                format!("CAD material `{}` does not exist", source.material_id)
            } else {
                format!("CAD material `{}` is ambiguous", source.material_id)
            },
            diagnostics,
        );
        return None;
    };
    if candidate.class != Ap242MaterialClass::Physical {
        error(
            declaration,
            "visual CAD materials cannot become physical analysis materials",
            diagnostics,
        );
        return None;
    }
    if !conflict_policy_is_reject(declaration) {
        error(
            declaration,
            "CAD material adoption requires `on_conflict = reject`",
            diagnostics,
        );
        return None;
    }
    if !validate_adoption_surface(declaration, diagnostics) {
        return None;
    }
    let requested = requested_properties(declaration, diagnostics);
    if requested.is_empty() {
        error(
            declaration,
            "CAD material adoption requires at least one `use <property>` command",
            diagnostics,
        );
        return None;
    }
    let mut adopted = Vec::new();
    for requested_kind in requested {
        let properties = candidate
            .property_ids
            .iter()
            .filter_map(|id| {
                inventory
                    .engineering_properties
                    .iter()
                    .find(|property| property.property_id == *id)
            })
            .filter(|property| property_kind(property.kind) == requested_kind)
            .collect::<Vec<_>>();
        let [property] = properties.as_slice() else {
            error(
                declaration,
                if properties.is_empty() {
                    format!(
                        "CAD material `{}` does not provide requested property `{}`",
                        source.material_id,
                        property_name(requested_kind)
                    )
                } else {
                    format!(
                        "CAD material property `{}` is ambiguous",
                        property_name(requested_kind)
                    )
                },
                diagnostics,
            );
            continue;
        };
        adopted.push((*property).clone());
    }
    if !environment_requirements_hold(declaration, &adopted, diagnostics) {
        return None;
    }
    let value = |kind| {
        adopted
            .iter()
            .find(|property| property_kind(property.kind) == kind)
            .map(|property| property.value.canonical_value())
    };
    let elasticity = match (
        value(ImportedCadMaterialPropertyKind::YoungsModulus),
        value(ImportedCadMaterialPropertyKind::PoissonRatio),
    ) {
        (Some(youngs_modulus_pa), Some(poisson_ratio)) => Some(
            runmat_analysis_core::MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(
                        youngs_modulus_pa,
                    ),
                    poisson_ratio: runmat_types::physical::PoissonRatio::new(poisson_ratio),
                },
            ),
        ),
        (None, None) => None,
        _ => {
            error(
                declaration,
                "linear-elastic CAD adoption requires both `elastic.youngs_modulus` and `elastic.poisson_ratio`",
                diagnostics,
            );
            None
        }
    };
    Some(MaterialModel {
        fluid: None,
        material_id: declaration.key.name.clone(),
        name: candidate.name.clone(),
        provenance: MaterialProvenance::ImportedCad(ImportedCadMaterialProvenance {
            geometry_module_path: geometry_module.as_str().to_string(),
            geometry_id,
            source_material_id: candidate.material_id.clone(),
            source_material_name: candidate.name.clone(),
            source_entity_id: candidate.source_entity_id,
            persistent_id: candidate.persistent_id.clone(),
            specification: candidate.specification.clone(),
            associations: associations(inventory, &candidate.material_id),
            adopted_properties: adopted.iter().map(provenance).collect(),
        }),
        mass_density_kg_per_m3: (value(ImportedCadMaterialPropertyKind::MassDensity))
            .map(runmat_types::physical::MassDensity::new),
        elasticity,
        damping: None,
        thermal: None,
        acoustic: None,
        electrical: None,
        plastic: None,
    })
}

fn resolve_geometry(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    name: &str,
) -> Option<(LogicalSourcePath, String)> {
    let key = FeaSymbolKey {
        kind: FeaSymbolKind::Geometry,
        name: name.to_string(),
    };
    let symbol = bound.symbols.get(&declaration.module)?.get(&key)?;
    Some(match &symbol.origin {
        FeaSymbolOrigin::Local => (declaration.module.clone(), name.to_string()),
        FeaSymbolOrigin::Import {
            target_module,
            target,
        } => (target_module.clone(), target.name.clone()),
        FeaSymbolOrigin::Function => return None,
    })
}

fn requested_properties(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeSet<ImportedCadMaterialPropertyKind> {
    let mut result = BTreeSet::new();
    for command in declaration.statements.iter().filter_map(|statement| {
        let CheckedFeaStatementKind::Use(property) = &statement.kind else {
            return None;
        };
        Some(property.join("."))
    }) {
        let Some(kind) = parse_property(&command) else {
            error(
                declaration,
                format!("unsupported CAD material property `{command}`"),
                diagnostics,
            );
            continue;
        };
        if !result.insert(kind) {
            error(
                declaration,
                format!("CAD material property `{command}` is requested more than once"),
                diagnostics,
            );
        }
    }
    result
}

fn conflict_policy_is_reject(declaration: &CheckedFeaDeclaration) -> bool {
    declaration.statements.iter().any(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return false;
        };
        target.as_slice() == ["on_conflict"]
            && matches!(&value.kind, CheckedExpressionKind::Enum(value) if value == "reject")
    })
}

fn validate_adoption_surface(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> bool {
    let mut valid = true;
    for statement in &declaration.statements {
        let supported = match &statement.kind {
            CheckedFeaStatementKind::Use(_) => true,
            CheckedFeaStatementKind::Requirement { target, .. } => {
                target.as_slice() == ["operating_temperature"]
            }
            CheckedFeaStatementKind::Assignment { target, .. } => {
                target.as_slice() == ["on_conflict"]
            }
            _ => false,
        };
        if !supported {
            error(
                declaration,
                "CAD material adoption accepts only typed `use`, operating-temperature `require`, and `on_conflict = reject` declarations",
                diagnostics,
            );
            valid = false;
        }
    }
    valid
}

fn environment_requirements_hold(
    declaration: &CheckedFeaDeclaration,
    properties: &[Ap242EngineeringProperty],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> bool {
    let requirements = declaration
        .statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Requirement { target, value } = &statement.kind else {
                return None;
            };
            (target.as_slice() == ["operating_temperature"]).then_some(value)
        })
        .collect::<Vec<_>>();
    let mut valid = true;
    if requirements.len() > 1 {
        error(
            declaration,
            "operating temperature is required more than once",
            diagnostics,
        );
        valid = false;
    }
    for requirement in requirements {
        let Some(required) =
            super::values::typed_quantity(requirement, PhysicalDimension::TEMPERATURE)
        else {
            error(
                declaration,
                "operating temperature requires a temperature quantity",
                diagnostics,
            );
            valid = false;
            continue;
        };
        let matches = properties.iter().all(|property| {
            property.environment.iter().any(|condition| {
                condition.value.kind() == QuantityKind::Temperature
                    && (condition.value.canonical_value() - required).abs() <= 1e-9
            })
        });
        if !matches {
            error(
                declaration,
                "adopted CAD properties do not carry the required operating temperature",
                diagnostics,
            );
            valid = false;
        }
    }
    valid
}

fn parse_property(value: &str) -> Option<ImportedCadMaterialPropertyKind> {
    Some(match value {
        "mass_density" => ImportedCadMaterialPropertyKind::MassDensity,
        "elastic.youngs_modulus" => ImportedCadMaterialPropertyKind::YoungsModulus,
        "elastic.poisson_ratio" => ImportedCadMaterialPropertyKind::PoissonRatio,
        "thermal.conductivity" => ImportedCadMaterialPropertyKind::ThermalConductivity,
        "thermal.specific_heat" => ImportedCadMaterialPropertyKind::SpecificHeat,
        "thermal.expansion_coefficient" => {
            ImportedCadMaterialPropertyKind::ThermalExpansionCoefficient
        }
        "electrical.conductivity" => ImportedCadMaterialPropertyKind::ElectricalConductivity,
        "electrical.relative_permittivity" => ImportedCadMaterialPropertyKind::RelativePermittivity,
        "electrical.relative_permeability" => ImportedCadMaterialPropertyKind::RelativePermeability,
        "acoustic.impedance" => ImportedCadMaterialPropertyKind::AcousticImpedance,
        "fluid.dynamic_viscosity" => ImportedCadMaterialPropertyKind::DynamicViscosity,
        _ => return None,
    })
}

fn property_kind(value: Ap242EngineeringPropertyKind) -> ImportedCadMaterialPropertyKind {
    match value {
        Ap242EngineeringPropertyKind::MassDensity => ImportedCadMaterialPropertyKind::MassDensity,
        Ap242EngineeringPropertyKind::YoungsModulus => {
            ImportedCadMaterialPropertyKind::YoungsModulus
        }
        Ap242EngineeringPropertyKind::PoissonRatio => ImportedCadMaterialPropertyKind::PoissonRatio,
        Ap242EngineeringPropertyKind::ThermalConductivity => {
            ImportedCadMaterialPropertyKind::ThermalConductivity
        }
        Ap242EngineeringPropertyKind::SpecificHeat => ImportedCadMaterialPropertyKind::SpecificHeat,
        Ap242EngineeringPropertyKind::ThermalExpansionCoefficient => {
            ImportedCadMaterialPropertyKind::ThermalExpansionCoefficient
        }
        Ap242EngineeringPropertyKind::ElectricalConductivity => {
            ImportedCadMaterialPropertyKind::ElectricalConductivity
        }
        Ap242EngineeringPropertyKind::RelativePermittivity => {
            ImportedCadMaterialPropertyKind::RelativePermittivity
        }
        Ap242EngineeringPropertyKind::RelativePermeability => {
            ImportedCadMaterialPropertyKind::RelativePermeability
        }
        Ap242EngineeringPropertyKind::AcousticImpedance => {
            ImportedCadMaterialPropertyKind::AcousticImpedance
        }
        Ap242EngineeringPropertyKind::DynamicViscosity => {
            ImportedCadMaterialPropertyKind::DynamicViscosity
        }
    }
}

fn property_name(value: ImportedCadMaterialPropertyKind) -> &'static str {
    match value {
        ImportedCadMaterialPropertyKind::MassDensity => "mass_density",
        ImportedCadMaterialPropertyKind::YoungsModulus => "elastic.youngs_modulus",
        ImportedCadMaterialPropertyKind::PoissonRatio => "elastic.poisson_ratio",
        ImportedCadMaterialPropertyKind::ThermalConductivity => "thermal.conductivity",
        ImportedCadMaterialPropertyKind::SpecificHeat => "thermal.specific_heat",
        ImportedCadMaterialPropertyKind::ThermalExpansionCoefficient => {
            "thermal.expansion_coefficient"
        }
        ImportedCadMaterialPropertyKind::ElectricalConductivity => "electrical.conductivity",
        ImportedCadMaterialPropertyKind::RelativePermittivity => "electrical.relative_permittivity",
        ImportedCadMaterialPropertyKind::RelativePermeability => "electrical.relative_permeability",
        ImportedCadMaterialPropertyKind::AcousticImpedance => "acoustic.impedance",
        ImportedCadMaterialPropertyKind::DynamicViscosity => "fluid.dynamic_viscosity",
    }
}

fn provenance(property: &Ap242EngineeringProperty) -> ImportedCadMaterialProperty {
    ImportedCadMaterialProperty {
        property: property_kind(property.kind),
        source_property_id: property.property_id.clone(),
        source_name: property.name.clone(),
        source_entity_id: property.source_entity_id,
        representation_entity_id: property.representation_entity_id,
        value: property.value.clone(),
        specification: property.specification.clone(),
        environment: property
            .environment
            .iter()
            .map(|condition| ImportedCadEnvironmentCondition {
                name: condition.name.clone(),
                value: condition.value.clone(),
            })
            .collect(),
    }
}

fn associations(
    inventory: &Ap242ExchangeInventory,
    material_id: &str,
) -> Vec<ImportedCadMaterialAssociation> {
    inventory
        .associations
        .iter()
        .filter(|association| association.material_id == material_id)
        .map(|association| ImportedCadMaterialAssociation {
            association_id: association.association_id.clone(),
            source_entity_id: association.source_entity_id,
            target: match &association.target {
                Ap242AssociationTarget::ProductEntity(value) => {
                    ImportedCadMaterialAssociationTarget::ProductEntity(*value)
                }
                Ap242AssociationTarget::ProductDefinitionEntity(value) => {
                    ImportedCadMaterialAssociationTarget::ProductDefinitionEntity(*value)
                }
                Ap242AssociationTarget::OccurrenceEntity(value) => {
                    ImportedCadMaterialAssociationTarget::OccurrenceEntity(*value)
                }
                Ap242AssociationTarget::ShapeAspectEntity(value) => {
                    ImportedCadMaterialAssociationTarget::ShapeAspectEntity(*value)
                }
                Ap242AssociationTarget::PersistentId(value) => {
                    ImportedCadMaterialAssociationTarget::PersistentId(value.clone())
                }
            },
        })
        .collect()
}
