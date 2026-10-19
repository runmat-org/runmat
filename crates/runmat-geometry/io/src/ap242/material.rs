use runmat_geometry_core::{
    Ap242Association, Ap242AssociationTarget, Ap242EngineeringProperty,
    Ap242EngineeringPropertyKind, Ap242EnvironmentCondition, Ap242MaterialCandidate,
    Ap242MaterialClass, Ap242PersistentId, Ap242PersistentIdRole, Ap242UnsupportedConcept,
};

use super::{p21, Ap242InspectionError};

const MATERIAL_CATEGORIES: &[&str] = &[
    "isotropic_material",
    "anisotropic_material",
    "filament_assembly",
    "discontinuous_fiber_assembly",
    "braided_assembly",
    "woven_assembly",
    "stock_core",
];

pub(super) struct MaterialExtraction {
    pub properties: Vec<Ap242EngineeringProperty>,
    pub materials: Vec<Ap242MaterialCandidate>,
    pub associations: Vec<Ap242Association>,
    pub unsupported: Vec<Ap242UnsupportedConcept>,
}

pub(super) fn extract(
    file: &p21::PhysicalFile,
    persistent_ids: &mut [Ap242PersistentId],
) -> MaterialExtraction {
    let material_products = material_products(file);
    for persistent_id in persistent_ids.iter_mut().filter(|persistent_id| {
        persistent_id
            .owner_entity_id
            .is_some_and(|owner| material_products.contains(&owner))
    }) {
        persistent_id.role = Ap242PersistentIdRole::Material;
    }
    let (properties, unsupported) = engineering_properties(file);
    let (mut materials, mut associations) =
        recommended_materials(file, persistent_ids, &properties, &material_products);
    legacy_material_designations(
        file,
        persistent_ids,
        &properties,
        &mut materials,
        &mut associations,
    );
    MaterialExtraction {
        properties,
        materials,
        associations,
        unsupported,
    }
}

fn material_products(file: &p21::PhysicalFile) -> Vec<u64> {
    let mut products = file
        .entities
        .values()
        .filter_map(|entity| {
            let category = entity.component("PRODUCT_RELATED_PRODUCT_CATEGORY")?;
            let name = category.arguments.first()?.as_string()?;
            MATERIAL_CATEGORIES
                .contains(&name.to_ascii_lowercase().as_str())
                .then(|| {
                    category
                        .arguments
                        .iter()
                        .find_map(p21::P21Value::as_list)
                        .into_iter()
                        .flatten()
                        .filter_map(p21::P21Value::as_reference)
                })
        })
        .flatten()
        .collect::<Vec<_>>();
    products.sort_unstable();
    products.dedup();
    products
}

fn engineering_properties(
    file: &p21::PhysicalFile,
) -> (Vec<Ap242EngineeringProperty>, Vec<Ap242UnsupportedConcept>) {
    let mut properties = Vec::new();
    let mut unsupported = Vec::new();
    for (relation_id, relation) in file.entities.values().filter_map(|entity| {
        entity
            .component("MATERIAL_PROPERTY_REPRESENTATION")
            .or_else(|| entity.component("FEA_MATERIAL_PROPERTY_REPRESENTATION"))
            .map(|component| (entity.id, component))
    }) {
        let property_id = relation
            .arguments
            .first()
            .and_then(p21::P21Value::as_reference);
        let representation_id = relation
            .arguments
            .get(1)
            .and_then(p21::P21Value::as_reference);
        let environment_id = relation
            .arguments
            .get(2)
            .and_then(p21::P21Value::as_reference);
        let result = property_id
            .zip(representation_id)
            .ok_or_else(|| {
                Ap242InspectionError::Inconsistent(format!(
                    "material property relation #{relation_id} is incomplete"
                ))
            })
            .and_then(|(property_id, representation_id)| {
                engineering_property(file, property_id, representation_id, environment_id)
            });
        match result {
            Ok(property) => properties.push(property),
            Err(error) => unsupported.push(Ap242UnsupportedConcept {
                entity_type: "MATERIAL_PROPERTY_REPRESENTATION".into(),
                entity_ids: vec![relation_id],
                reason: error.to_string(),
            }),
        }
    }
    (properties, unsupported)
}

fn engineering_property(
    file: &p21::PhysicalFile,
    property_id: u64,
    representation_id: u64,
    environment_id: Option<u64>,
) -> Result<Ap242EngineeringProperty, Ap242InspectionError> {
    let property = file
        .entities
        .get(&property_id)
        .and_then(|value| {
            value
                .component("MATERIAL_PROPERTY")
                .or_else(|| value.component("FEA_MATERIAL_PROPERTY"))
        })
        .ok_or_else(|| {
            Ap242InspectionError::Inconsistent(format!("missing material property #{property_id}"))
        })?;
    let name = property
        .arguments
        .first()
        .and_then(p21::P21Value::as_string)
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "material property #{property_id} has no name"
            ))
        })?;
    let kind = property_kind(name).ok_or_else(|| {
        Ap242InspectionError::Unsupported(format!(
            "material property `{name}` is not a supported constitutive quantity"
        ))
    })?;
    let representation = file.entities.get(&representation_id).ok_or_else(|| {
        Ap242InspectionError::Inconsistent(format!(
            "missing property representation #{representation_id}"
        ))
    })?;
    let item_ids = super::profile::representation_item_ids(representation)?;
    let [item_id] = item_ids.as_slice() else {
        return Err(Ap242InspectionError::Unsupported(format!(
            "property representation #{representation_id} must contain exactly one value"
        )));
    };
    let target = material_property_target(file, property_id);
    Ok(Ap242EngineeringProperty {
        property_id: format!("step_property_{property_id}"),
        kind,
        name: name.to_string(),
        source_entity_id: property_id,
        representation_entity_id: representation_id,
        value: super::units::measure(file, *item_id)?,
        specification: target.and_then(|target| material_specification(file, target)),
        environment: environment_id
            .map(|environment| environment_conditions(file, environment))
            .transpose()?
            .unwrap_or_default(),
    })
}

fn environment_conditions(
    file: &p21::PhysicalFile,
    environment_id: u64,
) -> Result<Vec<Ap242EnvironmentCondition>, Ap242InspectionError> {
    let environment = file
        .entities
        .get(&environment_id)
        .and_then(|entity| entity.component("DATA_ENVIRONMENT"))
        .ok_or_else(|| {
            Ap242InspectionError::Inconsistent(format!(
                "missing data environment #{environment_id}"
            ))
        })?;
    let definitions = environment
        .arguments
        .iter()
        .find_map(p21::P21Value::as_list)
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "data environment #{environment_id} has no typed condition elements"
            ))
        })?;
    if definitions.is_empty() {
        return Err(Ap242InspectionError::Unsupported(format!(
            "data environment #{environment_id} has no typed condition elements"
        )));
    }
    definitions
        .iter()
        .filter_map(p21::P21Value::as_reference)
        .map(|definition_id| environment_condition(file, definition_id))
        .collect()
}

fn environment_condition(
    file: &p21::PhysicalFile,
    definition_id: u64,
) -> Result<Ap242EnvironmentCondition, Ap242InspectionError> {
    let definition = file
        .entities
        .get(&definition_id)
        .and_then(|entity| entity.component("PROPERTY_DEFINITION"))
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "environment element #{definition_id} is not a property definition"
            ))
        })?;
    let name = definition
        .arguments
        .first()
        .and_then(p21::P21Value::as_string)
        .filter(|value| !value.is_empty())
        .or_else(|| {
            definition
                .arguments
                .get(1)
                .and_then(p21::P21Value::as_string)
        })
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "environment property #{definition_id} has no name"
            ))
        })?;
    let representation_id = file
        .entities
        .values()
        .filter_map(|entity| entity.component("PROPERTY_DEFINITION_REPRESENTATION"))
        .find_map(|relation| {
            (relation
                .arguments
                .first()
                .and_then(p21::P21Value::as_reference)
                == Some(definition_id))
            .then(|| relation.arguments.get(1)?.as_reference())?
        })
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "environment property #{definition_id} has no representation"
            ))
        })?;
    let representation = file.entities.get(&representation_id).ok_or_else(|| {
        Ap242InspectionError::Inconsistent(format!(
            "missing environment representation #{representation_id}"
        ))
    })?;
    let item_ids = super::profile::representation_item_ids(representation)?;
    let [item_id] = item_ids.as_slice() else {
        return Err(Ap242InspectionError::Unsupported(format!(
            "environment representation #{representation_id} must contain exactly one value"
        )));
    };
    Ok(Ap242EnvironmentCondition {
        name: name.to_string(),
        value: super::units::measure(file, *item_id)?,
    })
}

fn recommended_materials(
    file: &p21::PhysicalFile,
    persistent_ids: &[Ap242PersistentId],
    properties: &[Ap242EngineeringProperty],
    material_products: &[u64],
) -> (Vec<Ap242MaterialCandidate>, Vec<Ap242Association>) {
    let mut materials = Vec::new();
    let mut definitions = Vec::new();
    for product_id in material_products {
        let product_definitions = product_definitions(file, *product_id);
        let persistent_id = persistent_ids
            .iter()
            .find(|value| value.owner_entity_id == Some(*product_id))
            .map(|value| value.persistent_id.clone());
        let material_id = persistent_id
            .clone()
            .unwrap_or_else(|| format!("step_material_product_{product_id}"));
        let property_ids = properties
            .iter()
            .filter(|property| {
                material_property_target(file, property.source_entity_id)
                    .is_some_and(|target| product_definitions.contains(&target))
            })
            .map(|property| property.property_id.clone())
            .collect();
        let specification = product_definitions
            .iter()
            .find_map(|definition| material_specification(file, *definition));
        materials.push(Ap242MaterialCandidate {
            material_id: material_id.clone(),
            name: product_name(file, *product_id),
            source_entity_id: *product_id,
            persistent_id,
            class: Ap242MaterialClass::Physical,
            property_ids,
            specification,
        });
        definitions.extend(
            product_definitions
                .into_iter()
                .map(|definition| (definition, material_id.clone())),
        );
    }
    let associations = file
        .entities
        .values()
        .filter_map(|entity| {
            let usage = entity.component("MAKE_FROM_USAGE_OPTION")?;
            let references = usage
                .arguments
                .iter()
                .filter_map(p21::P21Value::as_reference)
                .collect::<Vec<_>>();
            let [.., relating, related] = references.as_slice() else {
                return None;
            };
            let material_id = definitions
                .iter()
                .find_map(|(definition, material)| (definition == related).then_some(material))?;
            Some(Ap242Association {
                association_id: format!("step_material_callout_{}", entity.id),
                source_entity_id: entity.id,
                material_id: material_id.clone(),
                target: Ap242AssociationTarget::ProductDefinitionEntity(*relating),
            })
        })
        .collect();
    (materials, associations)
}

fn legacy_material_designations(
    file: &p21::PhysicalFile,
    persistent_ids: &[Ap242PersistentId],
    properties: &[Ap242EngineeringProperty],
    materials: &mut Vec<Ap242MaterialCandidate>,
    associations: &mut Vec<Ap242Association>,
) {
    for entity in file.entities.values() {
        let Some(designation) = entity.component("MATERIAL_DESIGNATION") else {
            continue;
        };
        let Some(name) = designation
            .arguments
            .first()
            .and_then(p21::P21Value::as_string)
        else {
            continue;
        };
        let Some(target) = designation
            .arguments
            .get(1)
            .and_then(p21::P21Value::as_reference)
        else {
            continue;
        };
        let persistent_id = persistent_ids
            .iter()
            .find(|value| value.owner_entity_id == Some(entity.id))
            .map(|value| value.persistent_id.clone());
        let material_id = persistent_id
            .clone()
            .unwrap_or_else(|| format!("step_material_designation_{}", entity.id));
        materials.push(Ap242MaterialCandidate {
            material_id: material_id.clone(),
            name: name.to_string(),
            source_entity_id: entity.id,
            persistent_id,
            class: Ap242MaterialClass::Physical,
            property_ids: properties
                .iter()
                .filter(|property| {
                    material_property_target(file, property.source_entity_id) == Some(target)
                })
                .map(|property| property.property_id.clone())
                .collect(),
            specification: material_specification(file, target),
        });
        associations.push(Ap242Association {
            association_id: format!("step_material_designation_callout_{}", entity.id),
            source_entity_id: entity.id,
            material_id,
            target: Ap242AssociationTarget::ProductDefinitionEntity(target),
        });
    }
}

fn product_definitions(file: &p21::PhysicalFile, product_id: u64) -> Vec<u64> {
    let formations = file
        .entities
        .values()
        .filter_map(|entity| {
            let formation = entity.component("PRODUCT_DEFINITION_FORMATION")?;
            (formation
                .arguments
                .iter()
                .filter_map(p21::P21Value::as_reference)
                .next_back()
                == Some(product_id))
            .then_some(entity.id)
        })
        .collect::<Vec<_>>();
    file.entities
        .values()
        .filter_map(|entity| {
            let definition = entity.component("PRODUCT_DEFINITION")?;
            definition
                .arguments
                .iter()
                .filter_map(p21::P21Value::as_reference)
                .any(|reference| formations.contains(&reference))
                .then_some(entity.id)
        })
        .collect()
}

fn product_name(file: &p21::PhysicalFile, product_id: u64) -> String {
    file.entities
        .get(&product_id)
        .and_then(|entity| entity.component("PRODUCT"))
        .and_then(|product| {
            product
                .arguments
                .get(1)
                .and_then(p21::P21Value::as_string)
                .filter(|value| !value.is_empty())
                .or_else(|| product.arguments.first().and_then(p21::P21Value::as_string))
        })
        .unwrap_or("unnamed material")
        .to_string()
}

fn material_property_target(file: &p21::PhysicalFile, property_id: u64) -> Option<u64> {
    file.entities
        .get(&property_id)?
        .component("MATERIAL_PROPERTY")
        .or_else(|| {
            file.entities
                .get(&property_id)?
                .component("FEA_MATERIAL_PROPERTY")
        })?
        .arguments
        .iter()
        .filter_map(p21::P21Value::as_reference)
        .next_back()
}

fn material_specification(file: &p21::PhysicalFile, target: u64) -> Option<String> {
    file.entities.values().find_map(|entity| {
        let reference = entity.component("APPLIED_DOCUMENT_REFERENCE")?;
        let applies = reference
            .arguments
            .iter()
            .filter_map(p21::P21Value::as_list)
            .flatten()
            .filter_map(p21::P21Value::as_reference)
            .any(|item| item == target);
        if !applies {
            return None;
        }
        let document_id = reference
            .arguments
            .iter()
            .filter_map(p21::P21Value::as_reference)
            .find(|id| {
                file.entities
                    .get(id)
                    .is_some_and(|candidate| candidate.has("DOCUMENT"))
            })?;
        let document = file.entities.get(&document_id)?.component("DOCUMENT")?;
        document
            .arguments
            .iter()
            .filter_map(p21::P21Value::as_string)
            .find(|value| !value.is_empty())
            .map(str::to_string)
    })
}

fn property_kind(name: &str) -> Option<Ap242EngineeringPropertyKind> {
    let normalized = name
        .chars()
        .filter(|value| value.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    Some(match normalized.as_str() {
        "massdensity" | "density" => Ap242EngineeringPropertyKind::MassDensity,
        "youngsmodulus" | "elasticmodulus" => Ap242EngineeringPropertyKind::YoungsModulus,
        "poissonratio" | "poissonsratio" => Ap242EngineeringPropertyKind::PoissonRatio,
        "thermalconductivity" => Ap242EngineeringPropertyKind::ThermalConductivity,
        "specificheat" | "specificheatcapacity" => Ap242EngineeringPropertyKind::SpecificHeat,
        "thermalexpansioncoefficient" => Ap242EngineeringPropertyKind::ThermalExpansionCoefficient,
        "electricalconductivity" => Ap242EngineeringPropertyKind::ElectricalConductivity,
        "relativepermittivity" => Ap242EngineeringPropertyKind::RelativePermittivity,
        "relativepermeability" => Ap242EngineeringPropertyKind::RelativePermeability,
        "acousticimpedance" => Ap242EngineeringPropertyKind::AcousticImpedance,
        "dynamicviscosity" => Ap242EngineeringPropertyKind::DynamicViscosity,
        _ => return None,
    })
}
