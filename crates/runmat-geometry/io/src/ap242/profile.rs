use runmat_geometry_core::{
    Ap242Capability, Ap242CapabilityDisposition, Ap242CapabilityRecord, Ap242ExchangeInventory,
    Ap242Parameter, Ap242ParameterAccess, Ap242PersistentId, Ap242PersistentIdRole,
    Ap242UnsupportedConcept,
};

use super::{p21, Ap242InspectionError};

pub fn inspect_ap242(source: &str) -> Result<Ap242ExchangeInventory, Ap242InspectionError> {
    let file = p21::parse(source)?;
    let (mut persistent_ids, mut unsupported) = persistent_ids(&file);
    let parameter_representations = file
        .entities
        .values()
        .filter(|entity| entity.has("SHAPE_REPRESENTATION_WITH_PARAMETERS"))
        .collect::<Vec<_>>();
    let (parameters, parameter_unsupported) =
        parameters(&file, &persistent_ids, &parameter_representations);
    unsupported.extend(parameter_unsupported);
    let material = super::material::extract(&file, &mut persistent_ids);
    let engineering_properties = material.properties;
    let materials = material.materials;
    let associations = material.associations;
    unsupported.extend(material.unsupported);
    let capabilities = capability_records(
        &file,
        persistent_ids.len(),
        &parameter_representations,
        parameters.len(),
        materials
            .iter()
            .map(|material| material.source_entity_id)
            .collect(),
        engineering_properties.len(),
    );
    let inventory = Ap242ExchangeInventory {
        schemas: file.schemas,
        capabilities,
        persistent_ids,
        parameters,
        engineering_properties,
        materials,
        associations,
        unsupported,
    };
    inventory
        .validate()
        .map_err(|error| Ap242InspectionError::Inconsistent(error.to_string()))?;
    Ok(inventory)
}

fn parameters(
    file: &p21::PhysicalFile,
    persistent_ids: &[Ap242PersistentId],
    representations: &[&p21::Entity],
) -> (Vec<Ap242Parameter>, Vec<Ap242UnsupportedConcept>) {
    let mut parameters = Vec::new();
    let mut unsupported = Vec::new();
    for representation in representations {
        let item_ids = match representation_item_ids(representation) {
            Ok(item_ids) if !item_ids.is_empty() => item_ids,
            Ok(_) => {
                unsupported.push(Ap242UnsupportedConcept {
                    entity_type: "SHAPE_REPRESENTATION_WITH_PARAMETERS".into(),
                    entity_ids: vec![representation.id],
                    reason: "parameter representation has no items".into(),
                });
                continue;
            }
            Err(error) => {
                unsupported.push(Ap242UnsupportedConcept {
                    entity_type: "SHAPE_REPRESENTATION_WITH_PARAMETERS".into(),
                    entity_ids: vec![representation.id],
                    reason: error.to_string(),
                });
                continue;
            }
        };
        for item_id in item_ids {
            match parameter(file, persistent_ids, representation.id, item_id) {
                Ok(parameter) => parameters.push(parameter),
                Err(error) => unsupported.push(Ap242UnsupportedConcept {
                    entity_type: "MEASURE_REPRESENTATION_ITEM".into(),
                    entity_ids: vec![item_id],
                    reason: error.to_string(),
                }),
            }
        }
    }
    (parameters, unsupported)
}

fn parameter(
    file: &p21::PhysicalFile,
    persistent_ids: &[Ap242PersistentId],
    representation_id: u64,
    item_id: u64,
) -> Result<Ap242Parameter, Ap242InspectionError> {
    let item = file.entities.get(&item_id).ok_or_else(|| {
        Ap242InspectionError::Inconsistent(format!("missing parameter item #{item_id}"))
    })?;
    let name = representation_item_name(item).ok_or_else(|| {
        Ap242InspectionError::Unsupported(format!("parameter item #{item_id} has no name"))
    })?;
    let persistent_id = persistent_ids
        .iter()
        .find(|value| value.owner_entity_id == Some(item_id))
        .map(|value| value.persistent_id.clone());
    Ok(Ap242Parameter {
        parameter_id: persistent_id
            .clone()
            .unwrap_or_else(|| format!("step_parameter_{item_id}")),
        name,
        source_entity_id: item_id,
        representation_entity_id: representation_id,
        owner_entity_ids: representation_owners(file, representation_id),
        persistent_id,
        value: super::units::measure(file, item_id)?,
        minimum: None,
        maximum: None,
        access: Ap242ParameterAccess::ReadOnly,
    })
}

pub(super) fn representation_item_ids(
    entity: &p21::Entity,
) -> Result<Vec<u64>, Ap242InspectionError> {
    let component = entity
        .component("REPRESENTATION")
        .or_else(|| entity.component("SHAPE_REPRESENTATION_WITH_PARAMETERS"))
        .or_else(|| {
            entity.components.iter().find(|component| {
                component
                    .arguments
                    .get(1)
                    .and_then(p21::P21Value::as_list)
                    .is_some()
            })
        })
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "representation #{} has no items component",
                entity.id
            ))
        })?;
    let values = component
        .arguments
        .get(1)
        .and_then(p21::P21Value::as_list)
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "representation #{} has no item list",
                entity.id
            ))
        })?;
    values
        .iter()
        .map(|value| {
            value.as_reference().ok_or_else(|| {
                Ap242InspectionError::Unsupported(format!(
                    "representation #{} contains a non-reference item",
                    entity.id
                ))
            })
        })
        .collect()
}

fn representation_item_name(entity: &p21::Entity) -> Option<String> {
    entity
        .component("REPRESENTATION_ITEM")
        .or_else(|| entity.component("MEASURE_REPRESENTATION_ITEM"))?
        .arguments
        .first()?
        .as_string()
        .map(str::to_string)
}

fn representation_owners(file: &p21::PhysicalFile, representation_id: u64) -> Vec<u64> {
    let mut owners = file
        .entities
        .values()
        .filter_map(|entity| {
            let relation = entity.component("PROPERTY_DEFINITION_REPRESENTATION")?;
            (relation.arguments.get(1)?.as_reference()? == representation_id)
                .then(|| relation.arguments.first()?.as_reference())?
        })
        .collect::<Vec<_>>();
    owners.sort_unstable();
    owners.dedup();
    owners
}

fn persistent_ids(
    file: &p21::PhysicalFile,
) -> (Vec<Ap242PersistentId>, Vec<Ap242UnsupportedConcept>) {
    let mut values = Vec::new();
    let mut unsupported = Vec::new();
    for entity in file
        .entities
        .values()
        .filter(|entity| entity.has("ID_ATTRIBUTE"))
    {
        match persistent_id(file, entity) {
            Ok(value) => values.push(value),
            Err(error) => unsupported.push(Ap242UnsupportedConcept {
                entity_type: "ID_ATTRIBUTE".into(),
                entity_ids: vec![entity.id],
                reason: error.to_string(),
            }),
        }
    }
    (values, unsupported)
}

fn persistent_id(
    file: &p21::PhysicalFile,
    entity: &p21::Entity,
) -> Result<Ap242PersistentId, Ap242InspectionError> {
    let component = entity.component("ID_ATTRIBUTE").ok_or_else(|| {
        Ap242InspectionError::Inconsistent(format!("missing ID attribute #{}", entity.id))
    })?;
    let owner_entity_id = component
        .arguments
        .iter()
        .find_map(|value| value.as_reference());
    let persistent_id = component
        .arguments
        .first()
        .and_then(p21::P21Value::as_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            Ap242InspectionError::Unsupported(format!(
                "ID attribute #{} has no nonempty identifier",
                entity.id
            ))
        })?;
    let owner_entity_id = owner_entity_id.ok_or_else(|| {
        Ap242InspectionError::Unsupported(format!("ID attribute #{} has no owner", entity.id))
    })?;
    Ok(Ap242PersistentId {
        persistent_id: persistent_id.to_string(),
        source_entity_id: entity.id,
        owner_entity_id: Some(owner_entity_id),
        role: file
            .entities
            .get(&owner_entity_id)
            .map(persistent_id_role)
            .unwrap_or(Ap242PersistentIdRole::RepresentationItem),
    })
}

fn persistent_id_role(owner: &p21::Entity) -> Ap242PersistentIdRole {
    if owner.has("PRODUCT") {
        Ap242PersistentIdRole::Product
    } else if owner.has("PRODUCT_DEFINITION") {
        Ap242PersistentIdRole::ProductDefinition
    } else if owner.has("NEXT_ASSEMBLY_USAGE_OCCURRENCE") || owner.has("ASSEMBLY_COMPONENT_USAGE") {
        Ap242PersistentIdRole::Occurrence
    } else if owner.has("SHAPE_ASPECT") {
        Ap242PersistentIdRole::ShapeAspect
    } else if owner.has("MATERIAL_DESIGNATION") {
        Ap242PersistentIdRole::Material
    } else if owner.has("MATERIAL_PROPERTY")
        || owner.has("FEA_MATERIAL_PROPERTY")
        || owner.has("PROPERTY_DEFINITION")
    {
        Ap242PersistentIdRole::Property
    } else {
        Ap242PersistentIdRole::RepresentationItem
    }
}

fn capability_records(
    file: &p21::PhysicalFile,
    readable_id_count: usize,
    parameter_representations: &[&p21::Entity],
    readable_parameter_count: usize,
    material_evidence: Vec<u64>,
    readable_property_count: usize,
) -> Vec<Ap242CapabilityRecord> {
    let ids = |name: &str| {
        file.entities
            .values()
            .filter(|entity| entity.has(name))
            .map(|entity| entity.id)
            .collect::<Vec<_>>()
    };
    let product_structure = [
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        "ASSEMBLY_COMPONENT_USAGE",
        "PRODUCT_DEFINITION_USAGE",
        "MAKE_FROM_USAGE_OPTION",
    ]
    .into_iter()
    .flat_map(ids)
    .collect::<Vec<_>>();
    let readable_material_count = material_evidence.len();
    let properties = [
        "MATERIAL_PROPERTY_REPRESENTATION",
        "FEA_MATERIAL_PROPERTY_REPRESENTATION",
    ]
    .into_iter()
    .flat_map(ids)
    .collect::<Vec<_>>();
    [
        (
            Ap242Capability::ProductStructure,
            product_structure,
            "product structure",
        ),
        (
            Ap242Capability::PersistentIds,
            ids("ID_ATTRIBUTE"),
            "persistent ids",
        ),
        (
            Ap242Capability::Parameters,
            parameter_representations
                .iter()
                .map(|value| value.id)
                .collect(),
            "shape parameters",
        ),
        (
            Ap242Capability::Materials,
            material_evidence,
            "physical materials",
        ),
        (
            Ap242Capability::EngineeringProperties,
            properties,
            "engineering properties",
        ),
    ]
    .into_iter()
    .map(
        |(capability, evidence_entity_ids, label)| Ap242CapabilityRecord {
            capability,
            disposition: if evidence_entity_ids.is_empty() {
                Ap242CapabilityDisposition::Absent
            } else if capability == Ap242Capability::ProductStructure
                || (capability == Ap242Capability::PersistentIds && readable_id_count > 0)
                || (capability == Ap242Capability::Parameters && readable_parameter_count > 0)
                || (capability == Ap242Capability::Materials && readable_material_count > 0)
                || (capability == Ap242Capability::EngineeringProperties
                    && readable_property_count > 0)
            {
                Ap242CapabilityDisposition::ReadOnly
            } else {
                Ap242CapabilityDisposition::PresentUnsupported
            },
            evidence_entity_ids,
            detail: format!("STEP exchange {label} inventory"),
        },
    )
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{Ap242CapabilityDisposition, Ap242PersistentIdRole, QuantityKind};

    const AP242: &str = include_str!("../../tests/fixtures/ap242_exchange_metadata.step");

    const LEGACY_DESIGNATION: &str = r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('legacy material designation fixture'),'2;1');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));
ENDSEC;
DATA;
#1=PRODUCT('part','Part','',());
#2=PRODUCT_DEFINITION_FORMATION('','',#1);
#3=PRODUCT_DEFINITION('design','',#2,#4);
#4=PRODUCT_DEFINITION_CONTEXT('part definition',#5,'design');
#5=APPLICATION_CONTEXT('mechanical design');
#6=MATERIAL_DESIGNATION('Legacy steel',#3);
#7=ID_ATTRIBUTE('legacy-steel',#6);
ENDSEC;
END-ISO-10303-21;"#;

    const MALFORMED_DISCOVERABLE_FACTS: &str = r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('malformed AP242 facts fixture'),'2;1');
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));
ENDSEC;
DATA;
#1=PRODUCT('part','Part','',());
#2=ID_ATTRIBUTE('',#1);
#3=SHAPE_REPRESENTATION_WITH_PARAMETERS('design parameters',(),#4);
#4=REPRESENTATION_CONTEXT('','');
ENDSEC;
END-ISO-10303-21;"#;

    #[test]
    fn resolves_typed_parameters_material_properties_and_associations() {
        let inventory = inspect_ap242(AP242).unwrap();
        assert_eq!(inventory.parameters.len(), 1);
        assert_eq!(
            inventory.parameters[0].parameter_id,
            "parameter-wall-thickness"
        );
        assert_eq!(inventory.parameters[0].value.kind(), QuantityKind::Length);
        assert!((inventory.parameters[0].value.canonical_value() - 0.008).abs() < 1e-12);
        assert_eq!(inventory.engineering_properties.len(), 1);
        assert_eq!(
            inventory.engineering_properties[0].value.kind(),
            QuantityKind::MassDensity
        );
        assert_eq!(
            inventory.engineering_properties[0].value.canonical_value(),
            7850.0
        );
        assert_eq!(inventory.materials.len(), 1);
        assert_eq!(inventory.materials[0].property_ids, ["step_property_37"]);
        assert_eq!(
            inventory.materials[0].specification.as_deref(),
            Some("AMS 6414")
        );
        assert_eq!(inventory.engineering_properties[0].environment.len(), 1);
        assert_eq!(
            inventory.engineering_properties[0].environment[0]
                .value
                .kind(),
            QuantityKind::Temperature
        );
        assert_eq!(
            inventory.engineering_properties[0].environment[0]
                .value
                .canonical_value(),
            293.15
        );
        assert_eq!(
            inventory
                .persistent_ids
                .iter()
                .find(|value| value.persistent_id == "material-structural-steel")
                .unwrap()
                .role,
            Ap242PersistentIdRole::Material
        );
        assert_eq!(inventory.associations.len(), 1);
        assert_eq!(
            inventory.capability(Ap242Capability::Parameters),
            Ap242CapabilityDisposition::ReadOnly
        );
        assert_eq!(
            inventory.capability(Ap242Capability::Regeneration),
            Ap242CapabilityDisposition::Absent
        );
        assert!(inventory.unsupported.is_empty());
    }

    // witness: FEA-AUTHORING-GEOMETRY-002/physical_material_graph @
    //   shared_infrastructure/mesh_independent/source_check; coverage: partial;
    //   boundary: The standard legacy material designation is retained as an explicit alternate to the
    //   recommended-practice graph; the primary graph walk is covered by the resolves_typed_parameters
    //   test.
    #[test]
    fn retains_the_standard_legacy_material_designation_as_an_explicit_alternate() {
        let inventory = inspect_ap242(LEGACY_DESIGNATION).unwrap();
        assert_eq!(inventory.materials.len(), 1);
        assert_eq!(inventory.materials[0].material_id, "legacy-steel");
        assert_eq!(inventory.materials[0].name, "Legacy steel");
        assert_eq!(inventory.associations.len(), 1);
        assert_eq!(
            inventory.capability(Ap242Capability::Materials),
            Ap242CapabilityDisposition::ReadOnly
        );
    }

    #[test]
    fn discovered_but_unreadable_exchange_facts_are_reported_not_silently_dropped() {
        let inventory = inspect_ap242(MALFORMED_DISCOVERABLE_FACTS).unwrap();
        assert_eq!(inventory.unsupported.len(), 2);
        assert_eq!(
            inventory.capability(Ap242Capability::PersistentIds),
            Ap242CapabilityDisposition::PresentUnsupported
        );
        assert_eq!(
            inventory.capability(Ap242Capability::Parameters),
            Ap242CapabilityDisposition::PresentUnsupported
        );
    }
}
