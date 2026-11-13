use runmat_geometry_core::{
    Ap242Change, Ap242ChangeSet, Ap242ExchangeInventory, Ap242ExportDisposition,
    Ap242ExportItemReport, Ap242ExportReport,
};

use super::Ap242InspectionError;

/// Validates a typed AP242 change set against the admitted source inventory.
///
/// The current adapter deliberately produces sidecar evidence only. No STEP
/// output digest is published because this crate does not yet own a qualified
/// AP242 writer/regenerator for the normalized concepts.
pub fn evaluate_change_set(
    inventory: &Ap242ExchangeInventory,
    change_set: &Ap242ChangeSet,
) -> Result<Ap242ExportReport, Ap242InspectionError> {
    inventory
        .validate()
        .map_err(|error| Ap242InspectionError::Inconsistent(error.to_string()))?;
    change_set
        .validate()
        .map_err(|error| Ap242InspectionError::Inconsistent(error.to_string()))?;
    let items = change_set
        .changes
        .iter()
        .enumerate()
        .map(|(index, change)| {
            validate_change(inventory, change)?;
            Ok(Ap242ExportItemReport {
                change_index: index as u64,
                disposition: Ap242ExportDisposition::SidecarOnly,
                detail: "validated against the admitted source; no qualified AP242 writer or regenerative evaluator is available".to_string(),
            })
        })
        .collect::<Result<Vec<_>, Ap242InspectionError>>()?;
    let report = Ap242ExportReport {
        source_digest: change_set.source_digest,
        output_digest: None,
        source_preserved: true,
        lossless: false,
        items,
    };
    report
        .validate()
        .map_err(|error| Ap242InspectionError::Inconsistent(error.to_string()))?;
    Ok(report)
}

fn validate_change(
    inventory: &Ap242ExchangeInventory,
    change: &Ap242Change,
) -> Result<(), Ap242InspectionError> {
    match change {
        Ap242Change::SetParameter {
            parameter_id,
            previous,
            ..
        } => {
            let source = inventory
                .parameters
                .iter()
                .find(|parameter| parameter.parameter_id == *parameter_id)
                .ok_or_else(|| missing("parameter", parameter_id))?;
            if source.value != *previous {
                return Err(stale("parameter", parameter_id));
            }
        }
        Ap242Change::SetEngineeringProperty {
            property_id,
            previous,
            ..
        } => {
            let source = inventory
                .engineering_properties
                .iter()
                .find(|property| property.property_id == *property_id)
                .ok_or_else(|| missing("engineering property", property_id))?;
            if source.value != *previous {
                return Err(stale("engineering property", property_id));
            }
        }
        Ap242Change::SetMaterialAssociation {
            association_id,
            material_id,
        } => {
            if !inventory
                .associations
                .iter()
                .any(|association| association.association_id == *association_id)
            {
                return Err(missing("material association", association_id));
            }
            if !inventory
                .materials
                .iter()
                .any(|material| material.material_id == *material_id)
            {
                return Err(missing("material", material_id));
            }
        }
    }
    Ok(())
}

fn missing(kind: &str, id: &str) -> Ap242InspectionError {
    Ap242InspectionError::Inconsistent(format!("AP242 change references unknown {kind} `{id}`"))
}

fn stale(kind: &str, id: &str) -> Ap242InspectionError {
    Ap242InspectionError::Inconsistent(format!(
        "AP242 change has stale previous value for {kind} `{id}`"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        Ap242Measure, Ap242Parameter, Ap242ParameterAccess, GeometryDigest, QuantityKind,
    };

    // witness: FEA-AUTHORING-GEOMETRY-002/truthful_exchange_export @
    //   shared_infrastructure/mesh_independent/source_check; coverage: partial;
    //   boundary: Validated changes are reported as sidecar-only without claiming a CAD export.
    #[test]
    fn reports_validated_changes_as_sidecar_only_without_claiming_export() {
        let inventory = inventory();
        let previous = inventory.parameters[0].value.clone();
        let change_set = Ap242ChangeSet {
            source_digest: GeometryDigest::from_bytes([7; 32]),
            changes: vec![Ap242Change::SetParameter {
                parameter_id: "wall".to_string(),
                previous: previous.clone(),
                value: previous.with_source_value(10.0).unwrap(),
            }],
        };
        let report = evaluate_change_set(&inventory, &change_set).unwrap();
        assert_eq!(report.output_digest, None);
        assert!(report.source_preserved);
        assert!(!report.lossless);
        assert_eq!(
            report.items[0].disposition,
            Ap242ExportDisposition::SidecarOnly
        );
    }

    #[test]
    fn rejects_a_change_set_with_stale_source_evidence() {
        let inventory = inventory();
        let mut previous = inventory.parameters[0].value.clone();
        previous = previous.with_source_value(99.0).unwrap();
        let change_set = Ap242ChangeSet {
            source_digest: GeometryDigest::from_bytes([7; 32]),
            changes: vec![Ap242Change::SetParameter {
                parameter_id: "wall".to_string(),
                previous: previous.clone(),
                value: previous,
            }],
        };
        assert!(evaluate_change_set(&inventory, &change_set)
            .unwrap_err()
            .to_string()
            .contains("stale previous value"));
    }

    fn inventory() -> Ap242ExchangeInventory {
        Ap242ExchangeInventory {
            schemas: vec!["AP242".to_string()],
            parameters: vec![Ap242Parameter {
                parameter_id: "wall".to_string(),
                name: "WallThickness".to_string(),
                source_entity_id: 11,
                representation_entity_id: 12,
                owner_entity_ids: vec![2],
                persistent_id: Some("wall".to_string()),
                value: Ap242Measure::try_new(
                    QuantityKind::Length,
                    8.0,
                    *runmat_geometry_core::LengthUnit::Millimeter.definition(),
                    "mm",
                    runmat_geometry_core::physical::UnitAttribution::FileDeclaration {
                        context: "STEP fixture #11".into(),
                    },
                )
                .unwrap(),
                minimum: None,
                maximum: None,
                access: Ap242ParameterAccess::ReadOnly,
            }],
            ..Ap242ExchangeInventory::default()
        }
    }
}
