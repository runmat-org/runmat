//! Borrow original declarations; admit the shared exact identity inventory before validation.
use super::*;

pub(super) fn declaration_inventory_resident_bytes(count: usize) -> Result<u64, ObservationError> {
    crate::topology::storage::slots::<(crate::results::field::RecoveredFieldIdentity, ())>(count)
        .map_err(|error| ObservationError::Topology(error.into()))
}

pub(super) fn validate_field_catalog<'a>(
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition> + Clone,
    maximum_resident_bytes: u64,
) -> Result<(), ObservationError> {
    let count = fields.clone().count();
    let bytes = declaration_inventory_resident_bytes(count)?;
    if bytes > maximum_resident_bytes {
        return Err(ObservationError::Limit(
            "recovered-field declaration inventory",
        ));
    }
    let identities = crate::topology::storage::SortedValues::derive(
        fields.clone().map(|field| (field.identity(), ())),
        count,
    )
    .map_err(|error| match error {
        crate::topology::storage::TopologyStorageError::Index => ObservationError::Invalid(
            "recovered-field declarations require unique identities".into(),
        ),
        _ => ObservationError::Topology(error.into()),
    })?;
    for field in fields {
        if !matches!(
            field.location(),
            FieldLocation::Node
                | FieldLocation::Element
                | FieldLocation::Cell
                | FieldLocation::Face
                | FieldLocation::IntegrationPoint
                | FieldLocation::Connector
                | FieldLocation::Constraint
                | FieldLocation::Edge
        ) {
            return Err(ObservationError::Unsupported(
                "this evaluator accepts only recovered nodal, element, integration-point, connector, and constraint-group fields",
            ));
        }
        shape_component_count(field.value_shape())?;
    }
    drop(identities);
    Ok(())
}

pub(super) fn bind_field<'a>(
    spec: &ObservationSpec,
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition> + Clone,
) -> Result<&'a RecoveredFieldDefinition, ObservationError> {
    match bind_declared_field(spec, fields.clone()) {
        Err(ObservationError::Field { .. })
            if spec.recovery == RecoveryPolicy::Native
                || spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation =>
        {
            bind_located_field(spec, fields)
        }
        result => result,
    }
}

/// A field named at a location the solver provides only by transfer (nodal von Mises is the
/// volume-weighted average of element values) binds to that single declared transfer. The
/// bound declaration, and so the observation's identity, records how the values were produced.
fn bind_located_field<'a>(
    spec: &ObservationSpec,
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition>,
) -> Result<&'a RecoveredFieldDefinition, ObservationError> {
    let mut matches = fields.filter(|field| {
        matches!(
            field.derivation(),
            RecoveredFieldDerivation::LocationTransfer { .. }
        ) && field.location() == spec.source_location
            && field.quantity() == spec.quantity
            && field.value_shape() == &spec.value_shape
    });
    match (matches.next(), matches.next()) {
        (Some(field), None) => Ok(field),
        (None, _) => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason: "no declaration matches quantity, location, and value shape".into(),
        }),
        _ => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason:
                "multiple declarations match; recovery must provide an unambiguous field catalog"
                    .into(),
        }),
    }
}

fn bind_declared_field<'a>(
    spec: &ObservationSpec,
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition>,
) -> Result<&'a RecoveredFieldDefinition, ObservationError> {
    let mut matches = fields.filter(|field| {
        let derivation_matches = if spec.recovery == RecoveryPolicy::ShapeFunctionInterpolation
            && spec.output_location == FieldLocation::Point
        {
            field.derivation() == RecoveredFieldDerivation::Native
                && field.location() == spec.source_location
        } else {
            field.satisfies_recovery(spec.source_location, spec.output_location, spec.recovery)
        };
        field.quantity() == spec.quantity
            && derivation_matches
            && field.value_shape() == &spec.value_shape
    });
    match (matches.next(), matches.next()) {
        (Some(field), None) => Ok(field),
        (None, _) => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason: "no declaration matches quantity, location, and value shape".into(),
        }),
        _ => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason:
                "multiple declarations match; recovery must provide an unambiguous field catalog"
                    .into(),
        }),
    }
}

#[cfg(test)]
mod tests;
