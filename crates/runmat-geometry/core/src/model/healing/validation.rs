use super::{
    GeometryHealingFailure, GeometryHealingOperation, GeometryHealingOperationKind,
    GeometryHealingReport, GEOMETRY_HEALING_REPORT_SCHEMA_VERSION,
};
use crate::{
    model::analysis_identity::validate_token, GeometryContractError, GeometryHealingPolicy,
};

const MAX_HEALING_OPERATIONS: usize = 10_000_000;
const MAX_AFFECTED_ENTITIES: usize = 1_000_000;

impl GeometryHealingReport {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        if self.schema_version != GEOMETRY_HEALING_REPORT_SCHEMA_VERSION {
            return Err(invalid(
                "geometry healing report schema",
                "unsupported version",
            ));
        }
        self.policy.validate()?;
        self.tolerance.validate()?;
        self.revision_map.validate()?;
        self.original_topology_digest
            .validate_nonzero("original topology digest")?;
        self.healed_topology_digest
            .validate_nonzero("healed topology digest")?;
        if self.original_topology_digest != self.revision_map.source_geometry_digest
            || self.healed_topology_digest != self.revision_map.target_geometry_digest
        {
            return Err(invalid(
                "geometry healing topology identity",
                "the report and persistent revision map must bind the same original and healed topology",
            ));
        }
        if self.operations.is_empty() || self.operations.len() > MAX_HEALING_OPERATIONS {
            return Err(invalid(
                "geometry healing operations",
                "the operation inventory must be non-empty and within its hard bound",
            ));
        }
        for (index, operation) in self.operations.iter().enumerate() {
            if operation.sequence != index as u64 {
                return Err(invalid(
                    "geometry healing operation sequence",
                    "operations must use contiguous zero-based deterministic sequence numbers",
                ));
            }
            validate_operation(
                operation,
                &self.policy,
                self.tolerance.maximum_healing_displacement_m,
                &self.revision_map,
            )?;
        }
        if self.operations[0].before_validity != self.original_validity
            || self.operations.last().map(|value| value.after_validity)
                != Some(self.healed_validity)
        {
            return Err(invalid(
                "geometry healing validity chain",
                "operation endpoints must match the report validity summaries",
            ));
        }
        for pair in self.operations.windows(2) {
            if pair[0].after_validity != pair[1].before_validity {
                return Err(invalid(
                    "geometry healing validity chain",
                    "each operation must start from the preceding after-validity state",
                ));
            }
        }
        if !self.healed_validity.is_valid() {
            return Err(invalid(
                "healed topology validity",
                "successful healing must finish kernel-valid with consistent incidence, orientation, closure, and nesting",
            ));
        }
        Ok(())
    }
}

impl GeometryHealingFailure {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        validate_entity_list("healing failure entities", &self.affected_entities, false)?;
        validate_displacement(
            "measured healing displacement",
            self.measured_displacement_m,
        )?;
        validate_displacement(
            "permitted healing displacement",
            self.permitted_displacement_m,
        )?;
        if self.measured_displacement_m <= self.permitted_displacement_m {
            return Err(invalid(
                "healing displacement failure",
                "a limit witness must strictly exceed the permitted displacement",
            ));
        }
        for point in [self.original_point_m, self.proposed_point_m] {
            if point.iter().any(|coordinate| !coordinate.is_finite()) {
                return Err(invalid(
                    "healing displacement witness",
                    "witness points must be finite",
                ));
            }
        }
        let witnessed_displacement_m = self
            .original_point_m
            .iter()
            .zip(self.proposed_point_m)
            .map(|(original, proposed)| (proposed - original).powi(2))
            .sum::<f64>()
            .sqrt();
        let witness_error = (witnessed_displacement_m - self.measured_displacement_m).abs();
        let witness_scale = self.measured_displacement_m.abs().max(1.0);
        if witness_error > f64::EPSILON * witness_scale * 8.0 {
            return Err(invalid(
                "healing displacement witness",
                "witness points must realize the reported limiting displacement",
            ));
        }
        validate_token("healing failure reason", &self.reason, 512)
    }
}

fn validate_operation(
    operation: &GeometryHealingOperation,
    policy: &GeometryHealingPolicy,
    maximum_healing_displacement_m: f64,
    revision_map: &crate::GeometryRevisionMap,
) -> Result<(), GeometryContractError> {
    validate_entity_list(
        "healing affected-before entities",
        &operation.affected_before,
        false,
    )?;
    validate_entity_list(
        "healing affected-after entities",
        &operation.affected_after,
        matches!(
            operation.kind,
            GeometryHealingOperationKind::SimplifyShortEdge
                | GeometryHealingOperationKind::SimplifySliverFace
        ),
    )?;
    for source in &operation.affected_before {
        revision_map.resolve(source).map_err(|_| {
            invalid(
                "healing affected-before mapping",
                "every affected source must have an explicit revision-map disposition",
            )
        })?;
    }
    for target in &operation.affected_after {
        if !revision_map
            .operations
            .iter()
            .any(|mapping| mapping.targets().binary_search(target).is_ok())
        {
            return Err(invalid(
                "healing affected-after mapping",
                "every affected target must be produced by the revision map",
            ));
        }
    }
    validate_displacement(
        "healing operation displacement",
        operation.maximum_displacement_m,
    )?;
    if operation.maximum_displacement_m > maximum_healing_displacement_m {
        return Err(invalid(
            "healing operation displacement",
            "a successful operation cannot exceed the configured healing limit",
        ));
    }
    if !operation_enabled(operation.kind, policy) {
        return Err(invalid(
            "geometry healing policy",
            "the report contains an operation disabled by policy",
        ));
    }
    validate_token("geometry healing reason", &operation.reason, 512)
}

fn validate_entity_list(
    field: &str,
    entities: &[crate::PersistentEntityId],
    allow_empty: bool,
) -> Result<(), GeometryContractError> {
    if (!allow_empty && entities.is_empty()) || entities.len() > MAX_AFFECTED_ENTITIES {
        return Err(invalid(
            field,
            "the canonical entity inventory is empty or exceeds its hard bound",
        ));
    }
    let mut prior = None;
    for entity in entities {
        entity.validate()?;
        if prior.is_some_and(|value| value >= entity) {
            return Err(invalid(
                field,
                "entity inventory must be strictly canonical",
            ));
        }
        prior = Some(entity);
    }
    Ok(())
}

fn validate_displacement(field: &str, displacement_m: f64) -> Result<(), GeometryContractError> {
    if !displacement_m.is_finite() || displacement_m < 0.0 {
        return Err(invalid(
            field,
            "displacement must be finite and non-negative",
        ));
    }
    Ok(())
}

fn operation_enabled(kind: GeometryHealingOperationKind, policy: &GeometryHealingPolicy) -> bool {
    match kind {
        GeometryHealingOperationKind::Sew => policy.sew,
        GeometryHealingOperationKind::RepairOrientation => policy.repair_orientation,
        GeometryHealingOperationKind::ConsolidateDuplicate => policy.consolidate_duplicates,
        GeometryHealingOperationKind::RepairGap => policy.repair_tolerance_scale_gaps,
        GeometryHealingOperationKind::SimplifyShortEdge
        | GeometryHealingOperationKind::SimplifySliverFace => {
            policy.simplify_short_edges_and_sliver_faces
        }
    }
}

fn invalid(field: &str, reason: impl Into<String>) -> GeometryContractError {
    GeometryContractError::invalid(field, reason)
}
