use crate::{
    algebra::SparsePattern, constraints::CanonicalStructuralConstraintPlan,
    results::observation::ResolvedObservationPlan, topology::AnalysisProjection,
};

use super::contract::{
    ByteCount, ConstraintResourceFacts, Count, ObservationResourceFacts, ResourcePlanningError,
    SparseResourceFacts, TopologyResourceFacts,
};

impl TopologyResourceFacts {
    pub fn from_projection(projection: &AnalysisProjection) -> Result<Self, ResourcePlanningError> {
        let element_node_references = sum_lengths(
            projection
                .elements
                .iter()
                .map(|element| element.node_indices.len()),
            "element-node references",
        )?;
        let boundary_node_references = sum_lengths(
            projection
                .boundary_faces
                .iter()
                .map(|face| face.node_indices.len()),
            "boundary-node references",
        )?;
        let node_neighbor_references = sum_lengths(
            projection.node_neighbors.iter().map(Vec::len),
            "node-neighbor references",
        )?;
        Ok(Self {
            nodes: count(projection.nodes.len(), "node count")?,
            elements: count(projection.elements.len(), "element count")?,
            boundary_faces: count(projection.boundary_faces.len(), "boundary-face count")?,
            element_node_references: count(element_node_references, "element-node references")?,
            boundary_node_references: count(boundary_node_references, "boundary-node references")?,
            node_neighbor_references: count(node_neighbor_references, "node-neighbor references")?,
            estimated_resident_bytes: estimate_projection_resident_bytes(
                projection,
                element_node_references,
                boundary_node_references,
                node_neighbor_references,
            )?,
        })
    }
}

impl SparseResourceFacts {
    pub fn from_patterns<'a>(
        patterns: impl IntoIterator<Item = &'a SparsePattern>,
    ) -> Result<Self, ResourcePlanningError> {
        let mut patterns = patterns.into_iter();
        let first = patterns.next().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "at least one sparse operator pattern is required".into(),
            )
        })?;
        let rows = first.rows();
        let columns = first.columns();
        let mut operator_nonzeros =
            vec![count(first.column_indices().len(), "sparse nonzero count")?];
        for pattern in patterns {
            if pattern.rows() != rows || pattern.columns() != columns {
                return Err(ResourcePlanningError::Invalid(
                    "sparse operator patterns must have equal dimensions".into(),
                ));
            }
            operator_nonzeros.push(count(
                pattern.column_indices().len(),
                "sparse nonzero count",
            )?);
        }
        Ok(Self {
            rows: count(rows, "sparse row count")?,
            columns: count(columns, "sparse column count")?,
            operator_nonzeros,
        })
    }
}

impl ConstraintResourceFacts {
    pub fn from_plan(
        plan: &CanonicalStructuralConstraintPlan,
    ) -> Result<Self, ResourcePlanningError> {
        let canonical_length = u64::try_from(
            plan.canonical_bytes()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
                .len(),
        )
        .map_err(|_| ResourcePlanningError::Overflow("constraint canonical length"))?;
        Ok(Self {
            rows: count(plan.rows.len(), "constraint rows")?,
            coefficients: count(
                sum_lengths(
                    plan.rows.iter().map(|row| row.terms.len()),
                    "constraint coefficients",
                )?,
                "constraint coefficients",
            )?,
            reference_points: count(plan.reference_points.len(), "constraint reference points")?,
            estimated_resident_bytes: estimated_owned_bytes(
                canonical_length,
                "constraint resident estimate",
            )?,
        })
    }
}

fn estimate_projection_resident_bytes(
    projection: &AnalysisProjection,
    element_node_references: usize,
    boundary_node_references: usize,
    node_neighbor_references: usize,
) -> Result<ByteCount, ResourcePlanningError> {
    let node_count = u64::try_from(projection.nodes.len())
        .map_err(|_| ResourcePlanningError::Overflow("projection nodes"))?;
    let element_count = u64::try_from(projection.elements.len())
        .map_err(|_| ResourcePlanningError::Overflow("projection elements"))?;
    let boundary_count = u64::try_from(projection.boundary_faces.len())
        .map_err(|_| ResourcePlanningError::Overflow("projection boundary faces"))?;
    let references = [
        element_node_references,
        boundary_node_references,
        node_neighbor_references,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        let value = u64::try_from(value)
            .map_err(|_| ResourcePlanningError::Overflow("projection references"))?;
        checked_add(total, value, "projection references")
    })?;
    let structural_bytes = node_count
        .checked_mul(64)
        .and_then(|value| value.checked_add(element_count.checked_mul(128)?))
        .and_then(|value| value.checked_add(boundary_count.checked_mul(160)?))
        .and_then(|value| value.checked_add(references.checked_mul(8)?))
        .ok_or(ResourcePlanningError::Overflow(
            "projection resident estimate",
        ))?;
    estimated_owned_bytes(structural_bytes, "projection resident estimate")
}

fn estimated_owned_bytes(
    structural_bytes: u64,
    label: &'static str,
) -> Result<ByteCount, ResourcePlanningError> {
    structural_bytes
        .checked_mul(2)
        .and_then(|value| value.checked_add(4096))
        .map(ByteCount)
        .ok_or(ResourcePlanningError::Overflow(label))
}

impl ObservationResourceFacts {
    pub fn from_plan(plan: &ResolvedObservationPlan) -> Result<Self, ResourcePlanningError> {
        plan.validate_identity()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let total = plan.observations.iter().try_fold(
            ObservationResourceAccumulator::default(),
            |mut total, observation| {
                let cost = &observation.resolved_cost;
                let produced = u64::try_from(observation.sample_bounds.maximum_produced_samples)
                    .map_err(|_| ResourcePlanningError::Overflow("observation produced samples"))?;
                let retained = u64::try_from(observation.sample_bounds.maximum_retained_samples)
                    .map_err(|_| ResourcePlanningError::Overflow("observation retained samples"))?;
                total.maximum_samples_per_observation =
                    total.maximum_samples_per_observation.max(produced);
                total.produced_sample_count = checked_add(
                    total.produced_sample_count,
                    produced,
                    "observation produced samples",
                )?;
                total.retained_sample_count = checked_add(
                    total.retained_sample_count,
                    retained,
                    "observation retained samples",
                )?;
                total.operation_count = checked_add(
                    total.operation_count,
                    cost.operation_count,
                    "observation operations",
                )?;
                total.peak_resident_bytes = total.peak_resident_bytes.max(cost.peak_resident_bytes);
                total.stored_bytes =
                    checked_add(total.stored_bytes, cost.stored_bytes, "observation storage")?;
                total.transfer_bytes = checked_add(
                    total.transfer_bytes,
                    cost.transfer_bytes,
                    "observation transfer",
                )?;
                total.collective_count = checked_add(
                    total.collective_count,
                    cost.collective_count,
                    "observation collectives",
                )?;
                Ok::<_, ResourcePlanningError>(total)
            },
        )?;
        Ok(Self {
            maximum_samples_per_observation: Count(total.maximum_samples_per_observation),
            produced_sample_count: Count(total.produced_sample_count),
            retained_sample_count: Count(total.retained_sample_count),
            operation_count: Count(total.operation_count),
            peak_resident_bytes: ByteCount(total.peak_resident_bytes),
            stored_bytes: ByteCount(total.stored_bytes),
            transfer_bytes: ByteCount(total.transfer_bytes),
            collective_count: Count(total.collective_count),
        })
    }
}

#[derive(Default)]
struct ObservationResourceAccumulator {
    maximum_samples_per_observation: u64,
    produced_sample_count: u64,
    retained_sample_count: u64,
    operation_count: u64,
    peak_resident_bytes: u64,
    stored_bytes: u64,
    transfer_bytes: u64,
    collective_count: u64,
}

fn sum_lengths(
    values: impl IntoIterator<Item = usize>,
    label: &'static str,
) -> Result<usize, ResourcePlanningError> {
    values.into_iter().try_fold(0usize, |total, value| {
        total
            .checked_add(value)
            .ok_or(ResourcePlanningError::Overflow(label))
    })
}

fn count(value: usize, label: &'static str) -> Result<Count, ResourcePlanningError> {
    u64::try_from(value)
        .map(Count)
        .map_err(|_| ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
