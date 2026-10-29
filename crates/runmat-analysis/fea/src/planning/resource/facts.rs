use crate::execution::FieldRecoveryPlan;
use crate::physics::structural::topology::node_space::CanonicalStructuralNodeSpace;
use crate::{
    algebra::SparsePattern,
    constraints::CanonicalStructuralConstraintPlan,
    results::observation::{ObservationExecutionPolicy, ResolvedObservationPlan},
    topology::AnalysisProjection,
};
use runmat_analysis_core::{StructuralElementKind, StructuralModel};

use super::contract::{
    ByteCount, ConstraintResourceFacts, Count, FieldRecoveryResourceFacts,
    ObservationResourceFacts, ResourcePlanningError, SparseResourceFacts, TopologyResourceFacts,
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

    pub(crate) fn from_mixed_projection(
        projection: &AnalysisProjection,
        structural: &StructuralModel,
        topology: &CanonicalStructuralNodeSpace,
    ) -> Result<Self, ResourcePlanningError> {
        let solid_element_references = sum_lengths(
            projection
                .elements
                .iter()
                .map(|element| element.node_indices.len()),
            "solid element-node references",
        )?;
        let explicit_element_references = sum_lengths(
            structural
                .elements
                .iter()
                .map(|element| match &element.kind {
                    StructuralElementKind::Beam(beam) => beam.node_ids.len(),
                    StructuralElementKind::Shell(shell) => shell.node_ids.len(),
                }),
            "explicit element-node references",
        )?;
        let element_node_references = solid_element_references
            .checked_add(explicit_element_references)
            .ok_or(ResourcePlanningError::Overflow(
                "mixed element-node references",
            ))?;
        let boundary_node_references = sum_lengths(
            projection
                .boundary_faces
                .iter()
                .map(|face| face.node_indices.len()),
            "boundary-node references",
        )?;
        let explicit_neighbor_references =
            structural
                .elements
                .iter()
                .try_fold(0usize, |total, element| {
                    let node_count = match &element.kind {
                        StructuralElementKind::Beam(beam) => beam.node_ids.len(),
                        StructuralElementKind::Shell(shell) => shell.node_ids.len(),
                    };
                    total
                        .checked_add(node_count.saturating_mul(node_count.saturating_sub(1)))
                        .ok_or(ResourcePlanningError::Overflow(
                            "mixed node-neighbor references",
                        ))
                })?;
        let node_neighbor_references = sum_lengths(
            projection.node_neighbors.iter().map(Vec::len),
            "solid node-neighbor references",
        )?
        .checked_add(explicit_neighbor_references)
        .ok_or(ResourcePlanningError::Overflow(
            "mixed node-neighbor references",
        ))?;
        let node_count = topology.nodes.len();
        let element_count = projection
            .elements
            .len()
            .checked_add(structural.elements.len())
            .ok_or(ResourcePlanningError::Overflow("mixed element count"))?;
        Ok(Self {
            nodes: count(node_count, "mixed node count")?,
            elements: count(element_count, "mixed element count")?,
            boundary_faces: count(projection.boundary_faces.len(), "boundary-face count")?,
            element_node_references: count(
                element_node_references,
                "mixed element-node references",
            )?,
            boundary_node_references: count(boundary_node_references, "boundary-node references")?,
            node_neighbor_references: count(
                node_neighbor_references,
                "mixed node-neighbor references",
            )?,
            estimated_resident_bytes: estimate_topology_resident_bytes(
                node_count,
                element_count,
                projection.boundary_faces.len(),
                element_node_references,
                boundary_node_references,
                node_neighbor_references,
            )?,
        })
    }

    pub(crate) fn from_explicit(
        structural: &StructuralModel,
        topology: &CanonicalStructuralNodeSpace,
    ) -> Result<Self, ResourcePlanningError> {
        let element_node_references = sum_lengths(
            structural
                .elements
                .iter()
                .map(|element| match &element.kind {
                    StructuralElementKind::Beam(beam) => beam.node_ids.len(),
                    StructuralElementKind::Shell(shell) => shell.node_ids.len(),
                }),
            "explicit element-node references",
        )?;
        let node_neighbor_references =
            structural
                .elements
                .iter()
                .try_fold(0usize, |total, element| {
                    let node_count = match &element.kind {
                        StructuralElementKind::Beam(beam) => beam.node_ids.len(),
                        StructuralElementKind::Shell(shell) => shell.node_ids.len(),
                    };
                    total
                        .checked_add(node_count.saturating_mul(node_count.saturating_sub(1)))
                        .ok_or(ResourcePlanningError::Overflow(
                            "explicit node-neighbor references",
                        ))
                })?;
        Ok(Self {
            nodes: count(topology.nodes.len(), "explicit node count")?,
            elements: count(structural.elements.len(), "explicit element count")?,
            boundary_faces: Count(0),
            element_node_references: count(
                element_node_references,
                "explicit element-node references",
            )?,
            boundary_node_references: Count(0),
            node_neighbor_references: count(
                node_neighbor_references,
                "explicit node-neighbor references",
            )?,
            estimated_resident_bytes: estimate_topology_resident_bytes(
                topology.nodes.len(),
                structural.elements.len(),
                0,
                element_node_references,
                0,
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
    estimate_topology_resident_bytes(
        projection.nodes.len(),
        projection.elements.len(),
        projection.boundary_faces.len(),
        element_node_references,
        boundary_node_references,
        node_neighbor_references,
    )
}

fn estimate_topology_resident_bytes(
    node_count: usize,
    element_count: usize,
    boundary_count: usize,
    element_node_references: usize,
    boundary_node_references: usize,
    node_neighbor_references: usize,
) -> Result<ByteCount, ResourcePlanningError> {
    let node_count = u64::try_from(node_count)
        .map_err(|_| ResourcePlanningError::Overflow("projection nodes"))?;
    let element_count = u64::try_from(element_count)
        .map_err(|_| ResourcePlanningError::Overflow("projection elements"))?;
    let boundary_count = u64::try_from(boundary_count)
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
    pub fn from_plan(
        plan: &ResolvedObservationPlan,
        execution: ObservationExecutionPolicy,
    ) -> Result<Self, ResourcePlanningError> {
        let total = plan.observations.iter().try_fold(
            ObservationResourceAccumulator::default(),
            |mut total, observation| {
                let partition_plan = crate::results::observation::ObservationPartitionPlan::derive(
                    plan,
                    &observation.spec.observation_id,
                    execution,
                )
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                let cost = &partition_plan.resource_cost;
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

impl FieldRecoveryResourceFacts {
    pub fn from_plan(plan: &FieldRecoveryPlan) -> Result<Self, ResourcePlanningError> {
        plan.validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let mut facts = Self::default();
        for requirement in &plan.requirements {
            facts.requirement_count.0 = checked_add(
                facts.requirement_count.0,
                1,
                "field recovery requirement count",
            )?;
            facts.source_entity_count.0 = checked_add(
                facts.source_entity_count.0,
                requirement.cost.source_entity_count,
                "field recovery source entities",
            )?;
            facts.output_entity_count.0 = checked_add(
                facts.output_entity_count.0,
                requirement.cost.output_entity_count,
                "field recovery output entities",
            )?;
            facts.topology_incidence_count.0 = checked_add(
                facts.topology_incidence_count.0,
                requirement.cost.topology_incidence_count,
                "field recovery topology incidences",
            )?;
            facts.arithmetic_operation_count.0 = checked_add(
                facts.arithmetic_operation_count.0,
                requirement.cost.arithmetic_operation_count,
                "field recovery arithmetic operations",
            )?;
        }
        Ok(facts)
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
