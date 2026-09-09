use runmat_analysis_core::ObservationCostEstimate;

use crate::{
    algebra::SparsePattern, constraints::CanonicalStructuralConstraintPlan,
    results::observation::ResolvedObservationPlan, topology::AnalysisProjection,
};

use super::contract::{
    ByteCount, ConstraintResourceFacts, Count, ObservationResourceFacts, ResourcePlanningError,
    SparseResourceFacts, TopologyResourceFacts,
};

impl TopologyResourceFacts {
    pub fn from_projection(
        projection: &AnalysisProjection,
        resident_bytes_upper_bound: ByteCount,
    ) -> Result<Self, ResourcePlanningError> {
        Ok(Self {
            nodes: count(projection.nodes.len(), "node count")?,
            elements: count(projection.elements.len(), "element count")?,
            boundary_faces: count(projection.boundary_faces.len(), "boundary-face count")?,
            element_node_references: count(
                sum_lengths(
                    projection
                        .elements
                        .iter()
                        .map(|element| element.node_indices.len()),
                    "element-node references",
                )?,
                "element-node references",
            )?,
            boundary_node_references: count(
                sum_lengths(
                    projection
                        .boundary_faces
                        .iter()
                        .map(|face| face.node_indices.len()),
                    "boundary-node references",
                )?,
                "boundary-node references",
            )?,
            node_neighbor_references: count(
                sum_lengths(
                    projection.node_neighbors.iter().map(Vec::len),
                    "node-neighbor references",
                )?,
                "node-neighbor references",
            )?,
            resident_bytes_upper_bound,
        })
    }
}

impl SparseResourceFacts {
    pub fn from_pattern(
        pattern: &SparsePattern,
        value_array_count: u64,
    ) -> Result<Self, ResourcePlanningError> {
        Ok(Self {
            rows: count(pattern.rows(), "sparse row count")?,
            columns: count(pattern.columns(), "sparse column count")?,
            nonzeros: count(pattern.column_indices().len(), "sparse nonzero count")?,
            value_array_count: Count(value_array_count),
        })
    }
}

impl ConstraintResourceFacts {
    pub fn from_plan(
        plan: &CanonicalStructuralConstraintPlan,
        resident_bytes_upper_bound: ByteCount,
    ) -> Result<Self, ResourcePlanningError> {
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
            resident_bytes_upper_bound,
        })
    }
}

impl ObservationResourceFacts {
    pub fn from_plan(plan: &ResolvedObservationPlan) -> Result<Self, ResourcePlanningError> {
        let total = plan.observations.iter().try_fold(
            ObservationCostEstimate {
                operation_count: 0,
                peak_resident_bytes: 0,
                stored_bytes: 0,
                transfer_bytes: 0,
                collective_count: 0,
            },
            |mut total, observation| {
                let cost = &observation.resolved_cost;
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
            sample_count: Count(u64::from(!plan.observations.is_empty())),
            operation_count: Count(total.operation_count),
            peak_resident_bytes: ByteCount(total.peak_resident_bytes),
            stored_bytes: ByteCount(total.stored_bytes),
            transfer_bytes: ByteCount(total.transfer_bytes),
            collective_count: Count(total.collective_count),
        })
    }
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
