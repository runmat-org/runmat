//! Exact producer slice consumed by the legacy reduction expression. Source
//! selection shares the typed producer index and keeps request leaf identity.

use crate::fusion::{typed_producer::producer_indices, FusionGroupPlan};

pub(super) fn project(plan: &FusionGroupPlan) -> anyhow::Result<(FusionGroupPlan, Vec<usize>)> {
    let root = plan
        .reduction_data
        .ok_or_else(|| anyhow::anyhow!("reduction has no selected data producer"))?;
    let mut leaves = plan
        .inputs
        .iter()
        .chain(plan.const_values.keys())
        .copied()
        .collect::<Vec<_>>();
    leaves.sort_unstable();
    leaves.dedup();
    let selected = producer_indices(plan, &leaves, &[root])?;
    let indices: Vec<_> = (0..plan.operations.len())
        .filter(|index| selected.contains(index))
        .collect();
    let mut producer = plan.clone();
    producer.operations = indices
        .iter()
        .map(|&index| plan.operations[index].clone())
        .collect();
    producer.output = Some(root);
    producer.materialized_stores.clear();
    producer.reduction_data = None;
    producer.reduction_dim = None;
    producer.reduction_flavor = None;
    producer.reduction_axes = None;
    producer.pattern = None;
    Ok((producer, indices))
}
