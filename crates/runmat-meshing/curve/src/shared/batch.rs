use runmat_geometry_core::{
    ExactBRepTopology, ExactCurveEvaluator, ExactPcurveEvaluator, GeometryEvaluationControl,
};
use runmat_meshing_core::{CanonicalEntityRange, MeshingPartitionDescriptor, MeshingPartitionKind};

use super::{
    discretize::{
        discretize_edge, geometry_error, validate_options, CurveMetricField,
        SharedCurveDiscretizationOptions, SharedCurveEvaluationContext,
    },
    validation::validate_curve_against_topology,
    SharedCurveBatch, SharedCurveError, SharedCurveMesh, SHARED_CURVE_BATCH_SCHEMA_VERSION,
    SHARED_CURVE_MESH_SCHEMA_VERSION,
};

// One exact-geometry prerequisite plus this many partition roots fits the canonical 64-input join.
const MAX_CURVE_PARTITIONS: usize = 63;

pub fn curve_partition_descriptors(
    topology: &ExactBRepTopology,
    preferred_edges_per_partition: u32,
) -> Result<Vec<MeshingPartitionDescriptor>, SharedCurveError> {
    if preferred_edges_per_partition == 0 || topology.edges.is_empty() {
        return Err(SharedCurveError::invalid_request(
            "curve partition policy",
            "maximum edges and exact edge inventory must be nonzero",
        ));
    }
    let minimum_batch_size = topology.edges.len().div_ceil(MAX_CURVE_PARTITIONS);
    let batch_size = (preferred_edges_per_partition as usize).max(minimum_batch_size);
    let partition_count = topology.edges.len().div_ceil(batch_size);
    Ok(topology
        .edges
        .chunks(batch_size)
        .enumerate()
        .map(|(index, edges)| MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::CanonicalEntityBatch,
            partition_index: index as u32,
            partition_count: partition_count as u32,
            entity_range: Some(CanonicalEntityRange {
                first: edges[0].id.clone(),
                last: edges.last().expect("chunk is nonempty").id.clone(),
                entity_count: edges.len() as u64,
            }),
        })
        .collect())
}

pub fn discretize_shared_curve_partition(
    topology: &ExactBRepTopology,
    curves: &dyn ExactCurveEvaluator,
    pcurves: &dyn ExactPcurveEvaluator,
    metric_field: &dyn CurveMetricField,
    control: &dyn GeometryEvaluationControl,
    options: SharedCurveDiscretizationOptions,
    partition: MeshingPartitionDescriptor,
) -> Result<SharedCurveBatch, SharedCurveError> {
    validate_options(options)?;
    validate_partition(&partition)?;
    let context =
        SharedCurveEvaluationContext::new(topology, curves, pcurves, metric_field, control);
    let range = partition.entity_range.as_ref().expect("kind validated");
    let selected = context
        .topology
        .edges
        .iter()
        .filter(|edge| edge.id >= range.first && edge.id <= range.last)
        .collect::<Vec<_>>();
    if selected.len() as u64 != range.entity_count {
        return Err(SharedCurveError::invalid_request(
            "curve partition range",
            "entity range does not select its declared canonical edge count",
        ));
    }
    let mut edges = Vec::with_capacity(selected.len());
    for edge in selected {
        context
            .control
            .checkpoint()
            .map_err(|error| geometry_error(edge, error))?;
        edges.push(discretize_edge(context, edge, options)?);
    }
    let batch = SharedCurveBatch {
        schema_version: SHARED_CURVE_BATCH_SCHEMA_VERSION,
        partition,
        edges,
    };
    validate_batch(&batch, context.topology)?;
    Ok(batch)
}

pub fn join_shared_curve_batches(
    topology: &ExactBRepTopology,
    mut batches: Vec<SharedCurveBatch>,
) -> Result<SharedCurveMesh, SharedCurveError> {
    if batches.is_empty() || batches.len() > MAX_CURVE_PARTITIONS {
        return Err(SharedCurveError::invalid_contract(
            "curve batch join",
            "join requires a bounded nonempty partition set",
        ));
    }
    batches.sort_by_key(|batch| batch.partition.partition_index);
    for (index, batch) in batches.iter().enumerate() {
        validate_batch(batch, topology)?;
        if batch.partition.partition_index != index as u32
            || batch.partition.partition_count != batches.len() as u32
        {
            return Err(SharedCurveError::invalid_contract(
                "curve batch join",
                "partition indices and counts must form one complete canonical set",
            ));
        }
    }
    let mesh = SharedCurveMesh {
        schema_version: SHARED_CURVE_MESH_SCHEMA_VERSION,
        edges: batches.into_iter().flat_map(|batch| batch.edges).collect(),
    };
    mesh.validate_against(topology)?;
    Ok(mesh)
}

pub(super) fn validate_batch(
    batch: &SharedCurveBatch,
    topology: &ExactBRepTopology,
) -> Result<(), SharedCurveError> {
    if batch.schema_version != SHARED_CURVE_BATCH_SCHEMA_VERSION || batch.edges.is_empty() {
        return Err(SharedCurveError::invalid_contract(
            "curve batch",
            "schema and edge inventory must be valid",
        ));
    }
    validate_partition(&batch.partition)?;
    let range = batch
        .partition
        .entity_range
        .as_ref()
        .expect("kind validated");
    let expected = topology
        .edges
        .iter()
        .filter(|edge| edge.id >= range.first && edge.id <= range.last)
        .collect::<Vec<_>>();
    if expected.len() as u64 != range.entity_count
        || batch.edges.len() != expected.len()
        || batch
            .edges
            .iter()
            .zip(expected)
            .any(|(curve, edge)| curve.source_edge_id != edge.id)
    {
        return Err(SharedCurveError::invalid_contract(
            "curve batch range",
            "edges must exactly match the canonical partition range",
        ));
    }
    for curve in &batch.edges {
        validate_curve_against_topology(curve, topology)?;
    }
    Ok(())
}

fn validate_partition(partition: &MeshingPartitionDescriptor) -> Result<(), SharedCurveError> {
    partition.validate().map_err(|error| {
        SharedCurveError::invalid_contract("curve partition", error.to_string())
    })?;
    if partition.kind != MeshingPartitionKind::CanonicalEntityBatch
        || partition.partition_count as usize > MAX_CURVE_PARTITIONS
    {
        return Err(SharedCurveError::invalid_contract(
            "curve partition",
            "curve work requires a bounded canonical entity batch",
        ));
    }
    Ok(())
}
