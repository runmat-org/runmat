use runmat_execution::{DistributedPartitionLayout, PartitionRange, PartitionSelection};
use runmat_types::{DistributionScheme, LabCount, LabRank};
use runmat_value::{
    CellArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericStorage,
    SparseTensor, Tensor, Value,
};

use crate::indexing::plan::build_index_plan;
use crate::indexing::selectors::SliceSelector;
use crate::runtime_error::semantic_error;
use crate::RuntimeError;

use super::assembly;

/// One live partition paired with the immutable selection that locates it in
/// the global value. This is a runtime construction product, not a transport
/// object and not a second language value representation.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributedPartitionValue {
    pub layout: DistributedPartitionLayout,
    pub value: Value,
}

pub async fn partition_value(
    input: &Value,
    scheme: &DistributionScheme,
    count: LabCount,
) -> Result<(Vec<usize>, Vec<DistributedPartitionValue>), RuntimeError> {
    if count.0 == 0 {
        return Err(error("distributed values require at least one partition"));
    }
    if matches!(scheme, DistributionScheme::Replicated) {
        let shape = runtime_shape(input).await?;
        let selections = full_selections(&shape)?;
        let local_shape = usize_shape_to_u64(&shape)?;
        let partitions = (1..=count.0)
            .map(|rank| DistributedPartitionValue {
                layout: DistributedPartitionLayout {
                    rank: LabRank(rank),
                    selections: selections.clone(),
                    local_shape: local_shape.clone(),
                },
                value: input.clone(),
            })
            .collect();
        return Ok((shape, partitions));
    }

    if matches!(scheme, DistributionScheme::Custom { .. }) {
        return Err(error(
            "custom distributed partitioners require a registered partitioner service",
        ));
    }
    let shape = assembly::value_shape(input).ok_or_else(|| {
        error("this value does not support class-preserving distributed partitioning")
    })?;
    let dimension = scheme_dimension(scheme, &shape)?;
    let index_sets = partition_indices(shape[dimension], count, scheme);
    let mut partitions = Vec::with_capacity(index_sets.len());
    for (offset, indices) in index_sets.into_iter().enumerate() {
        let selectors = selectors_for_dimension(&shape, dimension, &indices);
        let plan = build_index_plan(&selectors, shape.len(), &shape)?;
        let value = assembly::read_with_plan(input, &plan)?;
        let rank = u32::try_from(offset + 1)
            .map_err(|_| error("distributed partition rank exceeds its portable representation"))?;
        let selections = layout_selections(&shape, dimension, &indices, scheme, rank, count)?;
        partitions.push(DistributedPartitionValue {
            layout: DistributedPartitionLayout {
                rank: LabRank(rank),
                selections,
                local_shape: usize_shape_to_u64(&plan.output_shape)?,
            },
            value,
        });
    }
    Ok((shape, partitions))
}

pub async fn materialize_partitions(
    global_shape: &[usize],
    scheme: &DistributionScheme,
    partitions: &[DistributedPartitionValue],
) -> Result<Value, RuntimeError> {
    let first = partitions
        .first()
        .ok_or_else(|| error("distributed value has no retained partitions"))?;
    if matches!(scheme, DistributionScheme::Replicated) {
        if partitions.iter().any(|part| part.value != first.value) {
            return Err(error("replicated partitions disagree at materialization"));
        }
        return Ok(first.value.clone());
    }
    let mut output = zero_like(&first.value, global_shape)?;
    for partition in partitions {
        let selectors = selectors_from_layout(&partition.layout)?;
        let plan = build_index_plan(&selectors, global_shape.len(), global_shape)?;
        if plan.output_shape != u64_shape_to_usize(&partition.layout.local_shape)? {
            return Err(error(
                "distributed partition payload shape disagrees with its immutable layout",
            ));
        }
        output = assembly::assign_with_plan(output, &plan, &partition.value).await?;
    }
    Ok(output)
}

async fn runtime_shape(value: &Value) -> Result<Vec<usize>, RuntimeError> {
    crate::builtins::common::shape::value_dimensions(value).await
}

fn scheme_dimension(scheme: &DistributionScheme, shape: &[usize]) -> Result<usize, RuntimeError> {
    let dimension = match scheme {
        DistributionScheme::Block { dimension } | DistributionScheme::Cyclic { dimension } => {
            *dimension
        }
        DistributionScheme::Replicated | DistributionScheme::Custom { .. } => unreachable!(),
    };
    usize::try_from(dimension)
        .ok()
        .and_then(|dimension| dimension.checked_sub(1))
        .filter(|dimension| *dimension < shape.len())
        .ok_or_else(|| error("distribution dimension lies outside the value rank"))
}

fn partition_indices(
    extent: usize,
    count: LabCount,
    scheme: &DistributionScheme,
) -> Vec<Vec<usize>> {
    match scheme {
        DistributionScheme::Block { .. } => {
            let count = count.0 as usize;
            let base = extent / count;
            let remainder = extent % count;
            let mut start = 0usize;
            (0..count)
                .map(|partition| {
                    let length = base + usize::from(partition < remainder);
                    let indices = (start..start + length).collect();
                    start += length;
                    indices
                })
                .collect()
        }
        DistributionScheme::Cyclic { .. } => (0..count.0 as usize)
            .map(|partition| (partition..extent).step_by(count.0 as usize).collect())
            .collect(),
        DistributionScheme::Replicated | DistributionScheme::Custom { .. } => unreachable!(),
    }
}

fn selectors_for_dimension(
    shape: &[usize],
    dimension: usize,
    zero_based: &[usize],
) -> Vec<SliceSelector> {
    (0..shape.len())
        .map(|current| {
            if current == dimension {
                SliceSelector::Indices(zero_based.iter().map(|index| index + 1).collect())
            } else {
                SliceSelector::Colon
            }
        })
        .collect()
}

fn full_selections(shape: &[usize]) -> Result<Vec<PartitionSelection>, RuntimeError> {
    shape
        .iter()
        .enumerate()
        .map(|(dimension, extent)| {
            Ok(PartitionSelection::Range(PartitionRange {
                dimension: portable_dimension(dimension)?,
                start: 0,
                end: u64::try_from(*extent)
                    .map_err(|_| error("value extent exceeds its portable representation"))?,
            }))
        })
        .collect()
}

fn layout_selections(
    shape: &[usize],
    partition_dimension: usize,
    indices: &[usize],
    scheme: &DistributionScheme,
    rank: u32,
    partition_count: LabCount,
) -> Result<Vec<PartitionSelection>, RuntimeError> {
    shape
        .iter()
        .enumerate()
        .map(|(dimension, extent)| {
            let dimension_id = portable_dimension(dimension)?;
            if dimension != partition_dimension {
                return Ok(PartitionSelection::Range(PartitionRange {
                    dimension: dimension_id,
                    start: 0,
                    end: u64::try_from(*extent)
                        .map_err(|_| error("value extent exceeds its portable representation"))?,
                }));
            }
            match scheme {
                DistributionScheme::Block { .. } => {
                    let start = indices.first().copied().unwrap_or(*extent);
                    Ok(PartitionSelection::Range(PartitionRange {
                        dimension: dimension_id,
                        start: u64::try_from(start)
                            .map_err(|_| error("partition offset exceeds u64"))?,
                        end: u64::try_from(start + indices.len())
                            .map_err(|_| error("partition extent exceeds u64"))?,
                    }))
                }
                DistributionScheme::Cyclic { .. } => Ok(PartitionSelection::Strided {
                    dimension: dimension_id,
                    start: u64::from(rank - 1),
                    step: u64::from(partition_count.0),
                    count: indices.len() as u64,
                }),
                DistributionScheme::Replicated | DistributionScheme::Custom { .. } => {
                    unreachable!()
                }
            }
        })
        .collect()
}

fn selectors_from_layout(
    layout: &DistributedPartitionLayout,
) -> Result<Vec<SliceSelector>, RuntimeError> {
    layout
        .selections
        .iter()
        .map(|selection| match selection {
            PartitionSelection::Range(range) => Ok(SliceSelector::Indices(
                (range.start..range.end)
                    .map(|index| {
                        usize::try_from(index + 1)
                            .map_err(|_| error("partition index exceeds this host"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            PartitionSelection::Strided {
                start, step, count, ..
            } => Ok(SliceSelector::Indices(
                (0..*count)
                    .map(|offset| {
                        start
                            .checked_add(step.checked_mul(offset).ok_or_else(|| {
                                error("partition selection multiplication overflowed")
                            })?)
                            .and_then(|index| index.checked_add(1))
                            .and_then(|index| usize::try_from(index).ok())
                            .ok_or_else(|| error("partition selection exceeds this host"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            PartitionSelection::Indices { indices, .. } => Ok(SliceSelector::Indices(
                indices
                    .iter()
                    .map(|index| {
                        index
                            .checked_add(1)
                            .and_then(|index| usize::try_from(index).ok())
                            .ok_or_else(|| error("partition index exceeds this host"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )),
        })
        .collect()
}

fn zero_like(prototype: &Value, shape: &[usize]) -> Result<Value, RuntimeError> {
    let length = shape
        .iter()
        .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))
        .ok_or_else(|| error("distributed global shape exceeds this host"))?;
    match prototype {
        Value::Tensor(value) => Tensor::from_numeric_storage(
            NumericStorage::zeros(value.numeric_dtype(), length),
            shape.to_vec(),
        )
        .map(Value::Tensor)
        .map_err(error),
        Value::ComplexTensor(value) => {
            let storage = match value.complex_storage() {
                ComplexStorage::F64(_) => {
                    ComplexStorage::F64(vec![runmat_value::ComplexElement(0.0, 0.0); length].into())
                }
                ComplexStorage::F32(_) => ComplexStorage::F32(
                    vec![runmat_value::ComplexElement(0.0f32, 0.0f32); length].into(),
                ),
                ComplexStorage::Integer(value) => ComplexStorage::Integer(
                    IntegerComplexStorage::new(
                        value.real.zeros_like(length),
                        value.imag.zeros_like(length),
                    )
                    .map_err(error)?,
                ),
            };
            ComplexTensor::from_complex_storage(storage, shape.to_vec())
                .map(Value::ComplexTensor)
                .map_err(error)
        }
        Value::SparseTensor(value) if shape.len() == 2 => {
            let sparse = if value.is_logical() {
                SparseTensor::zeros_logical(shape[0], shape[1])
            } else if value.is_complex() {
                SparseTensor::zeros_complex(shape[0], shape[1])
            } else if let Some(storage) = value.integer_storage() {
                SparseTensor::zeros_with_integer_storage(shape[0], shape[1], storage)
            } else if value.numeric_dtype() == Some(runmat_value::NumericDType::F32) {
                SparseTensor::zeros_f32(shape[0], shape[1])
            } else {
                SparseTensor::zeros(shape[0], shape[1])
            };
            Ok(Value::SparseTensor(sparse))
        }
        Value::LogicalArray(_) => Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec()))),
        Value::Cell(_) => CellArray::new_with_shape(vec![Value::Num(0.0); length], shape.to_vec())
            .map(Value::Cell)
            .map_err(error),
        _ => Err(error(
            "distributed materialization does not support this runtime storage family",
        )),
    }
}

fn portable_dimension(zero_based: usize) -> Result<u32, RuntimeError> {
    zero_based
        .checked_add(1)
        .and_then(|dimension| u32::try_from(dimension).ok())
        .ok_or_else(|| error("value rank exceeds its portable representation"))
}

fn usize_shape_to_u64(shape: &[usize]) -> Result<Vec<u64>, RuntimeError> {
    shape
        .iter()
        .map(|dimension| u64::try_from(*dimension).map_err(|_| error("value shape exceeds u64")))
        .collect()
}

fn u64_shape_to_usize(shape: &[u64]) -> Result<Vec<usize>, RuntimeError> {
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| error("value shape exceeds this host"))
        })
        .collect()
}

fn error(message: impl Into<String>) -> RuntimeError {
    semantic_error("RunMat:parallel:Distribution", message.into())
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::{IntegerStorage, Tensor};

    use super::*;

    #[test]
    fn block_and_cyclic_round_trips_preserve_wide_integer_storage() {
        let input = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, u64::MAX, 3, u64::MAX - 1, 5, 6]),
                vec![2, 3],
            )
            .unwrap(),
        );
        for scheme in [
            DistributionScheme::Block { dimension: 2 },
            DistributionScheme::Cyclic { dimension: 2 },
        ] {
            let (shape, parts) = block_on(partition_value(&input, &scheme, LabCount(2))).unwrap();
            assert_eq!(parts.len(), 2);
            let output = block_on(materialize_partitions(&shape, &scheme, &parts)).unwrap();
            assert_eq!(output, input);
        }
    }
}
