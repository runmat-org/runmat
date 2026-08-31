use runmat_execution::{DistributedPartitionLayout, PartitionRange, PartitionSelection};
use runmat_types::{DistributionScheme, LabCount, LabRank};
use runmat_value::{
    CellArray, ComplexStorage, ComplexTensor, IntegerComplexStorage, LogicalArray, NumericStorage,
    Tensor, Value,
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
    let shape = distributable_shape(input).ok_or_else(|| {
        error("this value does not support class-preserving distributed partitioning")
    })?;
    let layouts = partition_layouts_usize(&shape, scheme, count)?;
    let mut partitions = Vec::with_capacity(layouts.len());
    for layout in layouts {
        let value = extract_partition(input, scheme, &shape, &layout)?;
        partitions.push(DistributedPartitionValue { layout, value });
    }
    Ok((shape, partitions))
}

/// Extracts exactly the partition owned by `rank` while retaining the complete
/// immutable layout table. Worker-scoped construction uses this path so a
/// worker never stores remote payloads merely to construct its local shard.
pub async fn partition_local_value(
    input: &Value,
    scheme: &DistributionScheme,
    count: LabCount,
    rank: LabRank,
) -> Result<
    (
        Vec<u64>,
        Vec<DistributedPartitionLayout>,
        DistributedPartitionValue,
    ),
    RuntimeError,
> {
    let shape = distributable_shape(input).ok_or_else(|| {
        error("this value does not support class-preserving distributed partitioning")
    })?;
    let layouts = partition_layouts_usize(&shape, scheme, count)?;
    let layout = layouts
        .iter()
        .find(|layout| layout.rank == rank)
        .cloned()
        .ok_or_else(|| error("distributed rank lies outside the admitted partition layout"))?;
    let value = extract_partition(input, scheme, &shape, &layout)?;
    Ok((
        usize_shape_to_u64(&shape)?,
        layouts,
        DistributedPartitionValue { layout, value },
    ))
}

fn extract_partition(
    input: &Value,
    scheme: &DistributionScheme,
    shape: &[usize],
    layout: &DistributedPartitionLayout,
) -> Result<Value, RuntimeError> {
    if matches!(scheme, DistributionScheme::Replicated) {
        return Ok(input.clone());
    }
    let selectors = selectors_from_layout(layout)?;
    let plan = build_index_plan(&selectors, shape.len(), shape)?;
    let scalar_source = scalar_partition_source(input)?;
    assembly::read_with_plan(scalar_source.as_ref().unwrap_or(input), &plan)
}

fn distributable_shape(value: &Value) -> Option<Vec<usize>> {
    assembly::value_shape(value).or_else(|| {
        matches!(
            value,
            Value::Num(_) | Value::Int(_) | Value::Complex(_, _) | Value::Bool(_)
        )
        .then(|| vec![1, 1])
    })
}

fn scalar_partition_source(value: &Value) -> Result<Option<Value>, RuntimeError> {
    let source = match value {
        Value::Num(value) => {
            Tensor::from_numeric_storage(NumericStorage::F64(vec![*value]), vec![1, 1])
                .map(Value::Tensor)
                .map_err(error)?
        }
        Value::Int(value) => {
            let storage = match value {
                runmat_value::IntValue::I8(value) => NumericStorage::I8(vec![*value]),
                runmat_value::IntValue::I16(value) => NumericStorage::I16(vec![*value]),
                runmat_value::IntValue::I32(value) => NumericStorage::I32(vec![*value]),
                runmat_value::IntValue::I64(value) => NumericStorage::I64(vec![*value]),
                runmat_value::IntValue::U8(value) => NumericStorage::U8(vec![*value]),
                runmat_value::IntValue::U16(value) => NumericStorage::U16(vec![*value]),
                runmat_value::IntValue::U32(value) => NumericStorage::U32(vec![*value]),
                runmat_value::IntValue::U64(value) => NumericStorage::U64(vec![*value]),
            };
            Tensor::from_numeric_storage(storage, vec![1, 1])
                .map(Value::Tensor)
                .map_err(error)?
        }
        Value::Complex(real, imaginary) => ComplexTensor::from_complex_storage(
            ComplexStorage::F64(vec![runmat_value::ComplexElement(*real, *imaginary)].into()),
            vec![1, 1],
        )
        .map(Value::ComplexTensor)
        .map_err(error)?,
        Value::Bool(value) => LogicalArray::new(vec![u8::from(*value)], vec![1, 1])
            .map(Value::LogicalArray)
            .map_err(error)?,
        _ => return Ok(None),
    };
    Ok(Some(source))
}

pub fn partition_layouts(
    global_shape: &[u64],
    scheme: &DistributionScheme,
    count: LabCount,
) -> Result<Vec<DistributedPartitionLayout>, RuntimeError> {
    let shape = global_shape
        .iter()
        .map(|extent| {
            usize::try_from(*extent).map_err(|_| error("global extent exceeds this host"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    partition_layouts_usize(&shape, scheme, count)
}

fn partition_layouts_usize(
    shape: &[usize],
    scheme: &DistributionScheme,
    count: LabCount,
) -> Result<Vec<DistributedPartitionLayout>, RuntimeError> {
    if count.0 == 0 {
        return Err(error("distributed values require at least one partition"));
    }
    if matches!(scheme, DistributionScheme::Custom { .. }) {
        return Err(error(
            "custom distributed partitioners require a registered partitioner service",
        ));
    }
    if matches!(scheme, DistributionScheme::Replicated) {
        let selections = full_selections(shape)?;
        let local_shape = usize_shape_to_u64(shape)?;
        return Ok((1..=count.0)
            .map(|rank| DistributedPartitionLayout {
                rank: LabRank(rank),
                selections: selections.clone(),
                local_shape: local_shape.clone(),
            })
            .collect());
    }
    if let DistributionScheme::TwoDimensionalBlockCyclic {
        worker_grid,
        block_size,
        orientation,
    } = scheme
    {
        return two_dimensional_layouts(shape, count, *worker_grid, *block_size, *orientation);
    }
    let dimension = scheme_dimension(scheme, shape)?;
    partition_indices(shape[dimension], count, scheme)?
        .into_iter()
        .enumerate()
        .map(|(offset, indices)| {
            let rank = u32::try_from(offset + 1).map_err(|_| {
                error("distributed partition rank exceeds its portable representation")
            })?;
            let selections = layout_selections(shape, dimension, &indices, scheme, rank, count)?;
            Ok(DistributedPartitionLayout {
                rank: LabRank(rank),
                local_shape: selections
                    .iter()
                    .map(|selection| selection.element_count())
                    .collect(),
                selections,
            })
        })
        .collect()
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

pub async fn value_shape(value: &Value) -> Result<Vec<u64>, RuntimeError> {
    runtime_shape(value)
        .await?
        .into_iter()
        .map(|extent| {
            u64::try_from(extent)
                .map_err(|_| error("value extent exceeds its portable representation"))
        })
        .collect()
}

fn scheme_dimension(scheme: &DistributionScheme, shape: &[usize]) -> Result<usize, RuntimeError> {
    let dimension = match scheme {
        DistributionScheme::Block { dimension }
        | DistributionScheme::Cyclic { dimension }
        | DistributionScheme::OneDimensional { dimension, .. } => *dimension,
        DistributionScheme::Replicated
        | DistributionScheme::TwoDimensionalBlockCyclic { .. }
        | DistributionScheme::Custom { .. } => unreachable!(),
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
) -> Result<Vec<Vec<usize>>, RuntimeError> {
    match scheme {
        DistributionScheme::Block { .. } => {
            let count = count.0 as usize;
            let base = extent / count;
            let remainder = extent % count;
            let mut start = 0usize;
            Ok((0..count)
                .map(|partition| {
                    let length = base + usize::from(partition < remainder);
                    let indices = (start..start + length).collect();
                    start += length;
                    indices
                })
                .collect())
        }
        DistributionScheme::OneDimensional { partition, .. } => {
            let mut start = 0usize;
            let partitions = partition
                .iter()
                .map(|length| {
                    let length = usize::try_from(*length)
                        .map_err(|_| error("1-D partition length exceeds this host"))?;
                    let end = start
                        .checked_add(length)
                        .filter(|end| *end <= extent)
                        .ok_or_else(|| error("1-D partition lengths exceed the value extent"))?;
                    let indices = (start..end).collect();
                    start = end;
                    Ok(indices)
                })
                .collect::<Result<Vec<_>, RuntimeError>>()?;
            if start != extent || partitions.len() != count.0 as usize {
                return Err(error(
                    "1-D partition lengths must match the value extent and partition count",
                ));
            }
            Ok(partitions)
        }
        DistributionScheme::Cyclic { .. } => Ok((0..count.0 as usize)
            .map(|partition| (partition..extent).step_by(count.0 as usize).collect())
            .collect()),
        DistributionScheme::Replicated
        | DistributionScheme::TwoDimensionalBlockCyclic { .. }
        | DistributionScheme::Custom { .. } => unreachable!(),
    }
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
                DistributionScheme::Block { .. } | DistributionScheme::OneDimensional { .. } => {
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
                DistributionScheme::Replicated
                | DistributionScheme::TwoDimensionalBlockCyclic { .. }
                | DistributionScheme::Custom { .. } => {
                    unreachable!()
                }
            }
        })
        .collect()
}

fn two_dimensional_layouts(
    shape: &[usize],
    count: LabCount,
    worker_grid: [u32; 2],
    block_size: u64,
    orientation: runmat_types::WorkerGridOrientation,
) -> Result<Vec<DistributedPartitionLayout>, RuntimeError> {
    if shape.len() != 2 || worker_grid.contains(&0) || block_size == 0 {
        return Err(error(
            "2-D block-cyclic distribution requires a matrix, positive worker grid, and positive block size",
        ));
    }
    let grid_count = worker_grid[0]
        .checked_mul(worker_grid[1])
        .ok_or_else(|| error("2-D worker grid overflowed"))?;
    if grid_count != count.0 {
        return Err(error(
            "2-D worker grid must contain exactly one position per partition",
        ));
    }
    let block_size =
        usize::try_from(block_size).map_err(|_| error("2-D block size exceeds this host"))?;
    let mut partitions = Vec::with_capacity(count.0 as usize);
    for rank in 1..=count.0 {
        let (grid_row, grid_column) = worker_position(rank, worker_grid, orientation);
        let rows = block_cyclic_indices(shape[0], block_size, worker_grid[0], grid_row);
        let columns = block_cyclic_indices(shape[1], block_size, worker_grid[1], grid_column);
        let local_shape = vec![rows.len() as u64, columns.len() as u64];
        partitions.push(DistributedPartitionLayout {
            rank: LabRank(rank),
            selections: vec![
                PartitionSelection::Indices {
                    dimension: 1,
                    indices: rows
                        .into_iter()
                        .map(|index| {
                            u64::try_from(index).map_err(|_| error("row index exceeds u64"))
                        })
                        .collect::<Result<_, _>>()?,
                },
                PartitionSelection::Indices {
                    dimension: 2,
                    indices: columns
                        .into_iter()
                        .map(|index| {
                            u64::try_from(index).map_err(|_| error("column index exceeds u64"))
                        })
                        .collect::<Result<_, _>>()?,
                },
            ],
            local_shape,
        });
    }
    Ok(partitions)
}

fn worker_position(
    rank: u32,
    worker_grid: [u32; 2],
    orientation: runmat_types::WorkerGridOrientation,
) -> (u32, u32) {
    let zero_based = rank - 1;
    match orientation {
        runmat_types::WorkerGridOrientation::Row => {
            (zero_based / worker_grid[1], zero_based % worker_grid[1])
        }
        runmat_types::WorkerGridOrientation::Column => {
            (zero_based % worker_grid[0], zero_based / worker_grid[0])
        }
    }
}

fn block_cyclic_indices(
    extent: usize,
    block_size: usize,
    grid_extent: u32,
    grid_position: u32,
) -> Vec<usize> {
    (0..extent)
        .filter(|index| ((*index / block_size) % grid_extent as usize) == grid_position as usize)
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
        Value::Num(_) => {
            Tensor::from_numeric_storage(NumericStorage::F64(vec![0.0; length]), shape.to_vec())
                .map(Value::Tensor)
                .map_err(error)
        }
        Value::Int(value) => {
            let storage = match value {
                runmat_value::IntValue::I8(_) => NumericStorage::I8(vec![0; length]),
                runmat_value::IntValue::I16(_) => NumericStorage::I16(vec![0; length]),
                runmat_value::IntValue::I32(_) => NumericStorage::I32(vec![0; length]),
                runmat_value::IntValue::I64(_) => NumericStorage::I64(vec![0; length]),
                runmat_value::IntValue::U8(_) => NumericStorage::U8(vec![0; length]),
                runmat_value::IntValue::U16(_) => NumericStorage::U16(vec![0; length]),
                runmat_value::IntValue::U32(_) => NumericStorage::U32(vec![0; length]),
                runmat_value::IntValue::U64(_) => NumericStorage::U64(vec![0; length]),
            };
            Tensor::from_numeric_storage(storage, shape.to_vec())
                .map(Value::Tensor)
                .map_err(error)
        }
        Value::Complex(_, _) => ComplexTensor::from_complex_storage(
            ComplexStorage::F64(vec![runmat_value::ComplexElement(0.0, 0.0); length].into()),
            shape.to_vec(),
        )
        .map(Value::ComplexTensor)
        .map_err(error),
        Value::Bool(_) => Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec()))),
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
            let sparse = value.zeros_like(shape[0], shape[1]);
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

    #[test]
    fn public_codistributor_schemes_round_trip_exact_integer_storage() {
        let input = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    u64::MAX,
                ]),
                vec![4, 4],
            )
            .unwrap(),
        );
        let schemes = [
            DistributionScheme::OneDimensional {
                dimension: 2,
                partition: vec![1, 1, 1, 1],
            },
            DistributionScheme::TwoDimensionalBlockCyclic {
                worker_grid: [2, 2],
                block_size: 1,
                orientation: runmat_types::WorkerGridOrientation::Row,
            },
            DistributionScheme::TwoDimensionalBlockCyclic {
                worker_grid: [2, 2],
                block_size: 2,
                orientation: runmat_types::WorkerGridOrientation::Column,
            },
        ];
        for scheme in schemes {
            let (shape, parts) = block_on(partition_value(&input, &scheme, LabCount(4))).unwrap();
            assert_eq!(parts.len(), 4);
            let output = block_on(materialize_partitions(&shape, &scheme, &parts)).unwrap();
            assert_eq!(output, input);
        }
    }

    #[test]
    fn worker_partitioning_extracts_only_the_requested_rank_from_the_shared_layout() {
        let input = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 2, 3, u64::MAX]), vec![1, 4]).unwrap(),
        );
        let scheme = DistributionScheme::OneDimensional {
            dimension: 2,
            partition: vec![2, 2],
        };
        let (shape, layouts, local) = block_on(partition_local_value(
            &input,
            &scheme,
            LabCount(2),
            LabRank(2),
        ))
        .unwrap();

        assert_eq!(shape, vec![1, 4]);
        assert_eq!(layouts.len(), 2);
        assert_eq!(local.layout, layouts[1]);
        assert_eq!(
            local.value,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![3, u64::MAX]), vec![1, 2]).unwrap()
            )
        );
    }

    #[test]
    fn one_element_numeric_shards_materialize_from_their_exact_scalar_class() {
        let input = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![1, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .unwrap(),
        );
        let scheme = DistributionScheme::OneDimensional {
            dimension: 2,
            partition: vec![1, 1],
        };
        let (shape, partitions) = block_on(partition_value(&input, &scheme, LabCount(2))).unwrap();
        assert!(partitions
            .iter()
            .all(|partition| matches!(partition.value, Value::Int(_))));
        let output = block_on(materialize_partitions(&shape, &scheme, &partitions)).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn scalar_partitioning_retains_exact_class_and_typed_empty_shards() {
        let input = Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993));
        let scheme = DistributionScheme::Block { dimension: 1 };
        let (shape, partitions) = block_on(partition_value(&input, &scheme, LabCount(3))).unwrap();

        assert_eq!(shape, vec![1, 1]);
        assert_eq!(partitions[0].value, input);
        for partition in &partitions[1..] {
            assert_eq!(
                partition.value,
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 1]).unwrap()
                )
            );
        }
        assert_eq!(
            block_on(materialize_partitions(&shape, &scheme, &partitions)).unwrap(),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1],)
                    .unwrap()
            )
        );
    }
}
