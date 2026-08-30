use std::collections::HashMap;

use runmat_types::{CodistributorClass, DistributionScheme, LabCount, WorkerGridOrientation};
use runmat_value::{ObjectInstance, Tensor, Value};

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub const ONE_DIMENSIONAL_CLASS: runmat_types::StaticClassIdentity =
    CodistributorClass::OneDimensional.identity();
pub const TWO_DIMENSIONAL_CLASS: runmat_types::StaticClassIdentity =
    CodistributorClass::TwoDimensionalBlockCyclic.identity();

#[derive(Clone, Copy)]
enum CodistributorProperty {
    Dimension,
    Partition,
    GlobalSize,
    WorkerGrid,
    BlockSize,
    Orientation,
}

impl CodistributorProperty {
    const fn name(self) -> &'static str {
        match self {
            Self::Dimension => "Dimension",
            Self::Partition => "Partition",
            Self::GlobalSize => "GlobalSize",
            Self::WorkerGrid => "WorkerGrid",
            Self::BlockSize => "BlockSize",
            Self::Orientation => "Orientation",
        }
    }
}

pub fn is_supported_class(class_name: &runmat_types::ClassIdentity) -> bool {
    CodistributorClass::from_identity(class_name).is_some()
}

pub fn is_codistributor(value: &Value) -> bool {
    matches!(value, Value::Object(object) if is_supported_class(&object.class_name))
}

pub fn declared_global_shape(value: &Value) -> Result<Vec<u64>, RuntimeError> {
    let Value::Object(object) = value else {
        return Err(error("globalIndices requires a codistributor object"));
    };
    validate_definition(object)?;
    optional_vector_property(object, CodistributorProperty::GlobalSize)?
        .ok_or_else(|| error("globalIndices requires a complete codistributor"))
}

pub fn designated_worker(value: &Value) -> Result<runmat_types::LabRank, RuntimeError> {
    one_based_u32(value, "designated worker").map(runmat_types::LabRank)
}

pub fn distribution_dimension(value: &Value) -> Result<u32, RuntimeError> {
    one_based_u32(value, "distribution dimension")
}

fn one_based_u32(value: &Value, label: &str) -> Result<u32, RuntimeError> {
    let value = positive_scalar(value, label)?;
    u32::try_from(value).map_err(|_| error(format!("{label} exceeds u32")))
}

pub fn build_validation_option(value: &Value) -> Result<bool, RuntimeError> {
    let option = string_scalar(value)
        .ok_or_else(|| error("codistributed.build option must be 'noCommunication'"))?;
    if option.eq_ignore_ascii_case("noCommunication") {
        Ok(false)
    } else {
        Err(error(
            "codistributed.build option must be 'noCommunication'",
        ))
    }
}

pub fn default_scheme(
    global_shape: &[u64],
    labs: LabCount,
) -> Result<DistributionScheme, RuntimeError> {
    let Value::Object(object) = one_dimensional(&[])? else {
        unreachable!("one-dimensional factory always returns an object")
    };
    resolve_one_dimensional(&object, global_shape, labs)
}

pub fn resolve_local_parts(
    codistributor: Option<&Value>,
    local_shapes: &[Vec<u64>],
    labs: LabCount,
    validate_across_workers: bool,
) -> Result<(Vec<u64>, DistributionScheme), RuntimeError> {
    if labs.0 == 0 || local_shapes.len() != labs.0 as usize || local_shapes.is_empty() {
        return Err(error(
            "codistributed.build requires one local shape per admitted worker",
        ));
    }
    let rank = local_shapes[0].len();
    if rank == 0 || local_shapes.iter().any(|shape| shape.len() != rank) {
        return Err(error(
            "codistributed.build local parts must have a consistent array rank",
        ));
    }
    match codistributor {
        None => {
            resolve_one_dimensional_local_parts(None, local_shapes, labs, validate_across_workers)
        }
        Some(Value::Object(object)) if object.class_name.is(ONE_DIMENSIONAL_CLASS) => {
            validate_definition(object)?;
            resolve_one_dimensional_local_parts(
                Some(object),
                local_shapes,
                labs,
                validate_across_workers,
            )
        }
        Some(Value::Object(object)) if object.class_name.is(TWO_DIMENSIONAL_CLASS) => {
            validate_definition(object)?;
            let global_shape = optional_vector_property(object, CodistributorProperty::GlobalSize)?
                .ok_or_else(|| {
                    error("codistributor2dbc must declare GlobalSize for local-part construction")
                })?;
            let scheme = resolve_two_dimensional(object, &global_shape, labs)?;
            if validate_across_workers {
                let layouts =
                    crate::parallel::distribution::partition_layouts(&global_shape, &scheme, labs)?;
                if layouts
                    .iter()
                    .zip(local_shapes)
                    .any(|(layout, shape)| layout.local_shape != *shape)
                {
                    return Err(error(
                        "codistributed.build local shapes disagree with the 2-D block-cyclic layout",
                    ));
                }
            }
            Ok((global_shape, scheme))
        }
        Some(_) => Err(error(
            "codistributed.build requires a codistributor1d or codistributor2dbc object",
        )),
    }
}

fn resolve_one_dimensional_local_parts(
    object: Option<&ObjectInstance>,
    local_shapes: &[Vec<u64>],
    labs: LabCount,
    validate_across_workers: bool,
) -> Result<(Vec<u64>, DistributionScheme), RuntimeError> {
    let dimension = object
        .map(|object| optional_positive_property(object, CodistributorProperty::Dimension))
        .transpose()?
        .flatten()
        .unwrap_or_else(|| {
            local_shapes[0]
                .iter()
                .rposition(|extent| *extent != 1)
                .map(|axis| axis as u64 + 1)
                .unwrap_or(if local_shapes[0].len() >= 2 { 2 } else { 1 })
        });
    let axis = usize::try_from(dimension)
        .ok()
        .and_then(|dimension| dimension.checked_sub(1))
        .filter(|axis| *axis < local_shapes[0].len())
        .ok_or_else(|| error("codistributor1d dimension lies outside the local-part rank"))?;
    if validate_across_workers
        && local_shapes.iter().skip(1).any(|shape| {
            shape
                .iter()
                .enumerate()
                .any(|(current, extent)| current != axis && *extent != local_shapes[0][current])
        })
    {
        return Err(error(
            "codistributed.build local shapes disagree outside the distribution dimension",
        ));
    }
    let partition = local_shapes
        .iter()
        .map(|shape| shape[axis])
        .collect::<Vec<_>>();
    let mut global_shape = local_shapes[0].clone();
    global_shape[axis] = partition.iter().try_fold(0_u64, |sum, extent| {
        sum.checked_add(*extent)
            .ok_or_else(|| error("codistributed.build global extent overflowed"))
    })?;
    if let Some(object) = object {
        if let Some(declared) = optional_vector_property(object, CodistributorProperty::GlobalSize)?
        {
            if declared != global_shape {
                return Err(error(
                    "codistributed.build local shapes disagree with the declared GlobalSize",
                ));
            }
        }
        if let Some(declared) = optional_vector_property(object, CodistributorProperty::Partition)?
        {
            if declared.len() != labs.0 as usize
                || (validate_across_workers && declared != partition)
            {
                return Err(error(
                    "codistributed.build local shapes disagree with the declared partition",
                ));
            }
        }
    }
    Ok((
        global_shape,
        DistributionScheme::OneDimensional {
            dimension: u32::try_from(dimension)
                .map_err(|_| error("distribution dimension exceeds u32"))?,
            partition,
        },
    ))
}

pub fn factory(arguments: &[Value]) -> Result<Value, RuntimeError> {
    let Some((scheme, parameters)) = arguments.split_first() else {
        return one_dimensional(&[]);
    };
    let scheme = string_scalar(scheme).ok_or_else(|| {
        error("codistributor scheme must be the character vector or string scalar '1d' or '2dbc'")
    })?;
    match scheme.to_ascii_lowercase().as_str() {
        "1d" => {
            if parameters.len() > 2 {
                return Err(error(
                    "codistributor('1d', ...) accepts at most dimension and partition",
                ));
            }
            one_dimensional(parameters)
        }
        "2dbc" => {
            if parameters.len() > 2 {
                return Err(error(
                    "codistributor('2dbc', ...) accepts at most worker grid and block size",
                ));
            }
            two_dimensional(parameters)
        }
        _ => Err(error(
            "codistributor scheme must be the character vector or string scalar '1d' or '2dbc'",
        )),
    }
}

pub fn is_complete(value: &Value) -> Result<bool, RuntimeError> {
    let Value::Object(object) = value else {
        return Err(error("isComplete requires a codistributor object"));
    };
    validate_definition(object)?;
    Ok(optional_vector_property(object, CodistributorProperty::GlobalSize)?.is_some())
}

pub fn validate_definition(object: &ObjectInstance) -> Result<(), RuntimeError> {
    const ONE_DIMENSIONAL_PROPERTIES: &[CodistributorProperty] = &[
        CodistributorProperty::Dimension,
        CodistributorProperty::Partition,
        CodistributorProperty::GlobalSize,
    ];
    const TWO_DIMENSIONAL_PROPERTIES: &[CodistributorProperty] = &[
        CodistributorProperty::WorkerGrid,
        CodistributorProperty::BlockSize,
        CodistributorProperty::Orientation,
        CodistributorProperty::GlobalSize,
    ];
    let expected = match CodistributorClass::from_identity(&object.class_name) {
        Some(CodistributorClass::OneDimensional) => ONE_DIMENSIONAL_PROPERTIES,
        Some(CodistributorClass::TwoDimensionalBlockCyclic) => TWO_DIMENSIONAL_PROPERTIES,
        _ => return Err(error("unsupported codistributor class")),
    };
    if object.dynamic_properties.is_some()
        || object.properties.len() != expected.len()
        || expected
            .iter()
            .any(|property| !object.properties.contains_key(property.name()))
    {
        return Err(error(
            "codistributor properties do not match the immutable class schema",
        ));
    }
    match CodistributorClass::from_identity(&object.class_name) {
        Some(CodistributorClass::OneDimensional) => {
            optional_positive_property(object, CodistributorProperty::Dimension)?;
            optional_vector_property(object, CodistributorProperty::Partition)?;
            optional_vector_property(object, CodistributorProperty::GlobalSize)?;
        }
        Some(CodistributorClass::TwoDimensionalBlockCyclic) => {
            if let Some(values) =
                optional_vector_property(object, CodistributorProperty::WorkerGrid)?
            {
                pair_from_values(&values, "worker grid")?;
            }
            optional_positive_property(object, CodistributorProperty::BlockSize)?;
            parse_orientation(
                object
                    .properties
                    .get(CodistributorProperty::Orientation.name())
                    .expect("validated property set"),
            )?;
            if let Some(values) =
                optional_vector_property(object, CodistributorProperty::GlobalSize)?
            {
                pair_from_values(&values, "global size")?;
            }
        }
        _ => unreachable!("class identity was validated above"),
    }
    Ok(())
}

pub fn one_dimensional(arguments: &[Value]) -> Result<Value, RuntimeError> {
    if arguments.len() > 3 {
        return Err(error("codistributor1d accepts zero to three arguments"));
    }
    let dimension = arguments
        .first()
        .map(|value| positive_scalar(value, "distribution dimension"))
        .transpose()?;
    let partition = arguments
        .get(1)
        .map(|value| nonnegative_vector(value, "partition"))
        .transpose()?;
    let global_shape = arguments
        .get(2)
        .map(|value| nonnegative_vector(value, "global size"))
        .transpose()?;
    Ok(one_dimensional_object(dimension, partition, global_shape))
}

pub fn two_dimensional(arguments: &[Value]) -> Result<Value, RuntimeError> {
    if arguments.len() > 4 {
        return Err(error("codistributor2dbc accepts zero to four arguments"));
    }
    let worker_grid = arguments
        .first()
        .map(|value| pair(value, "worker grid"))
        .transpose()?;
    let block_size = arguments
        .get(1)
        .map(|value| positive_scalar(value, "block size"))
        .transpose()?;
    let orientation = arguments
        .get(2)
        .map(parse_orientation)
        .transpose()?
        .unwrap_or(WorkerGridOrientation::Row);
    let global_shape = arguments
        .get(3)
        .map(|value| pair(value, "global size"))
        .transpose()?;
    let mut properties = HashMap::new();
    properties.insert(
        CodistributorProperty::WorkerGrid.name().into(),
        worker_grid.map_or_else(empty_vector, |value| {
            vector_value(value.map(u64::from).to_vec())
        }),
    );
    properties.insert(
        CodistributorProperty::BlockSize.name().into(),
        block_size.map_or_else(empty_vector, exact_scalar),
    );
    properties.insert(
        CodistributorProperty::Orientation.name().into(),
        orientation_value(orientation),
    );
    properties.insert(
        CodistributorProperty::GlobalSize.name().into(),
        global_shape.map_or_else(empty_vector, |value| {
            vector_value(value.map(u64::from).to_vec())
        }),
    );
    Ok(object(TWO_DIMENSIONAL_CLASS, properties))
}

pub fn from_scheme(
    scheme: &DistributionScheme,
    global_shape: &[u64],
) -> Result<Value, RuntimeError> {
    let value = match scheme {
        DistributionScheme::Block { dimension } => one_dimensional_object(
            Some(u64::from(*dimension)),
            None,
            Some(global_shape.to_vec()),
        ),
        DistributionScheme::OneDimensional {
            dimension,
            partition,
        } => one_dimensional_object(
            Some(u64::from(*dimension)),
            Some(partition.clone()),
            Some(global_shape.to_vec()),
        ),
        DistributionScheme::TwoDimensionalBlockCyclic {
            worker_grid,
            block_size,
            orientation,
        } => {
            let mut properties = HashMap::new();
            properties.insert(
                CodistributorProperty::WorkerGrid.name().into(),
                vector_value(worker_grid.map(u64::from).to_vec()),
            );
            properties.insert(
                CodistributorProperty::BlockSize.name().into(),
                exact_scalar(*block_size),
            );
            properties.insert(
                CodistributorProperty::Orientation.name().into(),
                orientation_value(*orientation),
            );
            properties.insert(
                CodistributorProperty::GlobalSize.name().into(),
                vector_value(global_shape.to_vec()),
            );
            object(TWO_DIMENSIONAL_CLASS, properties)
        }
        DistributionScheme::Replicated
        | DistributionScheme::Cyclic { .. }
        | DistributionScheme::Custom { .. } => return Err(error(
            "the distributed value does not have a codistributor1d or codistributor2dbc representation",
        )),
    };
    Ok(value)
}

pub fn resolve(
    value: &Value,
    global_shape: &[u64],
    labs: LabCount,
) -> Result<DistributionScheme, RuntimeError> {
    if labs.0 == 0 {
        return Err(error("redistribution requires at least one lab"));
    }
    let Value::Object(object) = value else {
        return Err(error("redistribute requires a codistributor object"));
    };
    validate_definition(object)?;
    match CodistributorClass::from_identity(&object.class_name) {
        Some(CodistributorClass::OneDimensional) => {
            resolve_one_dimensional(object, global_shape, labs)
        }
        Some(CodistributorClass::TwoDimensionalBlockCyclic) => {
            resolve_two_dimensional(object, global_shape, labs)
        }
        _ => Err(error(
            "redistribute requires a codistributor1d or codistributor2dbc object",
        )),
    }
}

fn resolve_one_dimensional(
    object: &ObjectInstance,
    global_shape: &[u64],
    labs: LabCount,
) -> Result<DistributionScheme, RuntimeError> {
    let dimension = optional_positive_property(object, CodistributorProperty::Dimension)?
        .unwrap_or_else(|| {
            global_shape
                .iter()
                .rposition(|extent| *extent != 1)
                .map(|index| index as u64 + 1)
                .unwrap_or(if global_shape.len() >= 2 { 2 } else { 1 })
        });
    let dimension_index = usize::try_from(dimension)
        .ok()
        .and_then(|value| value.checked_sub(1))
        .filter(|value| *value < global_shape.len())
        .ok_or_else(|| error("codistributor1d dimension lies outside the global size"))?;
    validate_declared_shape(object, global_shape)?;
    let partition = optional_vector_property(object, CodistributorProperty::Partition)?
        .unwrap_or_else(|| default_partition(global_shape[dimension_index], labs));
    let partition_extent = partition.iter().try_fold(0_u64, |sum, length| {
        sum.checked_add(*length)
            .ok_or_else(|| error("codistributor1d partition lengths overflow the global extent"))
    })?;
    if partition.len() != labs.0 as usize || partition_extent != global_shape[dimension_index] {
        return Err(error(
            "codistributor1d partition must contain one length per lab and sum to the distribution extent",
        ));
    }
    Ok(DistributionScheme::OneDimensional {
        dimension: u32::try_from(dimension)
            .map_err(|_| error("distribution dimension exceeds u32"))?,
        partition,
    })
}

fn resolve_two_dimensional(
    object: &ObjectInstance,
    global_shape: &[u64],
    labs: LabCount,
) -> Result<DistributionScheme, RuntimeError> {
    if global_shape.len() != 2 {
        return Err(error("codistributor2dbc can distribute only matrices"));
    }
    validate_declared_shape(object, global_shape)?;
    let grid = optional_vector_property(object, CodistributorProperty::WorkerGrid)?
        .map(|values| pair_from_values(&values, "worker grid"))
        .transpose()?
        .unwrap_or_else(|| default_worker_grid(labs));
    if u64::from(grid[0]) * u64::from(grid[1]) != u64::from(labs.0) {
        return Err(error(
            "worker grid must contain exactly one position per lab",
        ));
    }
    let block_size =
        optional_positive_property(object, CodistributorProperty::BlockSize)?.unwrap_or(64);
    let orientation = object
        .properties
        .get(CodistributorProperty::Orientation.name())
        .map(parse_orientation)
        .transpose()?
        .unwrap_or(WorkerGridOrientation::Row);
    Ok(DistributionScheme::TwoDimensionalBlockCyclic {
        worker_grid: grid,
        block_size,
        orientation,
    })
}

fn one_dimensional_object(
    dimension: Option<u64>,
    partition: Option<Vec<u64>>,
    global_shape: Option<Vec<u64>>,
) -> Value {
    let mut properties = HashMap::new();
    properties.insert(
        CodistributorProperty::Dimension.name().into(),
        dimension.map_or_else(empty_vector, exact_scalar),
    );
    properties.insert(
        CodistributorProperty::Partition.name().into(),
        partition.map_or_else(empty_vector, vector_value),
    );
    properties.insert(
        CodistributorProperty::GlobalSize.name().into(),
        global_shape.map_or_else(empty_vector, vector_value),
    );
    object(ONE_DIMENSIONAL_CLASS, properties)
}

fn object(
    class_name: runmat_types::StaticClassIdentity,
    properties: HashMap<String, Value>,
) -> Value {
    Value::Object(ObjectInstance {
        class_name: class_name.into(),
        properties,
        dynamic_properties: None,
    })
}

fn empty_vector() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("canonical empty shape"))
}

fn vector_value(values: Vec<u64>) -> Value {
    let length = values.len();
    Value::Tensor(
        Tensor::new_integer(runmat_value::IntegerStorage::U64(values), vec![1, length])
            .expect("vector shape matches its values"),
    )
}

fn exact_scalar(value: u64) -> Value {
    Value::Int(runmat_value::IntValue::U64(value))
}

fn string_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::CharArray(value) => value.row_string(),
        _ => None,
    }
}

fn positive_scalar(value: &Value, name: &str) -> Result<u64, RuntimeError> {
    let values = nonnegative_vector(value, name)?;
    match values.as_slice() {
        [value] if *value > 0 => Ok(*value),
        _ => Err(error(format!("{name} must be a positive integer scalar"))),
    }
}

fn nonnegative_vector(value: &Value, name: &str) -> Result<Vec<u64>, RuntimeError> {
    match value {
        Value::Int(value) => value
            .try_to_u64()
            .map(|value| vec![value])
            .ok_or_else(|| error(format!("{name} must contain nonnegative integers"))),
        Value::Num(value) => exact_f64_u64(*value, name).map(|value| vec![value]),
        Value::Tensor(tensor) => (0..tensor.len())
            .map(|index| match tensor.numeric_value_at(index) {
                Some(runmat_value::NumericScalar::F64(value)) => exact_f64_u64(value, name),
                Some(runmat_value::NumericScalar::F32(value)) => exact_f32_u64(value, name),
                Some(value) => value
                    .into_int_value()
                    .and_then(|value| value.try_to_u64())
                    .ok_or_else(|| error(format!("{name} must contain nonnegative integers"))),
                None => Err(error(format!("{name} contains invalid numeric storage"))),
            })
            .collect(),
        _ => Err(error(format!("{name} must be a numeric integer vector"))),
    }
}

fn exact_f64_u64(value: f64, name: &str) -> Result<u64, RuntimeError> {
    const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= MAX_EXACT_INTEGER {
        Ok(value as u64)
    } else {
        Err(error(format!(
            "{name} must contain exactly representable nonnegative integers"
        )))
    }
}

fn exact_f32_u64(value: f32, name: &str) -> Result<u64, RuntimeError> {
    const MAX_EXACT_INTEGER: f32 = 16_777_216.0;
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= MAX_EXACT_INTEGER {
        Ok(value as u64)
    } else {
        Err(error(format!(
            "{name} must contain exactly representable nonnegative integers"
        )))
    }
}

fn pair(value: &Value, name: &str) -> Result<[u32; 2], RuntimeError> {
    pair_from_values(&nonnegative_vector(value, name)?, name)
}

fn pair_from_values(values: &[u64], name: &str) -> Result<[u32; 2], RuntimeError> {
    let [first, second] = values else {
        return Err(error(format!("{name} must contain exactly two integers")));
    };
    let first = u32::try_from(*first).map_err(|_| error(format!("{name} exceeds u32")))?;
    let second = u32::try_from(*second).map_err(|_| error(format!("{name} exceeds u32")))?;
    if first == 0 || second == 0 {
        return Err(error(format!("{name} entries must be positive")));
    }
    Ok([first, second])
}

fn parse_orientation(value: &Value) -> Result<WorkerGridOrientation, RuntimeError> {
    let text = String::try_from(value)
        .map_err(|_| error("codistributor2dbc orientation must be 'row' or 'col'"))?;
    match text.to_ascii_lowercase().as_str() {
        "row" => Ok(WorkerGridOrientation::Row),
        "col" => Ok(WorkerGridOrientation::Column),
        _ => Err(error(
            "codistributor2dbc orientation must be 'row' or 'col'",
        )),
    }
}

fn orientation_value(orientation: WorkerGridOrientation) -> Value {
    Value::String(
        match orientation {
            WorkerGridOrientation::Row => "row",
            WorkerGridOrientation::Column => "col",
        }
        .into(),
    )
}

fn optional_positive_property(
    object: &ObjectInstance,
    property: CodistributorProperty,
) -> Result<Option<u64>, RuntimeError> {
    let name = property.name();
    let Some(value) = object.properties.get(name) else {
        return Err(error(format!(
            "{} is missing its {name} property",
            object.class_name
        )));
    };
    if is_empty(value) {
        Ok(None)
    } else {
        positive_scalar(value, name).map(Some)
    }
}

fn optional_vector_property(
    object: &ObjectInstance,
    property: CodistributorProperty,
) -> Result<Option<Vec<u64>>, RuntimeError> {
    let name = property.name();
    let Some(value) = object.properties.get(name) else {
        return Err(error(format!(
            "{} is missing its {name} property",
            object.class_name
        )));
    };
    if is_empty(value) {
        Ok(None)
    } else {
        nonnegative_vector(value, name).map(Some)
    }
}

fn validate_declared_shape(object: &ObjectInstance, shape: &[u64]) -> Result<(), RuntimeError> {
    if optional_vector_property(object, CodistributorProperty::GlobalSize)?
        .is_some_and(|declared| declared != shape)
    {
        return Err(error(
            "codistributor global size does not match the distributed value",
        ));
    }
    Ok(())
}

fn is_empty(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.len() == 0)
}

fn default_partition(extent: u64, labs: LabCount) -> Vec<u64> {
    let base = extent / u64::from(labs.0);
    let remainder = extent % u64::from(labs.0);
    (0..labs.0)
        .map(|index| base + u64::from(u64::from(index) < remainder))
        .collect()
}

fn default_worker_grid(labs: LabCount) -> [u32; 2] {
    let mut rows = (f64::from(labs.0).sqrt().floor() as u32).max(1);
    while !labs.0.is_multiple_of(rows) {
        rows -= 1;
    }
    [rows, labs.0 / rows]
}

fn error(message: impl Into<String>) -> RuntimeError {
    semantic_error("RunMat:parallel:Codistributor", message.into())
}

#[cfg(test)]
mod tests {
    use runmat_value::{IntValue, IntegerStorage};

    use super::*;

    #[test]
    fn one_dimensional_resolution_preserves_exact_partition_lengths() {
        let distributor = one_dimensional(&[
            Value::Int(IntValue::U32(2)),
            vector_value(vec![1, 2, u64::MAX - 3]),
            vector_value(vec![4, u64::MAX]),
        ])
        .expect("typed constructor");
        assert_eq!(
            resolve(&distributor, &[4, u64::MAX], LabCount(3)).unwrap(),
            DistributionScheme::OneDimensional {
                dimension: 2,
                partition: vec![1, 2, u64::MAX - 3],
            }
        );
    }

    #[test]
    fn structural_double_inputs_must_be_exact_integers() {
        let beyond_flintmax = Value::Num(9_007_199_254_740_994.0);
        let error = one_dimensional(&[beyond_flintmax]).unwrap_err();
        assert!(error
            .to_string()
            .contains("exactly representable nonnegative integers"));

        let tensor = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
        );
        assert!(one_dimensional(&[Value::Int(IntValue::U32(1)), tensor]).is_ok());
    }

    #[test]
    fn generic_factory_preserves_scheme_identity_and_completeness() {
        let default = factory(&[]).expect("default codistributor");
        assert!(matches!(
            default,
            Value::Object(ref object) if object.class_name == ONE_DIMENSIONAL_CLASS
        ));
        assert!(!is_complete(&default).unwrap());

        let two_dimensional = factory(&[
            Value::String("2dbc".into()),
            vector_value(vec![1, 2]),
            Value::Int(IntValue::U32(8)),
        ])
        .expect("generic 2dbc codistributor");
        assert!(matches!(
            two_dimensional,
            Value::Object(ref object) if object.class_name == TWO_DIMENSIONAL_CLASS
        ));
        assert!(!is_complete(&two_dimensional).unwrap());

        let complete = one_dimensional(&[
            Value::Int(IntValue::U32(1)),
            vector_value(vec![2, 2]),
            vector_value(vec![4, 3]),
        ])
        .unwrap();
        assert!(is_complete(&complete).unwrap());
        assert!(factory(&[Value::String("cyclic".into())]).is_err());
    }

    #[test]
    fn declared_shape_partition_and_grid_are_validated_together() {
        let wrong_partition = one_dimensional(&[
            Value::Int(IntValue::U32(2)),
            vector_value(vec![2, 2]),
            vector_value(vec![4, 5]),
        ])
        .unwrap();
        assert!(resolve(&wrong_partition, &[4, 5], LabCount(2)).is_err());

        let wrong_shape = one_dimensional(&[
            Value::Int(IntValue::U32(2)),
            vector_value(vec![2, 3]),
            vector_value(vec![4, 6]),
        ])
        .unwrap();
        assert!(resolve(&wrong_shape, &[4, 5], LabCount(2)).is_err());

        let wrong_grid =
            two_dimensional(&[vector_value(vec![2, 2]), Value::Int(IntValue::U32(1))]).unwrap();
        assert!(resolve(&wrong_grid, &[4, 4], LabCount(3)).is_err());
    }

    #[test]
    fn schemes_without_public_codistributor_forms_are_not_stringified() {
        assert!(from_scheme(&DistributionScheme::Replicated, &[2, 2]).is_err());
        assert!(from_scheme(&DistributionScheme::Cyclic { dimension: 1 }, &[2, 2]).is_err());
    }
}
