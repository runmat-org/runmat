use std::collections::HashMap;

use runmat_types::{DistributionScheme, LabCount, WorkerGridOrientation};
use runmat_value::{ObjectInstance, Tensor, Value};

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub const ONE_DIMENSIONAL_CLASS: &str = "codistributor1d";
pub const TWO_DIMENSIONAL_CLASS: &str = "codistributor2dbc";

pub fn is_supported_class(class_name: &str) -> bool {
    matches!(class_name, ONE_DIMENSIONAL_CLASS | TWO_DIMENSIONAL_CLASS)
}

pub fn validate_definition(object: &ObjectInstance) -> Result<(), RuntimeError> {
    const ONE_DIMENSIONAL_PROPERTIES: &[&str] = &["Dimension", "Partition", "GlobalSize"];
    const TWO_DIMENSIONAL_PROPERTIES: &[&str] =
        &["WorkerGrid", "BlockSize", "Orientation", "GlobalSize"];
    let expected = match object.class_name.as_str() {
        ONE_DIMENSIONAL_CLASS => ONE_DIMENSIONAL_PROPERTIES,
        TWO_DIMENSIONAL_CLASS => TWO_DIMENSIONAL_PROPERTIES,
        _ => return Err(error("unsupported codistributor class")),
    };
    if object.dynamic_properties.is_some()
        || object.properties.len() != expected.len()
        || expected
            .iter()
            .any(|property| !object.properties.contains_key(*property))
    {
        return Err(error(
            "codistributor properties do not match the immutable class schema",
        ));
    }
    match object.class_name.as_str() {
        ONE_DIMENSIONAL_CLASS => {
            optional_positive_property(object, "Dimension")?;
            optional_vector_property(object, "Partition")?;
            optional_vector_property(object, "GlobalSize")?;
        }
        TWO_DIMENSIONAL_CLASS => {
            if let Some(values) = optional_vector_property(object, "WorkerGrid")? {
                pair_from_values(&values, "worker grid")?;
            }
            optional_positive_property(object, "BlockSize")?;
            parse_orientation(
                object
                    .properties
                    .get("Orientation")
                    .expect("validated property set"),
            )?;
            if let Some(values) = optional_vector_property(object, "GlobalSize")? {
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
        "WorkerGrid".into(),
        worker_grid.map_or_else(empty_vector, |value| {
            vector_value(value.map(u64::from).to_vec())
        }),
    );
    properties.insert(
        "BlockSize".into(),
        block_size.map_or_else(empty_vector, exact_scalar),
    );
    properties.insert("Orientation".into(), orientation_value(orientation));
    properties.insert(
        "GlobalSize".into(),
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
                "WorkerGrid".into(),
                vector_value(worker_grid.map(u64::from).to_vec()),
            );
            properties.insert("BlockSize".into(), exact_scalar(*block_size));
            properties.insert("Orientation".into(), orientation_value(*orientation));
            properties.insert("GlobalSize".into(), vector_value(global_shape.to_vec()));
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
    match object.class_name.as_str() {
        ONE_DIMENSIONAL_CLASS => resolve_one_dimensional(object, global_shape, labs),
        TWO_DIMENSIONAL_CLASS => resolve_two_dimensional(object, global_shape, labs),
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
    let dimension = optional_positive_property(object, "Dimension")?.unwrap_or_else(|| {
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
    let partition = optional_vector_property(object, "Partition")?
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
    let grid = optional_vector_property(object, "WorkerGrid")?
        .map(|values| pair_from_values(&values, "worker grid"))
        .transpose()?
        .unwrap_or_else(|| default_worker_grid(labs));
    if u64::from(grid[0]) * u64::from(grid[1]) != u64::from(labs.0) {
        return Err(error(
            "worker grid must contain exactly one position per lab",
        ));
    }
    let block_size = optional_positive_property(object, "BlockSize")?.unwrap_or(64);
    let orientation = object
        .properties
        .get("Orientation")
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
        "Dimension".into(),
        dimension.map_or_else(empty_vector, exact_scalar),
    );
    properties.insert(
        "Partition".into(),
        partition.map_or_else(empty_vector, vector_value),
    );
    properties.insert(
        "GlobalSize".into(),
        global_shape.map_or_else(empty_vector, vector_value),
    );
    object(ONE_DIMENSIONAL_CLASS, properties)
}

fn object(class_name: &str, properties: HashMap<String, Value>) -> Value {
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
    name: &str,
) -> Result<Option<u64>, RuntimeError> {
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
    name: &str,
) -> Result<Option<Vec<u64>>, RuntimeError> {
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
    if optional_vector_property(object, "GlobalSize")?.is_some_and(|declared| declared != shape) {
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
