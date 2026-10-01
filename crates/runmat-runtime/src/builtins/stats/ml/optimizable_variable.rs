//! MATLAB-compatible `optimizableVariable` metadata constructor.

use std::collections::{BTreeMap, HashSet};

use runmat_builtins::{
    BuiltinErrorDescriptor, OptimizableVariableOption, OptimizationVariableTransform as Transform,
    OptimizationVariableType as VariableType, OPTIMIZABLE_VARIABLE_ERROR_FLOW as ERROR_FLOW,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_OPTION as ERROR_INVALID_OPTION,
    OPTIMIZABLE_VARIABLE_ERROR_INVALID_RANGE as ERROR_INVALID_RANGE,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, LogicalArray, ObjectInstance, StringArray, Tensor, Value,
};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor as tensor_helpers;
use crate::builtins::common::validation::integer_is_exact_f64;
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, GpuGatherRetry, RuntimeError,
};

const NAME: &str = "optimizableVariable";
const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::standard::OPTIMIZABLE_VARIABLE;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::stats::ml::optimizable_variable"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("optimizable-variable-metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host metadata construction for bayesopt; gpuArray range inputs are gathered before validation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::stats::ml::optimizable_variable"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "optimizableVariable constructs host metadata and terminates fusion planning.",
};

#[runtime_builtin(
    name = "optimizableVariable",
    category = "stats/ml",
    summary = "Create Bayesian optimization variable metadata.",
    keywords = "optimizableVariable,bayesopt,optimization,variable,Type,Transform,Optimize",
    accel = "cpu",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::optimizable_variable"
)]
async fn optimizable_variable_builtin(
    name: Value,
    range: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !rest.len().is_multiple_of(2) {
        return Err(error(
            "optimizableVariable: name-value options must be paired",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    // Reject host-visible invalid controls before requesting any provider data.
    for pair in rest.chunks_exact(2) {
        if !matches!(pair[0], Value::GpuTensor(_)) {
            let name = parse_text_scalar(&pair[0], "option name", &ERROR_INVALID_OPTION)?;
            if OptimizableVariableOption::from_public_name(&name).is_none() {
                return Err(error(
                    format!("optimizableVariable: unknown option '{name}'"),
                    &ERROR_INVALID_OPTION,
                ));
            }
        }
    }
    if !matches!(name, Value::GpuTensor(_)) {
        let variable_name = parse_text_scalar(&name, "Name", &ERROR_INVALID_ARGUMENT)?;
        if variable_name.trim().is_empty() {
            return Err(error(
                "optimizableVariable: Name must be a nonempty text scalar",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
    }
    let name = gather_if_needed_async(&name)
        .await
        .map_err(|err| remap_flow(err, "name"))?;
    let range = gather_if_needed_async(&range)
        .await
        .map_err(|err| remap_flow(err, "range"))?;
    let mut gathered = Vec::with_capacity(rest.len());
    for value in rest {
        gathered.push(
            gather_if_needed_async(&value)
                .await
                .map_err(|err| remap_flow(err, "name-value argument"))?,
        );
    }

    let variable_name = parse_text_scalar(&name, "Name", &ERROR_INVALID_ARGUMENT)?;
    if variable_name.trim().is_empty() {
        return Err(error(
            "optimizableVariable: Name must be a nonempty text scalar",
            &ERROR_INVALID_ARGUMENT,
        ));
    }
    let options = Options::parse(gathered)?;
    let range = OptimizableRange::parse(range, options.var_type)?;
    let var_type = options.var_type.unwrap_or(range.inferred_type());
    range.validate_for(var_type, options.transform)?;

    let mut object = ObjectInstance::new(CLASS_IDENTITY);
    object
        .properties
        .insert("Name".to_string(), Value::String(variable_name));
    object
        .properties
        .insert("Range".to_string(), range.value().clone());
    object.properties.insert(
        "Type".to_string(),
        Value::String(var_type.public_name().to_string()),
    );
    object.properties.insert(
        "Transform".to_string(),
        Value::String(options.transform.public_name().to_string()),
    );
    object
        .properties
        .insert("Optimize".to_string(), Value::Bool(options.optimize));
    Ok(Value::Object(object))
}

fn variable_type(value: &Value) -> BuiltinResult<VariableType> {
    let text = parse_text_scalar(value, "Type", &ERROR_INVALID_OPTION)?;
    VariableType::from_public_name(&text).ok_or_else(|| {
        error(
            format!("optimizableVariable: unsupported Type '{text}'"),
            &ERROR_INVALID_OPTION,
        )
    })
}

fn transform(value: &Value) -> BuiltinResult<Transform> {
    let text = parse_text_scalar(value, "Transform", &ERROR_INVALID_OPTION)?;
    Transform::from_public_name(&text).ok_or_else(|| {
        error(
            format!("optimizableVariable: unsupported Transform '{text}'"),
            &ERROR_INVALID_OPTION,
        )
    })
}

#[derive(Debug)]
struct Options {
    var_type: Option<VariableType>,
    transform: Transform,
    optimize: bool,
}

impl Options {
    fn parse(args: Vec<Value>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(error(
                "optimizableVariable: name-value options must be paired",
                &ERROR_INVALID_ARGUMENT,
            ));
        }
        let mut options = Self {
            var_type: None,
            transform: Transform::None,
            optimize: true,
        };
        let mut pairs = BTreeMap::new();
        for pair in args.chunks_exact(2) {
            let name = parse_text_scalar(&pair[0], "option name", &ERROR_INVALID_OPTION)?
                .trim()
                .to_ascii_lowercase();
            let option = OptimizableVariableOption::from_public_name(&name).ok_or_else(|| {
                error(
                    format!("optimizableVariable: unknown option '{name}'"),
                    &ERROR_INVALID_OPTION,
                )
            })?;
            pairs.insert(option, pair[1].clone());
        }
        for (option, value) in pairs {
            match option {
                OptimizableVariableOption::Type => options.var_type = Some(variable_type(&value)?),
                OptimizableVariableOption::Transform => options.transform = transform(&value)?,
                OptimizableVariableOption::Optimize => {
                    options.optimize = parse_bool(&value, "Optimize")?
                }
            }
        }
        Ok(options)
    }
}

#[derive(Debug, Clone)]
enum OptimizableRange {
    Numeric {
        value: Value,
        lower: f64,
        upper: f64,
    },
    Categorical {
        value: Value,
        categories: Vec<String>,
    },
}

impl OptimizableRange {
    fn parse(value: Value, explicit_type: Option<VariableType>) -> BuiltinResult<Self> {
        match value {
            Value::Tensor(tensor) => {
                if tensor.integer_storage().is_some_and(|storage| {
                    (0..storage.len()).any(|index| {
                        storage
                            .value_at(index)
                            .is_none_or(|value| !integer_is_exact_f64(&value))
                    })
                }) {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be exactly representable as double",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                let values = tensor_helpers::tensor_values_f64(&tensor);
                Self::numeric_from_values(Value::Tensor(tensor), &values)
            }
            Value::Num(n) => Self::numeric_from_values(
                Value::Tensor(Tensor::new(vec![n], vec![1, 1]).unwrap()),
                &[n],
            ),
            Value::Int(i) => {
                if !integer_is_exact_f64(&i) {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be exactly representable as double",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                let n = i.to_f64();
                Self::numeric_from_values(
                    Value::Tensor(Tensor::new(vec![n], vec![1, 1]).unwrap()),
                    &[n],
                )
            }
            Value::StringArray(array) => {
                let categories = text_categories_from_string_array(&array)?;
                Ok(Self::Categorical {
                    value: Value::StringArray(array),
                    categories,
                })
            }
            Value::String(text) if explicit_type == Some(VariableType::Categorical) => {
                check_categories(std::slice::from_ref(&text))?;
                Ok(Self::Categorical {
                    value: Value::String(text.clone()),
                    categories: vec![text],
                })
            }
            Value::Cell(cell) => {
                let categories = text_categories_from_cell(&cell)?;
                Ok(Self::Categorical {
                    value: Value::Cell(cell),
                    categories,
                })
            }
            Value::CharArray(chars) if explicit_type == Some(VariableType::Categorical) => {
                let category = char_row_to_string(&chars, "Range", &ERROR_INVALID_RANGE)?;
                check_categories(std::slice::from_ref(&category))?;
                let value = Value::Cell(
                    CellArray::new(vec![Value::String(category.clone())], 1, 1)
                        .map_err(|err| error(err, &ERROR_INVALID_RANGE))?,
                );
                Ok(Self::Categorical {
                    value,
                    categories: vec![category],
                })
            }
            other => Err(error(
                format!("optimizableVariable: unsupported Range value {other:?}"),
                &ERROR_INVALID_RANGE,
            )),
        }
    }

    fn numeric_from_values(value: Value, values: &[f64]) -> BuiltinResult<Self> {
        if values.len() != 2 {
            return Err(error(
                "optimizableVariable: numeric Range must contain exactly two elements",
                &ERROR_INVALID_RANGE,
            ));
        }
        let lower = values[0];
        let upper = values[1];
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(error(
                "optimizableVariable: numeric Range bounds must be finite and increasing",
                &ERROR_INVALID_RANGE,
            ));
        }
        Ok(Self::Numeric {
            value,
            lower,
            upper,
        })
    }

    fn inferred_type(&self) -> VariableType {
        match self {
            Self::Numeric { .. } => VariableType::Real,
            Self::Categorical { .. } => VariableType::Categorical,
        }
    }

    fn value(&self) -> &Value {
        match self {
            Self::Numeric { value, .. } | Self::Categorical { value, .. } => value,
        }
    }

    fn validate_for(&self, var_type: VariableType, transform: Transform) -> BuiltinResult<()> {
        match (self, var_type) {
            (Self::Numeric { lower, .. }, VariableType::Real) => {
                validate_real_transform(*lower, transform)
            }
            (Self::Numeric { lower, upper, .. }, VariableType::Integer) => {
                if lower.fract().abs() > f64::EPSILON || upper.fract().abs() > f64::EPSILON {
                    return Err(error(
                        "optimizableVariable: integer Range bounds must be whole numbers",
                        &ERROR_INVALID_RANGE,
                    ));
                }
                validate_integer_transform(*lower, transform)
            }
            (Self::Numeric { .. }, VariableType::Categorical) => Err(error(
                "optimizableVariable: categorical variables require a categorical text Range",
                &ERROR_INVALID_RANGE,
            )),
            (Self::Categorical { categories, .. }, VariableType::Categorical) => {
                check_categories(categories)?;
                if transform != Transform::None {
                    return Err(error(
                        "optimizableVariable: categorical variables only support Transform 'none'",
                        &ERROR_INVALID_OPTION,
                    ));
                }
                Ok(())
            }
            (Self::Categorical { .. }, _) => Err(error(
                "optimizableVariable: real and integer variables require a two-element numeric Range",
                &ERROR_INVALID_RANGE,
            )),
        }
    }
}

fn validate_real_transform(lower: f64, transform: Transform) -> BuiltinResult<()> {
    if transform == Transform::Log && lower <= 0.0 {
        return Err(error(
            "optimizableVariable: log transform requires positive real Range bounds",
            &ERROR_INVALID_RANGE,
        ));
    }
    Ok(())
}

fn validate_integer_transform(lower: f64, transform: Transform) -> BuiltinResult<()> {
    if transform == Transform::Log && lower < 0.0 {
        return Err(error(
            "optimizableVariable: log transform requires nonnegative integer Range bounds",
            &ERROR_INVALID_RANGE,
        ));
    }
    Ok(())
}

fn text_categories_from_string_array(array: &StringArray) -> BuiltinResult<Vec<String>> {
    let categories = array.data.clone();
    check_categories(&categories)?;
    Ok(categories)
}

fn text_categories_from_cell(cell: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut categories = Vec::with_capacity(cell.data.len());
    for value in &cell.data {
        categories.push(parse_text_scalar(value, "Range", &ERROR_INVALID_RANGE)?);
    }
    check_categories(&categories)?;
    Ok(categories)
}

fn check_categories(categories: &[String]) -> BuiltinResult<()> {
    if categories.is_empty() {
        return Err(error(
            "optimizableVariable: categorical Range must contain at least one category",
            &ERROR_INVALID_RANGE,
        ));
    }
    let mut seen = HashSet::with_capacity(categories.len());
    for category in categories {
        if category.trim().is_empty() {
            return Err(error(
                "optimizableVariable: categorical Range values must be nonempty text",
                &ERROR_INVALID_RANGE,
            ));
        }
        if !seen.insert(category.clone()) {
            return Err(error(
                "optimizableVariable: categorical Range values must be unique",
                &ERROR_INVALID_RANGE,
            ));
        }
    }
    Ok(())
}

fn parse_text_scalar(
    value: &Value,
    label: &str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => char_row_to_string(chars, label, descriptor),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(error(
            format!("optimizableVariable: {label} must be a text scalar, got {other:?}"),
            descriptor,
        )),
    }
}

fn char_row_to_string(
    chars: &CharArray,
    label: &str,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    if chars.rows != 1 {
        return Err(error(
            format!("optimizableVariable: {label} char array must be a row vector"),
            descriptor,
        ));
    }
    Ok(chars.data.iter().collect())
}

fn parse_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) if i.to_f64() == 0.0 || i.to_f64() == 1.0 => Ok(i.to_f64() != 0.0),
        other => Err(error(
            format!("optimizableVariable: {label} must be a scalar logical value, got {other:?}"),
            &ERROR_INVALID_OPTION,
        )),
    }
}

fn error(message: impl Into<String>, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_flow(err: RuntimeError, label: &str) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "optimizableVariable: failed to gather {label}: {}",
        err.message()
    ))
    .with_builtin(NAME)
    .with_source(err)
    .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = ERROR_FLOW.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::IntegerStorage;

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        object
    }

    fn typed_integer_range(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![1, len]).expect("integer range");
        Value::Tensor(tensor)
    }

    #[test]
    fn builds_real_variable_with_defaults() {
        let out = block_on(optimizable_variable_builtin(
            Value::String("depth".into()),
            Value::Tensor(Tensor::new(vec![1.0, 10.0], vec![1, 2]).unwrap()),
            vec![],
        ))
        .expect("optimizableVariable");
        let object = object(out);
        assert!(object.class_name.is(CLASS_IDENTITY));
        assert_eq!(
            object.properties.get("Name"),
            Some(&Value::String("depth".into()))
        );
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("real".into()))
        );
        assert_eq!(
            object.properties.get("Transform"),
            Some(&Value::String("none".into()))
        );
        assert_eq!(object.properties.get("Optimize"), Some(&Value::Bool(true)));
    }

    #[test]
    fn builds_integer_log_variable_with_optimize_false() {
        let out = block_on(optimizable_variable_builtin(
            Value::String("trees".into()),
            Value::Tensor(Tensor::new(vec![0.0, 1000.0], vec![1, 2]).unwrap()),
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
                Value::String("Transform".into()),
                Value::String("log".into()),
                Value::String("Optimize".into()),
                Value::Bool(false),
            ],
        ))
        .expect("optimizableVariable integer");
        let object = object(out);
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("integer".into()))
        );
        assert_eq!(
            object.properties.get("Transform"),
            Some(&Value::String("log".into()))
        );
        assert_eq!(object.properties.get("Optimize"), Some(&Value::Bool(false)));
    }

    #[test]
    fn typed_integer_range_bounds_are_read_from_exact_storage() {
        let range = typed_integer_range(IntegerStorage::I16(vec![0, 1000]));
        let out = block_on(optimizable_variable_builtin(
            Value::String("trees".into()),
            range,
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
                Value::String("Transform".into()),
                Value::String("log".into()),
            ],
        ))
        .expect("optimizableVariable integer");
        let object = object(out);
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("integer".into()))
        );
        let Some(Value::Tensor(range)) = object.properties.get("Range") else {
            panic!("expected tensor Range");
        };
        assert_eq!(
            range.integer_storage(),
            Some(&IntegerStorage::I16(vec![0, 1000]))
        );
        assert_eq!(range.materialize_f64(), vec![0.0, 1000.0]);
    }

    #[test]
    fn builds_categorical_variable_from_cellstr() {
        let categories = Value::Cell(
            CellArray::new(
                vec![Value::String("linear".into()), Value::String("rbf".into())],
                1,
                2,
            )
            .unwrap(),
        );
        let out = block_on(optimizable_variable_builtin(
            Value::String("kernel".into()),
            categories.clone(),
            vec![
                Value::String("Type".into()),
                Value::String("categorical".into()),
            ],
        ))
        .expect("optimizableVariable categorical");
        let object = object(out);
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("categorical".into()))
        );
        assert_eq!(object.properties.get("Range"), Some(&categories));
    }

    #[test]
    fn infers_categorical_type_from_text_range() {
        let out = block_on(optimizable_variable_builtin(
            Value::String("method".into()),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            vec![],
        ))
        .expect("optimizableVariable categorical inferred");
        let object = object(out);
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("categorical".into()))
        );
    }

    #[test]
    fn accepts_explicit_categorical_scalar_string_range() {
        let out = block_on(optimizable_variable_builtin(
            Value::String("method".into()),
            Value::String("rbf".into()),
            vec![
                Value::String("Type".into()),
                Value::String("categorical".into()),
            ],
        ))
        .expect("optimizableVariable scalar categorical");
        let object = object(out);
        assert_eq!(
            object.properties.get("Range"),
            Some(&Value::String("rbf".into()))
        );
        assert_eq!(
            object.properties.get("Type"),
            Some(&Value::String("categorical".into()))
        );
    }

    #[test]
    fn rejects_invalid_ranges_and_options() {
        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            vec![],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("finite and increasing"));

        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::Tensor(Tensor::new(vec![1.5, 3.0], vec![1, 2]).unwrap()),
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("whole numbers"));

        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::Tensor(Tensor::new(vec![0.0, 3.0], vec![1, 2]).unwrap()),
            vec![
                Value::String("Transform".into()),
                Value::String("log".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("positive real"));

        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::Tensor(Tensor::new(vec![-1.0, 3.0], vec![1, 2]).unwrap()),
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
                Value::String("Transform".into()),
                Value::String("log".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("nonnegative integer"));

        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            vec![Value::String("Type".into()), Value::String("int".into())],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("unsupported Type"));

        let err = block_on(optimizable_variable_builtin(
            Value::String("bad".into()),
            Value::StringArray(StringArray::new(vec!["a".into(), "a".into()], vec![1, 2]).unwrap()),
            vec![
                Value::String("Type".into()),
                Value::String("categorical".into()),
            ],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("unique"));
    }

    #[test]
    fn wide_integer_bounds_require_exact_binary64_conversion() {
        let exact_lower = 1_u64 << 54;
        let exact_upper = exact_lower + 4;
        let accepted = block_on(optimizable_variable_builtin(
            Value::String("wide".into()),
            typed_integer_range(IntegerStorage::U64(vec![exact_lower, exact_upper])),
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
            ],
        ))
        .expect("aligned wide bounds remain exact");
        assert!(object(accepted).class_name.is(CLASS_IDENTITY));

        let error = block_on(optimizable_variable_builtin(
            Value::String("rounded".into()),
            typed_integer_range(IntegerStorage::U64(vec![
                (1_u64 << 53) + 1,
                (1_u64 << 53) + 3,
            ])),
            vec![
                Value::String("Type".into()),
                Value::String("integer".into()),
            ],
        ))
        .expect_err("rounded bounds must not become a different search domain");
        assert_eq!(error.identifier(), ERROR_INVALID_RANGE.identifier);
    }
}
