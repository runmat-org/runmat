//! MATLAB-compatible `issortedrows` builtin.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ComplexStorage, ComplexTensor, IntegerStorage, LogicalArray, NumericScalar,
    NumericStorage, Value,
};
use runmat_macros::runtime_builtin;

use super::{float_order::SetFloat, integer_order, type_resolvers::bool_output_type};
use crate::build_runtime_error;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::array::sorting_sets::issortedrows"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issortedrows",
    op_kind: GpuOpKind::Custom("predicate"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: true,
    notes: "Interactive GPU input is a mode-gated RunMat extension and uses authoritative typed gather before the scalar predicate.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::array::sorting_sets::issortedrows"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issortedrows",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "`issortedrows` is a scalar predicate and terminates fusion chains.",
};

const BUILTIN_NAME: &str = "issortedrows";

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when rows are sorted according to the requested row order.",
}];

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input column vector, matrix, or table.",
    },
    BuiltinParamDescriptor {
        name: "args",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Column selectors, direction modes, ComparisonMethod, and MissingPlacement.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "tf = issortedrows(A, column, direction, options)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTEDROWS.INVALID_INPUT",
    identifier: Some("RunMat:issortedrows:InvalidInput"),
    when: "Input or row-sorting arguments are invalid.",
    message: "issortedrows: invalid input",
};

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSORTEDROWS.INVALID_ARGUMENT",
    identifier: Some("RunMat:issortedrows:InvalidArgument"),
    when: "A column, direction, comparison, missing-placement, or output argument is invalid.",
    message: "issortedrows: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INVALID_ARGUMENT];

const GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "issortedrows-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "issortedrows with an interactive resident GPU input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IssortedrowsGpuInputExtension"),
};

const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GPU_INPUT_EXTENSION];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented matrix domain includes all eight real integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "column",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented nonzero integer column scalar or vector accepts every integer class and integer-valued double.",
    },
];

const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = issortedrows(integer_A, integer_columns, direction, options)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer rows and column selectors are compared exactly across ascending, descending, monotonic, strict, and per-column direction forms. tf is scalar logical. Interactive GPU input is a mode-gated RunMat extension that gathers exact typed storage.",
    }];

pub const ISSORTEDROWS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> crate::RuntimeError {
    error(&ERROR_INVALID_ARGUMENT, message)
}

fn invalid_input(message: impl Into<String>) -> crate::RuntimeError {
    error(&ERROR_INVALID_INPUT, message)
}

#[runtime_builtin(
    name = "issortedrows",
    category = "array/sorting_sets",
    summary = "Determine whether matrix or table rows are sorted.",
    keywords = "issortedrows,sortrows,rows,sorted,monotonic,strict,table",
    accel = "sink",
    sink = true,
    type_resolver(bool_output_type),
    descriptor(crate::builtins::array::sorting_sets::issortedrows::ISSORTEDROWS_DESCRIPTOR),
    extensions(EXTENSIONS),
    integer_capabilities(INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::sorting_sets::issortedrows"
)]
async fn issortedrows_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 1) {
        return Err(invalid_argument(
            "issortedrows: too many output arguments; maximum is 1",
        ));
    }
    let value = match value {
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &GPU_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
            gpu_helpers::gather_value_async(&Value::GpuTensor(handle)).await?
        }
        other => other,
    };
    if matches!(&value, Value::Object(object) if object.is_class(crate::builtins::table::TABLE_CLASS))
    {
        let evaluation =
            crate::builtins::array::sorting_sets::sortrows::evaluate(value, &rest).await?;
        return Ok(Value::Bool(indices_are_identity(
            evaluation.indices_value(),
        )));
    }
    let shape = value_shape(&value);
    ensure_matrix(&shape)?;
    let rows = rows(&shape);
    let cols = cols(&shape);
    let args = Args::parse(&rest, cols)?;
    let sorted = match value {
        Value::Tensor(tensor) => {
            let storage = tensor
                .into_numeric_storage()
                .map_err(|message| invalid_input(format!("issortedrows: {message}")))?;
            check_numeric(&storage, rows, cols, &args)
        }
        Value::Num(value) => {
            let storage = NumericStorage::F64(vec![value]);
            check_numeric(&storage, 1, 1, &args)
        }
        Value::Int(value) => {
            let storage = NumericStorage::from_integer_storage(IntegerStorage::from_scalar(value));
            check_numeric(&storage, 1, 1, &args)
        }
        Value::LogicalArray(array) => check_logical(&array, rows, cols, &args),
        Value::Bool(value) => {
            let array = LogicalArray::new(vec![u8::from(value)], vec![1, 1])
                .map_err(|message| invalid_input(format!("issortedrows: {message}")))?;
            check_logical(&array, 1, 1, &args)
        }
        Value::ComplexTensor(tensor) => check_complex(&tensor, rows, cols, &args),
        Value::Complex(real, imaginary) => {
            let tensor = ComplexTensor::new(vec![(real, imaginary)], vec![1, 1])
                .map_err(|message| invalid_input(format!("issortedrows: {message}")))?;
            check_complex(&tensor, 1, 1, &args)
        }
        Value::CharArray(array) => check_char(&array, rows, cols, &args),
        other => {
            return Err(invalid_input(format!(
                "issortedrows: unsupported input type {other:?}"
            )))
        }
    };
    Ok(Value::Bool(sorted))
}

fn indices_are_identity(indices: Value) -> bool {
    match indices {
        Value::Tensor(tensor) => tensor.as_f64_slice().is_some_and(|values| {
            values
                .iter()
                .enumerate()
                .all(|(index, value)| *value == index as f64 + 1.0)
        }),
        Value::Num(value) => value == 1.0,
        Value::Int(value) => value.to_i64() == 1,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Ascend,
    Descend,
    Monotonic,
    StrictAscend,
    StrictDescend,
    StrictMonotonic,
}

impl Direction {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "ascend" => Some(Self::Ascend),
            "descend" => Some(Self::Descend),
            "monotonic" => Some(Self::Monotonic),
            "strictascend" => Some(Self::StrictAscend),
            "strictdescend" => Some(Self::StrictDescend),
            "strictmonotonic" => Some(Self::StrictMonotonic),
            _ => None,
        }
    }

    fn fixed(self) -> Option<BaseDirection> {
        match self {
            Self::Ascend | Self::StrictAscend => Some(BaseDirection::Ascend),
            Self::Descend | Self::StrictDescend => Some(BaseDirection::Descend),
            Self::Monotonic | Self::StrictMonotonic => None,
        }
    }

    fn strict(self) -> bool {
        matches!(
            self,
            Self::StrictAscend | Self::StrictDescend | Self::StrictMonotonic
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BaseDirection {
    Ascend,
    Descend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparisonMethod {
    Auto,
    Real,
    Abs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissingPlacement {
    Auto,
    First,
    Last,
}

#[derive(Clone, Copy, Debug)]
struct ColumnCheck {
    index: usize,
    direction: Direction,
}

#[derive(Clone, Debug)]
struct Args {
    columns: Vec<ColumnCheck>,
    comparison: ComparisonMethod,
    missing: MissingPlacement,
}

impl Args {
    fn parse(rest: &[Value], num_cols: usize) -> crate::BuiltinResult<Self> {
        let mut columns: Option<Vec<ColumnCheck>> = None;
        let mut directions: Option<Vec<Direction>> = None;
        let mut comparison = ComparisonMethod::Auto;
        let mut missing = MissingPlacement::Auto;
        let mut index = 0usize;
        while index < rest.len() {
            if columns.is_none() {
                if let Some(parsed) = parse_columns(&rest[index], num_cols)? {
                    columns = Some(parsed);
                    index += 1;
                    continue;
                }
            }
            if let Some(parsed) = parse_direction_list(&rest[index])? {
                if directions.is_some() {
                    return Err(invalid_argument(
                        "issortedrows: sorting direction specified more than once",
                    ));
                }
                directions = Some(parsed);
                index += 1;
                continue;
            }
            let Some(keyword) = tensor::value_to_string(&rest[index]) else {
                return Err(invalid_argument(format!(
                    "issortedrows: invalid argument {:?}",
                    rest[index]
                )));
            };
            match keyword.trim().to_ascii_lowercase().as_str() {
                "comparisonmethod" => {
                    index += 1;
                    let value = rest
                        .get(index)
                        .and_then(tensor::value_to_string)
                        .ok_or_else(|| {
                            invalid_argument(
                                "issortedrows: ComparisonMethod expects a string value",
                            )
                        })?;
                    comparison = match value.trim().to_ascii_lowercase().as_str() {
                        "auto" => ComparisonMethod::Auto,
                        "real" => ComparisonMethod::Real,
                        "abs" | "magnitude" => ComparisonMethod::Abs,
                        other => {
                            return Err(invalid_argument(format!(
                                "issortedrows: unsupported ComparisonMethod '{other}'"
                            )))
                        }
                    };
                    index += 1;
                }
                "missingplacement" => {
                    index += 1;
                    let value = rest
                        .get(index)
                        .and_then(tensor::value_to_string)
                        .ok_or_else(|| {
                            invalid_argument(
                                "issortedrows: MissingPlacement expects a string value",
                            )
                        })?;
                    missing = match value.trim().to_ascii_lowercase().as_str() {
                        "auto" => MissingPlacement::Auto,
                        "first" => MissingPlacement::First,
                        "last" => MissingPlacement::Last,
                        other => {
                            return Err(invalid_argument(format!(
                                "issortedrows: unsupported MissingPlacement '{other}'"
                            )))
                        }
                    };
                    index += 1;
                }
                other => {
                    return Err(invalid_argument(format!(
                        "issortedrows: unexpected argument '{other}'"
                    )))
                }
            }
        }

        let mut columns = columns.unwrap_or_else(|| {
            (0..num_cols)
                .map(|index| ColumnCheck {
                    index,
                    direction: Direction::Ascend,
                })
                .collect()
        });
        if let Some(directions) = directions {
            if directions.len() == 1 {
                for column in &mut columns {
                    column.direction = directions[0];
                }
            } else if directions.len() == columns.len() {
                for (column, direction) in columns.iter_mut().zip(directions) {
                    column.direction = direction;
                }
            } else {
                return Err(invalid_argument(format!(
                    "issortedrows: direction list length {} must be 1 or match {} selected columns",
                    directions.len(),
                    columns.len()
                )));
            }
        }
        Ok(Self {
            columns,
            comparison,
            missing,
        })
    }

    fn strict(&self) -> bool {
        self.columns.iter().any(|column| column.direction.strict())
    }
}

fn parse_direction_list(value: &Value) -> crate::BuiltinResult<Option<Vec<Direction>>> {
    let strings = match value {
        Value::StringArray(array) => Some(array.data.clone()),
        Value::Cell(cell) => {
            let mut strings = Vec::with_capacity(cell.data.len());
            for value in &cell.data {
                let Some(value) = tensor::value_to_string(value) else {
                    return Err(invalid_argument(
                        "issortedrows: direction cell arrays must contain character vectors or strings",
                    ));
                };
                strings.push(value);
            }
            Some(strings)
        }
        _ => tensor::value_to_string(value).map(|value| vec![value]),
    };
    let Some(strings) = strings else {
        return Ok(None);
    };
    if strings.is_empty() {
        return Err(invalid_argument(
            "issortedrows: direction list must not be empty",
        ));
    }
    let mut directions = Vec::with_capacity(strings.len());
    for value in strings {
        let Some(direction) = Direction::parse(&value) else {
            return Ok(None);
        };
        directions.push(direction);
    }
    Ok(Some(directions))
}

fn parse_columns(value: &Value, num_cols: usize) -> crate::BuiltinResult<Option<Vec<ColumnCheck>>> {
    let values = match value {
        Value::Int(value) => vec![value
            .try_to_i64()
            .ok_or_else(|| invalid_argument("issortedrows: column index is out of range"))?],
        Value::Num(value) => vec![floating_column(*value)?],
        Value::Tensor(tensor) => {
            if !is_vector(&tensor.shape) {
                return Err(invalid_argument(
                    "issortedrows: column specification must be a vector",
                ));
            }
            let mut values = Vec::with_capacity(tensor.len());
            for index in 0..tensor.len() {
                let value = tensor.numeric_value_at(index).ok_or_else(|| {
                    invalid_argument("issortedrows: column storage is inconsistent")
                })?;
                values.push(numeric_column(value)?);
            }
            values
        }
        _ => return Ok(None),
    };
    let mut columns = Vec::with_capacity(values.len());
    for value in values {
        if value == 0 {
            return Err(invalid_argument(
                "issortedrows: column indices must be nonzero",
            ));
        }
        let absolute = usize::try_from(value.unsigned_abs())
            .map_err(|_| invalid_argument("issortedrows: column index is out of range"))?;
        if absolute == 0 || absolute > num_cols {
            return Err(invalid_argument(format!(
                "issortedrows: column index {absolute} exceeds matrix with {num_cols} columns"
            )));
        }
        columns.push(ColumnCheck {
            index: absolute - 1,
            direction: if value > 0 {
                Direction::Ascend
            } else {
                Direction::Descend
            },
        });
    }
    Ok(Some(columns))
}

fn numeric_column(value: NumericScalar) -> crate::BuiltinResult<i64> {
    match value {
        NumericScalar::F64(value) => floating_column(value),
        NumericScalar::F32(value) => floating_column(f64::from(value)),
        value => value
            .into_int_value()
            .and_then(|value| value.try_to_i64())
            .ok_or_else(|| invalid_argument("issortedrows: column index is out of range")),
    }
}

fn floating_column(value: f64) -> crate::BuiltinResult<i64> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(invalid_argument(
            "issortedrows: column indices must be finite integers",
        ));
    }
    if value < i64::MIN as f64 || value >= i64::MAX as f64 {
        return Err(invalid_argument(
            "issortedrows: column index is out of range",
        ));
    }
    let integer = value as i64;
    if integer as f64 != value {
        return Err(invalid_argument(
            "issortedrows: column index is not exactly representable",
        ));
    }
    Ok(integer)
}

fn check_numeric(storage: &NumericStorage, rows: usize, _cols: usize, args: &Args) -> bool {
    if args.strict()
        && args.columns.iter().any(|column| {
            (0..rows).any(|row| {
                storage
                    .value_at(row + column.index * rows)
                    .is_some_and(numeric_is_missing)
            })
        })
    {
        return false;
    }
    check_adjacent(rows, args, |row, column, direction| {
        let left = storage
            .value_at(row + column * rows)
            .expect("validated issortedrows numeric index");
        let right = storage
            .value_at(row + 1 + column * rows)
            .expect("validated issortedrows numeric index");
        compare_numeric(left, right, direction, args)
    })
}

fn check_logical(array: &LogicalArray, rows: usize, cols: usize, args: &Args) -> bool {
    let storage = NumericStorage::from_integer_storage(IntegerStorage::U8(array.data.clone()));
    check_numeric(&storage, rows, cols, args)
}

fn check_complex(tensor: &ComplexTensor, rows: usize, cols: usize, args: &Args) -> bool {
    match tensor.complex_storage() {
        ComplexStorage::F64(values) => check_complex_values(values, rows, cols, args),
        ComplexStorage::F32(values) => check_complex_values(values, rows, cols, args),
        ComplexStorage::Integer(_) => {
            let values = tensor.materialize_f64();
            check_complex_values(&values, rows, cols, args)
        }
    }
}

fn check_complex_values<T: SetFloat>(
    values: &[(T, T)],
    rows: usize,
    _cols: usize,
    args: &Args,
) -> bool {
    if args.strict()
        && args.columns.iter().any(|column| {
            (0..rows).any(|row| complex_is_missing(values[row + column.index * rows]))
        })
    {
        return false;
    }
    check_adjacent(rows, args, |row, column, direction| {
        compare_complex(
            values[row + column * rows],
            values[row + 1 + column * rows],
            direction,
            args,
        )
    })
}

fn check_char(array: &CharArray, rows: usize, cols: usize, args: &Args) -> bool {
    check_adjacent(rows, args, |row, column, direction| {
        let left = array.data[row * cols + column];
        let right = array.data[(row + 1) * cols + column];
        apply_direction(left.cmp(&right), direction)
    })
}

fn check_adjacent(
    rows: usize,
    args: &Args,
    mut compare: impl FnMut(usize, usize, BaseDirection) -> Ordering,
) -> bool {
    if rows <= 1 || args.columns.is_empty() {
        return true;
    }
    let mut monotonic = vec![None; args.columns.len()];
    for row in 0..rows - 1 {
        let mut row_order = Ordering::Equal;
        for (index, column) in args.columns.iter().enumerate() {
            let direction = match column.direction.fixed() {
                Some(direction) => direction,
                None => match monotonic[index] {
                    Some(direction) => direction,
                    None => {
                        let ascending = compare(row, column.index, BaseDirection::Ascend);
                        if ascending == Ordering::Equal {
                            continue;
                        }
                        let direction = if ascending == Ordering::Less {
                            BaseDirection::Ascend
                        } else {
                            BaseDirection::Descend
                        };
                        monotonic[index] = Some(direction);
                        direction
                    }
                },
            };
            row_order = compare(row, column.index, direction);
            if row_order != Ordering::Equal {
                break;
            }
        }
        if row_order == Ordering::Greater || (row_order == Ordering::Equal && args.strict()) {
            return false;
        }
    }
    true
}

fn compare_numeric(
    left: NumericScalar,
    right: NumericScalar,
    direction: BaseDirection,
    args: &Args,
) -> Ordering {
    match (left, right) {
        (NumericScalar::F64(left), NumericScalar::F64(right)) => {
            compare_real(left, right, direction, args)
        }
        (NumericScalar::F32(left), NumericScalar::F32(right)) => {
            compare_real(left, right, direction, args)
        }
        (left, right) => integer_order::compare(
            &left.into_int_value().expect("homogeneous integer storage"),
            &right.into_int_value().expect("homogeneous integer storage"),
            matches!(direction, BaseDirection::Descend),
            matches!(args.comparison, ComparisonMethod::Abs),
        ),
    }
}

fn compare_real<T: SetFloat>(left: T, right: T, direction: BaseDirection, args: &Args) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => match resolve_missing(args.missing, direction) {
            MissingPlacement::First => Ordering::Less,
            _ => Ordering::Greater,
        },
        (false, true) => match resolve_missing(args.missing, direction) {
            MissingPlacement::First => Ordering::Greater,
            _ => Ordering::Less,
        },
        (false, false) => {
            let ordering = if matches!(args.comparison, ComparisonMethod::Abs) {
                let magnitude = left.abs().compare(right.abs());
                if magnitude == Ordering::Equal {
                    real_phase(left).compare(real_phase(right))
                } else {
                    magnitude
                }
            } else {
                left.compare(right)
            };
            apply_direction(ordering, direction)
        }
    }
}

fn compare_complex<T: SetFloat>(
    left: (T, T),
    right: (T, T),
    direction: BaseDirection,
    args: &Args,
) -> Ordering {
    match (complex_is_missing(left), complex_is_missing(right)) {
        (true, true) => Ordering::Equal,
        (true, false) => match resolve_missing(args.missing, direction) {
            MissingPlacement::First => Ordering::Less,
            _ => Ordering::Greater,
        },
        (false, true) => match resolve_missing(args.missing, direction) {
            MissingPlacement::First => Ordering::Greater,
            _ => Ordering::Less,
        },
        (false, false) => {
            let ordering = match args.comparison {
                ComparisonMethod::Real => {
                    let real = left.0.compare(right.0);
                    if real == Ordering::Equal {
                        left.1.compare(right.1)
                    } else {
                        real
                    }
                }
                ComparisonMethod::Auto | ComparisonMethod::Abs => {
                    let magnitude = left.0.hypot(left.1).compare(right.0.hypot(right.1));
                    if magnitude == Ordering::Equal {
                        complex_phase(left).compare(complex_phase(right))
                    } else {
                        magnitude
                    }
                }
            };
            apply_direction(ordering, direction)
        }
    }
}

fn resolve_missing(missing: MissingPlacement, direction: BaseDirection) -> MissingPlacement {
    match missing {
        MissingPlacement::Auto => match direction {
            BaseDirection::Ascend => MissingPlacement::Last,
            BaseDirection::Descend => MissingPlacement::First,
        },
        other => other,
    }
}

fn apply_direction(ordering: Ordering, direction: BaseDirection) -> Ordering {
    match direction {
        BaseDirection::Ascend => ordering,
        BaseDirection::Descend => ordering.reverse(),
    }
}

fn numeric_is_missing(value: NumericScalar) -> bool {
    match value {
        NumericScalar::F64(value) => value.is_nan(),
        NumericScalar::F32(value) => value.is_nan(),
        _ => false,
    }
}

fn complex_is_missing<T: SetFloat>(value: (T, T)) -> bool {
    value.0.is_nan() || value.1.is_nan()
}

fn real_phase<T: SetFloat>(value: T) -> T {
    T::default().atan2(value)
}

fn complex_phase<T: SetFloat>((real, imaginary): (T, T)) -> T {
    let imaginary = if imaginary == T::default() {
        T::default()
    } else {
        imaginary
    };
    imaginary.atan2(real)
}

fn value_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::LogicalArray(array) => array.shape.clone(),
        Value::ComplexTensor(tensor) => tensor.shape.clone(),
        Value::CharArray(array) => array.shape.clone(),
        _ => vec![1, 1],
    }
}

fn ensure_matrix(shape: &[usize]) -> crate::BuiltinResult<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(invalid_input(
            "issortedrows: input must be a column vector or matrix",
        ))
    }
}

fn rows(shape: &[usize]) -> usize {
    match shape {
        [] | [_] => 1,
        [rows, ..] => *rows,
    }
}

fn cols(shape: &[usize]) -> usize {
    match shape {
        [] => 1,
        [length] => *length,
        [_, cols, ..] => *cols,
    }
}

fn is_vector(shape: &[usize]) -> bool {
    match shape {
        [] | [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{CellArray, IntValue, StringArray, Tensor};

    fn builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(issortedrows_builtin(value, rest))
    }

    fn directions(values: &[&str]) -> Value {
        Value::Cell(
            CellArray::new(
                values.iter().map(|value| Value::from(*value)).collect(),
                1,
                values.len(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn issortedrows_detects_sorted_and_unsorted_numeric_rows() {
        let sorted = Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap());
        assert_eq!(builtin(sorted, Vec::new()).unwrap(), Value::Bool(true));
        let unsorted = Value::Tensor(Tensor::new(vec![2.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
        assert_eq!(builtin(unsorted, Vec::new()).unwrap(), Value::Bool(false));
        assert_eq!(
            builtin(Value::Num(1.0), Vec::new()).unwrap(),
            Value::Bool(true)
        );
        let row = Tensor::new_integer(IntegerStorage::I16(vec![3, 2, 1]), vec![3]).unwrap();
        assert_eq!(
            builtin(Value::Tensor(row), Vec::new()).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn issortedrows_supports_all_direction_modes_exactly_for_wide_integers() {
        let ascending = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX, 4, 3, 2]),
            vec![3, 2],
        )
        .unwrap();
        assert_eq!(
            builtin(
                Value::Tensor(ascending.clone()),
                vec![Value::from("strictascend")]
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            builtin(
                Value::Tensor(ascending.clone()),
                vec![Value::from("monotonic")]
            )
            .unwrap(),
            Value::Bool(true)
        );
        let descending = Tensor::new_integer(
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 0, 2, 3, 4]),
            vec![3, 2],
        )
        .unwrap();
        assert_eq!(
            builtin(
                Value::Tensor(descending),
                vec![Value::from("strictmonotonic")]
            )
            .unwrap(),
            Value::Bool(true)
        );
        let duplicate =
            Tensor::new_integer(IntegerStorage::I64(vec![1, 1, 2, 2]), vec![2, 2]).unwrap();
        assert_eq!(
            builtin(Value::Tensor(duplicate), vec![Value::from("strictascend")]).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn issortedrows_accepts_every_integer_class_for_data_and_columns() {
        let data = vec![
            IntegerStorage::I8(vec![1, 2]),
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I32(vec![1, 2]),
            IntegerStorage::I64(vec![1, 2]),
            IntegerStorage::U8(vec![1, 2]),
            IntegerStorage::U16(vec![1, 2]),
            IntegerStorage::U32(vec![1, 2]),
            IntegerStorage::U64(vec![1, 2]),
        ];
        let columns = vec![
            IntegerStorage::I8(vec![1]),
            IntegerStorage::I16(vec![1]),
            IntegerStorage::I32(vec![1]),
            IntegerStorage::I64(vec![1]),
            IntegerStorage::U8(vec![1]),
            IntegerStorage::U16(vec![1]),
            IntegerStorage::U32(vec![1]),
            IntegerStorage::U64(vec![1]),
        ];
        for (data, column) in data.into_iter().zip(columns) {
            let data = Tensor::new_integer(data, vec![2, 1]).unwrap();
            let column = Tensor::new_integer(column, vec![1, 1]).unwrap();
            assert_eq!(
                builtin(
                    Value::Tensor(data),
                    vec![Value::Tensor(column), Value::from("strictascend")]
                )
                .unwrap(),
                Value::Bool(true)
            );
        }
    }

    #[test]
    fn issortedrows_supports_typed_columns_and_per_column_directions() {
        let values =
            Tensor::new_integer(IntegerStorage::I64(vec![3, 2, 1, 1, 1, 2]), vec![3, 2]).unwrap();
        let columns = Tensor::new_integer(IntegerStorage::U64(vec![1, 2]), vec![1, 2]).unwrap();
        assert_eq!(
            builtin(
                Value::Tensor(values),
                vec![Value::Tensor(columns), directions(&["descend", "ascend"])]
            )
            .unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn explicit_directions_override_column_signs_and_validate_list_length() {
        let values =
            Tensor::new_integer(IntegerStorage::I16(vec![2, 1, 1, 2]), vec![2, 2]).unwrap();
        let columns = Tensor::new_integer(IntegerStorage::I8(vec![-1, 2]), vec![1, 2]).unwrap();
        assert_eq!(
            builtin(
                Value::Tensor(values.clone()),
                vec![Value::Tensor(columns.clone())]
            )
            .unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            builtin(
                Value::Tensor(values.clone()),
                vec![
                    Value::Tensor(columns.clone()),
                    directions(&["ascend", "ascend"])
                ]
            )
            .unwrap(),
            Value::Bool(false)
        );
        let error = builtin(
            Value::Tensor(values),
            vec![
                Value::Tensor(columns),
                directions(&["ascend", "ascend", "descend"]),
            ],
        )
        .expect_err("direction length mismatch");
        assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn issortedrows_strict_modes_reject_missing_and_excess_outputs() {
        let values = Tensor::new(vec![1.0, f64::NAN], vec![2, 1]).unwrap();
        assert_eq!(
            builtin(Value::Tensor(values), vec![Value::from("strictascend")]).unwrap(),
            Value::Bool(false)
        );
        let _guard = crate::output_count::push_output_count(Some(2));
        let error = builtin(Value::Int(IntValue::I8(1)), Vec::new())
            .expect_err("excess output must reject");
        assert_eq!(error.identifier(), ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn issortedrows_direction_string_arrays_match_cell_arrays() {
        let values =
            Tensor::new_integer(IntegerStorage::I32(vec![3, 2, 1, 1, 1, 2]), vec![3, 2]).unwrap();
        let directions = Value::StringArray(
            StringArray::new(
                vec!["descend".to_string(), "ascend".to_string()],
                vec![1, 2],
            )
            .unwrap(),
        );
        assert_eq!(
            builtin(Value::Tensor(values), vec![directions]).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn resident_input_is_mode_gated_before_exact_gather() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new_integer(
                IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
                vec![3, 1],
            )
            .unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let error = builtin(Value::GpuTensor(handle.clone()), Vec::new())
                    .expect_err("MATLAB mode must reject interactive GPU input");
                assert_eq!(error.identifier(), GPU_INPUT_EXTENSION.error_identifier);
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                assert_eq!(
                    builtin(Value::GpuTensor(handle), Vec::new()).unwrap(),
                    Value::Bool(true)
                );
            }
        });
    }
}
