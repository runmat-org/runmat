//! Dummy-variable and one-hot encoding helpers.

use std::{cmp::Ordering, collections::HashMap};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, LogicalArray, ObjectInstance, ResolveContext, StringArray, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MAX_ONEHOT_CELLS: usize = 50_000_000;
const MAX_ENCODING_RANK: usize = 32;

const PARAM_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input labels.",
};

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input labels to encode.",
};

const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-hot encoded data.",
};

const PARAM_FEATURE_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "featureDim",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dimension containing or receiving one-hot feature vectors.",
};

const PARAM_CLASSES: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "classes",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class names corresponding to the one-hot feature dimension.",
};

const PARAM_TYPENAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "typename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Requested output type.",
};

const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "options",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Name-value options such as ClassNames or OutputType.",
};

const OUTPUT_DUMMYVAR: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dummy-variable design matrix.",
}];

const OUTPUT_ONEHOT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One-hot encoded array.",
}];

const OUTPUT_DECODED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Decoded labels.",
}];

const INPUT_X: [BuiltinParamDescriptor; 1] = [PARAM_X];
const INPUT_A_DIM: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_FEATURE_DIM];
const INPUT_A_DIM_TYPENAME: [BuiltinParamDescriptor; 3] =
    [PARAM_A, PARAM_FEATURE_DIM, PARAM_TYPENAME];
const INPUT_A_DIM_OPTIONS: [BuiltinParamDescriptor; 3] =
    [PARAM_A, PARAM_FEATURE_DIM, PARAM_OPTIONS];
const INPUT_B_CLASSES_DIM: [BuiltinParamDescriptor; 3] =
    [PARAM_B, PARAM_CLASSES, PARAM_FEATURE_DIM];
const INPUT_B_CLASSES_DIM_TYPENAME: [BuiltinParamDescriptor; 4] =
    [PARAM_B, PARAM_CLASSES, PARAM_FEATURE_DIM, PARAM_TYPENAME];
const INPUT_B_CLASSES_DIM_OPTIONS: [BuiltinParamDescriptor; 4] =
    [PARAM_B, PARAM_CLASSES, PARAM_FEATURE_DIM, PARAM_OPTIONS];

const DUMMYVAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = dummyvar(group)",
    inputs: &INPUT_X,
    outputs: &OUTPUT_DUMMYVAR,
}];

const ONEHOTENCODE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim)",
        inputs: &INPUT_A_DIM,
        outputs: &OUTPUT_ONEHOT,
    },
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim, typename)",
        inputs: &INPUT_A_DIM_TYPENAME,
        outputs: &OUTPUT_ONEHOT,
    },
    BuiltinSignatureDescriptor {
        label: "B = onehotencode(A, featureDim, Name, Value)",
        inputs: &INPUT_A_DIM_OPTIONS,
        outputs: &OUTPUT_ONEHOT,
    },
];

const ONEHOTDECODE_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim)",
        inputs: &INPUT_B_CLASSES_DIM,
        outputs: &OUTPUT_DECODED,
    },
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim, typename)",
        inputs: &INPUT_B_CLASSES_DIM_TYPENAME,
        outputs: &OUTPUT_DECODED,
    },
    BuiltinSignatureDescriptor {
        label: "A = onehotdecode(B, classes, featureDim, Name, Value)",
        inputs: &INPUT_B_CLASSES_DIM_OPTIONS,
        outputs: &OUTPUT_DECODED,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENCODING.INVALID_ARGUMENT",
    identifier: None,
    when: "Inputs, dimensions, classes, output types, or options are malformed.",
    message: "encoding: invalid argument",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENCODING.INTERNAL",
    identifier: None,
    when: "Internal conversion or allocation fails.",
    message: "encoding: internal error",
};

macro_rules! encoding_descriptor {
    ($name:literal, $signatures:expr) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: ERROR_INVALID_ARGUMENT.when,
                message: ERROR_INVALID_ARGUMENT.message,
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

mod dummyvar_meta {
    use super::*;
    encoding_descriptor!("dummyvar", DUMMYVAR_SIGNATURES);
}

mod onehotencode_meta {
    use super::*;
    encoding_descriptor!("onehotencode", ONEHOTENCODE_SIGNATURES);
}

mod onehotdecode_meta {
    use super::*;
    encoding_descriptor!("onehotdecode", ONEHOTDECODE_SIGNATURES);
}

fn tensor_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Tensor {
        shape: Some(vec![None, None]),
    }
}

fn unknown_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn invalid(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    builder = builder.with_identifier(match name {
        "dummyvar" => "RunMat:dummyvar:InvalidArgument",
        "onehotencode" => "RunMat:onehotencode:InvalidArgument",
        "onehotdecode" => "RunMat:onehotdecode:InvalidArgument",
        _ => "RunMat:encoding:InvalidArgument",
    });
    builder.build()
}

fn internal(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    builder = builder.with_identifier(match name {
        "dummyvar" => "RunMat:dummyvar:Internal",
        "onehotencode" => "RunMat:onehotencode:Internal",
        "onehotdecode" => "RunMat:onehotdecode:Internal",
        _ => "RunMat:encoding:Internal",
    });
    builder.build()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Logical,
    String,
    Char,
    Cell,
    Categorical,
}

#[derive(Clone, Debug)]
struct LabelArray {
    labels: Vec<Option<Label>>,
    shape: Vec<usize>,
    kind: LabelKind,
    categories: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Label {
    Numeric(f64),
    Logical(bool),
    Text(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputKind {
    Double,
    Logical,
    String,
    Cell,
    Categorical,
}

#[runtime_builtin(
    name = "dummyvar",
    category = "stats/summary",
    summary = "Create dummy variables for grouping variables.",
    keywords = "dummyvar,dummy variables,one hot,statistics,categorical",
    type_resolver(tensor_type),
    descriptor(crate::builtins::stats::summary::encoding::dummyvar_meta::DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn dummyvar_builtin(group: Value) -> BuiltinResult<Value> {
    let group = gather("dummyvar", group).await?;
    dummyvar_compute(group)
}

#[runtime_builtin(
    name = "onehotencode",
    category = "stats/summary",
    summary = "Encode labels as one-hot vectors.",
    keywords = "onehotencode,one hot,classification,categorical,statistics",
    type_resolver(unknown_type),
    descriptor(crate::builtins::stats::summary::encoding::onehotencode_meta::DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn onehotencode_builtin(
    input: Value,
    feature_dim: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let input = gather("onehotencode", input).await?;
    let feature_dim = gather("onehotencode", feature_dim).await?;
    let rest = gather_values("onehotencode", rest).await?;
    let feature_dim = positive_dim("onehotencode", &feature_dim)?;
    let options = parse_onehotencode_options(rest)?;
    onehotencode_compute(input, feature_dim, options)
}

#[runtime_builtin(
    name = "onehotdecode",
    category = "stats/summary",
    summary = "Decode one-hot vectors into class labels.",
    keywords = "onehotdecode,one hot,classification,categorical,statistics",
    type_resolver(unknown_type),
    descriptor(crate::builtins::stats::summary::encoding::onehotdecode_meta::DESCRIPTOR),
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn onehotdecode_builtin(
    encoded: Value,
    classes: Value,
    feature_dim: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let encoded = gather("onehotdecode", encoded).await?;
    let classes = gather("onehotdecode", classes).await?;
    let feature_dim = gather("onehotdecode", feature_dim).await?;
    let rest = gather_values("onehotdecode", rest).await?;
    let feature_dim = positive_dim("onehotdecode", &feature_dim)?;
    let options = parse_onehotdecode_options(rest)?;
    onehotdecode_compute(encoded, classes, feature_dim, options)
}

async fn gather(name: &str, value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value)
        .await
        .map_err(|err| invalid(name, format!("{name}: {err}")))
}

async fn gather_values(name: &str, values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(name, value).await?);
    }
    Ok(out)
}

fn dummyvar_compute(group: Value) -> BuiltinResult<Value> {
    let groups = dummyvar_groups(group)?;
    let rows = groups.first().map(|group| group.labels.len()).unwrap_or(0);
    let mut total_cols = 0usize;
    let mut specs = Vec::with_capacity(groups.len());
    for group in groups {
        if group.labels.len() != rows {
            return Err(invalid(
                "dummyvar",
                "dummyvar: all grouping variables must have the same number of rows",
            ));
        }
        let classes = classes_for_dummyvar(&group, rows)?;
        total_cols = total_cols
            .checked_add(classes.len())
            .ok_or_else(|| invalid("dummyvar", "dummyvar: too many dummy variables"))?;
        specs.push((group, classes));
    }
    let cell_count = rows
        .checked_mul(total_cols)
        .ok_or_else(|| invalid("dummyvar", "dummyvar: output is too large"))?;
    if cell_count > MAX_ONEHOT_CELLS {
        return Err(invalid("dummyvar", "dummyvar: output is too large"));
    }
    let mut data = vec![0.0; cell_count];
    let mut col_offset = 0usize;
    for (group, classes) in specs {
        let index = class_index(&classes)?;
        for (row, label) in group.labels.iter().enumerate() {
            if label.is_none() {
                for col in 0..classes.len() {
                    data[row + (col_offset + col) * rows] = f64::NAN;
                }
                continue;
            }
            if let Some(class_idx) = label
                .as_ref()
                .and_then(|label| index.get(&label_key(label)))
            {
                data[row + (col_offset + *class_idx) * rows] = 1.0;
            }
        }
        col_offset += classes.len();
    }
    Tensor::new(data, vec![rows, total_cols])
        .map(Value::Tensor)
        .map_err(|err| internal("dummyvar", format!("dummyvar: {err}")))
}

#[derive(Default)]
struct OnehotEncodeOptions {
    classes: Option<LabelArray>,
    output: Option<OutputKind>,
}

fn onehotencode_compute(
    input: Value,
    feature_dim: usize,
    options: OnehotEncodeOptions,
) -> BuiltinResult<Value> {
    let labels = labels_from_value("onehotencode", input)?;
    let classes = if let Some(class_names) = options.classes {
        if !label_kinds_compatible(labels.kind, class_names.kind) {
            return Err(invalid(
                "onehotencode",
                "onehotencode: ClassNames must match input label type",
            ));
        }
        nonmissing_classes("onehotencode", &class_names)?
    } else if labels.kind == LabelKind::Categorical {
        classes_for_labels(&labels)
    } else {
        return Err(invalid(
            "onehotencode",
            "onehotencode: ClassNames is required for noncategorical input",
        ));
    };
    if classes.is_empty() {
        return Err(invalid(
            "onehotencode",
            "onehotencode: class names cannot be empty",
        ));
    }
    let shape = onehot_encoded_shape("onehotencode", &labels.shape, feature_dim, classes.len())?;
    let cell_count = element_count(&shape)?;
    if cell_count > MAX_ONEHOT_CELLS {
        return Err(invalid("onehotencode", "onehotencode: output is too large"));
    }
    let mut values = vec![0.0; cell_count];
    let mut has_nan_vector = false;
    let index = class_index(&classes)?;
    let out_strides = strides(&shape);
    let input_shape = normalized_shape(&labels.shape);
    let input_strides = strides(&input_shape);
    let feature_axis = feature_dim - 1;
    for idx in 0..labels.labels.len() {
        let Some(label) = &labels.labels[idx] else {
            fill_feature_vector(
                &mut values,
                &shape,
                &out_strides,
                &input_shape,
                &input_strides,
                feature_axis,
                idx,
                classes.len(),
                f64::NAN,
            );
            has_nan_vector = true;
            continue;
        };
        let Some(class_idx) = index.get(&label_key(label)).copied() else {
            fill_feature_vector(
                &mut values,
                &shape,
                &out_strides,
                &input_shape,
                &input_strides,
                feature_axis,
                idx,
                classes.len(),
                f64::NAN,
            );
            has_nan_vector = true;
            continue;
        };
        let mut coords = coords_for_linear(idx, &input_shape, &input_strides);
        coords.resize(shape.len(), 0);
        coords[feature_axis] = class_idx;
        let out_idx = linear_for_coords(&coords, &out_strides);
        values[out_idx] = 1.0;
    }
    match options.output.unwrap_or(OutputKind::Double) {
        OutputKind::Logical => {
            if has_nan_vector {
                return Err(invalid(
                    "onehotencode",
                    "onehotencode: logical output cannot represent missing or excluded labels",
                ));
            }
            LogicalArray::new(
                values
                    .into_iter()
                    .map(|value| u8::from(value != 0.0))
                    .collect(),
                shape,
            )
            .map(Value::LogicalArray)
            .map_err(|err| internal("onehotencode", format!("onehotencode: {err}")))
        }
        OutputKind::Double => Tensor::new(values, shape)
            .map(Value::Tensor)
            .map_err(|err| internal("onehotencode", format!("onehotencode: {err}"))),
        _ => Err(invalid(
            "onehotencode",
            "onehotencode: OutputType must be double, single, or logical",
        )),
    }
}

#[derive(Default)]
struct OnehotDecodeOptions {
    output: Option<OutputKind>,
}

fn onehotdecode_compute(
    encoded: Value,
    classes: Value,
    feature_dim: usize,
    options: OnehotDecodeOptions,
) -> BuiltinResult<Value> {
    let encoded = encoded_values(encoded)?;
    let class_labels = labels_from_value("onehotdecode", classes)?;
    let classes = nonmissing_classes("onehotdecode", &class_labels)?;
    if classes.is_empty() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: classes cannot be empty",
        ));
    }
    let shape = normalized_shape(&encoded.shape);
    let feature_axis = feature_dim - 1;
    if feature_axis >= shape.len() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: featureDim exceeds encoded data dimensions",
        ));
    }
    if shape[feature_axis] != classes.len() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: classes length must match the feature dimension",
        ));
    }
    validate_encoded_range(&encoded.data)?;
    let mut out_shape = shape.clone();
    out_shape[feature_axis] = 1;
    let out_count = element_count(&out_shape)?;
    let out_strides = strides(&out_shape);
    let encoded_strides = strides(&shape);
    let mut decoded = Vec::with_capacity(out_count);
    for out_idx in 0..out_count {
        let mut coords = coords_for_linear(out_idx, &out_shape, &out_strides);
        let mut best_idx = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for class_idx in 0..classes.len() {
            coords[feature_axis] = class_idx;
            let value = encoded.data[linear_for_coords(&coords, &encoded_strides)];
            if value.is_nan() {
                continue;
            }
            if value > best_value {
                best_value = value;
                best_idx = class_idx;
            }
        }
        decoded.push(classes[best_idx].clone());
    }
    labels_to_value(
        "onehotdecode",
        &decoded,
        out_shape,
        options.output.unwrap_or(OutputKind::Categorical),
        Some(&classes),
    )
}

fn parse_onehotencode_options(rest: Vec<Value>) -> BuiltinResult<OnehotEncodeOptions> {
    let mut options = OnehotEncodeOptions::default();
    if rest.is_empty() {
        return Ok(options);
    }
    if rest.len() == 1 {
        options.output = Some(parse_output_kind(
            "onehotencode",
            &scalar_text(&rest[0], "option")?,
        )?);
        return Ok(options);
    }
    let mut idx = 0usize;
    if !rest.len().is_multiple_of(2) {
        let typename = scalar_text(&rest[0], "typename")?;
        options.output = Some(parse_output_kind("onehotencode", &typename)?);
        idx = 1;
    }
    if !(rest.len() - idx).is_multiple_of(2) {
        return Err(invalid(
            "onehotencode",
            "onehotencode: name-value options must be paired",
        ));
    }
    while idx < rest.len() {
        let name = canonical(&scalar_text(&rest[idx], "option name")?);
        match name.as_str() {
            "classnames" | "classes" => {
                options.classes = Some(labels_from_value("onehotencode", rest[idx + 1].clone())?);
            }
            "outputtype" | "typename" => {
                let text = scalar_text(&rest[idx + 1], "OutputType")?;
                options.output = Some(parse_output_kind("onehotencode", &text)?);
            }
            other => {
                return Err(invalid(
                    "onehotencode",
                    format!("onehotencode: unsupported option '{other}'"),
                ))
            }
        }
        idx += 2;
    }
    Ok(options)
}

fn parse_onehotdecode_options(rest: Vec<Value>) -> BuiltinResult<OnehotDecodeOptions> {
    let mut options = OnehotDecodeOptions::default();
    if rest.is_empty() {
        return Ok(options);
    }
    if rest.len() == 1 {
        options.output = Some(parse_output_kind(
            "onehotdecode",
            &scalar_text(&rest[0], "typename")?,
        )?);
        return Ok(options);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: name-value options must be paired",
        ));
    }
    let mut idx = 0usize;
    while idx < rest.len() {
        let name = canonical(&scalar_text(&rest[idx], "option name")?);
        match name.as_str() {
            "outputtype" | "typename" => {
                let text = scalar_text(&rest[idx + 1], "OutputType")?;
                options.output = Some(parse_output_kind("onehotdecode", &text)?);
            }
            other => {
                return Err(invalid(
                    "onehotdecode",
                    format!("onehotdecode: unsupported option '{other}'"),
                ))
            }
        }
        idx += 2;
    }
    Ok(options)
}

fn labels_from_value(name: &str, value: Value) -> BuiltinResult<LabelArray> {
    match value {
        Value::Num(value) => Ok(LabelArray {
            labels: vec![numeric_label(value)],
            shape: vec![1, 1],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Int(value) => Ok(LabelArray {
            labels: vec![numeric_label(value.to_f64())],
            shape: vec![1, 1],
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Tensor(tensor) => Ok(LabelArray {
            labels: tensor.data.iter().copied().map(numeric_label).collect(),
            shape: normalized_shape(&tensor.shape),
            kind: LabelKind::Numeric,
            categories: None,
        }),
        Value::Bool(value) => Ok(LabelArray {
            labels: vec![Some(Label::Logical(value))],
            shape: vec![1, 1],
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::LogicalArray(array) => Ok(LabelArray {
            labels: array
                .data
                .iter()
                .map(|value| Some(Label::Logical(*value != 0)))
                .collect(),
            shape: normalized_shape(&array.shape),
            kind: LabelKind::Logical,
            categories: None,
        }),
        Value::String(text) => Ok(LabelArray {
            labels: vec![text_label(text)],
            shape: vec![1, 1],
            kind: LabelKind::String,
            categories: None,
        }),
        Value::StringArray(array) => Ok(LabelArray {
            labels: array.data.into_iter().map(text_label).collect(),
            shape: normalized_shape(&array.shape),
            kind: LabelKind::String,
            categories: None,
        }),
        Value::CharArray(array) => Ok(LabelArray {
            labels: char_rows(&array).into_iter().map(text_label).collect(),
            shape: vec![array.rows, 1],
            kind: LabelKind::Char,
            categories: None,
        }),
        Value::Cell(cell) => {
            let mut labels = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                labels.push(text_label(scalar_text(&item, name)?));
            }
            Ok(LabelArray {
                labels,
                shape: vec![cell.rows, cell.cols],
                kind: LabelKind::Cell,
                categories: None,
            })
        }
        Value::Object(object) if object.is_class("categorical") => {
            let labels =
                crate::builtins::table::categorical_labels(&Value::Object(object.clone()))?
                    .into_iter()
                    .map(text_label)
                    .collect();
            let shape = categorical_shape(&object);
            Ok(LabelArray {
                labels,
                shape,
                kind: LabelKind::Categorical,
                categories: Some(crate::builtins::table::categorical_categories(&object)?),
            })
        }
        other => Err(invalid(
            name,
            format!("{name}: unsupported label input {other:?}"),
        )),
    }
}

fn dummyvar_groups(value: Value) -> BuiltinResult<Vec<LabelArray>> {
    match value {
        Value::Tensor(tensor)
            if tensor.shape.len() <= 2 && tensor.rows() > 1 && tensor.cols() > 1 =>
        {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let mut groups = Vec::with_capacity(cols);
            for col in 0..cols {
                let mut labels = Vec::with_capacity(rows);
                for row in 0..rows {
                    labels.push(numeric_label(tensor.data[row + col * rows]));
                }
                groups.push(LabelArray {
                    labels,
                    shape: vec![rows, 1],
                    kind: LabelKind::Numeric,
                    categories: None,
                });
            }
            Ok(groups)
        }
        Value::Cell(cell) if cell.rows == 1 || cell.cols == 1 => {
            if cell.rows == 1 && cell.cols > 1 && cell.data.iter().any(is_grouping_container) {
                let mut groups = Vec::with_capacity(cell.cols);
                for value in cell.data {
                    groups.push(labels_from_value("dummyvar", value)?);
                }
                Ok(groups)
            } else {
                Ok(vec![labels_from_value("dummyvar", Value::Cell(cell))?])
            }
        }
        other => Ok(vec![labels_from_value("dummyvar", other)?]),
    }
}

fn classes_for_dummyvar(group: &LabelArray, rows: usize) -> BuiltinResult<Vec<Label>> {
    match group.kind {
        LabelKind::Numeric => numeric_dummyvar_classes(&group.labels, rows),
        _ => Ok(classes_for_labels(group)),
    }
}

fn numeric_dummyvar_classes(labels: &[Option<Label>], rows: usize) -> BuiltinResult<Vec<Label>> {
    let mut max_level = 0usize;
    for label in labels.iter().flatten() {
        let Label::Numeric(value) = label else {
            continue;
        };
        if *value <= 0.0 || value.fract() != 0.0 || *value > usize::MAX as f64 {
            return Err(invalid(
                "dummyvar",
                "dummyvar: numeric groups must contain positive integer levels or NaN",
            ));
        }
        max_level = max_level.max(*value as usize);
    }
    let row_count = rows.max(1);
    if max_level > MAX_ONEHOT_CELLS / row_count {
        return Err(invalid("dummyvar", "dummyvar: output is too large"));
    }
    Ok((1..=max_level)
        .map(|value| Label::Numeric(value as f64))
        .collect())
}

fn classes_for_labels(labels: &LabelArray) -> Vec<Label> {
    match labels.kind {
        LabelKind::Numeric => {
            let mut classes = dedupe(labels.labels.iter().flatten().cloned());
            classes.sort_by(|left, right| match (left, right) {
                (Label::Numeric(a), Label::Numeric(b)) => {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal)
                }
                _ => Ordering::Equal,
            });
            classes
        }
        LabelKind::Logical => vec![Label::Logical(false), Label::Logical(true)],
        LabelKind::Categorical => {
            let mut classes = Vec::new();
            if let Some(categories) = &labels.categories {
                for category in categories {
                    classes.push(Label::Text(category.clone()));
                }
            }
            for label in labels.labels.iter().flatten() {
                if !classes.iter().any(|existing| same_label(existing, label)) {
                    classes.push(label.clone());
                }
            }
            classes
        }
        LabelKind::String | LabelKind::Char | LabelKind::Cell => {
            dedupe(labels.labels.iter().flatten().cloned())
        }
    }
}

fn nonmissing_classes(name: &str, labels: &LabelArray) -> BuiltinResult<Vec<Label>> {
    if labels.labels.iter().any(Option::is_none) {
        return Err(invalid(
            name,
            format!("{name}: class names cannot contain missing values"),
        ));
    }
    let classes = dedupe(labels.labels.iter().flatten().cloned());
    if classes.len() != labels.labels.iter().flatten().count() {
        return Err(invalid(name, format!("{name}: class names must be unique")));
    }
    Ok(classes)
}

fn class_index(classes: &[Label]) -> BuiltinResult<HashMap<String, usize>> {
    let mut index = HashMap::with_capacity(classes.len());
    for (idx, class) in classes.iter().enumerate() {
        if index.insert(label_key(class), idx).is_some() {
            return Err(error("encoding", "encoding: duplicate class labels"));
        }
    }
    Ok(index)
}

fn fill_feature_vector(
    values: &mut [f64],
    output_shape: &[usize],
    output_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    feature_axis: usize,
    input_idx: usize,
    class_count: usize,
    value: f64,
) {
    let mut coords = coords_for_linear(input_idx, input_shape, input_strides);
    coords.resize(output_shape.len(), 0);
    for class_idx in 0..class_count {
        coords[feature_axis] = class_idx;
        let out_idx = linear_for_coords(&coords, output_strides);
        values[out_idx] = value;
    }
}

struct EncodedArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

fn encoded_values(value: Value) -> BuiltinResult<EncodedArray> {
    match value {
        Value::Tensor(tensor) => Ok(EncodedArray {
            data: tensor.data,
            shape: normalized_shape(&tensor.shape),
        }),
        Value::LogicalArray(array) => Ok(EncodedArray {
            data: array.data.into_iter().map(|value| value as f64).collect(),
            shape: normalized_shape(&array.shape),
        }),
        Value::Num(value) => Ok(EncodedArray {
            data: vec![value],
            shape: vec![1, 1],
        }),
        Value::Bool(value) => Ok(EncodedArray {
            data: vec![if value { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
        }),
        other => Err(invalid(
            "onehotdecode",
            format!("onehotdecode: encoded data must be numeric or logical, got {other:?}"),
        )),
    }
}

fn validate_encoded_range(values: &[f64]) -> BuiltinResult<()> {
    if values
        .iter()
        .any(|value| !value.is_nan() && !(0.0..=1.0).contains(value))
    {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: encoded values must be probabilities between 0 and 1",
        ));
    }
    Ok(())
}

fn onehot_encoded_shape(
    name: &str,
    input_shape: &[usize],
    feature_dim: usize,
    classes: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut shape = normalized_shape(input_shape);
    if feature_dim == 0 {
        return Err(invalid(
            name,
            format!("{name}: featureDim must be positive"),
        ));
    }
    let feature_axis = feature_dim - 1;
    if feature_axis >= MAX_ENCODING_RANK || shape.len() > MAX_ENCODING_RANK {
        return Err(invalid(name, format!("{name}: featureDim is too large")));
    }
    if feature_axis >= shape.len() {
        shape.resize(feature_axis + 1, 1);
    }
    if shape[feature_axis] != 1 {
        return Err(invalid(
            name,
            format!("{name}: input dimension featureDim must be singleton"),
        ));
    }
    shape[feature_axis] = classes;
    Ok(shape)
}

fn labels_to_value(
    name: &str,
    labels: &[Label],
    shape: Vec<usize>,
    output: OutputKind,
    category_order: Option<&[Label]>,
) -> BuiltinResult<Value> {
    match output {
        OutputKind::Double => {
            let mut data = Vec::with_capacity(labels.len());
            for label in labels {
                match label {
                    Label::Numeric(value) => data.push(*value),
                    _ => {
                        return Err(invalid(
                            name,
                            format!("{name}: numeric output requires numeric classes"),
                        ))
                    }
                }
            }
            Tensor::new(data, shape)
                .map(Value::Tensor)
                .map_err(|err| internal(name, format!("{name}: {err}")))
        }
        OutputKind::Logical => {
            let mut data = Vec::with_capacity(labels.len());
            for label in labels {
                match label {
                    Label::Logical(value) => data.push(u8::from(*value)),
                    _ => {
                        return Err(invalid(
                            name,
                            format!("{name}: logical output requires logical classes"),
                        ))
                    }
                }
            }
            LogicalArray::new(data, shape)
                .map(Value::LogicalArray)
                .map_err(|err| internal(name, format!("{name}: {err}")))
        }
        OutputKind::String => StringArray::new(text_values(labels), shape)
            .map(Value::StringArray)
            .map_err(|err| internal(name, format!("{name}: {err}"))),
        OutputKind::Cell => {
            let rows = shape.first().copied().unwrap_or(labels.len());
            let cols = if rows == 0 { 0 } else { labels.len() / rows };
            CellArray::new(
                text_values(labels)
                    .into_iter()
                    .map(|text| Value::CharArray(CharArray::new_row(&text)))
                    .collect(),
                rows,
                cols,
            )
            .map(Value::Cell)
            .map_err(|err| internal(name, format!("{name}: {err}")))
        }
        OutputKind::Categorical => {
            let values = text_values(labels);
            let category_values = category_order
                .map(text_values)
                .unwrap_or_else(|| dedupe_text(values.clone()));
            let value_array = Value::StringArray(
                StringArray::new(values.clone(), shape)
                    .map_err(|err| internal(name, format!("{name}: {err}")))?,
            );
            let categories = Value::StringArray(
                StringArray::new(category_values.clone(), vec![1, category_values.len()])
                    .map_err(|err| internal(name, format!("{name}: {err}")))?,
            );
            crate::builtins::table::categorical_from_args(vec![value_array, categories])
        }
    }
}

fn parse_output_kind(name: &str, text: &str) -> BuiltinResult<OutputKind> {
    match canonical(text).as_str() {
        "double" | "single" => Ok(OutputKind::Double),
        "logical" => Ok(OutputKind::Logical),
        "string" => Ok(OutputKind::String),
        "cell" | "cellstr" => Ok(OutputKind::Cell),
        "categorical" => Ok(OutputKind::Categorical),
        other => Err(invalid(
            name,
            format!("{name}: unsupported output type '{other}'"),
        )),
    }
}

fn positive_dim(name: &str, value: &Value) -> BuiltinResult<usize> {
    let dim = match value {
        Value::Num(value) => *value,
        Value::Int(value) => value.to_f64(),
        _ => {
            return Err(invalid(
                name,
                format!("{name}: featureDim must be a positive integer scalar"),
            ))
        }
    };
    if dim.fract() != 0.0 || dim < 1.0 || dim > usize::MAX as f64 {
        return Err(invalid(
            name,
            format!("{name}: featureDim must be a positive integer scalar"),
        ));
    }
    Ok(dim as usize)
}

fn numeric_label(value: f64) -> Option<Label> {
    (!value.is_nan()).then_some(Label::Numeric(if value == 0.0 { 0.0 } else { value }))
}

fn text_label(value: String) -> Option<Label> {
    (!value.is_empty() && value != "<missing>").then_some(Label::Text(value))
}

fn same_label(left: &Label, right: &Label) -> bool {
    match (left, right) {
        (Label::Numeric(a), Label::Numeric(b)) => a == b,
        (Label::Logical(a), Label::Logical(b)) => a == b,
        (Label::Text(a), Label::Text(b)) => a == b,
        _ => false,
    }
}

fn label_key(label: &Label) -> String {
    match label {
        Label::Numeric(value) if *value == 0.0 => "n:0".to_string(),
        Label::Numeric(value) => format!("n:{:016x}", value.to_bits()),
        Label::Logical(value) => format!("l:{}", u8::from(*value)),
        Label::Text(value) => format!("t:{value}"),
    }
}

fn dedupe<I>(labels: I) -> Vec<Label>
where
    I: IntoIterator<Item = Label>,
{
    let mut out = Vec::new();
    for label in labels {
        if !out.iter().any(|existing| same_label(existing, &label)) {
            out.push(label);
        }
    }
    out
}

fn text_values(labels: &[Label]) -> Vec<String> {
    labels
        .iter()
        .map(|label| match label {
            Label::Numeric(value) => format_number(*value),
            Label::Logical(value) => {
                if *value {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            Label::Text(value) => value.clone(),
        })
        .collect()
}

fn dedupe_text(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for value in values {
        if !out.iter().any(|existing| existing == &value) {
            out.push(value);
        }
    }
    out
}

fn label_kinds_compatible(left: LabelKind, right: LabelKind) -> bool {
    left == right || (is_text_kind(left) && is_text_kind(right))
}

fn is_grouping_container(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.data.len() > 1,
        Value::LogicalArray(array) => array.data.len() > 1,
        Value::StringArray(array) => array.data.len() > 1,
        Value::CharArray(array) => array.rows > 1,
        Value::Cell(cell) => cell.data.len() > 1,
        Value::Object(object) if object.is_class("categorical") => {
            matches!(object.properties.get("Codes"), Some(Value::Tensor(tensor)) if tensor.data.len() > 1)
        }
        _ => false,
    }
}

fn is_text_kind(kind: LabelKind) -> bool {
    matches!(
        kind,
        LabelKind::String | LabelKind::Char | LabelKind::Cell | LabelKind::Categorical
    )
}

fn char_rows(array: &CharArray) -> Vec<String> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        rows.push(array.data[start..start + array.cols].iter().collect());
    }
    rows
}

fn scalar_text(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid(
            name,
            format!("{name}: expected string scalar, got {other:?}"),
        )),
    }
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

fn categorical_shape(object: &ObjectInstance) -> Vec<usize> {
    match object.properties.get("Codes") {
        Some(Value::Tensor(tensor)) => normalized_shape(&tensor.shape),
        _ => vec![1, 1],
    }
}

fn normalized_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![shape[0], 1],
        _ => shape.to_vec(),
    }
}

fn element_count(shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| error("encoding", "encoding: array is too large"))
    })
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1usize;
    for dim in shape {
        strides.push(acc);
        acc = acc.saturating_mul(*dim);
    }
    strides
}

fn coords_for_linear(mut idx: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let stride = strides[axis];
        coords[axis] = if stride == 0 { 0 } else { idx / stride };
        idx = if stride == 0 { idx } else { idx % stride };
    }
    coords
}

fn linear_for_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(coord, stride)| coord * stride)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new(data, shape).unwrap())
    }

    fn assert_f64_slice_eq_nan(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (idx, (a, b)) in left.iter().zip(right).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || (*a - *b).abs() < 1.0e-12,
                "mismatch at {idx}: {a:?} != {b:?}"
            );
        }
    }

    #[test]
    fn dummyvar_encodes_numeric_columns_and_missing() {
        let Value::Tensor(out) = block_on(dummyvar_builtin(tensor(
            vec![1.0, 2.0, f64::NAN, 1.0, 2.0, 2.0],
            vec![3, 2],
        )))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 4]);
        assert_f64_slice_eq_nan(
            &out.data,
            &[
                1.0,
                0.0,
                f64::NAN,
                0.0,
                1.0,
                f64::NAN,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        );
    }

    #[test]
    fn dummyvar_encodes_cell_grouping_variables() {
        let phone = Value::Cell(
            CellArray::new(
                vec![
                    Value::from("mobile"),
                    Value::from("landline"),
                    Value::from("mobile"),
                ],
                3,
                1,
            )
            .unwrap(),
        );
        let region = tensor(vec![2.0, 1.0, 2.0], vec![3, 1]);
        let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
            CellArray::new(vec![phone, region], 1, 2).unwrap(),
        )))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 4]);
        assert_eq!(
            out.data,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn dummyvar_treats_row_cellstr_as_one_grouping_vector() {
        let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
            CellArray::new(
                vec![Value::from("red"), Value::from("blue"), Value::from("red")],
                1,
                3,
            )
            .unwrap(),
        )))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn dummyvar_treats_numeric_row_vector_as_one_grouping_vector() {
        let Value::Tensor(out) =
            block_on(dummyvar_builtin(tensor(vec![1.0, 2.0, 1.0], vec![1, 3]))).unwrap()
        else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn dummyvar_rejects_huge_numeric_level_before_allocating_classes() {
        let err =
            block_on(dummyvar_builtin(tensor(vec![1_000_000_000.0], vec![1, 1]))).unwrap_err();
        assert!(err.message.contains("output is too large"));
    }

    #[test]
    fn categorical_classes_preserve_unused_categories() {
        let categories = Value::StringArray(
            StringArray::new(
                vec!["low".into(), "medium".into(), "high".into()],
                vec![1, 3],
            )
            .unwrap(),
        );
        let group = crate::builtins::table::categorical_from_args(vec![
            Value::StringArray(
                StringArray::new(vec!["high".into(), "low".into()], vec![2, 1]).unwrap(),
            ),
            categories.clone(),
        ])
        .unwrap();
        let Value::Tensor(dummy) = block_on(dummyvar_builtin(group.clone())).unwrap() else {
            panic!("dummy tensor");
        };
        assert_eq!(dummy.shape, vec![2, 3]);
        assert_eq!(dummy.data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

        let Value::Tensor(encoded) =
            block_on(onehotencode_builtin(group, Value::Num(2.0), Vec::new())).unwrap()
        else {
            panic!("encoded tensor");
        };
        assert_eq!(encoded.shape, vec![2, 3]);
        assert_eq!(encoded.data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn onehotencode_uses_feature_dimension_and_class_names() {
        let Value::LogicalArray(out) = block_on(onehotencode_builtin(
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
            ),
            Value::Num(2.0),
            vec![
                Value::from("ClassNames"),
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
                ),
                Value::from("OutputType"),
                Value::from("logical"),
            ],
        ))
        .unwrap() else {
            panic!("logical");
        };
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn onehotencode_accepts_typename_before_name_value_options() {
        let Value::Tensor(out) = block_on(onehotencode_builtin(
            Value::StringArray(
                StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
            ),
            Value::Num(2.0),
            vec![
                Value::from("single"),
                Value::from("ClassNames"),
                Value::StringArray(
                    StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap(),
                ),
            ],
        ))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn onehotencode_requires_classnames_for_noncategorical_and_marks_excluded_labels() {
        let err = block_on(onehotencode_builtin(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("ClassNames is required"));

        let Value::Tensor(out) = block_on(onehotencode_builtin(
            tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]),
            Value::Num(2.0),
            vec![
                Value::from("ClassNames"),
                tensor(vec![1.0, 2.0], vec![1, 2]),
            ],
        ))
        .unwrap() else {
            panic!("tensor");
        };
        assert_eq!(out.shape, vec![3, 2]);
        assert_f64_slice_eq_nan(
            &out.data,
            &[1.0, f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN],
        );

        let err = block_on(onehotencode_builtin(
            tensor(vec![1.0, 3.0], vec![2, 1]),
            Value::Num(2.0),
            vec![
                Value::from("ClassNames"),
                tensor(vec![1.0, 2.0], vec![1, 2]),
                Value::from("OutputType"),
                Value::from("logical"),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("logical output cannot represent"));
    }

    #[test]
    fn onehotencode_rejects_pathological_feature_dimension_before_resizing_shape() {
        let err = block_on(onehotencode_builtin(
            tensor(vec![1.0], vec![1, 1]),
            Value::Num(1_000_000_000.0),
            vec![
                Value::from("ClassNames"),
                tensor(vec![1.0, 2.0], vec![1, 2]),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("featureDim is too large"));
    }

    #[test]
    fn onehotdecode_round_trips_string_classes() {
        let encoded =
            Value::LogicalArray(LogicalArray::new(vec![0, 1, 1, 1, 0, 0], vec![3, 2]).unwrap());
        let Value::StringArray(out) = block_on(onehotdecode_builtin(
            encoded,
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
            Value::Num(2.0),
            vec![Value::from("string")],
        ))
        .unwrap() else {
            panic!("strings");
        };
        assert_eq!(out.shape, vec![3, 1]);
        assert_eq!(out.data, vec!["b", "a", "a"]);
    }

    #[test]
    fn onehotdecode_returns_categorical_for_categorical_classes() {
        let categories = Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        );
        let classes = crate::builtins::table::categorical_from_args(vec![
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
            ),
            categories,
        ])
        .unwrap();
        let encoded = Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap());
        let Value::Object(out) = block_on(onehotdecode_builtin(
            encoded,
            classes,
            Value::Num(1.0),
            Vec::new(),
        ))
        .unwrap() else {
            panic!("categorical");
        };
        assert!(out.is_class("categorical"));
        let labels = crate::builtins::table::categorical_labels(&Value::Object(out)).unwrap();
        assert_eq!(labels, vec!["high", "low"]);
    }

    #[test]
    fn onehotdecode_defaults_to_categorical_and_preserves_class_order() {
        let encoded = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap());
        let classes = Value::StringArray(
            StringArray::new(
                vec!["low".into(), "high".into(), "unused".into()],
                vec![1, 3],
            )
            .unwrap(),
        );
        let err = block_on(onehotdecode_builtin(
            encoded.clone(),
            classes.clone(),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("classes length"));

        let classes = Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        );
        let Value::Object(out) = block_on(onehotdecode_builtin(
            encoded,
            classes,
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap() else {
            panic!("categorical");
        };
        let categories = match out.properties.get("Categories") {
            Some(Value::StringArray(array)) => array.data.clone(),
            other => panic!("categories {other:?}"),
        };
        assert_eq!(categories, vec!["low", "high"]);
    }

    #[test]
    fn onehotdecode_categorical_output_allows_repeated_decoded_labels() {
        let encoded =
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![3, 2]).unwrap());
        let classes = Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        );
        let Value::Object(out) = block_on(onehotdecode_builtin(
            encoded,
            classes,
            Value::Num(2.0),
            vec![Value::from("categorical")],
        ))
        .unwrap() else {
            panic!("categorical");
        };
        let labels = crate::builtins::table::categorical_labels(&Value::Object(out)).unwrap();
        assert_eq!(labels, vec!["low", "low", "low"]);
    }

    #[test]
    fn rejects_bad_dimensions_and_unknown_labels() {
        let err = block_on(onehotencode_builtin(
            tensor(vec![1.0, 2.0], vec![2, 1]),
            Value::Num(1.0),
            vec![
                Value::from("ClassNames"),
                tensor(vec![1.0, 2.0], vec![1, 2]),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("singleton"));

        let err = block_on(onehotencode_builtin(
            tensor(vec![1.0], vec![1, 1]),
            Value::Num(2.0),
            vec![
                Value::from("ClassNames"),
                tensor(vec![1.0, f64::NAN], vec![1, 2]),
            ],
        ))
        .unwrap_err();
        assert!(err.message.contains("cannot contain missing"));

        let err = block_on(onehotdecode_builtin(
            tensor(vec![1.2, 0.0], vec![1, 2]),
            tensor(vec![1.0, 2.0], vec![1, 2]),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap_err();
        assert!(err.message.contains("between 0 and 1"));
    }
}
