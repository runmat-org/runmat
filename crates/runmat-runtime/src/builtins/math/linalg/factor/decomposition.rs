//! MATLAB-compatible `decomposition` objects for reusable linear solves.

use std::cell::Cell;
use std::collections::HashMap;

use num_complex::Complex64;
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ClassDef, ComplexTensor, MethodDef, NumericDType, ObjectInstance, PropertyDef, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::{random_args::complex_tensor_into_value, tensor};
use crate::builtins::math::linalg::ops::{mldivide::mldivide_eval, mrdivide::mrdivide_eval};
use crate::{build_runtime_error, BuiltinResult, RuntimeError, OBJECT_INDEX_MEMBER};

const NAME: &str = "decomposition";
const CLASS_NAME: &str = "decomposition";

const MATRIX_FIELD: &str = "__matrix";
const TYPE_FIELD: &str = "Type";
const MATRIX_SIZE_FIELD: &str = "MatrixSize";
const CHECK_CONDITION_FIELD: &str = "CheckCondition";
const DATATYPE_FIELD: &str = "Datatype";
const IS_CONJUGATE_TRANSPOSED_FIELD: &str = "IsConjugateTransposed";
const IS_REAL_FIELD: &str = "IsReal";
const IS_SPARSE_FIELD: &str = "IsSparse";
const SCALE_FACTOR_FIELD: &str = "ScaleFactor";
const RANK_TOLERANCE_FIELD: &str = "RankTolerance";

thread_local! {
    static DECOMPOSITION_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecompositionType {
    Auto,
    Lu,
    Qr,
    Chol,
    Triangular,
    Diagonal,
}

#[derive(Clone, Debug)]
struct DecompositionSpec {
    matrix: Value,
    decomp_type: DecompositionType,
    check_condition: bool,
    rank_tolerance: Option<f64>,
    is_conjugate_transposed: bool,
    is_sparse: bool,
    scale: (f64, f64),
}

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dA",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix decomposition object.",
}];

const SOLVE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric solution.",
}];

const CONSTRUCTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Coefficient matrix, optional type, triangular flag, and name-value options.",
}];

const METHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

const UNARY_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dA",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Matrix decomposition object.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Matrix decomposition object.",
    },
    BuiltinParamDescriptor {
        name: "kind",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing kind.",
    },
    BuiltinParamDescriptor {
        name: "payload",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Indexing payload.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A, type)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(A, type, triangularFlag)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "dA = decomposition(___, Name, Value)",
        inputs: &CONSTRUCTOR_INPUTS,
        outputs: &OUTPUT,
    },
];

const BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.method(lhs, rhs)",
    inputs: &METHOD_INPUTS,
    outputs: &OUTPUT,
}];

const SOLVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = decomposition.solve(lhs, rhs)",
    inputs: &METHOD_INPUTS,
    outputs: &SOLVE_OUTPUT,
}];

const UNARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.method(dA)",
    inputs: &UNARY_INPUT,
    outputs: &OUTPUT,
}];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = decomposition.subsref(obj, kind, payload)",
    inputs: &SUBSREF_INPUTS,
    outputs: &OUTPUT,
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECOMPOSITION.INVALID_INPUT",
    identifier: Some("RunMat:decomposition:InvalidInput"),
    when: "The matrix, type, option, or object operation is invalid.",
    message: "decomposition: invalid input",
};

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECOMPOSITION.INTERNAL",
    identifier: Some("RunMat:decomposition:Internal"),
    when: "Runtime cannot materialize a decomposition object or transformed coefficient matrix.",
    message: "decomposition: internal runtime failure",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const DECOMPOSITION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BINARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_SOLVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SOLVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_UNARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DECOMPOSITION_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn decomposition_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid(message: impl Into<String>) -> RuntimeError {
    decomposition_error(&ERROR_INVALID_INPUT, message)
}

fn internal(message: impl Into<String>) -> RuntimeError {
    decomposition_error(&ERROR_INTERNAL, message)
}

fn ensure_decomposition_class_registered() {
    DECOMPOSITION_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }

        let mut properties = HashMap::new();
        for name in [
            MATRIX_SIZE_FIELD,
            TYPE_FIELD,
            CHECK_CONDITION_FIELD,
            DATATYPE_FIELD,
            IS_CONJUGATE_TRANSPOSED_FIELD,
            IS_REAL_FIELD,
            IS_SPARSE_FIELD,
            SCALE_FACTOR_FIELD,
        ] {
            properties.insert(
                name.to_string(),
                PropertyDef {
                    name: name.to_string(),
                    is_static: false,
                    is_constant: false,
                    is_dependent: false,
                    get_access: Access::Public,
                    set_access: Access::Private,
                    default_value: None,
                },
            );
        }
        properties.insert(
            MATRIX_FIELD.to_string(),
            PropertyDef {
                name: MATRIX_FIELD.to_string(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: Access::Private,
                set_access: Access::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        for method_name in [
            "subsref",
            "uplus",
            "uminus",
            "ctranspose",
            "times",
            "mtimes",
            "rdivide",
            "mrdivide",
            "ldivide",
            "mldivide",
        ] {
            methods.insert(
                method_name.to_string(),
                MethodDef {
                    name: method_name.to_string(),
                    is_static: false,
                    is_abstract: false,
                    is_sealed: false,
                    access: Access::Public,
                    function_name: format!("{CLASS_NAME}.{method_name}"),
                    implicit_class_argument: None,
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: CLASS_NAME.to_string(),
            parent: None,
            properties,
            methods,
        });
        registered.set(true);
    });
}

#[runtime_builtin(
    name = "decomposition",
    category = "math/linalg/factor",
    summary = "Create a matrix decomposition object for reusable linear solves.",
    keywords = "decomposition,linear solve,lu,qr,chol,triangular,diagonal",
    descriptor(crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_DESCRIPTOR),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    ensure_decomposition_class_registered();
    let spec = parse_constructor(args).await?;
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.subsref",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_SUBSREF_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_subsref(obj: Value, kind: String, payload: Value) -> BuiltinResult<Value> {
    if kind != OBJECT_INDEX_MEMBER {
        return Err(invalid(
            "decomposition objects only support property indexing",
        ));
    }
    let field = text_from_value(&payload)
        .ok_or_else(|| invalid("decomposition property name must be a text scalar"))?;
    let object = expect_object(&obj)?;
    public_property_value(object, &field)
}

#[runtime_builtin(
    name = "decomposition.uplus",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_UNARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_uplus(d_a: Value) -> BuiltinResult<Value> {
    let spec = object_to_spec(&d_a)?;
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.uminus",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_UNARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_uminus(d_a: Value) -> BuiltinResult<Value> {
    let mut spec = object_to_spec(&d_a)?;
    spec.scale = complex_mul(spec.scale, (-1.0, 0.0));
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.ctranspose",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_UNARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_ctranspose(d_a: Value) -> BuiltinResult<Value> {
    let mut spec = object_to_spec(&d_a)?;
    spec.is_conjugate_transposed = !spec.is_conjugate_transposed;
    spec.scale = complex_conj(spec.scale);
    Ok(Value::Object(spec_to_object(&spec)?))
}

#[runtime_builtin(
    name = "decomposition.times",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_BINARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_times(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    scale_product(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mtimes",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_BINARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    scale_product(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.rdivide",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_BINARY_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    divide_or_solve(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mrdivide",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_SOLVE_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    divide_or_solve(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.ldivide",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_SOLVE_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_ldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    left_solve(lhs, rhs).await
}

#[runtime_builtin(
    name = "decomposition.mldivide",
    descriptor(
        crate::builtins::math::linalg::factor::decomposition::DECOMPOSITION_SOLVE_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::math::linalg::factor::decomposition"
)]
async fn decomposition_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    left_solve(lhs, rhs).await
}

async fn parse_constructor(args: Vec<Value>) -> BuiltinResult<DecompositionSpec> {
    if args.is_empty() {
        return Err(invalid("decomposition: expected coefficient matrix"));
    }
    let host_args = gather_values(args).await?;
    let mut iter = host_args.into_iter();
    let raw_matrix = iter
        .next()
        .ok_or_else(|| invalid("decomposition: expected coefficient matrix"))?;
    let (matrix, is_sparse) = normalize_matrix(raw_matrix)?;
    let mut decomp_type = DecompositionType::Auto;
    let mut triangular_flag: Option<TriangularFlag> = None;
    let mut options = Vec::new();

    if let Some(next) = iter.next() {
        if let Some(text) = text_from_value(&next) {
            if is_decomposition_type(&text) {
                decomp_type = parse_decomposition_type(&text)?;
                if let Some(candidate) = iter.next() {
                    if let Some(flag) = text_from_value(&candidate)
                        .and_then(|value| parse_triangular_flag(&value).ok())
                    {
                        triangular_flag = Some(flag);
                    } else {
                        options.push(candidate);
                    }
                }
            } else if is_decomposition_option(&text) {
                options.push(next);
            } else {
                return Err(parse_decomposition_type(&text).unwrap_err());
            }
        } else {
            options.push(next);
        }
    }
    options.extend(iter);

    let (check_condition, rank_tolerance) = parse_options(&options)?;
    let matrix = apply_constructor_structure(matrix, decomp_type, triangular_flag)?;
    let decomp_type = if decomp_type == DecompositionType::Auto {
        infer_decomposition_type(&matrix)
    } else {
        validate_type_compatibility(&matrix, decomp_type)?;
        if decomp_type == DecompositionType::Chol {
            validate_chol_positive_definite(&matrix).await?;
        }
        decomp_type
    };

    Ok(DecompositionSpec {
        matrix,
        decomp_type,
        check_condition,
        rank_tolerance,
        is_conjugate_transposed: false,
        is_sparse,
        scale: (1.0, 0.0),
    })
}

async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            crate::dispatcher::gather_if_needed_async(&value)
                .await
                .map_err(|err| {
                    build_runtime_error(err.message())
                        .with_builtin(NAME)
                        .with_source(err)
                        .build()
                })?,
        );
    }
    Ok(out)
}

fn normalize_matrix(value: Value) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::Tensor(tensor) => {
            validate_matrix_shape(&tensor.shape)?;
            Ok((Value::Tensor(tensor), false))
        }
        Value::ComplexTensor(tensor) => {
            validate_matrix_shape(&tensor.shape)?;
            Ok((Value::ComplexTensor(tensor), false))
        }
        Value::SparseTensor(_) => Err(invalid(
            "decomposition: sparse coefficient matrices are not supported yet",
        )),
        Value::Num(n) => Ok((
            Value::Tensor(
                Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Int(i) => Ok((
            Value::Tensor(
                Tensor::new(vec![i.to_f64()], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Bool(b) => Ok((
            Value::Tensor(
                Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Complex(re, im) => Ok((
            Value::ComplexTensor(
                ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        other => Err(invalid(format!(
            "decomposition: coefficient matrix must be numeric, got {other:?}"
        ))),
    }
}

fn validate_matrix_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() > 2 {
        return Err(invalid(
            "decomposition: coefficient matrix must be two-dimensional",
        ));
    }
    Ok(())
}

fn parse_options(args: &[Value]) -> BuiltinResult<(bool, Option<f64>)> {
    let mut check_condition = true;
    let mut rank_tolerance = None;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid(
                "decomposition: name-value options must be provided in pairs",
            ));
        }
        let name = text_from_value(&args[idx])
            .ok_or_else(|| invalid("decomposition: option name must be text"))?;
        match name.to_ascii_lowercase().as_str() {
            "checkcondition" => check_condition = bool_scalar(&args[idx + 1], "CheckCondition")?,
            "ranktolerance" => {
                let tol = real_scalar(&args[idx + 1], "RankTolerance")?;
                if tol < 0.0 || tol.is_nan() {
                    return Err(invalid(
                        "decomposition: RankTolerance must be a nonnegative scalar",
                    ));
                }
                rank_tolerance = Some(tol);
            }
            other => {
                return Err(invalid(format!(
                    "decomposition: unsupported option '{other}'"
                )))
            }
        }
        idx += 2;
    }
    Ok((check_condition, rank_tolerance))
}

fn is_decomposition_type(value: &str) -> bool {
    parse_decomposition_type(value).is_ok()
}

fn is_decomposition_option(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "checkcondition" | "ranktolerance"
    )
}

fn parse_decomposition_type(value: &str) -> BuiltinResult<DecompositionType> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(DecompositionType::Auto),
        "lu" => Ok(DecompositionType::Lu),
        "qr" => Ok(DecompositionType::Qr),
        "chol" => Ok(DecompositionType::Chol),
        "triangular" => Ok(DecompositionType::Triangular),
        "diagonal" => Ok(DecompositionType::Diagonal),
        other => Err(invalid(format!(
            "decomposition: unsupported decomposition type '{other}'"
        ))),
    }
}

fn decomposition_type_name(value: DecompositionType) -> &'static str {
    match value {
        DecompositionType::Auto => "auto",
        DecompositionType::Lu => "lu",
        DecompositionType::Qr => "qr",
        DecompositionType::Chol => "chol",
        DecompositionType::Triangular => "triangular",
        DecompositionType::Diagonal => "diagonal",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TriangularFlag {
    Upper,
    Lower,
}

fn parse_triangular_flag(value: &str) -> BuiltinResult<TriangularFlag> {
    match value.to_ascii_lowercase().as_str() {
        "upper" => Ok(TriangularFlag::Upper),
        "lower" => Ok(TriangularFlag::Lower),
        other => Err(invalid(format!(
            "decomposition: unsupported triangular flag '{other}'"
        ))),
    }
}

fn apply_constructor_structure(
    matrix: Value,
    decomp_type: DecompositionType,
    triangular_flag: Option<TriangularFlag>,
) -> BuiltinResult<Value> {
    match (decomp_type, triangular_flag) {
        (DecompositionType::Triangular, Some(flag)) => keep_triangle(matrix, flag),
        (DecompositionType::Chol, Some(flag)) => symmetrize_from_triangle(matrix, flag),
        (DecompositionType::Triangular, None)
        | (DecompositionType::Auto, None)
        | (DecompositionType::Lu, None)
        | (DecompositionType::Qr, None)
        | (DecompositionType::Chol, None)
        | (DecompositionType::Diagonal, None) => Ok(matrix),
        (_, Some(_)) => Err(invalid(
            "decomposition: triangularFlag is supported only with types 'triangular' and 'chol'",
        )),
    }
}

fn keep_triangle(matrix: Value, flag: TriangularFlag) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(mut tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        tensor.set2(row, col, 0.0).map_err(internal)?;
                    }
                }
            }
            Ok(Value::Tensor(tensor))
        }
        Value::ComplexTensor(mut tensor) => {
            for col in 0..tensor.cols {
                for row in 0..tensor.rows {
                    let zero = match flag {
                        TriangularFlag::Upper => row > col,
                        TriangularFlag::Lower => row < col,
                    };
                    if zero {
                        tensor.data[row + col * tensor.rows] = (0.0, 0.0);
                    }
                }
            }
            Ok(Value::ComplexTensor(tensor))
        }
        other => Err(invalid(format!(
            "decomposition: unsupported coefficient matrix {other:?}"
        ))),
    }
}

fn symmetrize_from_triangle(matrix: Value, flag: TriangularFlag) -> BuiltinResult<Value> {
    match matrix {
        Value::Tensor(mut tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row == col {
                        continue;
                    }
                    let value = match flag {
                        TriangularFlag::Upper if row > col => tensor.get2(col, row),
                        TriangularFlag::Lower if row < col => tensor.get2(col, row),
                        _ => continue,
                    }
                    .map_err(internal)?;
                    tensor.set2(row, col, value).map_err(internal)?;
                }
            }
            Ok(Value::Tensor(tensor))
        }
        Value::ComplexTensor(mut tensor) => {
            for col in 0..tensor.cols {
                for row in 0..tensor.rows {
                    if row == col {
                        continue;
                    }
                    let maybe_value = match flag {
                        TriangularFlag::Upper if row > col => {
                            let (re, im) = tensor.data[col + row * tensor.rows];
                            Some((re, -im))
                        }
                        TriangularFlag::Lower if row < col => {
                            let (re, im) = tensor.data[col + row * tensor.rows];
                            Some((re, -im))
                        }
                        _ => None,
                    };
                    if let Some(value) = maybe_value {
                        tensor.data[row + col * tensor.rows] = value;
                    }
                }
            }
            Ok(Value::ComplexTensor(tensor))
        }
        other => Err(invalid(format!(
            "decomposition: unsupported coefficient matrix {other:?}"
        ))),
    }
}

fn infer_decomposition_type(matrix: &Value) -> DecompositionType {
    let (rows, cols) = matrix_dims(matrix);
    if is_diagonal_matrix(matrix) {
        DecompositionType::Diagonal
    } else if rows == cols && is_triangular_matrix(matrix) {
        DecompositionType::Triangular
    } else if rows == cols {
        DecompositionType::Lu
    } else {
        DecompositionType::Qr
    }
}

fn validate_type_compatibility(
    matrix: &Value,
    decomp_type: DecompositionType,
) -> BuiltinResult<()> {
    let (rows, cols) = matrix_dims(matrix);
    match decomp_type {
        DecompositionType::Lu
        | DecompositionType::Chol
        | DecompositionType::Triangular
        | DecompositionType::Diagonal
            if rows != cols =>
        {
            Err(invalid(format!(
                "decomposition: type '{}' requires a square matrix",
                decomposition_type_name(decomp_type)
            )))
        }
        DecompositionType::Diagonal if !is_diagonal_matrix(matrix) => Err(invalid(
            "decomposition: type 'diagonal' requires a diagonal matrix",
        )),
        DecompositionType::Triangular if !is_triangular_matrix(matrix) => Err(invalid(
            "decomposition: type 'triangular' requires a triangular matrix",
        )),
        DecompositionType::Chol if !is_hermitian_matrix(matrix) => Err(invalid(
            "decomposition: type 'chol' requires a symmetric or Hermitian matrix",
        )),
        _ => Ok(()),
    }
}

async fn validate_chol_positive_definite(matrix: &Value) -> BuiltinResult<()> {
    crate::call_builtin_async("chol", std::slice::from_ref(matrix))
        .await
        .map(|_| ())
        .map_err(|_| {
            invalid("decomposition: type 'chol' requires a positive definite coefficient matrix")
        })
}

fn matrix_dims(matrix: &Value) -> (usize, usize) {
    match matrix {
        Value::Tensor(tensor) => (tensor.rows(), tensor.cols()),
        Value::ComplexTensor(tensor) => (tensor.rows, tensor.cols),
        _ => (1, 1),
    }
}

fn matrix_shape_value(matrix: &Value) -> BuiltinResult<Value> {
    let (rows, cols) = matrix_dims(matrix);
    let tensor = Tensor::new(vec![rows as f64, cols as f64], vec![1, 2])
        .map_err(|e| internal(format!("decomposition: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn is_diagonal_matrix(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..tensor.rows() {
                    if row != col && tensor.data[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            for col in 0..tensor.cols {
                for row in 0..tensor.rows {
                    if row != col && tensor.data[row + col * tensor.rows] != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        _ => true,
    }
}

fn is_triangular_matrix(matrix: &Value) -> bool {
    is_upper_triangular(matrix) || is_lower_triangular(matrix)
}

fn is_upper_triangular(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            for col in 0..tensor.cols() {
                for row in (col + 1)..tensor.rows() {
                    if tensor.data[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            for col in 0..tensor.cols {
                for row in (col + 1)..tensor.rows {
                    if tensor.data[row + col * tensor.rows] != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        _ => true,
    }
}

fn is_lower_triangular(matrix: &Value) -> bool {
    match matrix {
        Value::Tensor(tensor) => {
            for col in 0..tensor.cols() {
                for row in 0..col.min(tensor.rows()) {
                    if tensor.data[row + col * tensor.rows()] != 0.0 {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            for col in 0..tensor.cols {
                for row in 0..col.min(tensor.rows) {
                    if tensor.data[row + col * tensor.rows] != (0.0, 0.0) {
                        return false;
                    }
                }
            }
            true
        }
        _ => true,
    }
}

fn is_hermitian_matrix(matrix: &Value) -> bool {
    let (rows, cols) = matrix_dims(matrix);
    if rows != cols {
        return false;
    }
    match matrix {
        Value::Tensor(tensor) => {
            for col in 0..cols {
                for row in 0..rows {
                    if tensor.data[row + col * rows] != tensor.data[col + row * rows] {
                        return false;
                    }
                }
            }
            true
        }
        Value::ComplexTensor(tensor) => {
            for col in 0..cols {
                for row in 0..rows {
                    let lhs = tensor.data[row + col * rows];
                    let rhs = tensor.data[col + row * rows];
                    if lhs != (rhs.0, -rhs.1) {
                        return false;
                    }
                }
            }
            true
        }
        _ => false,
    }
}

fn spec_to_object(spec: &DecompositionSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CLASS_NAME.to_string());
    let public_matrix = oriented_matrix(spec)?;
    object
        .properties
        .insert(MATRIX_FIELD.to_string(), spec.matrix.clone());
    object.properties.insert(
        TYPE_FIELD.to_string(),
        Value::String(decomposition_type_name(spec.decomp_type).to_string()),
    );
    object.properties.insert(
        MATRIX_SIZE_FIELD.to_string(),
        matrix_shape_value(&public_matrix)?,
    );
    object.properties.insert(
        CHECK_CONDITION_FIELD.to_string(),
        Value::Bool(spec.check_condition),
    );
    object.properties.insert(
        DATATYPE_FIELD.to_string(),
        Value::String(matrix_datatype(&spec.matrix).to_string()),
    );
    object.properties.insert(
        IS_CONJUGATE_TRANSPOSED_FIELD.to_string(),
        Value::Bool(spec.is_conjugate_transposed),
    );
    object.properties.insert(
        IS_REAL_FIELD.to_string(),
        Value::Bool(matrix_is_real(&public_matrix) && spec.scale.1 == 0.0),
    );
    object
        .properties
        .insert(IS_SPARSE_FIELD.to_string(), Value::Bool(spec.is_sparse));
    object.properties.insert(
        SCALE_FACTOR_FIELD.to_string(),
        complex_scalar_to_value(spec.scale),
    );
    if let Some(tol) = spec.rank_tolerance {
        object
            .properties
            .insert(RANK_TOLERANCE_FIELD.to_string(), Value::Num(tol));
    }
    Ok(object)
}

fn public_property_value(object: &ObjectInstance, field: &str) -> BuiltinResult<Value> {
    match field {
        MATRIX_SIZE_FIELD
        | TYPE_FIELD
        | CHECK_CONDITION_FIELD
        | DATATYPE_FIELD
        | IS_CONJUGATE_TRANSPOSED_FIELD
        | IS_REAL_FIELD
        | IS_SPARSE_FIELD
        | SCALE_FACTOR_FIELD => object
            .properties
            .get(field)
            .cloned()
            .ok_or_else(|| internal(format!("decomposition object is missing {field}"))),
        _ => Err(invalid(format!("unknown decomposition property '{field}'"))),
    }
}

fn object_to_spec(value: &Value) -> BuiltinResult<DecompositionSpec> {
    let object = expect_object(value)?;
    let matrix = object
        .properties
        .get(MATRIX_FIELD)
        .cloned()
        .ok_or_else(|| internal("decomposition object is missing coefficient matrix"))?;
    let decomp_type = object
        .properties
        .get(TYPE_FIELD)
        .and_then(text_from_value)
        .map(|value| parse_decomposition_type(&value))
        .transpose()?
        .ok_or_else(|| internal("decomposition object is missing Type"))?;
    let check_condition = object
        .properties
        .get(CHECK_CONDITION_FIELD)
        .map(|value| bool_scalar(value, CHECK_CONDITION_FIELD))
        .transpose()?
        .unwrap_or(true);
    let is_conjugate_transposed = object
        .properties
        .get(IS_CONJUGATE_TRANSPOSED_FIELD)
        .map(|value| bool_scalar(value, IS_CONJUGATE_TRANSPOSED_FIELD))
        .transpose()?
        .unwrap_or(false);
    let is_sparse = object
        .properties
        .get(IS_SPARSE_FIELD)
        .map(|value| bool_scalar(value, IS_SPARSE_FIELD))
        .transpose()?
        .unwrap_or(false);
    let scale = object
        .properties
        .get(SCALE_FACTOR_FIELD)
        .map(|value| complex_scalar(value, SCALE_FACTOR_FIELD))
        .transpose()?
        .unwrap_or((1.0, 0.0));
    let rank_tolerance = object
        .properties
        .get(RANK_TOLERANCE_FIELD)
        .map(|value| real_scalar(value, RANK_TOLERANCE_FIELD))
        .transpose()?;
    Ok(DecompositionSpec {
        matrix,
        decomp_type,
        check_condition,
        rank_tolerance,
        is_conjugate_transposed,
        is_sparse,
        scale,
    })
}

fn expect_object(value: &Value) -> BuiltinResult<&ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(CLASS_NAME) => Ok(object),
        _ => Err(invalid("receiver must be a decomposition object")),
    }
}

async fn scale_product(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if is_decomposition_value(&lhs) {
        let mut spec = object_to_spec(&lhs)?;
        let scalar = complex_scalar_async(&rhs, "scale factor").await?;
        spec.scale = complex_mul(spec.scale, scalar);
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    if is_decomposition_value(&rhs) {
        let mut spec = object_to_spec(&rhs)?;
        let scalar = complex_scalar_async(&lhs, "scale factor").await?;
        spec.scale = complex_mul(scalar, spec.scale);
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    Err(invalid(
        "decomposition: multiplication requires one decomposition object and one scalar",
    ))
}

async fn divide_or_solve(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if is_decomposition_value(&lhs) {
        let mut spec = object_to_spec(&lhs)?;
        let scalar = complex_scalar_async(&rhs, "scale divisor").await?;
        spec.scale = complex_div(spec.scale, scalar)?;
        return Ok(Value::Object(spec_to_object(&spec)?));
    }
    if is_decomposition_value(&rhs) {
        let spec = object_to_spec(&rhs)?;
        let matrix = effective_matrix(&spec)?;
        return mrdivide_eval(&lhs, &matrix).await;
    }
    Err(invalid(
        "decomposition: right division requires a decomposition object",
    ))
}

async fn left_solve(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if !is_decomposition_value(&lhs) {
        return Err(invalid(
            "decomposition: left division requires a decomposition object on the left",
        ));
    }
    let spec = object_to_spec(&lhs)?;
    let matrix = effective_matrix(&spec)?;
    mldivide_eval(&matrix, &rhs).await
}

fn is_decomposition_value(value: &Value) -> bool {
    matches!(value, Value::Object(object) if object.is_class(CLASS_NAME))
}

fn effective_matrix(spec: &DecompositionSpec) -> BuiltinResult<Value> {
    let matrix = oriented_matrix(spec)?;
    scale_matrix(matrix, spec.scale)
}

fn oriented_matrix(spec: &DecompositionSpec) -> BuiltinResult<Value> {
    if spec.is_conjugate_transposed {
        conjugate_transpose(spec.matrix.clone())
    } else {
        Ok(spec.matrix.clone())
    }
}

fn conjugate_transpose(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => {
            let rows = tensor.rows();
            let cols = tensor.cols();
            let dtype = tensor.dtype;
            let values = tensor::tensor_into_values_f64(tensor);
            let mut data = vec![0.0; values.len()];
            for row in 0..rows {
                for col in 0..cols {
                    data[col + row * cols] = values[row + col * rows];
                }
            }
            Ok(Value::Tensor(
                Tensor::new_with_dtype(data, vec![cols, rows], dtype)
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::ComplexTensor(tensor) => {
            let rows = tensor.rows;
            let cols = tensor.cols;
            let values = tensor::complex_tensor_into_values_complex64(tensor);
            let mut data = vec![(0.0, 0.0); values.len()];
            for row in 0..rows {
                for col in 0..cols {
                    let value = values[row + col * rows];
                    data[col + row * cols] = (value.re, -value.im);
                }
            }
            Ok(Value::ComplexTensor(
                ComplexTensor::new(data, vec![cols, rows])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        other => Err(internal(format!(
            "decomposition: unsupported stored matrix {other:?}"
        ))),
    }
}

fn scale_matrix(value: Value, scale: (f64, f64)) -> BuiltinResult<Value> {
    if scale == (1.0, 0.0) {
        return Ok(value);
    }
    match value {
        Value::Tensor(tensor) if scale.1 == 0.0 => {
            let shape = tensor.shape.clone();
            let dtype = tensor.dtype;
            Ok(Value::Tensor(
                Tensor::new_with_dtype(
                    tensor::tensor_into_values_f64(tensor)
                        .into_iter()
                        .map(|value| value * scale.0)
                        .collect(),
                    shape,
                    dtype,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::Tensor(tensor) => {
            let shape = tensor.shape.clone();
            Ok(Value::ComplexTensor(
                ComplexTensor::new(
                    tensor::tensor_into_values_f64(tensor)
                        .into_iter()
                        .map(|value| complex_mul((value, 0.0), scale))
                        .collect(),
                    shape,
                )
                .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        Value::ComplexTensor(tensor) => {
            let shape = tensor.shape.clone();
            let data = tensor::complex_tensor_into_values_complex64(tensor)
                .into_iter()
                .map(|value| complex_mul((value.re, value.im), scale))
                .collect();
            Ok(complex_tensor_into_value(
                ComplexTensor::new(data, shape)
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ))
        }
        other => Err(internal(format!(
            "decomposition: unsupported stored matrix {other:?}"
        ))),
    }
}

fn matrix_datatype(matrix: &Value) -> &'static str {
    match matrix {
        Value::Tensor(tensor) if tensor.dtype == NumericDType::F32 => "single",
        _ => "double",
    }
}

fn matrix_is_real(matrix: &Value) -> bool {
    match matrix {
        Value::ComplexTensor(tensor) => tensor.data.iter().all(|(_, im)| *im == 0.0),
        _ => true,
    }
}

async fn complex_scalar_async(value: &Value, label: &str) -> BuiltinResult<(f64, f64)> {
    let host = crate::dispatcher::gather_if_needed_async(value)
        .await
        .map_err(|err| {
            build_runtime_error(err.message())
                .with_builtin(NAME)
                .with_source(err)
                .build()
        })?;
    complex_scalar(&host, label)
}

fn complex_scalar(value: &Value, label: &str) -> BuiltinResult<(f64, f64)> {
    match value {
        Value::Num(n) => Ok((*n, 0.0)),
        Value::Int(i) => Ok((i.to_f64(), 0.0)),
        Value::Bool(b) => Ok((if *b { 1.0 } else { 0.0 }, 0.0)),
        Value::Complex(re, im) => Ok((*re, *im)),
        Value::Tensor(tensor) if tensor_element_len(tensor) == 1 => {
            Ok((tensor::tensor_value_f64(tensor, 0), 0.0))
        }
        Value::ComplexTensor(tensor) if complex_tensor_element_len(tensor) == 1 => {
            if let Some(storage) = tensor.integer_data.as_ref() {
                let re = storage
                    .real
                    .value_at(0)
                    .expect("one-element complex integer real storage")
                    .to_f64();
                let im = storage
                    .imag
                    .value_at(0)
                    .expect("one-element complex integer imaginary storage")
                    .to_f64();
                return Ok((re, im));
            }
            Ok(tensor.data[0])
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            Ok((if array.data[0] != 0 { 1.0 } else { 0.0 }, 0.0))
        }
        other => Err(invalid(format!(
            "decomposition: {label} must be scalar, got {other:?}"
        ))),
    }
}

fn tensor_element_len(tensor: &Tensor) -> usize {
    tensor
        .integer_storage()
        .map_or(tensor.data.len(), |storage| storage.len())
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor
        .integer_data
        .as_ref()
        .map_or(tensor.data.len(), |storage| storage.real.len())
}

fn real_scalar(value: &Value, label: &str) -> BuiltinResult<f64> {
    let (re, im) = complex_scalar(value, label)?;
    if im != 0.0 {
        return Err(invalid(format!(
            "decomposition: {label} must be a real scalar"
        )));
    }
    Ok(re)
}

fn bool_scalar(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        _ => {
            let scalar = real_scalar(value, label)?;
            Ok(scalar != 0.0)
        }
    }
}

fn text_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn complex_scalar_to_value(value: (f64, f64)) -> Value {
    if value.1 == 0.0 && !value.1.is_nan() {
        Value::Num(value.0)
    } else {
        Value::Complex(value.0, value.1)
    }
}

fn complex_conj(value: (f64, f64)) -> (f64, f64) {
    (value.0, -value.1)
}

fn complex_mul(lhs: (f64, f64), rhs: (f64, f64)) -> (f64, f64) {
    (
        lhs.0.mul_add(rhs.0, -(lhs.1 * rhs.1)),
        lhs.0.mul_add(rhs.1, lhs.1 * rhs.0),
    )
}

fn complex_div(lhs: (f64, f64), rhs: (f64, f64)) -> BuiltinResult<(f64, f64)> {
    if rhs.0 == 0.0 && rhs.1 == 0.0 {
        return Err(invalid("decomposition: division by zero scale factor"));
    }
    let result = Complex64::new(lhs.0, lhs.1) / Complex64::new(rhs.0, rhs.1);
    Ok((result.re, result.im))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerComplexStorage, IntegerStorage};

    fn tensor(data: &[f64], rows: usize, cols: usize) -> Value {
        Value::Tensor(Tensor::new(data.to_vec(), vec![rows, cols]).unwrap())
    }

    fn complex_tensor(data: &[(f64, f64)], rows: usize, cols: usize) -> Value {
        Value::ComplexTensor(ComplexTensor::new(data.to_vec(), vec![rows, cols]).unwrap())
    }

    fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        let mut tensor = Tensor::new_integer(storage, shape).unwrap();
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    fn mirrorless_complex_int_tensor(
        real: IntegerStorage,
        imag: IntegerStorage,
        shape: Vec<usize>,
    ) -> Value {
        let mut tensor =
            ComplexTensor::new_integer(IntegerComplexStorage::new(real, imag).unwrap(), shape)
                .unwrap();
        tensor.data.clear();
        Value::ComplexTensor(tensor)
    }

    fn call_constructor(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(decomposition_builtin(args))
    }

    fn call_mldivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(decomposition_mldivide(lhs, rhs))
    }

    fn call_mrdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(decomposition_mrdivide(lhs, rhs))
    }

    fn call_mtimes(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(decomposition_mtimes(lhs, rhs))
    }

    fn call_rdivide(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
        block_on(decomposition_rdivide(lhs, rhs))
    }

    fn call_ctranspose(value: Value) -> BuiltinResult<Value> {
        block_on(decomposition_ctranspose(value))
    }

    fn approx_vec(value: Value) -> Vec<f64> {
        match value {
            Value::Num(n) => vec![n],
            Value::Tensor(t) => t.data,
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn decomposition_left_solve_reuses_mldivide_semantics() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let b = tensor(&[8.0, 12.0], 2, 1);
        let d = call_constructor(vec![a]).expect("decomposition");

        let result = call_mldivide(d, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 4.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn decomposition_right_solve_uses_rhs_object() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let b = tensor(&[8.0, 12.0], 1, 2);
        let d = call_constructor(vec![a]).expect("decomposition");

        let result = call_mrdivide(b, d).expect("right solve");

        let values = approx_vec(result);
        assert!((values[0] - 4.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn scalar_multiply_and_divide_adjust_solve_matrix() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let b = tensor(&[8.0, 12.0], 2, 1);
        let d = call_constructor(vec![a]).expect("decomposition");
        let scaled = call_mtimes(Value::Num(2.0), d).expect("scale");
        let divided = call_rdivide(scaled, Value::Num(4.0)).expect("divide");

        let result = call_mldivide(divided, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 8.0).abs() < 1e-12);
        assert!((values[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn ctranspose_toggles_property_and_solves_transposed_system() {
        let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
        let b = tensor(&[3.0, 1.0], 2, 1);
        let d = call_constructor(vec![a]).expect("decomposition");
        let dt = call_ctranspose(d).expect("ctranspose");

        let prop = block_on(decomposition_subsref(
            dt.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(IS_CONJUGATE_TRANSPOSED_FIELD.to_string()),
        ))
        .expect("property");
        assert_eq!(prop, Value::Bool(true));

        let result = call_mldivide(dt, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 3.0).abs() < 1e-12);
        assert!((values[1] + 5.0).abs() < 1e-12);
    }

    #[test]
    fn stored_matrix_helpers_read_typed_integer_storage_exactly() {
        let matrix = mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4, 5, 6]), vec![2, 3]);

        match conjugate_transpose(matrix).expect("ctranspose") {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![3, 2]);
                assert_eq!(
                    tensor::tensor_values_f64(&tensor),
                    vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let scaled = scale_matrix(
            mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]),
            (2.0, 0.0),
        )
        .expect("scale");
        match scaled {
            Value::Tensor(tensor) => {
                assert_eq!(tensor::tensor_values_f64(&tensor), vec![2.0, 4.0, 6.0, 8.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn stored_matrix_helpers_read_typed_complex_integer_storage_exactly() {
        let matrix = mirrorless_complex_int_tensor(
            IntegerStorage::I16(vec![1, 2, 3, 4]),
            IntegerStorage::I16(vec![5, 6, 7, 8]),
            vec![2, 2],
        );

        match conjugate_transpose(matrix).expect("ctranspose") {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(
                    tensor.data,
                    vec![(1.0, -5.0), (3.0, -7.0), (2.0, -6.0), (4.0, -8.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }

        let scaled = scale_matrix(
            mirrorless_complex_int_tensor(
                IntegerStorage::I16(vec![1, 2]),
                IntegerStorage::I16(vec![3, 4]),
                vec![2, 1],
            ),
            (2.0, 0.0),
        )
        .expect("scale");
        match scaled {
            Value::ComplexTensor(tensor) => {
                assert_eq!(tensor.data, vec![(2.0, 6.0), (4.0, 8.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn properties_expose_type_size_and_flags() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let d = call_constructor(vec![a, Value::String("diagonal".to_string())])
            .expect("decomposition");

        let ty = block_on(decomposition_subsref(
            d.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(TYPE_FIELD.to_string()),
        ))
        .expect("Type");
        assert_eq!(ty, Value::String("diagonal".to_string()));

        let size = block_on(decomposition_subsref(
            d,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(MATRIX_SIZE_FIELD.to_string()),
        ))
        .expect("MatrixSize");
        assert_eq!(approx_vec(size), vec![2.0, 2.0]);
    }

    #[test]
    fn triangular_flag_ignores_opposite_triangle() {
        let a = tensor(&[2.0, 99.0, 0.0, 4.0], 2, 2);
        let b = tensor(&[8.0, 12.0], 2, 1);
        let d = call_constructor(vec![
            a,
            Value::String("triangular".to_string()),
            Value::String("upper".to_string()),
        ])
        .expect("decomposition");

        let result = call_mldivide(d, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 4.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn complex_decomposition_solves_complex_rhs() {
        let a = complex_tensor(&[(2.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 2.0)], 2, 2);
        let b = complex_tensor(&[(8.0, 2.0), (4.0, 2.0)], 2, 1);
        let d = call_constructor(vec![a]).expect("decomposition");

        let result = call_mldivide(d, b).expect("complex solve");

        match result {
            Value::ComplexTensor(tensor) => {
                assert!((tensor.data[0].0 - 4.0).abs() < 1e-12);
                assert!((tensor.data[0].1 - 1.0).abs() < 1e-12);
                assert!((tensor.data[1].0 - 1.0).abs() < 1e-12);
                assert!((tensor.data[1].1 + 2.0).abs() < 1e-12);
            }
            other => panic!("expected complex result, got {other:?}"),
        }
    }

    #[test]
    fn registered_constructor_and_ctranspose_dispatch_work_together() {
        let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
        let d = block_on(crate::call_builtin_async("decomposition", &[a])).expect("constructor");
        let dt = block_on(crate::call_builtin_async("ctranspose", &[d])).expect("dispatch");
        let b = tensor(&[3.0, 1.0], 2, 1);

        let result = call_mldivide(dt, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 3.0).abs() < 1e-12);
        assert!((values[1] + 5.0).abs() < 1e-12);
    }

    #[test]
    fn decomposition_scalar_operands_read_typed_integer_storage_exactly() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let d = call_constructor(vec![a]).expect("decomposition");
        let mut scale =
            Tensor::new_integer(IntegerStorage::I16(vec![3]), vec![1, 1]).expect("scale");
        scale.data.clear();

        let scaled = call_mtimes(d, Value::Tensor(scale.clone())).expect("scale");
        let prop = block_on(decomposition_subsref(
            scaled,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(SCALE_FACTOR_FIELD.to_string()),
        ))
        .expect("ScaleFactor");
        assert_eq!(approx_vec(prop), vec![3.0]);
        assert_eq!(
            complex_scalar(&Value::Tensor(scale), "scale").unwrap(),
            (3.0, 0.0)
        );

        let storage = runmat_builtins::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![4]),
            IntegerStorage::I16(vec![-2]),
        )
        .expect("typed complex storage");
        let mut complex_scale =
            ComplexTensor::new_integer(storage, vec![1, 1]).expect("complex scale");
        complex_scale.data.clear();
        assert_eq!(
            complex_scalar(&Value::ComplexTensor(complex_scale), "scale").unwrap(),
            (4.0, -2.0)
        );
    }

    #[test]
    fn chol_type_rejects_non_positive_definite_matrix() {
        let a = tensor(&[1.0, 0.0, 0.0, -1.0], 2, 2);

        let err =
            call_constructor(vec![a, Value::String("chol".to_string())]).expect_err("chol error");

        assert!(err.message().contains("positive definite"));
    }

    #[test]
    fn chol_lower_flag_uses_lower_triangle_as_hermitian_matrix() {
        let a = tensor(&[4.0, 1.0, 99.0, 3.0], 2, 2);
        let b = tensor(&[5.0, 4.0], 2, 1);
        let d = call_constructor(vec![
            a,
            Value::String("chol".to_string()),
            Value::String("lower".to_string()),
        ])
        .expect("chol decomposition");

        let result = call_mldivide(d, b).expect("solve");

        let values = approx_vec(result);
        assert!((values[0] - 1.0).abs() < 1e-12);
        assert!((values[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn private_matrix_storage_is_not_publicly_readable() {
        let d = call_constructor(vec![tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2)]).expect("decomposition");

        let subsref_err = block_on(decomposition_subsref(
            d.clone(),
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(MATRIX_FIELD.to_string()),
        ))
        .expect_err("private subsref should fail");
        assert!(subsref_err
            .message()
            .contains("unknown decomposition property"));

        let getfield_err = block_on(crate::call_builtin_async(
            "getfield",
            &[d, Value::String(MATRIX_FIELD.to_string())],
        ))
        .expect_err("private getfield should fail");
        assert!(
            getfield_err.message().contains("cannot get")
                || getfield_err.message().contains("No public property")
        );
    }

    #[test]
    fn sparse_inputs_are_rejected_until_sparse_solves_exist() {
        let sparse =
            runmat_builtins::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])
                .expect("sparse");

        let err = call_constructor(vec![Value::SparseTensor(sparse)]).expect_err("sparse reject");

        assert!(err.message().contains("sparse coefficient matrices"));
    }

    #[test]
    fn unsupported_decomposition_types_are_rejected() {
        let a = tensor(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        let err = call_constructor(vec![a, Value::String("banded".to_string())]).expect_err("type");

        assert!(err.message().contains("unsupported decomposition type"));
    }

    #[test]
    fn ctranspose_updates_rectangular_matrix_size_property() {
        let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let d = call_constructor(vec![a]).expect("decomposition");
        let dt = call_ctranspose(d).expect("ctranspose");

        let size = block_on(decomposition_subsref(
            dt,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(MATRIX_SIZE_FIELD.to_string()),
        ))
        .expect("MatrixSize");

        assert_eq!(approx_vec(size), vec![3.0, 2.0]);
    }

    #[test]
    fn complex_scale_marks_decomposition_nonreal() {
        let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
        let d = call_constructor(vec![a]).expect("decomposition");
        let scaled = call_mtimes(Value::Complex(0.0, 1.0), d).expect("complex scale");

        let is_real = block_on(decomposition_subsref(
            scaled,
            OBJECT_INDEX_MEMBER.to_string(),
            Value::String(IS_REAL_FIELD.to_string()),
        ))
        .expect("IsReal");

        assert_eq!(is_real, Value::Bool(false));
    }
}
