use runmat_builtins::{
    builtin_function_by_name, builtin_functions, AccelTag, BuiltinCompletionPolicy,
    BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor, BuiltinExtensionMode,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Tensor;

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];
const ADD_SUB_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left integer input.",
    },
    BuiltinParamDescriptor {
        name: "y",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right integer input.",
    },
];
const MATRIX_SUM_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix.",
}];
const STR_LENGTH_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "s",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input string.",
}];
const ADD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = add(x, y)",
    inputs: &ADD_SUB_INPUTS,
    outputs: &OUT_ANY,
}];
const SUB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = sub(x, y)",
    inputs: &ADD_SUB_INPUTS,
    outputs: &OUT_ANY,
}];
const MATRIX_SUM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = matrix_sum(A)",
    inputs: &MATRIX_SUM_INPUTS,
    outputs: &OUT_ANY,
}];
const STR_LENGTH_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = str_length(s)",
    inputs: &STR_LENGTH_INPUTS,
    outputs: &OUT_ANY,
}];
const MUL_ADD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = mul_add(x, y)",
    inputs: &[
        BuiltinParamDescriptor {
            name: "x",
            ty: BuiltinParamType::NumericScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Left numeric input.",
        },
        BuiltinParamDescriptor {
            name: "y",
            ty: BuiltinParamType::NumericScalar,
            arity: BuiltinParamArity::Required,
            default: None,
            description: "Right numeric input.",
        },
    ],
    outputs: &OUT_ANY,
}];
const ADD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const ADD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BuiltinExtensionDescriptor {
    id: "test-add-extension",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "test-only add extension",
    error_identifier: None,
}];
const SUB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const MATRIX_SUM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATRIX_SUM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const STR_LENGTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STR_LENGTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const MUL_ADD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MUL_ADD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};

#[runtime_builtin(
    name = "add",
    descriptor(crate::native_registration::ADD_DESCRIPTOR),
    extensions(crate::native_registration::ADD_EXTENSIONS),
    builtin_path = "tests::native_registration::add"
)]
fn add(x: i32, y: i32) -> Result<i32, String> {
    Ok(x + y)
}

#[runtime_builtin(
    name = "sub",
    descriptor(crate::native_registration::SUB_DESCRIPTOR),
    builtin_path = "tests::native_registration::sub"
)]
fn sub(x: i32, y: i32) -> Result<i32, String> {
    Ok(x - y)
}

#[runtime_builtin(
    name = "matrix_sum",
    descriptor(crate::native_registration::MATRIX_SUM_DESCRIPTOR),
    builtin_path = "tests::native_registration::matrix_sum"
)]
fn matrix_sum(m: Tensor) -> Result<f64, String> {
    Ok(m.materialize_f64().iter().sum())
}

#[runtime_builtin(
    name = "str_length",
    descriptor(crate::native_registration::STR_LENGTH_DESCRIPTOR),
    builtin_path = "tests::native_registration::str_length"
)]
fn str_length(s: String) -> Result<i32, String> {
    Ok(s.len() as i32)
}

#[runtime_builtin(
    name = "mul_add",
    accel = "binary",
    descriptor(crate::native_registration::MUL_ADD_DESCRIPTOR),
    builtin_path = "tests::native_registration::mul_add"
)]
fn mul_add(x: f64, y: f64) -> Result<f64, String> {
    Ok(x + y)
}

#[test]
fn contains_registered_functions() {
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
    assert!(names.contains(&"matrix_sum"));
    assert!(names.contains(&"str_length"));
    assert!(names.contains(&"mul_add"));
}

#[test]
fn binary_accel_metadata_maps_to_elementwise_tag() {
    let builtin = builtin_function_by_name("mul_add").expect("registered builtin");
    assert_eq!(builtin.accel_tags, &[AccelTag::Elementwise]);
}

#[test]
fn extension_metadata_is_registered_declaratively() {
    let add = builtin_function_by_name("add").expect("registered builtin");
    assert_eq!(add.extensions, &ADD_EXTENSIONS);
    let sub = builtin_function_by_name("sub").expect("registered builtin");
    assert!(sub.extensions.is_empty());
}
