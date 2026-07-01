use runmat_builtins::{
    builtin_function_by_name, builtin_functions, AccelTag, BuiltinCompletionPolicy,
    BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, CellArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

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
    descriptor(crate::ADD_DESCRIPTOR),
    builtin_path = "tests::add"
)]
fn add(x: i32, y: i32) -> Result<i32, String> {
    Ok(x + y)
}

#[runtime_builtin(
    name = "sub",
    descriptor(crate::SUB_DESCRIPTOR),
    builtin_path = "tests::sub"
)]
fn sub(x: i32, y: i32) -> Result<i32, String> {
    Ok(x - y)
}

#[runtime_builtin(
    name = "matrix_sum",
    descriptor(crate::MATRIX_SUM_DESCRIPTOR),
    builtin_path = "tests::matrix_sum"
)]
fn matrix_sum(m: Tensor) -> Result<f64, String> {
    Ok(m.data.iter().sum())
}

#[runtime_builtin(
    name = "str_length",
    descriptor(crate::STR_LENGTH_DESCRIPTOR),
    builtin_path = "tests::str_length"
)]
fn str_length(s: String) -> Result<i32, String> {
    Ok(s.len() as i32)
}

#[runtime_builtin(
    name = "mul_add",
    accel = "binary",
    descriptor(crate::MUL_ADD_DESCRIPTOR),
    builtin_path = "tests::mul_add"
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
fn test_value_conversions() {
    // Test basic types
    let int_val = Value::Int(runmat_builtins::IntValue::I32(42));
    let num_val = Value::Num(3.15);
    let bool_val = Value::Bool(true);
    let str_val = Value::String("hello".to_string());

    // Test From implementations
    assert_eq!(Value::from(42), int_val);
    assert_eq!(Value::from(3.15), num_val);
    assert_eq!(Value::from(true), bool_val);
    assert_eq!(Value::from("hello"), str_val);

    // Test TryFrom implementations
    use std::convert::TryInto;
    assert_eq!((&int_val).try_into(), Ok(42i32));
    assert_eq!((&num_val).try_into(), Ok(3.15f64));
    assert_eq!((&bool_val).try_into(), Ok(true));
    assert_eq!((&str_val).try_into(), Ok("hello".to_string()));
}

#[test]
fn test_matrix_operations() {
    let mut matrix = Tensor::zeros2(2, 3);
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    assert_eq!(matrix.data.len(), 6);

    // Test setting and getting values (0-based helpers)
    matrix.set2(1, 2, 5.0).unwrap();
    assert_eq!(matrix.get2(1, 2).unwrap(), 5.0);

    // Test bounds checking
    assert!(matrix.get2(2, 0).is_err());
    assert!(matrix.set2(0, 3, 1.0).is_err());

    // Test matrix creation
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let matrix2 = Tensor::new_2d(data, 2, 2).unwrap();
    // Column-major: data = [1,2,3,4] laid out as [[1,3];[2,4]] for 2x2
    assert_eq!(matrix2.get2(0, 1).unwrap(), 3.0);
    assert_eq!(matrix2.get2(1, 1).unwrap(), 4.0);

    // Test invalid dimensions
    assert!(Tensor::new_2d(vec![1.0, 2.0], 2, 2).is_err());
}

#[test]
fn test_cell_arrays() {
    let cell = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(runmat_builtins::IntValue::I32(1)),
                Value::String("test".to_string()),
                Value::Bool(false),
            ],
            1,
            3,
        )
        .unwrap(),
    );

    if let Value::Cell(contents) = cell {
        assert_eq!(contents.data.len(), 3);
        assert_eq!(
            &contents.data[0],
            &Value::Int(runmat_builtins::IntValue::I32(1))
        );
        assert_eq!(&contents.data[1], &Value::String("test".to_string()));
        assert_eq!(&contents.data[2], &Value::Bool(false));
    } else {
        panic!("Expected Cell value");
    }
}
