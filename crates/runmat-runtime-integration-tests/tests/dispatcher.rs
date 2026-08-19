use runmat_accelerate::simple_provider::register_inprocess_provider;
use runmat_builtins::{
    builtin_functions, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_runtime::call_builtin;
use runmat_value::{Tensor, Value};

#[derive(Debug)]
struct TestBuiltinError {
    message: String,
    retry: runmat_runtime::GpuGatherRetry,
}

impl TestBuiltinError {
    fn terminal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retry: runmat_runtime::GpuGatherRetry::Never,
        }
    }

    fn request_gpu_gather(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retry: runmat_runtime::GpuGatherRetry::Requested,
        }
    }
}

impl std::fmt::Display for TestBuiltinError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TestBuiltinError {}

impl From<TestBuiltinError> for runmat_runtime::RuntimeError {
    fn from(error: TestBuiltinError) -> Self {
        runmat_runtime::build_runtime_error(error.message)
            .with_gpu_gather_retry(error.retry)
            .build()
    }
}

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const OUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output value.",
}];
const DOUBLE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input integer.",
}];
const HOST_TRACE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value.",
}];
const HOST_ADD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left input tensor.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right input tensor.",
    },
];
const DOUBLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = double(x)",
    inputs: &DOUBLE_INPUTS,
    outputs: &OUT_VALUE,
}];
const HOST_TRACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = host_only_trace(value)",
    inputs: &HOST_TRACE_INPUTS,
    outputs: &OUT_VALUE,
}];
const HOST_ADD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = host_only_add_tensors(A, B)",
    inputs: &HOST_ADD_INPUTS,
    outputs: &OUT_VALUE,
}];
const DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DOUBLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const HOST_TRACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOST_TRACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const HOST_ADD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOST_ADD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};

#[runtime_builtin(
    name = "double",
    descriptor(crate::DOUBLE_DESCRIPTOR),
    builtin_path = "tests::double_fn"
)]
fn double_fn(x: i32) -> Result<i32, String> {
    Ok(x * 2)
}

#[runtime_builtin(
    name = "host_only_trace",
    descriptor(crate::HOST_TRACE_DESCRIPTOR),
    builtin_path = "tests::host_only_trace"
)]
fn host_only_trace(value: Value) -> Result<Value, TestBuiltinError> {
    match value {
        Value::Tensor(t) => {
            let sum: f64 = t.materialize_f64().iter().copied().sum();
            Ok(Value::Num(sum))
        }
        other => Err(TestBuiltinError::request_gpu_gather(format!(
            "host_only_trace: unsupported input {other:?}"
        ))),
    }
}

#[runtime_builtin(
    name = "host_only_add_tensors",
    descriptor(crate::HOST_ADD_DESCRIPTOR),
    builtin_path = "tests::host_only_add_tensors"
)]
fn host_only_add_tensors(a: Value, b: Value) -> Result<Value, TestBuiltinError> {
    match (a, b) {
        (Value::Tensor(ta), Value::Tensor(tb)) => {
            if ta.shape != tb.shape {
                return Err(TestBuiltinError::terminal(
                    "host_only_add_tensors: shape mismatch",
                ));
            }
            let data: Vec<f64> = ta
                .materialize_f64()
                .iter()
                .zip(tb.materialize_f64().iter())
                .map(|(x, y)| x + y)
                .collect();
            let tensor = Tensor::new(data, ta.shape.clone())
                .map_err(|error| TestBuiltinError::terminal(error.to_string()))?;
            Ok(Value::Tensor(tensor))
        }
        (lhs, rhs) => Err(TestBuiltinError::request_gpu_gather(format!(
            "host_only_add_tensors: unsupported inputs {lhs:?} and {rhs:?}"
        ))),
    }
}

#[test]
fn call_registered_builtin() {
    let result = call_builtin("double", &[Value::Int(runmat_value::IntValue::I32(4))]).unwrap();
    if let Value::Int(n) = result {
        assert_eq!(n.to_i64(), 8);
    } else {
        panic!();
    }
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"double"));
}

fn mark_automatic(value: Value) -> Value {
    match value {
        Value::GpuTensor(mut handle) => {
            runmat_accelerate_api::mark_handle_automatic(&mut handle);
            Value::GpuTensor(handle)
        }
        other => other,
    }
}

#[test]
fn dispatcher_gathers_gpu_argument_for_host_builtin() {
    register_inprocess_provider();

    let cpu_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let gpu_value =
        mark_automatic(call_builtin("gpuArray", &[Value::Tensor(cpu_tensor.clone())]).unwrap());
    let result = call_builtin("host_only_trace", &[gpu_value]).unwrap();

    match result {
        Value::Num(sum) => assert!((sum - 10.0).abs() < 1e-9),
        other => panic!("expected numeric result, got {other:?}"),
    }
}

#[test]
fn dispatcher_gathers_multiple_gpu_arguments() {
    register_inprocess_provider();

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let ga = mark_automatic(call_builtin("gpuArray", &[Value::Tensor(a.clone())]).unwrap());
    let gb = mark_automatic(call_builtin("gpuArray", &[Value::Tensor(b.clone())]).unwrap());

    let result = call_builtin("host_only_add_tensors", &[ga, gb]).unwrap();

    match result {
        Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![6.0, 8.0, 10.0, 12.0]),
        other => panic!("expected tensor result, got {other:?}"),
    }
}
