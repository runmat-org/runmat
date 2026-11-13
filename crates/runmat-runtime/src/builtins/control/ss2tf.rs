//! State-space matrix conversion through the domain-owned Markov polynomial algorithm.
use crate::builtins::control::model_polynomial::{
    allocation_error, preallocate_polynomial_buffers, selected_input_polynomials,
    SsPolynomialGeometry,
};
use crate::builtins::control::tf_model::control_error;
use crate::context::host_allocation::HostAllocationReceipt;
use crate::{dispatcher, BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{ExactHostTensor2dAllocation, NumericDType, Tensor, Value};

const NAME: &str = "ss2tf";

fn invalid(message: impl Into<String>) -> RuntimeError {
    control_error(NAME, "RunMat:ss2tf:InvalidArgument", message)
}

#[runtime_builtin(
    name = "ss2tf",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::ss2tf"
)]
async fn ss2tf_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    a: Value,
    b: Value,
    c: Value,
    d: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count().unwrap_or(1);
    if requested > 2 || rest.len() > 1 {
        return Err(invalid(
            "ss2tf expects four matrices, an optional input index, and at most two outputs",
        ));
    }
    let inputs = [a, b, c, d];
    if inputs
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::SS2TF_EXPLICIT_GPU_EXTENSION,
            NAME,
        )?;
    }
    let single = inputs.iter().all(is_single_matrix);
    for (name, value) in ["A", "B", "C", "D"].into_iter().zip(&inputs) {
        require_real_floating_matrix(name, value)?;
    }
    let geometry =
        SsPolynomialGeometry::from_values([&inputs[0], &inputs[1], &inputs[2], &inputs[3]], NAME)?;
    let dtype = if single {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let plan = geometry
        .selected_input_plan(dtype, requested)
        .map_err(|error| allocation_error(NAME, runtime.as_ref(), error))?;
    let receipt = HostAllocationReceipt::admit(runtime.as_ref(), plan)
        .map_err(|error| allocation_error(NAME, runtime.as_ref(), error))?;
    let buffers = preallocate_polynomial_buffers(
        receipt.as_ref(),
        runtime.as_ref(),
        geometry.states(),
        NAME,
    )?;
    let input = match rest.first() {
        None => 1,
        Some(value) => {
            require_real_floating_matrix("ni", value)?;
            crate::builtins::common::tensor::dimension_from_value_async(value, "ni", false)
                .await
                .map_err(invalid)?
                .ok_or_else(|| invalid("ni must be a positive integer scalar"))?
        }
    };
    if input > geometry.inputs() {
        return Err(invalid("ni exceeds the number of system inputs"));
    }
    let [a, b, c, d] = inputs;
    let a = host_matrix(a).await?;
    let b = host_matrix(b).await?;
    let c = host_matrix(c).await?;
    let d = host_matrix(d).await?;
    let (numerators, denominator) =
        selected_input_polynomials([&a, &b, &c, &d], input - 1, NAME, runtime.as_ref(), buffers)
            .await?;
    let columns = denominator.len();
    let rows = numerators.len();
    let numerator = if requested > 0 {
        Some(coefficient_tensor_from_elements(
            rows,
            columns,
            single,
            receipt.as_ref().zip(runtime.as_ref()),
            |index| numerators[index % rows][index / rows],
        )?)
    } else {
        None
    };
    let denominator = if requested >= 2 {
        Some(coefficient_tensor_from_elements(
            1,
            columns,
            single,
            receipt.as_ref().zip(runtime.as_ref()),
            |index| denominator[index],
        )?)
    } else {
        None
    };
    match requested {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(numerator.expect("one output requested")),
        2 => Ok(Value::output_list(vec![
            numerator.expect("numerator requested"),
            denominator.expect("denominator requested"),
        ])),
        _ => unreachable!(),
    }
}

async fn host_matrix(value: Value) -> BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        dispatcher::gather_if_needed_async(&value).await
    } else {
        Ok(value)
    }
}

fn is_single_matrix(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

fn require_real_floating_matrix(name: &str, value: &Value) -> BuiltinResult<()> {
    let valid = match value {
        Value::Num(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::Real
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(format!(
            "ss2tf {name} must be a real single- or double-precision matrix"
        )))
    }
}

fn real_coefficient(value: num_complex::Complex64) -> BuiltinResult<f64> {
    if value.im == 0.0 && value.re.is_finite() {
        Ok(value.re)
    } else {
        Err(control_error(
            NAME,
            "RunMat:ss2tf:Internal",
            "state-space conversion produced a non-real or non-finite coefficient",
        ))
    }
}

fn coefficient_tensor_from_elements(
    rows: usize,
    columns: usize,
    single: bool,
    allocation: Option<(&HostAllocationReceipt, &crate::context::RuntimeContext)>,
    element: impl Fn(usize) -> num_complex::Complex64,
) -> BuiltinResult<Value> {
    let len = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("coefficient matrix size overflows"))?;
    for index in 0..len {
        let value = real_coefficient(element(index))?;
        if single && !(value as f32).is_finite() {
            return Err(control_error(
                NAME,
                "RunMat:ss2tf:Internal",
                "single-precision coefficient exceeds its finite range",
            ));
        }
    }
    if let Some((receipt, runtime)) = allocation {
        let dtype = if single {
            NumericDType::F32
        } else {
            NumericDType::F64
        };
        let tensor = receipt
            .with_output_allocation(runtime, |permit| {
                let mut output = ExactHostTensor2dAllocation::new(dtype, rows, columns, permit)?;
                if single {
                    let values = output.as_f32_slice_mut().expect("F32 exact allocation");
                    for (index, slot) in values.iter_mut().enumerate() {
                        *slot = element(index).re as f32;
                    }
                } else {
                    let values = output.as_f64_slice_mut().expect("F64 exact allocation");
                    for (index, slot) in values.iter_mut().enumerate() {
                        *slot = element(index).re;
                    }
                }
                output.into_tensor()
            })
            .map_err(|error| allocation_error(NAME, Some(runtime), error))?;
        return Ok(Value::Tensor(tensor));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("coefficient output exceeds available host memory"))?;
    for index in 0..len {
        data.push(element(index).re);
    }
    coefficient_tensor(data, vec![rows, columns], single)
}

fn coefficient_tensor(data: Vec<f64>, shape: Vec<usize>, single: bool) -> BuiltinResult<Value> {
    let result = if single {
        if data.iter().any(|value| !(*value as f32).is_finite()) {
            return Err(control_error(
                NAME,
                "RunMat:ss2tf:Internal",
                "single-precision coefficient exceeds its finite range",
            ));
        }
        Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
    } else {
        Tensor::new(data, shape)
    };
    result
        .map(Value::Tensor)
        .map_err(|error| control_error(NAME, "RunMat:ss2tf:Internal", error))
}
