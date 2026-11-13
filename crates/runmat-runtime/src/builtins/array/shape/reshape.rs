//! MATLAB-compatible `reshape` builtin with GPU-aware semantics for RunMat.

use crate::aggregate::structure::operations as struct_ops;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::shape::value_numel;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, RuntimeError};
pub use runmat_builtins::RESHAPE_DESCRIPTOR;
use runmat_builtins::*;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value};

#[path = "reshape/admission.rs"]
mod admission;
#[path = "reshape/readback.rs"]
mod readback;

fn reshape_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("reshape").build()
}

fn reshape_size_error(message: impl Into<String>) -> RuntimeError {
    super::error::declared(reshape_error(message), &RESHAPE_ERROR_INVALID_SIZE)
}
fn reshape_dimension_error(error: tensor::DimensionDecodeError) -> RuntimeError {
    let message = format!("reshape: {error}");
    let owner = if matches!(&error, tensor::DimensionDecodeError::Gather(_)) {
        reshape_error(message)
    } else {
        reshape_size_error(message)
    };
    error.attach_cause(owner)
}

#[runtime_builtin(
    name = "reshape",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::shape::reshape"
)]
async fn reshape_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if !rest.is_empty()
        && (matches!(value, Value::Object(_) | Value::HandleObject(_))
            || !shape_transform_input_supported(
                &crate::value_fact::value_fact(&value),
                ArrayShapeTransform::Reshape,
            ))
    {
        return Err(super::error::declared(
            reshape_error(format!(
                "reshape: unsupported input type {:?}; expected numeric, logical, char, string, cell, or gpu array",
                value
            )),
            &RESHAPE_ERROR_UNSUPPORTED_INPUT,
        ));
    }

    crate::output_count::validate_fixed_outputs(
        &RESHAPE_DESCRIPTOR,
        &RESHAPE_ERROR_INVALID_SIZE,
        "reshape",
    )?;
    for control in &rest {
        if !shape_transform_control_supported(
            &crate::value_fact::value_fact(control),
            ArrayShapeTransform::Reshape,
            rest.len() > 1,
        ) {
            return Err(reshape_size_error(
                "reshape: unsupported size control class or geometry",
            ));
        }
    }

    if rest.is_empty() {
        return Err(super::error::declared(
            reshape_error(RESHAPE_ERROR_SIZE_MISSING.message),
            &RESHAPE_ERROR_SIZE_MISSING,
        ));
    }
    let runtime = crate::context::legacy::active();
    if matches!(
        &value,
        Value::Tensor(_) | Value::ComplexTensor(_) | Value::LogicalArray(_)
    ) {
        if let Some(runtime) = runtime.as_ref() {
            if let Some(mut admitted) = admission::ReshapeDimensions::prepare(runtime, rest.view())?
            {
                admitted.parse(rest.view(), runtime).await?;
                let numel = value_numel(&value).await?;
                admitted.finalize(numel)?;
                return admitted.publish(value, runtime);
            }
        }
    }
    let tokens = parse_size_arguments(rest.view()).await?;
    let numel = value_numel(&value).await?;
    let dims = finalize_dimensions(tokens, numel)?;
    reshape_value(value, &dims)
}

fn reshape_value(value: Value, dims: &[usize]) -> crate::BuiltinResult<Value> {
    match value {
        Value::Tensor(tensor) => tensor
            .reshape(dims.to_vec())
            .map(Value::Tensor)
            .map_err(|e| reshape_error(format!("reshape: {e}"))),
        Value::ComplexTensor(ct) => {
            ComplexTensor::from_complex_storage(ct.into_complex_storage(), dims.to_vec())
                .map(Value::ComplexTensor)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::LogicalArray(logical) => {
            let LogicalArray { data, .. } = logical;
            LogicalArray::from_host_buffer(data, dims.to_vec())
                .map(Value::LogicalArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::String(s) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::String(s.into()))
            } else {
                StringArray::from_elements_unbounded(vec![s], dims.to_vec())
                    .map(Value::StringArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::StringArray(strings) => {
            let (data, _, _, _) = strings.into_parts();
            StringArray::from_shared_storage(data, runmat_value::SharedShape::new_unmanaged(dims.to_vec()))
                .map(Value::StringArray)
                .map_err(|e| reshape_error(format!("reshape: {e}")))
        }
        Value::CharArray(chars) => reshape_char_array(chars, dims),
        Value::Cell(cell) => reshape_cell_array(cell, dims),
        Value::Struct(structure) => Ok(Value::Struct(structure)),
        Value::StructArray(array)=>struct_ops::reshape(array,if dims.len()==1{vec![dims[0],1]}else{dims.to_vec()}).map(Value::StructArray).map_err(|error|struct_ops::remap(error,|message|reshape_error(format!("reshape: {message}")))),
        Value::ObjectArray(array) => {
            let context = crate::context::legacy::active();
            crate::execution::object_array::reshape(array, dims, context.as_ref())
                .map(Value::ObjectArray)
        },
        Value::GpuTensor(handle) => reshape_gpu_tensor(handle, dims),
        Value::Num(n) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Num(n))
            } else {
                Tensor::new(vec![n], dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Int(i) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Int(i))
            } else {
                Tensor::new_integer(runmat_value::IntegerStorage::from_scalar(i), dims.to_vec())
                    .map(Value::Tensor)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Bool(b) => {
            if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
                Ok(Value::Bool(b))
            } else {
                let fill = if b { 1u8 } else { 0u8 };
                let total: usize = dims.iter().product();
                LogicalArray::new(vec![fill; total], dims.to_vec())
                    .map(Value::LogicalArray)
                    .map_err(|e| reshape_error(format!("reshape: {e}")))
            }
        }
        Value::Complex(re, im) => reshape_complex_scalar(re, im, dims),
        other => Err(reshape_error(format!(
            "reshape: unsupported input type {:?}; expected numeric, logical, char, string, cell, or gpu array",
            other
        ))),
    }
}

fn reshape_complex_scalar(re: f64, im: f64, dims: &[usize]) -> crate::BuiltinResult<Value> {
    let total: usize = dims.iter().copied().product();
    if total != 1 {
        return Err(reshape_error(format!(
            "reshape: product of dimensions ({total}) must equal numel(A) (1)"
        )));
    }

    if dims.len() <= 2 && dims.iter().all(|&d| d == 1) {
        Ok(Value::Complex(re, im))
    } else {
        ComplexTensor::new(vec![(re, im)], dims.to_vec())
            .map(Value::ComplexTensor)
            .map_err(|e| reshape_error(format!("reshape: {e}")))
    }
}

fn reshape_char_array(ca: CharArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }
    let shape = if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims.to_vec()
    };
    CharArray::from_column_major(ca.to_column_major(), shape)
        .map(Value::CharArray)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_cell_array(ca: CellArray, dims: &[usize]) -> crate::BuiltinResult<Value> {
    if dims.is_empty() {
        return Err(reshape_error(
            "reshape: size vector must contain at least one element",
        ));
    }
    let shape = if dims.len() == 1 {
        vec![dims[0], 1]
    } else {
        dims.to_vec()
    };
    CellArray::from_column_major(ca.to_column_major(), shape)
        .map(Value::Cell)
        .map_err(|e| reshape_error(format!("reshape: {e}")))
}

fn reshape_gpu_tensor(
    handle: runmat_accelerate_api::GpuTensorHandle,
    dims: &[usize],
) -> crate::BuiltinResult<Value> {
    let output = super::gpu_reshape::reshape(handle, dims, "reshape")?;
    gpu_helpers::resident_gpu_value(output)
}

#[derive(Clone, Copy, Debug)]
enum DimToken {
    Known(usize),
    Auto,
}

async fn parse_size_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Vec<DimToken>> {
    if args.len() == 1 {
        let value = &args[0];
        match value {
            Value::Tensor(t) => {
                if tensor::tensor_element_len(t) == 0 {
                    return Err(reshape_size_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async_typed(value)
                    .await
                    .map_err(reshape_dimension_error)?;
                let Some(dims) = dims else {
                    return Err(reshape_size_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::LogicalArray(la) => {
                if la.data.is_empty() {
                    return Err(reshape_size_error(
                        "reshape: size vector must contain at least one element",
                    ));
                }
                let dims = tensor::dims_from_value_async_typed(value)
                    .await
                    .map_err(reshape_dimension_error)?;
                let Some(dims) = dims else {
                    return Err(reshape_size_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::GpuTensor(_) => {
                let dims = tensor::dims_from_value_async_typed(value)
                    .await
                    .map_err(reshape_dimension_error)?;
                let Some(dims) = dims else {
                    return Err(reshape_size_error(
                        "reshape: size vector must be a row or column vector",
                    ));
                };
                Ok(dims.into_iter().map(DimToken::Known).collect())
            }
            Value::Int(_) | Value::Num(_) | Value::Bool(_) => {
                Ok(vec![parse_size_scalar(value).await?])
            }
            other => Err(reshape_size_error(format!(
                "reshape: size vector must be numeric, got {:?}",
                other
            ))),
        }
    } else {
        let mut tokens = Vec::with_capacity(args.len());
        for value in args {
            tokens.push(parse_size_scalar(value).await?);
        }
        Ok(tokens)
    }
}
async fn parse_size_scalar(value: &Value) -> crate::BuiltinResult<DimToken> {
    let gathered;
    let value = if matches!(value, Value::GpuTensor(_)) {
        gathered = crate::gather_if_needed_async(value)
            .await
            .map_err(|source| {
                build_runtime_error(format!("reshape: {source}"))
                    .with_builtin("reshape")
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .with_source(source)
                    .build()
            })?;
        &gathered
    } else {
        value
    };

    match value {
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            if len == 0 {
                return Ok(DimToken::Auto);
            }
            if len != 1 {
                return Err(reshape_size_error(
                    "reshape: size arguments must be scalars",
                ));
            }
        }
        Value::LogicalArray(la) => {
            if la.data.is_empty() {
                return Ok(DimToken::Auto);
            }
            if la.data.len() != 1 {
                return Err(reshape_size_error(
                    "reshape: size arguments must be scalars",
                ));
            }
        }
        _ => {}
    }

    let Some(dim) = tensor::dimension_from_value_async(value, "reshape", true)
        .await
        .map_err(|e| reshape_size_error(format!("reshape: {e}")))?
    else {
        return Err(reshape_size_error(format!(
            "reshape: size arguments must be numeric scalars, got {:?}",
            value
        )));
    };
    Ok(DimToken::Known(dim))
}

fn finalize_dimensions(tokens: Vec<DimToken>, numel: usize) -> crate::BuiltinResult<Vec<usize>> {
    if tokens.is_empty() {
        return Err(reshape_size_error(
            "reshape: size vector must contain at least one element",
        ));
    }

    let mut dims = Vec::with_capacity(tokens.len());
    let mut auto_index: Option<usize> = None;
    let mut duplicate_auto = false;
    for (idx, token) in tokens.iter().enumerate() {
        match token {
            DimToken::Known(value) => dims.push(*value),
            DimToken::Auto => {
                duplicate_auto |= auto_index.replace(idx).is_some();
                dims.push(1);
            }
        }
    }
    finish_dimensions_in_place(&mut dims, auto_index, duplicate_auto, numel)?;
    Ok(dims)
}

fn finish_dimensions_in_place(
    dims: &mut [usize],
    auto_index: Option<usize>,
    duplicate_auto: bool,
    numel: usize,
) -> crate::BuiltinResult<()> {
    if duplicate_auto {
        return Err(reshape_size_error(
            "reshape: can only specify a single [] dimension",
        ));
    }
    // A known zero makes the complete product empty even when an earlier
    // nonzero prefix cannot be represented. All tokens still get validated.
    let mut known_product: usize = if dims
        .iter()
        .enumerate()
        .any(|(index, &dim)| Some(index) != auto_index && dim == 0)
    {
        0
    } else {
        1
    };

    for (idx, &value) in dims.iter().enumerate() {
        if Some(idx) == auto_index {
            continue;
        }
        if value == 0 {
            known_product = 0;
        } else if known_product != 0 {
            known_product = known_product.checked_mul(value).ok_or_else(|| {
                reshape_size_error("reshape: product of dimensions exceeds usize range")
            })?;
        }
    }

    if let Some(auto) = auto_index {
        if known_product == 0 {
            if numel != 0 {
                return Err(reshape_size_error(format!(
                    "reshape: product of dimensions (0) must equal numel(A) ({numel})"
                )));
            }
            dims[auto] = 0;
        } else if !numel.is_multiple_of(known_product) {
            return Err(reshape_size_error(format!(
                "reshape: product of dimensions ({}) must equal numel(A) ({numel})",
                known_product
            )));
        } else {
            dims[auto] = numel / known_product;
        }
    } else if known_product != numel {
        return Err(reshape_size_error(format!(
            "reshape: product of dimensions ({known_product}) must equal numel(A) ({numel})"
        )));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;

    fn reshape_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::reshape_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }
    use crate::builtins::common::test_support;
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, LogicalArray};

    #[test]
    fn reshape_size_scalar_reads_typed_integer_storage_exactly() {
        let dim = Tensor::new_integer(IntegerStorage::U64(vec![3]), vec![1, 1])
            .expect("dimension tensor");

        match block_on(parse_size_scalar(&Value::Tensor(dim))).expect("dimension") {
            DimToken::Known(value) => assert_eq!(value, 3),
            other => panic!("expected known dimension, got {other:?}"),
        }
        #[cfg(target_pointer_width = "64")]
        test_support::with_test_provider(|provider| {
            let value = 9_007_199_254_740_993_u64;
            let handle = provider
                .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                    data: runmat_accelerate_api::HostIntegerDataView::U64(&[value]),
                    shape: &[1, 1],
                })
                .expect("resident dimension");
            let allocation = handle.allocation_identity();
            match block_on(parse_size_scalar(&Value::GpuTensor(handle.clone())))
                .expect("exact gathered dimension")
            {
                DimToken::Known(actual) => assert_eq!(actual, value as usize),
                other => panic!("expected known dimension, got {other:?}"),
            }
            assert_eq!(handle.allocation_identity(), allocation);
            handle
                .allocation_lease()
                .validate_live()
                .expect("borrowed control remains live");
        });
    }

    #[test]
    fn reshape_size_vector_reads_typed_integer_storage_without_mirror() {
        let dims = Tensor::new_integer(IntegerStorage::U64(vec![2, 3]), vec![1, 2])
            .expect("dimension tensor");

        let borrowed = tensor::borrowed_host_shape_dimensions(&Value::Tensor(dims.clone()))
            .expect("borrowed typed dimensions")
            .collect::<Result<Vec<_>, _>>()
            .expect("exact integer dimension values");
        assert_eq!(borrowed, vec![2, 3]);

        let parsed = block_on(parse_size_arguments(
            crate::call::FunctionArgumentView::from_slice(&[Value::Tensor(dims)]),
        ))
        .expect("dimension vector");
        assert!(matches!(
            parsed.as_slice(),
            [DimToken::Known(2), DimToken::Known(3)]
        ));
    }

    #[test]
    fn admitted_reshape_shape_outlives_dimension_workspace_and_receipt() {
        use crate::context::host_allocation::HostAllocationPlan;
        use crate::context::host_effect::ManagedHostEffectLedger;
        use crate::context::{
            ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
            RuntimeServicePorts,
        };
        use std::rc::Rc;

        struct FiniteParallel(ManagedHostEffectLedger);
        impl RuntimeParallelService for FiniteParallel {
            fn supports(&self, _: ParallelCapability) -> bool {
                false
            }

            fn placement_resources(&self) -> RuntimeParallelResources {
                let snapshot = self.0.snapshot();
                RuntimeParallelResources {
                    memory_available_bytes: Some(snapshot.free_bytes()),
                    epoch: snapshot.epoch,
                    ..RuntimeParallelResources::default()
                }
            }

            fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
                Some(self.0.clone())
            }
        }

        let ledger = ManagedHostEffectLedger::new(65_536);
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(FiniteParallel(ledger.clone()))),
                );
        let controls = [Value::Num(3.0), Value::Num(2.0)];
        let mut dimensions = admission::ReshapeDimensions::prepare(
            &runtime,
            crate::call::FunctionArgumentView::from_slice(&controls),
        )
        .expect("shape admission")
        .expect("finite shape receipt");
        block_on(dimensions.parse(
            crate::call::FunctionArgumentView::from_slice(&controls),
            &runtime,
        ))
        .expect("dimension parsing");
        dimensions.finalize(6).expect("matching element count");
        let result = dimensions
            .publish(
                Value::Tensor(tensor_from_slice(&[1.0; 6], &[6, 1])),
                &runtime,
            )
            .expect("admitted reshape");
        let shape_charge = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(runmat_value::SharedShape::requested_bytes_for_rank(2).unwrap()).unwrap(),
            1,
        )
        .unwrap();
        assert_eq!(ledger.snapshot().live_claimed_bytes, shape_charge);
        let Value::Tensor(tensor) = &result else {
            panic!("expected tensor result");
        };
        assert_eq!(tensor.shape(), &[3, 2]);
        drop(result);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    fn tensor_from_slice(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_vector_to_matrix() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(tensor_from_slice(&[3.0, 4.0], &[1, 2]))],
        )
        .expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), vec![3, 4]);
                assert_eq!(
                    out.materialize_f64(),
                    (1..=12).map(|v| v as f64).collect::<Vec<_>>()
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_with_auto_dimension() {
        let data: Vec<f64> = (1..=18).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[18, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let args = vec![Value::from(3.0), Value::Tensor(empty)];
        let result = reshape_builtin(Value::Tensor(tensor), args).expect("reshape");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), vec![3, 6]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_logical_array_preserves_type() {
        let logical = LogicalArray::new(vec![1, 0, 1, 0, 1, 0], vec![6, 1]).expect("logical");
        let result = reshape_builtin(
            Value::LogicalArray(logical),
            vec![Value::from(2.0), Value::from(3.0)],
        )
        .expect("reshape");
        match result {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 3]);
                assert_eq!(out.data, vec![1, 0, 1, 0, 1, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_preserves_all_exact_integer_classes() {
        for storage in [
            IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX]),
            IntegerStorage::U8(vec![0, 1, 7, u8::MAX]),
            IntegerStorage::U16(vec![0, 1, 700, u16::MAX]),
            IntegerStorage::U32(vec![0, 1, 9_007_199, u32::MAX]),
            IntegerStorage::U64(vec![0, 1, 9_007_199_254_740_993, u64::MAX]),
        ] {
            let expected = storage.clone();
            let tensor = Tensor::new_integer(storage, vec![4, 1]).expect("integer tensor");
            let result = reshape_builtin(
                Value::Tensor(tensor),
                vec![Value::from(2.0), Value::from(2.0)],
            )
            .expect("reshape");

            match result {
                Value::Tensor(out) => {
                    assert_eq!(out.shape(), vec![2, 2]);
                    assert_eq!(out.integer_storage(), Some(&expected));
                }
                other => panic!("expected typed integer tensor, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_preserves_typed_complex_integer_storage() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![0, 1, 9_007_199_254_740_993, u64::MAX]),
            IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993, 1, 0]),
        )
        .expect("complex integer storage");
        let tensor =
            ComplexTensor::new_integer(storage.clone(), vec![4, 1]).expect("complex integer");

        let result = reshape_builtin(
            Value::ComplexTensor(tensor),
            vec![Value::from(2.0), Value::from(2.0)],
        )
        .expect("reshape");

        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape(), vec![2, 2]);
                assert_eq!(out.integer_storage().cloned(), Some(storage));
            }
            other => panic!("expected typed complex integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_char_array_single_dimension_becomes_column() {
        let chars =
            CharArray::from_unicode_scalars("abcd".chars().collect(), 1, 4).expect("char array");
        let result =
            reshape_builtin(Value::CharArray(chars), vec![Value::from(4.0)]).expect("reshape");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 4);
                assert_eq!(out.cols, 1);
                let collected: String = runmat_value::decode_utf16_code_units(out.data())
                    .expect("valid UTF-16 fixture text");
                assert_eq!(collected, "abcd");
            }
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_cell_array_two_dimensional() {
        let cell = CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).expect("cell");
        let result = reshape_builtin(Value::Cell(cell), vec![Value::from(2.0), Value::from(1.0)])
            .expect("reshape");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 1);
                let first = out.get(0, 0).expect("first cell");
                let second = out.get(1, 0).expect("second cell");
                assert!(matches!(first, Value::Num(f) if (f - 1.0).abs() < 1e-12));
                assert!(matches!(second, Value::Num(f) if (f - 2.0).abs() < 1e-12));
            }
            other => panic!("expected CellArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_string_scalar_high_rank() {
        let result = reshape_builtin(
            Value::String("runmat".to_string().into()),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1, 1]);
                assert_eq!(sa.data(), vec!["runmat".to_string()]);
            }
            other => panic!("expected StringArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_gpu_preserves_handle_shape() {
        test_support::with_test_provider(|provider| {
            let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
            let tensor = tensor_from_slice(&data, &[3, 4]);
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(2.0), Value::from(6.0)],
            )
            .expect("reshape");
            match result {
                Value::GpuTensor(out) => {
                    assert_eq!(out.shape(), &[2, 6]);
                    assert_eq!(out.allocation_identity(), handle.allocation_identity());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn reshape_complex_gpu_preserves_storage_and_data() {
        test_support::with_test_provider(|provider| {
            let complex = ComplexTensor::new(
                vec![
                    (1.0, -1.0),
                    (2.0, -2.0),
                    (3.0, -3.0),
                    (4.0, -4.0),
                    (5.0, -5.0),
                    (6.0, -6.0),
                ],
                vec![2, 3],
            )
            .unwrap();
            let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
            let result = reshape_builtin(
                Value::GpuTensor(handle.clone()),
                vec![Value::from(3.0), Value::from(2.0)],
            )
            .expect("reshape");
            let Value::GpuTensor(reshaped) = result else {
                panic!("expected gpu tensor");
            };
            assert_eq!(reshaped.allocation_identity(), handle.allocation_identity());
            assert_eq!(reshaped.shape(), vec![3, 2]);
            assert_eq!(
                runmat_accelerate_api::handle_storage(&reshaped),
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            );
            let gathered = block_on(
                crate::builtins::math::fft::common::gather_gpu_complex_tensor(&reshaped, "reshape"),
            )
            .expect("gather complex");
            assert_eq!(gathered.shape(), vec![3, 2]);
            assert_eq!(gathered.materialize_f64(), complex.materialize_f64());
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_updates_provider_shape() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[3, 4]);
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(2.0), Value::from(6.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.shape(), vec![2, 6]);
        let host = test_support::gather(Value::GpuTensor(reshaped.clone())).expect("gather");
        assert_eq!(host.shape(), vec![2, 6]);
        assert_eq!(host.materialize_f64(), tensor.materialize_f64());
        drop(reshaped);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn reshape_wgpu_complex_updates_provider_shape_and_storage() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };
        let complex = ComplexTensor::new(
            vec![
                (1.0, -1.0),
                (2.0, -2.0),
                (3.0, -3.0),
                (4.0, -4.0),
                (5.0, -5.0),
                (6.0, -6.0),
            ],
            vec![2, 3],
        )
        .unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = reshape_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::from(3.0), Value::from(2.0)],
        )
        .expect("reshape");
        let Value::GpuTensor(reshaped) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(reshaped.allocation_identity(), handle.allocation_identity());
        assert_eq!(reshaped.shape(), vec![3, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_storage(&reshaped),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let host = block_on(
            crate::builtins::math::fft::common::gather_gpu_complex_tensor(&reshaped, "reshape"),
        )
        .expect("gather complex reshape");
        assert_eq!(host.shape(), vec![3, 2]);
        assert_eq!(host.materialize_f64(), complex.materialize_f64());
        drop(reshaped);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_mismatched_elements_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(4.0), Value::from(4.0)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("product of dimensions"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_multiple_auto_errors() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[6, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty.clone()), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(err.to_string().contains("single []"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_accepts_zero_sized_dimension() {
        let large = u32::MAX as usize;
        assert_eq!(
            finalize_dimensions(
                vec![
                    DimToken::Known(large),
                    DimToken::Known(large),
                    DimToken::Known(2),
                    DimToken::Known(0)
                ],
                0
            )
            .unwrap(),
            [large, large, 2, 0]
        );
        let tensor = tensor_from_slice(&[], &[0, 1]);
        let result = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(0.0), Value::from(3.0)],
        )
        .expect("reshape zero");
        match result {
            Value::Tensor(out) => assert_eq!(out.shape(), vec![0, 3]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_int_scalar_to_vector() {
        let value = Value::Int(IntValue::I32(5));
        let err = reshape_builtin(value.clone(), vec![Value::from(1.0), Value::from(5.0)])
            .expect_err("should fail because numel mismatch");
        assert!(err.to_string().contains("numel"));
        let ok = reshape_builtin(value, vec![Value::from(1.0), Value::from(1.0)])
            .expect("reshape scalar");
        assert!(matches!(ok, Value::Int(_)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_int_scalar_high_rank_preserves_exact_integer_storage() {
        let result = reshape_builtin(
            Value::Int(IntValue::U64(9_007_199_254_740_993)),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape scalar");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape(), vec![1, 1, 1]);
                assert_eq!(
                    out.integer_storage(),
                    Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
                );
            }
            other => panic!("expected typed integer tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_complex_scalar_high_rank() {
        let result = reshape_builtin(
            Value::Complex(1.0, 2.0),
            vec![Value::from(1.0), Value::from(1.0), Value::from(1.0)],
        )
        .expect("reshape complex");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape(), vec![1, 1, 1]);
                assert_eq!(ct.materialize_f64(), vec![(1.0, 2.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn reshape_auto_dimension_mismatch_reports_product() {
        let data: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let tensor = tensor_from_slice(&data, &[12, 1]);
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let err = reshape_builtin(
            Value::Tensor(tensor),
            vec![Value::from(5.0), Value::Tensor(empty)],
        )
        .expect_err("should fail");
        assert!(
            err.to_string().contains("5"),
            "expected product to appear in error message, got {err}"
        );
    }
}
