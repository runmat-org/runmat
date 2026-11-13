//! MATLAB-compatible `zeros` builtin with GPU-aware semantics.

use runmat_accelerate_api::{GpuTensorHandle, ProviderPrecision};
#[cfg(test)]
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView, IntegerElementType};
use runmat_builtins::BuiltinErrorDescriptor;
use runmat_builtins::{
    ZEROS_ERROR_CLASS_CONFLICT, ZEROS_ERROR_LIKE_DUPLICATE, ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE,
    ZEROS_ERROR_UNRECOGNIZED_OPTION, ZEROS_IMPLICIT_PROTOTYPE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_types::{standard, ClassIdentity, NumericClass};
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericStorage,
    SparseTensor, Value,
};
use std::sync::OnceLock;

use crate::build_runtime_error;
#[cfg(test)]
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionKernelTemplate,
    GpuOpKind, ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    random_args::{
        extract_constructor_dimensions, normalize_constructor_shape,
        validate_constructor_gpu_output,
    },
    shape::normalize_scalar_shape,
    tensor,
};
use runmat_value::NumericDType;

#[path = "zeros/gpu.rs"]
mod gpu;
#[path = "zeros/dimensions.rs"]
mod dimensions;
#[path = "zeros/prototype.rs"]
mod prototype;
pub(in crate::builtins) use dimensions::parse_creation_dimensions;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "zeros",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("zeros"),
        ProviderHook::Custom("zeros_like"),
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Allocates device zeros when providers expose dedicated hooks; otherwise falls back to host upload.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin("zeros").build()
}

fn zeros_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    zeros_error_with_message(error.message, error)
}

fn zeros_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    zeros_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn zeros_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("zeros");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::zeros")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "zeros",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: crate::builtins::common::spec::FusionExprBuilder::new(|ctx, writer| {
            let literal = match ctx.scalar_ty {
                            ScalarType::F32 => "0.0", ScalarType::F64 => "f64(0.0)",
                            ScalarType::I32 => "0", ScalarType::Bool => "false",
                        };
                        writer.push(literal)
        }),
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion planner materialises zeros as literal constants; providers may substitute inexpensive fill kernels.",
};

#[runtime_builtin(
    name = "zeros",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::array::creation::zeros"
)]
async fn zeros_dispatch(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    let selected = runtime.clone();
    crate::context::RuntimeScopeFactory::new(runtime, move || async move {
        use crate::builtin::CanonicalBuiltinOutput;
        if let Some(index) = prototype::like_prototype(&rest)? {
            prototype::invoke(selected.as_ref(), rest, index).await
        } else {
            zeros_builtin(rest).await?.into_rooted_sequence()
        }
    })
    .construct()
    .await
}

async fn zeros_builtin(rest: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let parsed = ParsedZeros::parse(rest).await?;
    build_output(parsed).await
}

struct ParsedZeros {
    shape: Vec<usize>,
    template: OutputTemplate,
}

#[derive(Clone)]
enum OutputTemplate {
    Double,
    /// Single-precision output request.
    Single,
    Logical,
    Integer(NumericDType),
    /// GPU residency is independent of the requested underlying element class.
    GpuArray(GpuElementClass),
    Like(Value),
}

#[derive(Clone, Copy)]
enum GpuElementClass {
    Numeric(NumericDType),
    Logical,
}

#[derive(Clone, Copy)]
enum ParsedClassKeyword {
    GpuArray,
    Logical,
    Numeric(NumericClass),
}

fn parse_class_keyword(keyword: &str) -> Option<ParsedClassKeyword> {
    let identity = if keyword.eq_ignore_ascii_case(standard::GPU_ARRAY.display_name()) {
        standard::GPU_ARRAY.owned()
    } else {
        ClassIdentity::new(keyword).ok()?
    };
    if identity.is(standard::GPU_ARRAY) {
        Some(ParsedClassKeyword::GpuArray)
    } else if identity.is(standard::LOGICAL) {
        Some(ParsedClassKeyword::Logical)
    } else {
        NumericClass::from_class_identity(&identity).map(ParsedClassKeyword::Numeric)
    }
}

impl ParsedZeros {
    async fn parse(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Self> {
        let mut dims: Vec<usize> = Vec::new();
        let mut saw_dims_arg = false;
        let mut shape_source: Option<Vec<usize>> = None;
        let mut like_proto: Option<Value> = None;
        let mut class_override: Option<OutputTemplate> = None;
        let mut gpu_output = false;
        let mut implicit_proto: Option<Value> = None;

        let mut idx = 0;
        while idx < args.len() {
            let arg = args[idx].clone();
            if let Some(keyword) =
                crate::builtins::common::arg_tokens::scalar_keyword_of(&arg, "zeros")?
            {
                if keyword == "like" {
                    if like_proto.is_some() {
                        return Err(zeros_error(&ZEROS_ERROR_LIKE_DUPLICATE));
                    }
                    if class_override.is_some() || gpu_output {
                        return Err(zeros_error(&ZEROS_ERROR_CLASS_CONFLICT));
                    }
                    let Some(proto) = args.get(idx + 1).cloned() else {
                        return Err(zeros_error(&ZEROS_ERROR_LIKE_EXPECTED_PROTOTYPE));
                    };
                    like_proto = Some(proto.clone());
                    if shape_source.is_none() && !saw_dims_arg {
                        shape_source = Some(shape_from_value(&proto)?);
                    }
                    idx += 2;
                    continue;
                }
                match parse_class_keyword(&keyword) {
                    Some(ParsedClassKeyword::Logical) => {
                        if like_proto.is_some() || class_override.is_some() || gpu_output {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "logical class override",
                            ));
                        }
                        class_override = Some(OutputTemplate::Logical);
                        idx += 1;
                        continue;
                    }
                    Some(ParsedClassKeyword::Numeric(class)) => {
                        if like_proto.is_some() || class_override.is_some() || gpu_output {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                format!("{keyword} class override"),
                            ));
                        }
                        class_override = Some(match NumericDType::from(class) {
                            NumericDType::F64 => OutputTemplate::Double,
                            NumericDType::F32 => OutputTemplate::Single,
                            dtype => OutputTemplate::Integer(dtype),
                        });
                        idx += 1;
                        continue;
                    }
                    Some(ParsedClassKeyword::GpuArray) => {
                        if like_proto.is_some() || gpu_output {
                            return Err(zeros_error_with_detail(
                                &ZEROS_ERROR_CLASS_CONFLICT,
                                "gpuArray class override",
                            ));
                        }
                        gpu_output = true;
                        idx += 1;
                        continue;
                    }
                    None => {
                        return Err(zeros_error_with_detail(
                            &ZEROS_ERROR_UNRECOGNIZED_OPTION,
                            format!("'{keyword}'"),
                        ));
                    }
                }
            }

            if let Some(parsed_dims) = dimensions::argument(&arg).await? {
                saw_dims_arg = true;
                if dims.is_empty() {
                    dims = parsed_dims;
                } else {
                    dims.extend(parsed_dims);
                }
                idx += 1;
                continue;
            }

            tracing::debug!(
                arg_type = value_tag(&arg),
                "zeros: argument did not parse as dimensions"
            );

            if shape_source.is_none() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &ZEROS_IMPLICIT_PROTOTYPE_EXTENSION,
                    "zeros",
                )?;
                shape_source = Some(shape_from_value(&arg)?);
            }
            if implicit_proto.is_none() {
                implicit_proto = Some(arg.clone());
            }
            idx += 1;
        }

        let shape = if saw_dims_arg {
            if dims.is_empty() {
                vec![0, 0]
            } else {
                normalize_constructor_shape(dims)
            }
        } else if let Some(shape) = shape_source {
            tracing::warn!(
                shape = ?shape,
                "zeros: falling back to shape source; no dimension arguments parsed"
            );
            shape
        } else {
            vec![1, 1]
        };

        tracing::trace!(
            "zeros: resolved output shape {:?} (saw_dims_arg={})",
            shape,
            saw_dims_arg
        );

        let template = if let Some(proto) = like_proto {
            OutputTemplate::Like(proto)
        } else if let Some(spec) = class_override {
            spec
        } else if let Some(proto) = implicit_proto {
            OutputTemplate::Like(proto)
        } else {
            OutputTemplate::Double
        };
        let template = if gpu_output {
            let class = match template {
                OutputTemplate::Double => GpuElementClass::Numeric(NumericDType::F64),
                OutputTemplate::Single => GpuElementClass::Numeric(NumericDType::F32),
                OutputTemplate::Logical => GpuElementClass::Logical,
                OutputTemplate::Integer(dtype) => GpuElementClass::Numeric(dtype),
                OutputTemplate::Like(_) | OutputTemplate::GpuArray(_) => {
                    return Err(zeros_error(&ZEROS_ERROR_CLASS_CONFLICT));
                }
            };
            OutputTemplate::GpuArray(class)
        } else {
            template
        };

        Ok(Self { shape, template })
    }
}

async fn build_output(parsed: ParsedZeros) -> crate::BuiltinResult<Value> {
    match parsed.template {
        OutputTemplate::Double => zeros_double(&parsed.shape),
        OutputTemplate::Single => zeros_single(&parsed.shape),
        OutputTemplate::Logical => zeros_logical(&parsed.shape),
        OutputTemplate::Integer(dtype) => {
            let prototype = NumericStorage::zeros(dtype, 0)
                .into_integer_storage()
                .expect("typed integer class control");
            zeros_integer_like(&prototype, &parsed.shape)
        }
        OutputTemplate::GpuArray(element_class) => gpu::build(&parsed.shape, element_class).await,
        OutputTemplate::Like(proto) => zeros_like(&proto, &parsed.shape).await,
    }
}

fn value_tag(value: &Value) -> &'static str {
    match value {
        Value::Num(_) => "Num",
        Value::Int(_) => "Int",
        Value::Bool(_) => "Bool",
        Value::Tensor(_) => "Tensor",
        Value::SparseTensor(_) => "SparseTensor",
        Value::LogicalArray(_) => "LogicalArray",
        Value::GpuTensor(_) => "GpuTensor",
        Value::Complex(_, _) => "Complex",
        Value::ComplexTensor(_) => "ComplexTensor",
        Value::String(_) => "String",
        Value::StringArray(_) => "StringArray",
        Value::CharArray(_) => "CharArray",
        Value::Symbolic(_) => "Symbolic",
        Value::SymbolicArray(_) => "SymbolicArray",
        Value::Cell(_) => "Cell",
        Value::Struct(_) | Value::StructArray(_) => "Struct",
        Value::ObjectArray(_) | Value::Object(_) => "Object",
        Value::HandleObject(_) => "HandleObject",
        Value::Listener(_) => "Listener",
        Value::ScopedFunctionHandle(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle(_) => "FunctionHandle",
        Value::Closure(_) => "Closure",
        Value::ClassRef(_) => "ClassRef",
        Value::MException(_) => "MException",
        Value::OutputList(_) => "OutputList",
        Value::Future(_) => "Future",
        Value::Task(_) => "Task",
        Value::Pool(_) => "Pool",
        Value::Job(_) => "Job",
        Value::Distributed(_) => "Distributed",
        Value::Composite(_) => "Composite",
        Value::Foreign(_) => "Foreign",
    }
}

fn zeros_double(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F64)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros(shape)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_single(shape: &[usize]) -> crate::BuiltinResult<Value> {
    if !force_host_allocation(shape) {
        if let Some(value) = zeros_gpu_alloc(shape, NumericDType::F32)? {
            return Ok(value);
        }
    }
    let tensor = tensor::zeros_with_dtype(shape, NumericDType::F32)?;
    Ok(tensor::tensor_into_value(tensor))
}

fn force_host_allocation(shape: &[usize]) -> bool {
    tensor::element_count(shape) <= 1
}

fn zeros_logical(shape: &[usize]) -> crate::BuiltinResult<Value> {
    Ok(Value::LogicalArray(LogicalArray::zeros(shape.to_vec())))
}

#[async_recursion::async_recursion(?Send)]
async fn zeros_like(proto: &Value, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match proto {
        Value::LogicalArray(_) | Value::Bool(_) => zeros_logical(shape),
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            zeros_complex_integer_like(
                tensor
                    .integer_storage()
                    .as_ref()
                    .expect("guarded typed complex integer storage"),
                shape,
            )
        }
        Value::ComplexTensor(_) | Value::Complex(_, _) => {
            let tensor = ComplexTensor::zeros(shape.to_vec());
            Ok(Value::ComplexTensor(tensor))
        }
        Value::GpuTensor(handle) => zeros_like_gpu(handle, shape).await,
        Value::SparseTensor(sparse) => zeros_sparse_like(sparse, shape),
        Value::Tensor(t) => match t.numeric_dtype() {
            NumericDType::F32 => zeros_single(shape),
            NumericDType::F64 => zeros_double(shape),
            dtype => tensor::zeros_with_dtype(shape, dtype)
                .map(Value::Tensor)
                .map_err(|e| builtin_error(format!("zeros: {e}"))),
        },
        Value::Int(value) => zeros_integer_like(&IntegerStorage::from_scalar(value.clone()), shape),
        Value::Num(_) => zeros_double(shape),
        Value::CharArray(_) | Value::Cell(_) => zeros_double(shape),
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) => Err(builtin_error(
            "zeros: object creation requires a trailing 'like', prototype pair",
        )),
        _ => zeros_double(shape),
    }
}

fn zeros_integer_like(storage: &IntegerStorage, shape: &[usize]) -> crate::BuiltinResult<Value> {
    let tensor = runmat_value::Tensor::new_integer(
        storage.zeros_like(tensor::element_count(shape)),
        shape.to_vec(),
    )
    .map_err(|e| builtin_error(format!("zeros: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn zeros_complex_integer_like(
    storage: &IntegerComplexStorage,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let len = tensor::element_count(shape);
    let storage = IntegerComplexStorage::new(
        storage.real().zeros_like(len),
        storage.imag().zeros_like(len),
    )
    .map_err(|e| builtin_error(format!("zeros: {e}")))?;
    ComplexTensor::new_integer(storage, shape.to_vec())
        .map(Value::ComplexTensor)
        .map_err(|e| builtin_error(format!("zeros: {e}")))
}

fn zeros_sparse_like(proto: &SparseTensor, shape: &[usize]) -> crate::BuiltinResult<Value> {
    match shape {
        [rows, cols] => Ok(Value::SparseTensor(match proto.numeric_dtype() {
            Some(NumericDType::F32) => SparseTensor::zeros_f32(*rows, *cols),
            Some(NumericDType::F64) => SparseTensor::zeros(*rows, *cols),
            None => SparseTensor::zeros_logical(*rows, *cols),
            Some(_) => SparseTensor::zeros_with_integer_storage(
                *rows,
                *cols,
                proto
                    .integer_storage()
                    .expect("integer sparse dtype has integer storage"),
            ),
        })),
        other => Err(builtin_error(format!(
            "zeros: sparse 'like' output must be 2-D, got {} dimensions",
            other.len()
        ))),
    }
}

async fn zeros_like_gpu(handle: &GpuTensorHandle, shape: &[usize]) -> crate::BuiltinResult<Value> {
    super::resident_constant::like(handle, shape, super::resident_constant::Constant::Zero)
}

fn zeros_gpu_alloc(shape: &[usize], dtype: NumericDType) -> crate::BuiltinResult<Option<Value>> {
    let Some(provider) = runmat_accelerate_api::provider() else {
        log_zeros_fallback(shape, dtype, "no-provider");
        return Ok(None);
    };
    let precision = match dtype {
        NumericDType::F32 => ProviderPrecision::F32,
        NumericDType::F64 => ProviderPrecision::F64,
        NumericDType::I8
        | NumericDType::I16
        | NumericDType::I32
        | NumericDType::I64
        | NumericDType::U8
        | NumericDType::U16
        | NumericDType::U32
        | NumericDType::U64 => {
            log_zeros_fallback(shape, dtype, "integer-dtype");
            return Ok(None);
        }
    };
    if provider.precision() != precision {
        log_zeros_fallback(shape, dtype, "precision-mismatch");
        return Ok(None);
    }
    match provider.zeros(shape) {
        Ok(handle) => {
            let handle = validate_constructor_gpu_output(
                "zeros",
                provider,
                handle,
                shape,
                runmat_accelerate_api::GpuTensorStorage::Real,
                Some(precision),
                None,
                false,
            )?;
            let handle = handle
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .map_err(|cause| {
                    crate::build_runtime_error(format!("zeros: output provenance failed: {cause}"))
                        .with_builtin("zeros")
                        .with_source(cause)
                        .build()
                })?;
            Ok(Some(Value::GpuTensor(handle)))
        }
        Err(err)
            if crate::builtins::common::gpu_helpers::provider_hook_allows_host_fallback(&err) =>
        {
            log::warn!("zeros: provider zeros failed ({err}); falling back to host tensor path");
            log_zeros_fallback(shape, dtype, "provider-error");
            Ok(None)
        }
        Err(cause) => Err(crate::build_runtime_error(format!(
            "zeros: provider allocation failed: {cause}"
        ))
        .with_builtin("zeros")
        .with_source(runmat_accelerate_api::ProviderExecutionError::from(cause))
        .build()),
    }
}

fn zeros_fallback_debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        matches!(
            std::env::var("RUNMAT_DEBUG_ZEROS_FALLBACK"),
            Ok(value)
                if value == "1"
                    || value.eq_ignore_ascii_case("true")
                    || value.eq_ignore_ascii_case("yes")
        )
    })
}

fn log_zeros_fallback(shape: &[usize], dtype: NumericDType, reason: &str) {
    if !zeros_fallback_debug_enabled() {
        return;
    }
    let elems = tensor::element_count(shape);
    tracing::debug!(
        dtype = ?dtype,
        elems,
        shape = ?shape,
        reason,
        "[zeros_debug] fallback"
    );
}

fn shape_from_value(value: &Value) -> Result<Vec<usize>, String> {
    match value {
        Value::Tensor(t) => Ok(t.shape().to_vec()),
        Value::SparseTensor(t) => Ok(t.shape()),
        Value::ComplexTensor(t) => Ok(t.shape().to_vec()),
        Value::LogicalArray(l) => Ok(l.shape.to_vec()),
        Value::GpuTensor(h) => Ok(normalize_scalar_shape(h.shape())),
        Value::CharArray(ca) => Ok(ca.shape().to_vec()),
        Value::Cell(cell) => Ok(cell.shape().to_vec()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        other => Err(format!("zeros: unsupported prototype {other:?}")),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, SparseTensor, Tensor};
    use std::sync::Mutex;

    struct InvalidAutomaticZerosProvider {
        owner: TestAllocationOwner,
        result: Mutex<Option<RetirementProbe>>,
    }

    impl runmat_accelerate_api::AccelProvider for InvalidAutomaticZerosProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<GpuTensorHandle> {
            unreachable!("no upload for invalid constructor result")
        }

        fn download<'a>(
            &'a self,
            _handle: &'a GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            unreachable!("no download for invalid constructor result")
        }

        fn device_info(&self) -> String {
            "invalid owned zeros result".into()
        }

        fn device_id(&self) -> u32 {
            self.owner.device_id()
        }

        fn zeros(&self, _shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
            let allocation = self.owner.allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )?;
            *self.result.lock().unwrap() = Some(allocation.probe);
            Ok(allocation.handle)
        }
    }

    fn clear_accel_provider_state() -> test_support::AccelTestGuard {
        test_support::accel_test_lock()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_default_scalar() {
        let _guard = clear_accel_provider_state();
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("zeros");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn automatic_zeros_rejects_and_frees_invalid_provider_output() {
        let _guard = clear_accel_provider_state();
        let provider = Box::leak(Box::new(InvalidAutomaticZerosProvider {
            owner: TestAllocationOwner::new(),
            result: Mutex::new(None),
        }));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

        let error = zeros_double(&[2, 2]).expect_err("invalid provider output must be terminal");
        assert!(error.message().contains("invalid constructor result"));
        assert_eq!(
            provider
                .result
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
    }

    #[test]
    fn explicit_gpu_zeros_retires_invalid_owned_provider_result() {
        let _guard = clear_accel_provider_state();
        let provider = Box::leak(Box::new(InvalidAutomaticZerosProvider {
            owner: TestAllocationOwner::new(),
            result: Mutex::new(None),
        }));
        // SAFETY: the leaked fixture has a unique immutable device id.
        unsafe { runmat_accelerate_api::register_device_provider(provider) };
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let error = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("gpuArray"),
            ]),
        ))
        .expect_err("malformed explicit provider result");
        assert!(error.message().contains("invalid constructor result"));
        assert_eq!(
            provider
                .result
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_square_from_single_dimension() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(3.0)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[3, 3]);
        assert!(tensor.materialize_f64().iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_rectangular_from_dims() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(2.0), Value::Num(4.0)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[2, 4]);
        assert_eq!(tensor.len(), 8);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_from_size_vector() {
        let _guard = clear_accel_provider_state();
        let size_vec = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let args = vec![Value::Tensor(size_vec)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_logical_output() {
        let _guard = clear_accel_provider_state();
        let args = vec![Value::Num(2.0), Value::Num(2.0), Value::from("logical")];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 2]);
                assert!(logical.data.iter().all(|&x| x == 0));
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_tensor_infers_shape() {
        let _guard = clear_accel_provider_state();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::Tensor(tensor)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert!(tensor.materialize_f64().iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_complex_scalar() {
        let _guard = clear_accel_provider_state();
        let args = vec![
            Value::Num(3.0),
            Value::from("like"),
            Value::Complex(1.0, 2.0),
        ];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape(), &[3, 3]);
                assert!(t
                    .materialize_f64()
                    .iter()
                    .all(|&(re, im)| re == 0.0 && im == 0.0));
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn zeros_like_typed_complex_uint64_keeps_exact_integer_storage() {
        let prototype = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
                IntegerStorage::U64(vec![u64::MAX, 1]),
            )
            .expect("typed complex prototype"),
            vec![1, 2],
        )
        .expect("typed complex tensor");
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::ComplexTensor(prototype),
            ]),
        ))
        .expect("zeros like");
        let Value::ComplexTensor(output) = result else {
            panic!("expected typed complex output");
        };
        assert_eq!(
            output.integer_storage().cloned(),
            Some(
                IntegerComplexStorage::new(
                    IntegerStorage::U64(vec![0; 4]),
                    IntegerStorage::U64(vec![0; 4]),
                )
                .expect("typed complex zeros"),
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_uses_shape_argument_when_combined_with_like() {
        let _guard = clear_accel_provider_state();
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let shape_source = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let proto = Tensor::new(vec![7.0, 8.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Tensor(shape_source.clone()),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.materialize_f64().iter().all(|&x| x == 0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let args = vec![Value::from("like"), Value::Tensor(proto)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        let tensor = test_support::gather(result).expect("gather tensor");
        assert_eq!(tensor.shape(), &[2, 2]);
        assert!(tensor.materialize_f64().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zeros_like_preserves_every_exact_integer_class() {
        let _guard = clear_accel_provider_state();
        let storages = vec![
            IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
            IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
            IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
            IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
            IntegerStorage::U8(vec![0, u8::MAX]),
            IntegerStorage::U16(vec![0, u16::MAX]),
            IntegerStorage::U32(vec![0, u32::MAX]),
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        ];

        for storage in storages {
            let prototype = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("prototype");
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("like"),
                    Value::Tensor(prototype),
                ]),
            ))
            .expect("zeros like");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor");
            };
            assert_eq!(output.shape(), &[1, 2]);
            assert_eq!(output.integer_storage(), Some(&storage.zeros_like(2)));
        }

        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(2.0),
                Value::from("like"),
                Value::Int(IntValue::I64(i64::MAX)),
            ]),
        ))
        .expect("integer scalar prototype");
        let Value::Tensor(output) = result else {
            panic!("expected int64 tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I64(vec![0; 4]))
        );
    }

    #[test]
    fn zeros_class_strings_create_exact_integer_storage() {
        let _guard = clear_accel_provider_state();
        let cases = [
            ("int8", IntegerStorage::I8(vec![0; 6])),
            ("int16", IntegerStorage::I16(vec![0; 6])),
            ("int32", IntegerStorage::I32(vec![0; 6])),
            ("int64", IntegerStorage::I64(vec![0; 6])),
            ("uint8", IntegerStorage::U8(vec![0; 6])),
            ("uint16", IntegerStorage::U16(vec![0; 6])),
            ("uint32", IntegerStorage::U32(vec![0; 6])),
            ("uint64", IntegerStorage::U64(vec![0; 6])),
        ];

        for (class_name, expected) in cases {
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                    Value::from(class_name),
                ]),
            ))
            .expect("zeros integer class");
            let Value::Tensor(output) = result else {
                panic!("expected integer tensor for {class_name}");
            };
            assert_eq!(output.shape(), &[2, 3]);
            assert_eq!(output.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn class_keyword_decoder_preserves_case_insensitive_public_forms() {
        let _guard = clear_accel_provider_state();
        for (spelling, expected) in [
            ("DOUBLE", NumericDType::F64),
            ("SiNgLe", NumericDType::F32),
            ("UINT64", NumericDType::U64),
        ] {
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(2.0),
                    Value::from(spelling),
                ]),
            ))
            .expect("canonical class option");
            let value = test_support::gather(result).expect("host value");
            assert_eq!(value.numeric_dtype(), expected);
        }
        assert!(matches!(
            parse_class_keyword("gpuarray"),
            Some(ParsedClassKeyword::GpuArray)
        ));
        assert!(matches!(
            parse_class_keyword("logical"),
            Some(ParsedClassKeyword::Logical)
        ));
        assert!(parse_class_keyword("notAClass").is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_sparse_preserves_sparse_storage_with_explicit_shape() {
        let _guard = clear_accel_provider_state();
        let proto = SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![10.0, 20.0]).unwrap();
        let args = vec![
            Value::Num(3.0),
            Value::Num(4.0),
            Value::from("like"),
            Value::SparseTensor(proto),
        ];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        match result {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.rows, 3);
                assert_eq!(sparse.cols, 4);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0]);
                assert!(sparse.row_indices.is_empty());
                assert!(sparse.materialize_f64().is_empty());
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_sparse_without_explicit_shape_uses_prototype_shape() {
        let _guard = clear_accel_provider_state();
        let proto = SparseTensor::zeros(2, 5);
        let args = vec![Value::from("like"), Value::SparseTensor(proto)];
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("zeros");
        match result {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.shape(), vec![2, 5]);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0, 0]);
                assert!(sparse.row_indices.is_empty());
                assert!(sparse.materialize_f64().is_empty());
            }
            other => panic!("expected sparse tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_like_single_sparse_preserves_native_single_storage() {
        let _guard = clear_accel_provider_state();
        let proto =
            SparseTensor::new_f32(2, 2, vec![0, 1, 1], vec![0], vec![1.25]).expect("single sparse");
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(3.0),
                Value::Num(4.0),
                Value::from("like"),
                Value::SparseTensor(proto),
            ]),
        ))
        .expect("zeros");
        let Value::SparseTensor(sparse) = result else {
            panic!("expected sparse tensor");
        };
        assert_eq!(sparse.shape(), vec![3, 4]);
        assert_eq!(sparse.numeric_dtype(), Some(NumericDType::F32));
        assert!(sparse.as_f32_slice().is_some_and(<[f32]>::is_empty));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_empty_input_returns_empty_matrix() {
        let _guard = clear_accel_provider_state();
        let empty = Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap();
        let result = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(empty)]),
        ))
        .expect("zeros");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), &[0, 0]);
                assert!(t.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_conflicting_like_and_logical_is_error() {
        let _guard = clear_accel_provider_state();
        let proto = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let args = vec![
            Value::Num(2.0),
            Value::from("logical"),
            Value::from("like"),
            Value::Tensor(proto),
        ];
        assert!(block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_like_alloc() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(handle),
            ];
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("zeros");
            match result {
                Value::GpuTensor(gpu) => {
                    assert_eq!(gpu.shape(), vec![2, 2]);
                    let gathered = test_support::gather(Value::GpuTensor(gpu)).expect("gather");
                    assert!(gathered.materialize_f64().iter().all(|&x| x == 0.0));
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn zeros_gpu_integer_like_preserves_exact_class_resident() {
        test_support::with_test_provider(|provider| {
            let prototype_values = [u64::MAX, 9_007_199_254_740_993];
            let prototype = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&prototype_values),
                    shape: &[1, 2],
                })
                .expect("upload uint64 prototype");
            let args = vec![
                Value::Num(2.0),
                Value::Num(2.0),
                Value::from("like"),
                Value::GpuTensor(prototype),
            ];
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect("zeros integer gpu like");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpuArray");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(&handle),
                Some(IntegerElementType::U64)
            );
            assert_eq!(handle.shape(), vec![2, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(
                gathered.integer_storage(),
                Some(&IntegerStorage::U64(vec![0; 4]))
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn explicit_gpu_class_is_not_erased_by_residency_selector() {
        test_support::with_test_provider(|_| {
            for (class, expected_precision, expected_integer, logical) in [
                ("double", Some(ProviderPrecision::F64), None, false),
                ("single", Some(ProviderPrecision::F32), None, false),
                ("uint64", None, Some(IntegerElementType::U64), false),
                ("logical", None, None, true),
            ] {
                let result = block_on(zeros_builtin(
                    crate::builtins::common::test_support::builtin_arguments(vec![
                        Value::Num(2.0),
                        Value::Num(3.0),
                        Value::from(class),
                        Value::from("gpuArray"),
                    ]),
                ))
                .expect("typed GPU zeros");
                let Value::GpuTensor(handle) = result else {
                    panic!("explicit GPU constructor must stay resident");
                };
                assert_eq!(handle.shape(), vec![2, 3]);
                if !logical {
                    assert_eq!(
                        runmat_accelerate_api::handle_precision(&handle),
                        expected_precision
                    );
                }
                assert_eq!(
                    runmat_accelerate_api::handle_integer_type(&handle),
                    expected_integer
                );
                assert_eq!(runmat_accelerate_api::handle_is_logical(&handle), logical);
            }
            let default = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("gpuArray"),
                ]),
            ))
            .expect("default GPU zeros");
            let Value::GpuTensor(default) = default else {
                panic!("default GPU zeros must stay resident");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&default),
                Some(ProviderPrecision::F64)
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn conflicting_gpu_class_controls_reject_before_provider_work() {
        let _guard = clear_accel_provider_state();
        for args in [
            vec![
                Value::Num(2.0),
                Value::from("single"),
                Value::from("double"),
                Value::from("gpuArray"),
            ],
            vec![
                Value::Num(2.0),
                Value::from("gpuArray"),
                Value::from("single"),
            ],
            vec![
                Value::Num(2.0),
                Value::from("gpuArray"),
                Value::from("gpuArray"),
            ],
        ] {
            let error = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect_err("conflicting class controls");
            assert!(error.to_string().contains("class"));
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn static_and_feval_gpu_zeros_use_one_class_selector() {
        test_support::with_test_provider(|_| {
            let args = [Value::Num(2.0), Value::Num(3.0), Value::from("single")];
            let direct = block_on(crate::call_builtin_async_with_outputs(
                "gpuArray.zeros",
                &args,
                1,
            ))
            .expect("dotted static zeros");
            let via_feval = block_on(crate::call_feval_async_with_outputs(
                Value::from("gpuArray.zeros"),
                &args,
                1,
            ))
            .expect("feval static zeros");
            for result in [direct, via_feval] {
                let Value::GpuTensor(handle) = result.value() else {
                    panic!("static GPU zeros must remain resident");
                };
                assert_eq!(handle.shape(), vec![2, 3]);
                assert_eq!(
                    runmat_accelerate_api::handle_precision(handle),
                    Some(ProviderPrecision::F32)
                );
            }
            let primitive = block_on(crate::call_builtin_async_with_outputs(
                "uint8.zeros",
                &[Value::Num(2.0)],
                1,
            ))
            .expect("primitive static zeros");
            assert!(
                matches!(primitive.value(), Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::U8)
            );
            let primitive_feval = block_on(crate::call_feval_async_with_outputs(
                Value::from("uint8.zeros"),
                &[Value::Num(2.0)],
                1,
            ))
            .expect("feval primitive static zeros");
            assert!(matches!(primitive_feval.value(), Value::Tensor(tensor)
                if tensor.numeric_dtype() == NumericDType::U8 && tensor.shape() == [2, 2]));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_gpu_class_never_falls_back_to_preferred_single() {
        test_support::with_f32_test_provider(|_| {
            let result = block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::from("gpuArray"),
                ]),
            ));
            assert!(
                result.is_err(),
                "F32-only provider must not publish single for default double request"
            );
        });
    }

    #[test]
    fn zeros_column_size_vector_is_independently_gated() {
        let size = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![2, 1]).expect("column size"),
        );
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(zeros_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![size.clone()]),
        ))
        .expect_err("strict gate");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ZerosColumnSizeVectorExtension")
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let output = test_support::gather(
            block_on(zeros_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![size]),
            ))
            .expect("extension"),
        )
        .expect("gather output");
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn zeros_wgpu_single_allocates_gpu_without_like() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        assert_eq!(
            runmat_accelerate_api::AccelProvider::precision(provider),
            ProviderPrecision::F32
        );
        let value = zeros_single(&[2, 2]).expect("zeros single");
        match value {
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather to host");
                assert_eq!(gathered.shape(), &[2, 2]);
                assert!(gathered.materialize_f64().iter().all(|&x| x == 0.0));
            }
            other => panic!("expected gpu tensor, got {other:?}"),
        }
    }
}
