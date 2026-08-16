//! MATLAB-compatible `pol2cart` builtin for RunMat.

use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::common::{broadcast, gpu_helpers};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::shape_rules::element_count_if_known;

const NAME: &str = "pol2cart";

const OUTPUT_X: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian x coordinate.",
};
const OUTPUT_Y: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian y coordinate.",
};
const OUTPUT_Z: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cartesian z coordinate.",
};
const OUTPUT_XY: [BuiltinParamDescriptor; 2] = [OUTPUT_X, OUTPUT_Y];
const OUTPUT_XYZ: [BuiltinParamDescriptor; 3] = [OUTPUT_X, OUTPUT_Y, OUTPUT_Z];

const INPUT_THETA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "theta",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polar or cylindrical angular coordinate in radians.",
};
const INPUT_RHO: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "rho",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Polar or cylindrical radial coordinate.",
};
const INPUT_Z: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cylindrical elevation coordinate.",
};
const INPUTS_POLAR: [BuiltinParamDescriptor; 2] = [INPUT_THETA, INPUT_RHO];
const INPUTS_CYLINDRICAL: [BuiltinParamDescriptor; 3] = [INPUT_THETA, INPUT_RHO, INPUT_Z];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[x, y] = pol2cart(theta, rho)",
        inputs: &INPUTS_POLAR,
        outputs: &OUTPUT_XY,
    },
    BuiltinSignatureDescriptor {
        label: "[x, y, z] = pol2cart(theta, rho, z)",
        inputs: &INPUTS_CYLINDRICAL,
        outputs: &OUTPUT_XYZ,
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.INVALID_INPUT",
    identifier: Some("RunMat:pol2cart:InvalidInput"),
    when: "An input is not a supported real floating-point scalar or array.",
    message: "pol2cart: invalid input",
};
const ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:pol2cart:ComplexUnsupported"),
    when: "At least one input is complex.",
    message: "pol2cart: complex inputs are not supported",
};
const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.SIZE_MISMATCH",
    identifier: Some("RunMat:pol2cart:SizeMismatch"),
    when: "Input arrays do not have compatible sizes for implicit expansion.",
    message: "pol2cart: size mismatch",
};
const ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.OUTPUT_COUNT",
    identifier: Some("RunMat:pol2cart:OutputCount"),
    when: "More outputs are requested than the selected syntax supports.",
    message: "pol2cart: too many output arguments",
};
const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POL2CART.INTERNAL",
    identifier: Some("RunMat:pol2cart:Internal"),
    when: "Internal gather/conversion/allocation flow failed.",
    message: "pol2cart: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    ERROR_INVALID_INPUT,
    ERROR_COMPLEX_UNSUPPORTED,
    ERROR_SIZE_MISMATCH,
    ERROR_OUTPUT_COUNT,
    ERROR_INTERNAL,
];

pub const POL2CART_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const POL2CART_INTEGER_THETA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pol2cart-integer-theta",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pol2cart accepts typed-integer angular coordinates as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pol2cartIntegerThetaExtension"),
};
const POL2CART_INTEGER_RHO_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pol2cart-integer-rho",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pol2cart accepts typed-integer radial coordinates as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pol2cartIntegerRhoExtension"),
};
const POL2CART_INTEGER_Z_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pol2cart-integer-z",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pol2cart accepts typed-integer elevation coordinates as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Pol2cartIntegerZExtension"),
};
pub const POL2CART_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    POL2CART_INTEGER_THETA_EXTENSION,
    POL2CART_INTEGER_RHO_EXTENSION,
    POL2CART_INTEGER_Z_EXTENSION,
];

const POL2CART_INTEGER_THETA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "theta",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double angular coordinates; RunMat admits typed integers only after exact binary64 conversion is proved.",
    }];
const POL2CART_INTEGER_RHO_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "rho",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double radial coordinates; RunMat admits typed integers only at its checked floating transform boundary.",
    }];
const POL2CART_INTEGER_Z_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "z",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents real single and double elevation coordinates; RunMat admits typed integers only when exact binary64 conversion is possible.",
    }];
pub const POL2CART_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y] = pol2cart(integer_theta,rho) or [x,y,z] = pol2cart(integer_theta,rho,z)",
        inputs: &POL2CART_INTEGER_THETA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "The angular role is independently gated before provider dispatch and crosses once into the trigonometric domain.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y] = pol2cart(theta,integer_rho) or [x,y,z] = pol2cart(theta,integer_rho,z)",
        inputs: &POL2CART_INTEGER_RHO_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "The radial role is independently gated before provider dispatch and floating multiplication.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[x,y,z] = pol2cart(theta,rho,integer_z)",
        inputs: &POL2CART_INTEGER_Z_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Elevation is independently gated and broadcast only after its exact floating boundary is proved.",
    },
];

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::trigonometry::pol2cart")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[
        ProviderHook::Unary { name: "unary_cos" },
        ProviderHook::Unary { name: "unary_sin" },
        ProviderHook::Binary {
            name: "elem_mul",
            commutative: true,
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat keeps real gpuArray coordinate transforms resident by composing provider sin/cos and broadcasted multiply hooks. Providers without those hooks fall back to the host implementation.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::trigonometry::pol2cart"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "pol2cart returns multiple outputs and is treated as a terminal coordinate transform.",
};

#[runtime_builtin(
    name = "pol2cart",
    category = "math/trigonometry",
    summary = "Transform polar or cylindrical coordinates to Cartesian coordinates.",
    keywords = "pol2cart,polar,cylindrical,cartesian,coordinate transform,gpu",
    sink = true,
    type_resolver(pol2cart_type),
    descriptor(crate::builtins::math::trigonometry::pol2cart::POL2CART_DESCRIPTOR),
    extensions(crate::builtins::math::trigonometry::pol2cart::POL2CART_EXTENSIONS),
    integer_capabilities(
        crate::builtins::math::trigonometry::pol2cart::POL2CART_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::trigonometry::pol2cart"
)]
async fn pol2cart_builtin(theta: Value, rho: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::builtins::common::validation::reject_typed_complex_integer(&theta, NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&rho, NAME)?;
    for value in &rest {
        crate::builtins::common::validation::reject_typed_complex_integer(value, NAME)?;
    }
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &theta,
        &POL2CART_INTEGER_THETA_EXTENSION,
        NAME,
        "theta",
    )
    .await?;
    crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
        &rho,
        &POL2CART_INTEGER_RHO_EXTENSION,
        NAME,
        "rho",
    )
    .await?;
    if let Some(z) = rest.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            z,
            &POL2CART_INTEGER_Z_EXTENSION,
            NAME,
            "z",
        )
        .await?;
    }
    let requested_outputs = crate::output_count::current_output_count();
    let mut inputs = Vec::with_capacity(2 + rest.len());
    inputs.push(theta);
    inputs.push(rho);
    inputs.extend(rest);

    let wants_z_output = matches!(requested_outputs, Some(count) if count >= 3);
    if inputs
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        if let Some(eval) = try_pol2cart_gpu(&inputs, wants_z_output).await? {
            return match requested_outputs {
                Some(0) => Ok(Value::OutputList(Vec::new())),
                Some(count) => eval.output_list(count),
                None => eval.x_value(),
            };
        }
    }

    let mut inputs: Vec<Value> =
        futures::future::try_join_all(inputs.into_iter().map(gather_if_gpu)).await?;

    let eval = match inputs.len() {
        2 => {
            let rho = inputs.pop().expect("rho");
            let theta = inputs.pop().expect("theta");
            Pol2CartEval::polar(theta, rho)?
        }
        3 => {
            let z = inputs.pop().expect("z");
            let rho = inputs.pop().expect("rho");
            let theta = inputs.pop().expect("theta");
            Pol2CartEval::cylindrical(theta, rho, z, wants_z_output)?
        }
        _ => {
            return Err(pol2cart_error(
                &ERROR_INVALID_INPUT,
                "expected two or three inputs",
            ))
        }
    };

    match requested_outputs {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(count) => eval.output_list(count),
        None => eval.x_value(),
    }
}

async fn try_pol2cart_gpu(
    inputs: &[Value],
    include_z_output: bool,
) -> BuiltinResult<Option<Pol2CartGpuEval>> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(pol2cart_error(
            &ERROR_INVALID_INPUT,
            "expected two or three inputs",
        ));
    }

    let Some(anchor) = inputs.iter().find_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }) else {
        return Ok(None);
    };

    let Some(provider) = runmat_accelerate_api::provider_for_handle(anchor) else {
        return Ok(None);
    };

    let theta = prepare_gpu_input(provider, &inputs[0])?;
    let rho = prepare_gpu_input(provider, &inputs[1])?;
    let z = if inputs.len() == 3 {
        Some(prepare_gpu_input(provider, &inputs[2])?)
    } else {
        None
    };

    let theta_rho = matlab_broadcast_shape(&theta.handle.shape, &rho.handle.shape)?;
    let final_shape = match &z {
        Some(z_input) => matlab_broadcast_shape(&theta_rho, &z_input.handle.shape)?,
        None => theta_rho,
    };

    let result = try_pol2cart_gpu_ops(
        provider,
        &theta.handle,
        &rho.handle,
        z.as_ref(),
        &final_shape,
    )
    .await;

    theta.free_if_temporary(provider);
    rho.free_if_temporary(provider);
    if let Some(z_input) = z {
        z_input.free_if_temporary(provider);
    }

    match result {
        Ok(eval) => {
            if include_z_output && eval.z.is_none() {
                Ok(None)
            } else {
                Ok(Some(eval))
            }
        }
        Err(_) => Ok(None),
    }
}

async fn try_pol2cart_gpu_ops(
    provider: &'static dyn AccelProvider,
    theta: &GpuTensorHandle,
    rho: &GpuTensorHandle,
    z: Option<&GpuInput>,
    final_shape: &[usize],
) -> anyhow::Result<Pol2CartGpuEval> {
    let cos_theta = provider.unary_cos(theta).await?;
    let sin_theta = match provider.unary_sin(theta).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&cos_theta);
            return Err(err);
        }
    };
    let x = match provider.elem_mul(rho, &cos_theta).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&cos_theta);
            let _ = provider.free(&sin_theta);
            return Err(err);
        }
    };
    let y = match provider.elem_mul(rho, &sin_theta).await {
        Ok(handle) => handle,
        Err(err) => {
            let _ = provider.free(&cos_theta);
            let _ = provider.free(&sin_theta);
            let _ = provider.free(&x);
            return Err(err);
        }
    };
    let _ = provider.free(&cos_theta);
    let _ = provider.free(&sin_theta);

    let z_output = match z {
        Some(z_input) => {
            let ones = match provider.fill(final_shape, 1.0) {
                Ok(handle) => handle,
                Err(err) => {
                    let _ = provider.free(&x);
                    let _ = provider.free(&y);
                    return Err(err);
                }
            };
            let out = match provider.elem_mul(&z_input.handle, &ones).await {
                Ok(handle) => handle,
                Err(err) => {
                    let _ = provider.free(&ones);
                    let _ = provider.free(&x);
                    let _ = provider.free(&y);
                    return Err(err);
                }
            };
            let _ = provider.free(&ones);
            Some(out)
        }
        None => None,
    };

    Ok(Pol2CartGpuEval {
        provider,
        x,
        y,
        z: z_output,
        max_outputs: if z.is_some() { 3 } else { 2 },
    })
}

#[derive(Debug)]
struct GpuInput {
    handle: GpuTensorHandle,
    temporary: bool,
}

impl GpuInput {
    fn free_if_temporary(self, provider: &'static dyn AccelProvider) {
        if self.temporary {
            let _ = provider.free(&self.handle);
        }
    }
}

fn prepare_gpu_input(
    provider: &'static dyn AccelProvider,
    value: &Value,
) -> BuiltinResult<GpuInput> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
            {
                return Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"));
            }
            Ok(GpuInput {
                handle: handle.clone(),
                temporary: false,
            })
        }
        Value::Num(value) => {
            let data = [*value];
            upload_gpu_input(provider, &data, &[1, 1])
        }
        Value::Tensor(tensor) => {
            let data = tensor::tensor_values_f64_cow(tensor);
            upload_gpu_input(provider, &data, &tensor.shape)
        }
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"))
        }
        other => Err(pol2cart_error(
            &ERROR_INVALID_INPUT,
            format!("expected real single or double input, got {other:?}"),
        )),
    }
}

fn upload_gpu_input(
    provider: &'static dyn AccelProvider,
    data: &[f64],
    shape: &[usize],
) -> BuiltinResult<GpuInput> {
    let view = HostTensorView { data, shape };
    let handle = provider
        .upload(&view)
        .map_err(|err| pol2cart_error(&ERROR_INTERNAL, format!("gpu upload failed: {err}")))?;
    Ok(GpuInput {
        handle,
        temporary: true,
    })
}

fn pol2cart_type(args: &[Type], _context: &runmat_builtins::ResolveContext) -> Type {
    let mut shape: Option<Vec<Option<usize>>> = None;
    let mut saw_array = false;
    for arg in args.iter().take(3) {
        match arg {
            Type::Tensor {
                shape: Some(arg_shape),
            } => {
                if element_count_if_known(arg_shape) == Some(1) {
                    continue;
                }
                saw_array = true;
                shape = Some(match shape.take() {
                    Some(existing) => matlab_broadcast_type_shape(&existing, arg_shape),
                    None => arg_shape.clone(),
                });
            }
            Type::Tensor { shape: None } => return Type::tensor(),
            Type::Num => {}
            Type::Unknown => return Type::Unknown,
            _ => return Type::Unknown,
        }
    }
    if saw_array {
        Type::Tensor { shape }
    } else if args.len() >= 2 {
        Type::Num
    } else {
        Type::Unknown
    }
}

async fn gather_if_gpu(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => gather_gpu(handle).await,
        other => Ok(other),
    }
}

async fn gather_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
        .await
        .map_err(|err| pol2cart_error(&ERROR_INTERNAL, err.message()))
}

#[derive(Debug)]
struct Pol2CartEval {
    x: Tensor,
    y: Tensor,
    z: Option<Tensor>,
    max_outputs: usize,
}

struct Pol2CartGpuEval {
    provider: &'static dyn AccelProvider,
    x: GpuTensorHandle,
    y: GpuTensorHandle,
    z: Option<GpuTensorHandle>,
    max_outputs: usize,
}

impl Pol2CartGpuEval {
    fn x_value(self) -> BuiltinResult<Value> {
        let Self {
            provider,
            x,
            y,
            z,
            max_outputs: _,
        } = self;
        let _ = provider.free(&y);
        if let Some(z) = &z {
            let _ = provider.free(z);
        }
        Ok(gpu_helpers::resident_gpu_value(x))
    }

    fn output_list(self, count: usize) -> BuiltinResult<Value> {
        let Self {
            provider,
            x,
            y,
            z,
            max_outputs,
        } = self;
        if count > max_outputs {
            let _ = provider.free(&x);
            let _ = provider.free(&y);
            if let Some(z) = &z {
                let _ = provider.free(z);
            }
            return Err(pol2cart_error(
                &ERROR_OUTPUT_COUNT,
                format!(
                    "requested {count} outputs but this syntax supports at most {}",
                    max_outputs
                ),
            ));
        }
        let mut outputs = Vec::with_capacity(count);
        if count >= 1 {
            outputs.push(gpu_helpers::resident_gpu_value(x));
        } else {
            let _ = provider.free(&x);
        }
        if count >= 2 {
            outputs.push(gpu_helpers::resident_gpu_value(y));
        } else {
            let _ = provider.free(&y);
        }
        if count >= 3 {
            outputs.push(gpu_helpers::resident_gpu_value(
                z.expect("z output available"),
            ));
        } else if let Some(z) = &z {
            let _ = provider.free(z);
        }
        Ok(Value::OutputList(outputs))
    }
}

impl Pol2CartEval {
    fn polar(theta: Value, rho: Value) -> BuiltinResult<Self> {
        let theta = value_into_real_tensor(theta)?;
        let rho = value_into_real_tensor(rho)?;
        let (x, y, _) = compute_pol2cart(&theta, &rho, None, false)?;
        Ok(Self {
            x,
            y,
            z: None,
            max_outputs: 2,
        })
    }

    fn cylindrical(
        theta: Value,
        rho: Value,
        z: Value,
        include_z_output: bool,
    ) -> BuiltinResult<Self> {
        let theta = value_into_real_tensor(theta)?;
        let rho = value_into_real_tensor(rho)?;
        let z = value_into_real_tensor(z)?;
        let (x, y, z) = compute_pol2cart(&theta, &rho, Some(&z), include_z_output)?;
        Ok(Self {
            x,
            y,
            z,
            max_outputs: 3,
        })
    }

    fn max_outputs(&self) -> usize {
        self.max_outputs
    }

    fn x_value(&self) -> BuiltinResult<Value> {
        Ok(tensor::tensor_into_value(self.x.clone()))
    }

    fn output_list(&self, count: usize) -> BuiltinResult<Value> {
        if count > self.max_outputs() {
            return Err(pol2cart_error(
                &ERROR_OUTPUT_COUNT,
                format!(
                    "requested {count} outputs but this syntax supports at most {}",
                    self.max_outputs()
                ),
            ));
        }
        let mut outputs = Vec::with_capacity(count);
        if count >= 1 {
            outputs.push(tensor::tensor_into_value(self.x.clone()));
        }
        if count >= 2 {
            outputs.push(tensor::tensor_into_value(self.y.clone()));
        }
        if count >= 3 {
            outputs.push(tensor::tensor_into_value(
                self.z.clone().expect("z output available"),
            ));
        }
        Ok(Value::OutputList(outputs))
    }
}

fn compute_pol2cart(
    theta: &Tensor,
    rho: &Tensor,
    z: Option<&Tensor>,
    include_z_output: bool,
) -> BuiltinResult<(Tensor, Tensor, Option<Tensor>)> {
    let theta_rho = matlab_broadcast_shape(&theta.shape, &rho.shape)?;
    let final_shape = match z {
        Some(z_tensor) => matlab_broadcast_shape(&theta_rho, &z_tensor.shape)?,
        None => theta_rho,
    };
    let len = final_shape.iter().copied().product::<usize>();

    let theta_plan = MatlabBroadcastIndexPlan::new(&theta.shape, &final_shape)?;
    let rho_plan = MatlabBroadcastIndexPlan::new(&rho.shape, &final_shape)?;
    let z_plan = match z {
        Some(z_tensor) => Some(MatlabBroadcastIndexPlan::new(
            &z_tensor.shape,
            &final_shape,
        )?),
        None => None,
    };
    let theta_data = tensor::tensor_values_f64_cow(theta);
    let rho_data = tensor::tensor_values_f64_cow(rho);
    let z_data = z.map(tensor::tensor_values_f64_cow);

    let mut x = vec![0.0; len];
    let mut y = vec![0.0; len];
    for out_idx in 0..len {
        let theta_idx = theta_plan.index(out_idx);
        let rho_idx = rho_plan.index(out_idx);
        let angle = theta_data[theta_idx];
        let radius = rho_data[rho_idx];
        x[out_idx] = radius * angle.cos();
        y[out_idx] = radius * angle.sin();
    }

    let z_output = match (z, z_plan, include_z_output) {
        (_, _, false) => None,
        (Some(_), Some(plan), true) => {
            let mut values = vec![0.0; len];
            let z_data = z_data.as_ref().expect("z values");
            for out_idx in 0..len {
                let z_idx = plan.index(out_idx);
                values[out_idx] = z_data[z_idx];
            }
            Some(
                Tensor::new(values, final_shape.clone())
                    .map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
            )
        }
        _ => None,
    };

    Ok((
        Tensor::new(x, final_shape.clone()).map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
        Tensor::new(y, final_shape).map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))?,
        z_output,
    ))
}

fn matlab_broadcast_shape(left: &[usize], right: &[usize]) -> BuiltinResult<Vec<usize>> {
    let rank = left.len().max(right.len());
    let left = broadcast::align_shape(left, rank);
    let right = broadcast::align_shape(right, rank);
    let mut shape = Vec::with_capacity(rank);
    for dim in 0..rank {
        let a = left[dim];
        let b = right[dim];
        if a == b {
            shape.push(a);
        } else if a == 1 {
            shape.push(b);
        } else if b == 1 {
            shape.push(a);
        } else {
            return Err(pol2cart_error(
                &ERROR_SIZE_MISMATCH,
                format!(
                    "non-singleton dimension mismatch (dimension {}: {} vs {})",
                    dim + 1,
                    a,
                    b
                ),
            ));
        }
    }
    Ok(shape)
}

fn matlab_broadcast_type_shape(
    left: &[Option<usize>],
    right: &[Option<usize>],
) -> Vec<Option<usize>> {
    runmat_builtins::shape_rules::broadcast_shapes(left, right)
}

#[derive(Debug)]
struct MatlabBroadcastIndexPlan {
    out_shape: Vec<usize>,
    in_shape: Vec<usize>,
    strides: Vec<usize>,
}

impl MatlabBroadcastIndexPlan {
    fn new(input_shape: &[usize], output_shape: &[usize]) -> BuiltinResult<Self> {
        let expected = matlab_broadcast_shape(input_shape, output_shape)?;
        if expected != output_shape {
            return Err(pol2cart_error(
                &ERROR_SIZE_MISMATCH,
                "input cannot expand to final output shape",
            ));
        }
        let mut in_shape = input_shape.to_vec();
        in_shape.resize(output_shape.len(), 1);
        let strides = column_major_strides(&in_shape);
        Ok(Self {
            out_shape: output_shape.to_vec(),
            in_shape,
            strides,
        })
    }

    fn index(&self, mut linear: usize) -> usize {
        let mut offset = 0usize;
        for dim in 0..self.out_shape.len() {
            let out_extent = self.out_shape[dim];
            let coord = if out_extent == 0 {
                0
            } else {
                let coord = linear % out_extent;
                linear /= out_extent;
                coord
            };
            if self.in_shape[dim] != 1 && out_extent != 0 {
                offset += coord * self.strides[dim];
            }
        }
        offset
    }
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride.saturating_mul(extent.max(1));
    }
    strides
}

fn value_into_real_tensor(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Num(value) => {
            Tensor::new(vec![value], vec![1, 1]).map_err(|err| pol2cart_error(&ERROR_INTERNAL, err))
        }
        Value::Tensor(tensor) => Ok(tensor),
        Value::Complex(_, _) | Value::ComplexTensor(_) => {
            Err(pol2cart_error(&ERROR_COMPLEX_UNSUPPORTED, "complex input"))
        }
        other => Err(pol2cart_error(
            &ERROR_INVALID_INPUT,
            format!("expected real single or double input, got {other:?}"),
        )),
    }
}

fn pol2cart_error(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {}", error.message, detail)).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntegerStorage, ResolveContext, Type};

    use crate::builtins::common::test_support;

    const EPS: f64 = 1.0e-12;

    fn call(theta: Value, rho: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::pol2cart_builtin(theta, rho, rest))
    }

    fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).expect("tensor")
    }

    fn output_list(value: Value) -> Vec<Value> {
        match value {
            Value::OutputList(values) => values,
            other => panic!("expected output list, got {other:?}"),
        }
    }

    fn data(value: &Value) -> Vec<f64> {
        match value {
            Value::Num(n) => vec![*n],
            Value::Tensor(tensor) => tensor.materialize_f64().clone(),
            Value::GpuTensor(handle) => test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather gpu output")
                .materialize_f64(),
            other => panic!("expected numeric output, got {other:?}"),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= EPS, "index {idx}: expected {e}, got {a}");
        }
    }

    #[test]
    fn descriptor_covers_polar_and_cylindrical_forms() {
        let labels: Vec<&str> = POL2CART_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"[x, y] = pol2cart(theta, rho)"));
        assert!(labels.contains(&"[x, y, z] = pol2cart(theta, rho, z)"));
        assert_eq!(
            POL2CART_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
    }

    #[test]
    fn type_resolver_broadcasts_primary_output_shape() {
        let out = pol2cart_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(1)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn type_resolver_includes_cylindrical_z_shape() {
        let out = pol2cart_type(
            &[
                Type::Num,
                Type::Tensor {
                    shape: Some(vec![Some(2), Some(1)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(3)])
            }
        );
    }

    #[test]
    fn type_resolver_uses_missing_trailing_dimensions() {
        let out = pol2cart_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(4)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(3), Some(4), Some(2)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(4), Some(2)])
            }
        );
    }

    #[test]
    fn scalar_no_output_context_returns_x() {
        let result = call(
            Value::Num(std::f64::consts::FRAC_PI_2),
            Value::Num(2.0),
            Vec::new(),
        )
        .expect("pol2cart");
        let Value::Num(x) = result else {
            panic!("expected scalar x");
        };
        assert!(x.abs() <= EPS);
    }

    #[test]
    fn polar_two_outputs_for_row_vectors() {
        let theta = tensor(
            vec![
                0.0,
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
            ],
            vec![1, 4],
        );
        let rho = tensor(vec![5.0, 5.0, 10.0, 10.0], vec![1, 4]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs =
            output_list(call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).unwrap());
        assert_close(&data(&outputs[0]), &[5.0, 3.5355339059327378, 0.0, -10.0]);
        assert_close(&data(&outputs[1]), &[0.0, 3.5355339059327373, 10.0, 0.0]);
    }

    #[test]
    fn cylindrical_expands_all_three_inputs() {
        let theta = tensor(vec![0.0, std::f64::consts::FRAC_PI_2], vec![2, 1]);
        let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let _guard = crate::output_count::push_output_count(Some(3));
        let outputs = output_list(
            call(
                Value::Tensor(theta),
                Value::Tensor(rho),
                vec![Value::Num(7.0)],
            )
            .expect("pol2cart"),
        );
        assert_close(&data(&outputs[0]), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        assert_close(&data(&outputs[1]), &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
        assert_close(&data(&outputs[2]), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn cylindrical_two_outputs_omits_z() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(
            call(Value::Num(0.0), Value::Num(3.0), vec![Value::Num(9.0)]).expect("pol2cart"),
        );
        assert_eq!(outputs.len(), 2);
        assert_close(&data(&outputs[0]), &[3.0]);
        assert_close(&data(&outputs[1]), &[0.0]);
    }

    #[test]
    fn cylindrical_broadcasts_missing_trailing_dimension() {
        let theta = tensor(vec![0.0; 12], vec![3, 4]);
        let rho = tensor((1..=24).map(|value| value as f64).collect(), vec![3, 4, 2]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs = output_list(
            call(
                Value::Tensor(theta),
                Value::Tensor(rho),
                vec![Value::Num(5.0)],
            )
            .expect("trailing singleton expansion"),
        );
        match &outputs[0] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4, 2]);
                assert_close(
                    &out.materialize_f64(),
                    &(1..=24).map(|value| value as f64).collect::<Vec<_>>(),
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 4, 2]);
                assert_close(&out.materialize_f64(), &[0.0; 24]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn rejects_trailing_aligned_non_matlab_shapes() {
        let theta = tensor(vec![0.0; 12], vec![3, 4]);
        let rho = tensor(vec![1.0; 12], vec![1, 3, 4]);
        let err = call(Value::Tensor(theta), Value::Tensor(rho), Vec::new())
            .expect_err("shape should not be MATLAB-compatible");
        assert_eq!(err.identifier(), ERROR_SIZE_MISMATCH.identifier);
    }

    #[test]
    fn polar_rejects_three_outputs() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = call(Value::Num(0.0), Value::Num(1.0), Vec::new()).expect_err("too many outputs");
        assert_eq!(err.identifier(), ERROR_OUTPUT_COUNT.identifier);
    }

    #[test]
    fn rejects_complex_inputs() {
        let err = call(Value::Complex(1.0, 1.0), Value::Num(1.0), Vec::new())
            .expect_err("complex should error");
        assert_eq!(err.identifier(), ERROR_COMPLEX_UNSUPPORTED.identifier);
    }

    #[test]
    fn rejects_non_floating_inputs() {
        let err =
            call(Value::Bool(true), Value::Num(1.0), Vec::new()).expect_err("bool should error");
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn rejects_incompatible_shapes() {
        let theta = tensor(vec![0.0, 1.0], vec![2, 1]);
        let rho = tensor(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let err =
            call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).expect_err("size mismatch");
        assert_eq!(err.identifier(), ERROR_SIZE_MISMATCH.identifier);
    }

    #[test]
    fn empty_inputs_preserve_broadcast_shape() {
        let theta = tensor(Vec::new(), vec![0, 3]);
        let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let _guard = crate::output_count::push_output_count(Some(2));
        let outputs =
            output_list(call(Value::Tensor(theta), Value::Tensor(rho), Vec::new()).unwrap());
        match &outputs[0] {
            Value::Tensor(out) => assert_eq!(out.shape, vec![0, 3]),
            other => panic!("expected empty tensor, got {other:?}"),
        }
        match &outputs[1] {
            Value::Tensor(out) => assert_eq!(out.shape, vec![0, 3]),
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[test]
    fn typed_integer_host_inputs_read_exact_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let theta =
            Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).expect("theta");
        let rho = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("rho");
        let z = Tensor::new_integer(IntegerStorage::U16(vec![7, 8]), vec![1, 2]).expect("z");

        let _guard = crate::output_count::push_output_count(Some(3));
        let outputs = output_list(
            call(
                Value::Tensor(theta),
                Value::Tensor(rho),
                vec![Value::Tensor(z)],
            )
            .expect("pol2cart"),
        );

        assert_close(&data(&outputs[0]), &[2.0, 3.0]);
        assert_close(&data(&outputs[1]), &[0.0, 0.0]);
        assert_close(&data(&outputs[2]), &[7.0, 8.0]);
    }

    #[test]
    fn gpu_inputs_keep_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let theta_view = HostTensorView {
                data: &[0.0, std::f64::consts::FRAC_PI_2],
                shape: &[1, 2],
            };
            let rho_view = HostTensorView {
                data: &[2.0, 3.0],
                shape: &[1, 2],
            };
            let theta = provider.upload(&theta_view).expect("upload theta");
            let rho = provider.upload(&rho_view).expect("upload rho");
            let _guard = crate::output_count::push_output_count(Some(2));
            let outputs = output_list(
                call(Value::GpuTensor(theta), Value::GpuTensor(rho), Vec::new()).expect("pol2cart"),
            );
            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
            assert_close(&data(&outputs[0]), &[2.0, 0.0]);
            assert_close(&data(&outputs[1]), &[0.0, 3.0]);
        });
    }

    #[test]
    fn mixed_gpu_typed_integer_inputs_upload_exact_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let theta_view = HostTensorView {
                data: &[0.0, 0.0],
                shape: &[1, 2],
            };
            let theta = provider.upload(&theta_view).expect("upload theta");

            let rho =
                Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).expect("rho");
            let z = Tensor::new_integer(IntegerStorage::U16(vec![7, 8]), vec![1, 2]).expect("z");

            let _guard = crate::output_count::push_output_count(Some(3));
            let outputs = output_list(
                call(
                    Value::GpuTensor(theta),
                    Value::Tensor(rho),
                    vec![Value::Tensor(z)],
                )
                .expect("pol2cart"),
            );

            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
            assert!(matches!(outputs[2], Value::GpuTensor(_)));
            assert_close(&data(&outputs[0]), &[2.0, 3.0]);
            assert_close(&data(&outputs[1]), &[0.0, 0.0]);
            assert_close(&data(&outputs[2]), &[7.0, 8.0]);
        });
    }

    #[test]
    fn mixed_gpu_typed_integer_theta_upload_reads_exact_storage() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let rho_view = HostTensorView {
                data: &[2.0, 3.0],
                shape: &[1, 2],
            };
            let rho = provider.upload(&rho_view).expect("upload rho");

            let theta =
                Tensor::new_integer(IntegerStorage::I16(vec![0, 0]), vec![1, 2]).expect("theta");

            let _guard = crate::output_count::push_output_count(Some(2));
            let outputs = output_list(
                call(Value::Tensor(theta), Value::GpuTensor(rho), Vec::new()).expect("pol2cart"),
            );

            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
            assert_close(&data(&outputs[0]), &[2.0, 3.0]);
            assert_close(&data(&outputs[1]), &[0.0, 0.0]);
        });
    }

    #[test]
    fn gpu_mixed_cylindrical_broadcasts_z_output() {
        test_support::with_test_provider(|provider| {
            let theta_view = HostTensorView {
                data: &[0.0, std::f64::consts::FRAC_PI_2],
                shape: &[2, 1],
            };
            let theta = provider.upload(&theta_view).expect("upload theta");
            let rho = tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
            let _guard = crate::output_count::push_output_count(Some(3));
            let outputs = output_list(
                call(
                    Value::GpuTensor(theta),
                    Value::Tensor(rho),
                    vec![Value::Num(7.0)],
                )
                .expect("pol2cart"),
            );
            assert!(matches!(outputs[0], Value::GpuTensor(_)));
            assert!(matches!(outputs[1], Value::GpuTensor(_)));
            assert!(matches!(outputs[2], Value::GpuTensor(_)));
            assert_close(&data(&outputs[0]), &[1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
            assert_close(&data(&outputs[1]), &[0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
            assert_close(&data(&outputs[2]), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
        });
    }
}
