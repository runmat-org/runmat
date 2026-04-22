use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::type_resolvers::numeric_unary_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::elementwise::nextpow2")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nextpow2",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Unary { name: "unary_nextpow2" }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may execute nextpow2 on-device via unary_nextpow2; otherwise the runtime gathers and applies MATLAB-style scalar semantics on the host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::elementwise::nextpow2"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nextpow2",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let input = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            Ok(format!(
                "select(ceil(log2(abs({input}))), 0.0, abs({input}) == 0.0)"
            ))
        },
    }),
    reduction: None,
    emits_nan: true,
    notes: "Fusion emits ceil(log2(abs(x))) with zero mapped to zero.",
};

const BUILTIN_NAME: &str = "nextpow2";

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "nextpow2",
    category = "math/elementwise",
    summary = "Return the exponent p such that 2^p is the next power of two greater than or equal to abs(n).",
    keywords = "nextpow2,power of two,fft,zero padding,gpu",
    accel = "unary",
    type_resolver(numeric_unary_type),
    builtin_path = "crate::builtins::math::elementwise::nextpow2"
)]
async fn nextpow2_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => nextpow2_gpu(handle).await,
        other => nextpow2_host(other),
    }
}

async fn nextpow2_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider_for_handle(&handle) {
        if let Ok(out) = provider.unary_nextpow2(&handle).await {
            return Ok(Value::GpuTensor(out));
        }
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    Ok(tensor::tensor_into_value(nextpow2_tensor(tensor)?))
}

fn nextpow2_host(value: Value) -> BuiltinResult<Value> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value)
        .map_err(|e| builtin_error(format!("nextpow2: {e}")))?;
    Ok(tensor::tensor_into_value(nextpow2_tensor(tensor)?))
}

fn nextpow2_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let data = tensor
        .data
        .iter()
        .map(|&x| nextpow2_scalar(x))
        .collect::<Vec<_>>();
    Tensor::new(data, tensor.shape.clone()).map_err(|e| builtin_error(format!("nextpow2: {e}")))
}

fn nextpow2_scalar(x: f64) -> f64 {
    let ax = x.abs();
    if ax == 0.0 {
        0.0
    } else {
        ax.log2().ceil()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{ResolveContext, Type};

    #[test]
    fn nextpow2_type_preserves_tensor_shape() {
        let out = numeric_unary_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(4), Some(1)])
            }
        );
    }

    #[test]
    fn nextpow2_returns_expected_scalars() {
        let Value::Num(v9) = block_on(nextpow2_builtin(Value::Num(9.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(v9, 4.0);
        let Value::Num(v0) = block_on(nextpow2_builtin(Value::Num(0.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(v0, 0.0);
        let Value::Num(vneg) = block_on(nextpow2_builtin(Value::Num(-3.0))).unwrap() else {
            panic!("expected scalar")
        };
        assert_eq!(vneg, 2.0);
    }

    #[test]
    fn nextpow2_handles_inf_and_nan() {
        let Value::Num(vinf) = block_on(nextpow2_builtin(Value::Num(f64::INFINITY))).unwrap()
        else {
            panic!("expected scalar")
        };
        assert!(vinf.is_infinite());
        let Value::Num(vnan) = block_on(nextpow2_builtin(Value::Num(f64::NAN))).unwrap() else {
            panic!("expected scalar")
        };
        assert!(vnan.is_nan());
    }

    #[test]
    fn nextpow2_tensor_matches_expected() {
        let value = block_on(super::nextpow2_builtin(Value::Tensor(
            Tensor::new(vec![0.0, 1.0, 3.0, 9.0], vec![4, 1]).unwrap(),
        )))
        .expect("nextpow2");
        let Value::Tensor(t) = value else {
            panic!("expected tensor")
        };
        assert_eq!(t.data, vec![0.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn nextpow2_gpu_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let host = vec![0.0, 1.0, 3.0, 9.0];
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &host,
                    shape: &[4, 1],
                })
                .expect("upload");
            let gpu =
                block_on(super::nextpow2_builtin(Value::GpuTensor(handle))).expect("gpu nextpow2");
            let t = test_support::gather(gpu).expect("gather");
            assert_eq!(t.data, vec![0.0, 0.0, 2.0, 4.0]);
        });
    }
}
