use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use runmat_macros::runtime_builtin;

use crate::builtins::math::signal::common::{
    parse_window_options, provider_precision_matches, window_tensor, WindowOutputType,
    WindowSampling,
};
use crate::builtins::math::signal::type_resolvers::window_vector_type;

const BUILTIN_NAME: &str = "blackman";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "blackman",
    op_kind: GpuOpKind::Custom("window"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("blackman_window")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Generates the Blackman window directly on the active provider when the custom hook is available; otherwise falls back to host construction.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::blackman")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "blackman",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "blackman materialises a new window vector and is not currently fused.",
};

#[runtime_builtin(
    name = "blackman",
    category = "math/signal",
    summary = "Generate a Blackman window as an N-by-1 real column vector.",
    keywords = "blackman,window,signal processing,dsp,fft",
    type_resolver(window_vector_type),
    builtin_path = "crate::builtins::math::signal::blackman"
)]
async fn blackman_builtin(
    n: runmat_builtins::Value,
    varargin: Vec<runmat_builtins::Value>,
) -> crate::BuiltinResult<runmat_builtins::Value> {
    let options = parse_window_options(BUILTIN_NAME, n, &varargin, true)?;
    if options.len > 1 && provider_precision_matches(options.output_type) {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if let Ok(handle) = provider.blackman_window(
                options.len,
                matches!(options.sampling, WindowSampling::Periodic),
            ) {
                let precision = match options.output_type {
                    WindowOutputType::Double => runmat_accelerate_api::ProviderPrecision::F64,
                    WindowOutputType::Single => runmat_accelerate_api::ProviderPrecision::F32,
                };
                runmat_accelerate_api::set_handle_precision(&handle, precision);
                return Ok(runmat_builtins::Value::GpuTensor(handle));
            }
        }
    }
    window_tensor(options, BUILTIN_NAME, |idx, total| {
        let denom = (total - 1) as f64;
        let phase = 2.0 * std::f64::consts::PI * idx as f64 / denom;
        0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::Value;

    #[test]
    fn blackman_returns_expected_values() {
        let value = block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman");
        let Value::Tensor(t) = value else {
            panic!("expected tensor")
        };
        let expected = [
            -1.3877787807814457e-17,
            0.09045342435412812,
            0.45918295754596355,
            0.9203636180999081,
            0.9203636180999083,
            0.45918295754596383,
            0.09045342435412818,
            -1.3877787807814457e-17,
        ];
        assert_eq!(t.shape, vec![8, 1]);
        for (got, want) in t.data.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn blackman_handles_zero_and_one_lengths() {
        let Value::Tensor(zero) =
            block_on(blackman_builtin(Value::Num(0.0), Vec::new())).expect("blackman(0)")
        else {
            panic!("expected tensor")
        };
        assert_eq!(zero.shape, vec![0, 1]);
        assert!(zero.data.is_empty());

        let Value::Tensor(one) =
            block_on(blackman_builtin(Value::Num(1.0), Vec::new())).expect("blackman(1)")
        else {
            panic!("expected tensor")
        };
        assert_eq!(one.shape, vec![1, 1]);
        assert_eq!(one.data, vec![1.0]);
    }

    #[test]
    fn blackman_rejects_invalid_lengths() {
        assert!(block_on(blackman_builtin(Value::Num(-1.0), Vec::new())).is_err());
        let Value::Tensor(rounded) =
            block_on(blackman_builtin(Value::Num(2.5), Vec::new())).expect("blackman rounded")
        else {
            panic!("expected tensor")
        };
        assert_eq!(rounded.shape, vec![3, 1]);
        assert!(block_on(blackman_builtin(
            Value::Tensor(runmat_builtins::Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
            Vec::new()
        ))
        .is_err());
    }

    #[test]
    fn blackman_supports_periodic_and_single_overloads() {
        let Value::Tensor(periodic) = block_on(blackman_builtin(
            Value::Num(4.0),
            vec![Value::from("periodic")],
        ))
        .expect("blackman periodic") else {
            panic!("expected tensor")
        };
        assert_eq!(periodic.shape, vec![4, 1]);
        assert!((periodic.data[1] - 0.34).abs() < 1e-12);

        let Value::Tensor(single) = block_on(blackman_builtin(
            Value::Num(4.0),
            vec![Value::from("single")],
        ))
        .expect("blackman single") else {
            panic!("expected tensor")
        };
        assert_eq!(single.dtype, runmat_builtins::NumericDType::F32);
    }

    #[test]
    fn blackman_gpu_matches_cpu() {
        test_support::with_test_provider(|_| {
            let value =
                block_on(blackman_builtin(Value::Num(8.0), Vec::new())).expect("blackman gpu");
            let tensor = test_support::gather(value).expect("gather");
            assert_eq!(tensor.shape, vec![8, 1]);
            assert!((tensor.data[3] - 0.9203636180999081).abs() < 1e-12);

            let periodic_one = block_on(blackman_builtin(
                Value::Num(1.0),
                vec![Value::from("periodic")],
            ))
            .expect("blackman periodic len1 gpu");
            let periodic_one = test_support::gather(periodic_one).expect("gather periodic len1");
            assert_eq!(periodic_one.data, vec![1.0]);
        });
    }
}
