macro_rules! define_integer_conversion_builtin {
    ($function:ident, $name:literal, $target:expr, $path:literal, $argument_error:path, $input_error:path, $internal_error:path) => {
        use runmat_builtins::BuiltinErrorDescriptor;
        use runmat_value::Value;

        use crate::builtins::common::integer_conversion::{cast_value, CastError};
        use crate::builtins::common::spec::{
            BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
            ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
        };
        use crate::{build_runtime_error, BuiltinResult, RuntimeError};

        #[runmat_macros::register_gpu_spec(builtin_path = $path)]
        pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
            name: $name,
            op_kind: GpuOpKind::Elementwise,
            supported_precisions: &[ScalarType::F32, ScalarType::F64],
            broadcast: BroadcastSemantics::Matlab,
            provider_hooks: &[ProviderHook::Custom("cast_to_integer")],
            constant_strategy: ConstantStrategy::InlineLiteral,
            residency: ResidencyPolicy::NewHandle,
            nan_mode: ReductionNaN::Include,
            two_pass_threshold: None,
            workgroup_size: None,
            accepts_nan_mode: false,
            notes: "Real gpuArray inputs use the provider-resident integer conversion hook. Paired-complex inputs use exact owner-resolved fallback and retain native integer storage.",
        };

        #[runmat_macros::register_fusion_spec(builtin_path = $path)]
        pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
            name: $name,
            shape: ShapeRequirements::BroadcastCompatible,
            constant_strategy: ConstantStrategy::InlineLiteral,
            elementwise: None,
            reduction: None,
            emits_nan: false,
            notes: "Resident integer conversion uses provider-native storage; fusion can target the typed provider hook when supported.",
        };

        #[runmat_macros::runtime_builtin(
            name = $name,
            binding_variant = "default",
            builtin_path = $path
        )]
        pub(crate) async fn $function(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
            if !rest.is_empty() {
                return Err(error(&$argument_error, "too many input arguments"));
            }
            cast_value(value, $target).await.map_err(|cause| match cause {
                CastError::Unsupported(class) => error(
                    &$input_error,
                    format!("conversion to {} from {class} is not possible", $name),
                ),
                CastError::Internal(detail) => error(&$internal_error, detail),
            })
        }

        fn error(
            descriptor: &'static BuiltinErrorDescriptor,
            detail: impl std::fmt::Display,
        ) -> RuntimeError {
            let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
                .with_builtin($name);
            if let Some(identifier) = descriptor.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        }
    };
}

pub(super) use define_integer_conversion_builtin;
