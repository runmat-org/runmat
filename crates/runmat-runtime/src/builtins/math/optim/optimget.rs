//! Legacy option query: preserve selected values, never materialize their data.
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    OptimParameterLookup, OptimParameterName, OPTIMGET_DESCRIPTOR, OPTIMGET_ERROR_INVALID_ARGUMENT,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Value, ValueSequence};
const NAME: &str = "optimget";
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::optim::optimget")]
pub const GPU_SPEC:BuiltinGpuSpec=BuiltinGpuSpec {
    name:NAME,op_kind:GpuOpKind::Custom("optimization-option-query"),supported_precisions:&[],broadcast:BroadcastSemantics::None,
    provider_hooks:&[],constant_strategy:ConstantStrategy::InlineLiteral,residency:ResidencyPolicy::InheritInputs,
    nan_mode:ReductionNaN::Include,two_pass_threshold:None,workgroup_size:None,accepts_nan_mode:false,
    notes:"Metadata projection preserves the selected resident value; no provider read or numerical kernel occurs.",
};
#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::optim::optimget")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Option lookup is a value projection, not an elementwise numerical expression.",
};
#[cfg(test)]
mod tests;

#[runtime_builtin(
    name = "optimget",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimget"
)]
async fn optimget_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    query(runtime.as_ref(), args)
}
fn query(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; legacy parameter text and result-root metadata have no complete domain bound")?;
    crate::output_count::validate_fixed_outputs(
        &OPTIMGET_DESCRIPTOR,
        &OPTIMGET_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    if !(2..=3).contains(&args.len()) {
        return Err(invalid(
            "expected options, parameter name and optional default",
        ));
    }
    let name = crate::builtins::common::validation::value_text_scalar_string(&args[1])
        .ok_or_else(|| invalid("parameter name must be scalar text"))?;
    let profile = explicit_profile(&args[0]);
    let parameter = match OptimParameterName::resolve_for_solver(&name, profile) {
        OptimParameterLookup::Matched(parameter) => parameter,
        OptimParameterLookup::Ambiguous => {
            return Err(invalid("ambiguous parameter-name abbreviation"))
        }
        OptimParameterLookup::Unknown => {
            return Err(invalid("unknown optimization parameter name"))
        }
    };
    let selected = match &args[0] {
        Value::Struct(options) => {
            let mut found=options.fields.iter().filter_map(|(name,value)|
                matches!(OptimParameterName::resolve_for_solver(name, profile),OptimParameterLookup::Matched(field) if field.equivalent(parameter, profile)).then_some(value));
            let selected = found.next();
            if found.next().is_some() {
                return Err(invalid(
                    "multiple stored fields identify the same parameter",
                ));
            }
            selected
        }
        Value::Tensor(value) if value.is_empty() => None,
        _ => return Err(invalid("options must be a scalar structure or numeric []")),
    };
    // No provider reads, integer conversion, closure inspection or callback
    // invocation: the original field/default's immutable carrier is projected.
    let selected = selected
        .filter(|value| !unset(value))
        .or_else(|| args.get(2));
    let value = match selected {
        Some(value) => value.clone(),
        None => super::common::output_callback::empty(runtime)?,
    };
    let sequence =
        ValueSequence::single(value).map_err(crate::sequence::sequence_error_to_runtime)?;
    // Input roots remain live through canonical output admission/publication.
    match runtime {
        Some(runtime) => crate::execution::RootedValueSequence::in_context(sequence, runtime),
        None => crate::execution::RootedValueSequence::new(sequence)
            .map_err(|source| source.into_runtime_error()),
    }
}
fn explicit_profile(value: &Value) -> Option<runmat_builtins::OptimSolver> {
    let Value::Struct(options) = value else {
        return None;
    };
    let mut profiles = options
        .fields
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("Solver"));
    let (_, value) = profiles.next()?;
    // An absent, ambiguous or unrecognized public profile is still a plain
    // legacy struct; never infer a profile from its other names or values.
    if profiles.next().is_some() {
        return None;
    }
    let name = crate::builtins::common::validation::value_text_scalar_string(value)?;
    runmat_builtins::OptimSolver::parse(&name)
        .filter(|solver| *solver != runmat_builtins::OptimSolver::Generic)
}

fn unset(value: &Value) -> bool {
    matches!(value,Value::Tensor(value)if value.is_empty())
}
fn invalid(detail: &str) -> RuntimeError {
    build_runtime_error(format!("optimget: {detail}"))
        .with_builtin(NAME)
        .with_identifier("RunMat:optimget:InvalidArgument")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
