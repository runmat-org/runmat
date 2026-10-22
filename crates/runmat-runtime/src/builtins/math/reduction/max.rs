//! MATLAB-compatible `max` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{MaxComparisonMethod as ComparisonMethod, MaxControl, ReductionNaN};
#[cfg(test)]
use runmat_builtins::{
    MAX_DESCRIPTOR, MAX_ERROR_INVALID_ARGUMENT, MAX_ERROR_INVALID_INPUT, MAX_ERROR_SIZE_MISMATCH,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use super::extrema_provider;
use crate::BuiltinResult;
pub(super) use runmat_builtins::MAX_ERROR_INTERNAL;

mod diagnostics;
use diagnostics::{max_internal_error, max_invalid_argument, max_invalid_input, max_size_mismatch};

mod specs;
#[cfg(target_arch = "wasm32")]
pub(crate) use specs::{
    __runmat_wasm_register_fusion_spec_FUSION_SPEC, __runmat_wasm_register_gpu_spec_GPU_SPEC,
};
#[path = "max/fusion_contract.rs"]
mod fusion_contract;
#[path = "max/fusion_pairwise.rs"]
mod fusion_pairwise;

mod output;
pub use output::MaxEvaluation;
use output::{builtin_demand, evaluation_to_value, reject_pairwise_multiple_outputs};

#[runtime_builtin(
    name = "max",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::reduction::max"
)]
pub(super) async fn max_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if let Some(eval) = crate::builtins::table::categorical_max_evaluate(&value, rest.view()).await
    {
        return crate::builtins::table::categorical_extrema_to_value(eval?);
    }
    reject_pairwise_multiple_outputs(rest.view())?;
    let eval = evaluate_for_demand(runtime.as_ref(), value, rest.view(), builtin_demand()).await?;
    evaluation_to_value(eval)
}

mod nanmax;
#[cfg(target_arch = "wasm32")]
pub(crate) use nanmax::__runmat_wasm_register_builtin_nanmax_builtin;
#[cfg(test)]
use nanmax::nanmax_builtin;

/// Evaluate the builtin once and expose both outputs (value + indices).
pub async fn evaluate(
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<MaxEvaluation> {
    let evaluated = evaluate_for_demand(
        None,
        value,
        rest,
        super::extrema_provider::ExtremaDemand::ValuesAndIndices,
    )
    .await?;
    let (values, indices) = evaluated.into_pair(super::extrema_provider::ExtremaKind::Max)?;
    Ok(MaxEvaluation { values, indices })
}

pub(super) async fn evaluate_for_demand(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    evaluate_for_missing(runtime, value, rest, demand, None).await
}
async fn evaluate_for_missing(
    runtime: Option<&crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::FunctionArgumentView<'_>,
    demand: super::extrema_provider::ExtremaDemand,
    missing: Option<ReductionNaN>,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    if crate::builtins::common::validation::is_typed_complex_integer(&value) {
        return Err(max_invalid_input(
            "operations involving complex numbers with integer types are not supported",
        ));
    }
    let mut parsed = parse_call(rest).await?;
    if let Some(missing) = missing {
        match &mut parsed {
            ParsedCall::Reduction(args) => args.nan_mode = missing,
            ParsedCall::Elementwise(args) => args.nan_mode = missing,
        }
    }
    if std::env::var("RUNMAT_DEBUG_MAX").is_ok() {
        let call_label = match &parsed {
            ParsedCall::Reduction(_) => "reduction",
            ParsedCall::Elementwise(_) => "elementwise",
        };
        let first_arg = rest.first().map(debug_value_kind).unwrap_or("None");
        tracing::debug!(
            call_type = call_label,
            rest_len = rest.len(),
            first_arg = first_arg,
            "[runmat-debug-max]"
        );
    }
    match parsed {
        ParsedCall::Elementwise(args) => elementwise_max(value, args, demand).await,
        ParsedCall::Reduction(args) => reduction_max(runtime, value, args, demand).await,
    }
}
mod options;
use options::{
    debug_value_kind, is_empty_placeholder, parse_call, DimSelection, ElementwiseArgs, ParsedCall,
    ReductionArgs,
};

mod reduction;
use reduction::{
    materialize_for_max, reduction_max, should_replace_complex, should_replace_real, InputData,
};

mod pairwise;
use pairwise::elementwise_max;
#[cfg(test)]
use pairwise::scalar_complex_value;

#[cfg(test)]
pub(crate) mod tests;
