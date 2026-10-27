//! Hypothesis-test compatibility helpers.

use runmat_builtins::{
    BuiltinErrorDescriptor, KstestNamedCdf, KstestOptionName, KstestTail, Ttest2OptionName,
    Ttest2Tail as Tail, Ttest2Variance as VarType, KSTEST_ERROR_INVALID_ARGUMENT,
    KSTEST_INTEGER_CDF_EXTENSION, KSTEST_INTEGER_SAMPLE_EXTENSION, KSTEST_RESIDENT_INPUT_EXTENSION,
    TTEST2_ERROR_INTERNAL, TTEST2_ERROR_INVALID_ARGUMENT, TTEST2_INTEGER_OPTION_EXTENSION,
    TTEST2_INTEGER_SAMPLE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::common::validation::value_contains_explicit_gpu as kstest_value_contains_explicit_gpu;
use crate::builtins::stats::summary::distribution_math::{
    student_t_cdf, student_t_cdf_upper, student_t_inv,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "ttest2";
const KSTEST_NAME: &str = "kstest";

#[path = "hypothesis/ttest2_options.rs"]
mod ttest2_options;
use ttest2_options::Ttest2OptionPlan;
#[path = "hypothesis/kstest_options.rs"]
mod kstest_options;
use kstest_options::KstestOptionPlan;
#[path = "hypothesis/kstest_domain.rs"]
mod kstest_domain;
use kstest_domain::*;
#[path = "hypothesis/kstest_admission.rs"]
mod kstest_admission;
use kstest_admission::*;
#[path = "hypothesis/kstest_cdf.rs"]
mod kstest_cdf;
use kstest_cdf::*;
#[path = "hypothesis/kstest_statistics.rs"]
mod kstest_statistics;
use kstest_statistics::*;
#[cfg(test)]
#[path = "hypothesis/ttest2_provider_tests.rs"]
mod ttest2_provider_tests;
#[cfg(test)]
#[path = "hypothesis/kstest_provider_tests.rs"]
mod kstest_provider_tests;

fn ttest2_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    ttest2_error(message, &TTEST2_ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    ttest2_error(message, &TTEST2_ERROR_INTERNAL)
}

#[runtime_builtin(
    name = "ttest2",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::hypothesis"
)]
pub(crate) async fn ttest2_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    x: Value,
    y: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(runtime.as_ref(), BUILTIN_NAME,
        "finite host-memory execution is unsupported; two-sample test materialization, lane and result workspaces have no complete resource bound")?;
    let mut cooperation = super::inferential::Cooperation::new(runtime.as_ref());
    cooperation.completed(0).await?;
    ensure_ttest2_sample_extension(&x, &y)?;
    let _control_sources = rest.fork();
    let options = Ttest2OptionPlan::admit(rest)?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(invalid_argument(
            "ttest2: too many output arguments; maximum is 4",
        ));
    }
    let x = gathered_tensor(x, runtime.as_ref()).await?;
    let y = gathered_tensor(y, runtime.as_ref()).await?;
    ensure_exact_ttest2_integer_tensor(&x, "sample x")?;
    ensure_exact_ttest2_integer_tensor(&y, "sample y")?;
    let options = options.resolve(runtime.as_ref()).await?;
    cooperation.completed(0).await?;
    let eval = evaluate(&x, &y, options)?;
    cooperation.completed(0).await?;
    match crate::output_count::current_output_count() {
        Some(out_count) => outputs_for_count(out_count, eval),
        None => Ok(eval.h),
    }
}

#[runtime_builtin(
    name = "kstest",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::hypothesis"
)]
pub(crate) async fn kstest_builtin(
    x: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let _control_sources = rest.fork();
    let options = KstestOptionPlan::admit(rest)?;
    ensure_kstest_sample_extension(&x)?;
    options.ensure_extensions(&x)?;
    let options = options.preflight_known()?;
    preflight_kstest_sample_metadata(&x)?;
    if crate::output_count::current_output_count().is_some_and(|count| count > 4) {
        return Err(kstest_invalid_argument(
            "kstest: too many output arguments; maximum is 4",
        ));
    }
    let x = kstest_gather(x).await?;
    let x = tensor::value_into_tensor_for(KSTEST_NAME, x).map_err(kstest_invalid_argument)?;
    ensure_exact_kstest_integer_tensor(&x, "sample")?;
    let options = options.resolve().await?;
    let eval = evaluate_kstest(&x, &options)?;
    match crate::output_count::current_output_count() {
        Some(out_count) => kstest_outputs_for_count(out_count, eval),
        None => Ok(eval.h),
    }
}

#[path = "hypothesis/ttest2_engine.rs"]
mod ttest2_engine;
use ttest2_engine::*;

#[cfg(test)]
#[path = "hypothesis/tests/mod.rs"]
mod tests;
