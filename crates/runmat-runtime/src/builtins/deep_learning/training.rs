//! Deep-learning training entrypoints.

use super::{
    autodiff, deep_learning_error, ensure_dlarray_class_registered, object, object_with_identity,
    parse_name_values, scalar_text, text_or_missing,
};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

mod adam;
mod dlupdate;
#[cfg(target_arch = "wasm32")]
pub(crate) use adam::__runmat_wasm_register_builtin_adamupdate_builtin;
#[cfg(test)]
pub(super) use adam::adamupdate_builtin;
#[cfg(target_arch = "wasm32")]
pub(crate) use dlupdate::__runmat_wasm_register_builtin_dlupdate_builtin_registered;
#[cfg(test)]
pub(super) use dlupdate::dlupdate_builtin;

#[runtime_builtin(
    name = "trainingOptions",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn training_options_builtin(
    solver: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "trainingOptions",
        &runmat_builtins::TRAINING_OPTIONS_INVALID_INPUT,
    )?;
    let solver_name = scalar_text(&solver, "trainingOptions")?;
    let solver = runmat_builtins::TrainingSolver::from_name(&solver_name).ok_or_else(|| {
        deep_learning_error(
            "trainingOptions",
            format!("trainingOptions: unsupported solver '{solver_name}'"),
        )
    })?;
    let mut properties = vec![
        (
            "SolverName".to_string(),
            Value::String(solver.as_str().into()),
        ),
        ("MaxEpochs".to_string(), Value::Num(30.0)),
        ("MiniBatchSize".to_string(), Value::Num(128.0)),
        (
            "InitialLearnRate".to_string(),
            Value::Num(solver.default_initial_learn_rate()),
        ),
        ("Shuffle".to_string(), Value::String("once".into())),
        ("Verbose".to_string(), Value::Bool(true)),
        ("L2Regularization".to_string(), Value::Num(0.0001)),
        ("Plots".to_string(), Value::String("none".into())),
        (
            "ExecutionEnvironment".to_string(),
            Value::String("auto".into()),
        ),
    ];
    if solver != runmat_builtins::TrainingSolver::Sgdm {
        properties.push((
            "SquaredGradientDecayFactor".to_string(),
            Value::Num(solver.default_squared_gradient_decay_factor()),
        ));
    }
    // An options object stores its values; construction has no reason to
    // download resident values just to copy them into metadata.
    for (key, value) in parse_name_values(rest, "trainingOptions")? {
        if runmat_builtins::TrainingOptionKey::from_name(&key)
            == Some(runmat_builtins::TrainingOptionKey::SolverName)
        {
            return Err(deep_learning_error(
                "trainingOptions",
                "trainingOptions: solver is supplied by the required positional argument, not SolverName",
            ));
        }
        properties.push((runmat_builtins::canonical_training_option_name(&key), value));
    }
    object_with_identity(solver.class_identity(), properties)
}

#[runtime_builtin(
    name = "dlarray",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlarray_builtin(
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output("dlarray", &runmat_builtins::DLARRAY_INVALID_INPUT)?;
    validate_dlarray_data(&data)?;
    if rest.len() > 1 {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: only the optional dimension-label format argument is supported",
        ));
    }
    // Format is host text, so a non-text resident value must fail without
    // downloading it. Data itself remains in its original owner/residency.
    let format = text_or_missing(rest.first(), "", "dlarray")?;
    ensure_dlarray_class_registered();
    autodiff::annotate_dlarray_value(object(
        runmat_types::standard::DL_ARRAY.display_name(),
        vec![
            (runmat_builtins::DLARRAY_DATA_PROPERTY, data),
            ("Format", Value::string(format.clone())),
            ("Labels", Value::string(format)),
        ],
    )?)
}

fn validate_dlarray_data(data: &Value) -> BuiltinResult<()> {
    if !crate::builtins::common::shape::array_payload_storage_matches(
        data,
        runmat_types::ClassArrayPayload::NumericOrLogical,
    ) {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: Data must be a numeric or logical scalar, array, sparse array, or gpuArray",
        ));
    }
    if crate::builtins::common::validation::value_has_native_integer_class(data) {
        return Err(deep_learning_error(
            "dlarray",
            "dlarray: integer data is not supported; use double, single, logical, or gpuArray data",
        ));
    }
    Ok(())
}

#[runtime_builtin(
    name = "dlfeval",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::training"
)]
async fn dlfeval_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    dlfeval_in_runtime(runtime.as_ref(), requested_outputs, args).await
}

#[cfg(test)]
pub(super) async fn dlfeval_builtin(
    args: Vec<Value>,
) -> BuiltinResult<crate::execution::RootedValue> {
    dlfeval_in_runtime(
        None,
        crate::output_count::current_output_count(),
        crate::builtins::common::test_support::builtin_arguments(args),
    )
    .await
}

async fn dlfeval_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    mut args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let Some(function) = args.first() else {
        return Err(super::errors::descriptor_error(
            "dlfeval",
            "dlfeval: expected a function handle followed by input arguments",
            &runmat_builtins::DLFEVAL_INVALID_FUNCTION,
        ));
    };
    if !matches!(
        function,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    ) {
        return Err(super::errors::descriptor_error(
            "dlfeval",
            format!("dlfeval: expected a function handle, got {function:?}"),
            &runmat_builtins::DLFEVAL_INVALID_FUNCTION,
        ));
    }
    let requested_outputs = requested_outputs.unwrap_or(1);
    // Admission is complete. Move the call arguments into the tape future
    // without an additional whole-tree clone at this boundary.
    let function = args.pop_front().expect("validated argument prefix");
    autodiff::with_tape_context(runtime, function, args, requested_outputs).await
}
