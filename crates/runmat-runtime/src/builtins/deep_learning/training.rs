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
pub(crate) use dlupdate::__runmat_wasm_register_builtin_dlupdate_builtin;
#[cfg(test)]
pub(super) use dlupdate::dlupdate_builtin;

#[runtime_builtin(
    name = "trainingOptions",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn training_options_builtin(
    solver: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    super::errors::admit_single_output(
        "trainingOptions",
        &runmat_builtins::TRAINING_OPTIONS_INVALID_INPUT,
    )?;
    let solver = scalar_text(&solver, "trainingOptions")?.to_ascii_lowercase();
    match solver.as_str() {
        "sgdm" | "adam" | "rmsprop" => {}
        other => {
            return Err(deep_learning_error(
                "trainingOptions",
                format!("trainingOptions: unsupported solver '{other}'"),
            ));
        }
    }
    let mut properties = vec![
        ("SolverName".to_string(), Value::String(solver.clone())),
        ("MaxEpochs".to_string(), Value::Num(30.0)),
        ("MiniBatchSize".to_string(), Value::Num(128.0)),
        (
            "InitialLearnRate".to_string(),
            Value::Num(if solver == "sgdm" { 0.01 } else { 0.001 }),
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
    // An options object stores its values; construction has no reason to
    // download resident values just to copy them into metadata.
    for (key, value) in parse_name_values(rest, "trainingOptions")? {
        properties.push((runmat_builtins::canonical_training_option_name(&key), value));
    }
    Ok(object_with_identity(
        super::TRAINING_OPTIONS_CLASS,
        properties,
    ))
}

#[runtime_builtin(
    name = "dlarray",
    binding_variant = "default",
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlarray_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
            ("Format", Value::String(format.clone())),
            ("Labels", Value::String(format)),
        ],
    ))
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
    builtin_path = "crate::builtins::deep_learning::training"
)]
pub(super) async fn dlfeval_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    let Some(function) = args.first() else {
        return Err(super::errors::descriptor_error(
            "dlfeval",
            "dlfeval: expected a function handle followed by input arguments",
            &runmat_builtins::DLFEVAL_INVALID_FUNCTION,
        ));
    };
    if !matches!(
        function,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    ) {
        return Err(super::errors::descriptor_error(
            "dlfeval",
            format!("dlfeval: expected a function handle, got {function:?}"),
            &runmat_builtins::DLFEVAL_INVALID_FUNCTION,
        ));
    }
    let requested_outputs = crate::output_count::current_output_count().unwrap_or(1);
    // Admission is complete. Move the call arguments into the tape future
    // without an additional whole-tree clone at this boundary.
    let function = args.remove(0);
    autodiff::with_tape_context(function, args, requested_outputs).await
}
