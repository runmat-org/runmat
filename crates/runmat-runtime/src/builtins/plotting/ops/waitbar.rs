//! MATLAB-compatible `waitbar` progress figure builtin.

use runmat_builtins::{
    WAITBAR_ERROR_INVALID_ARGUMENT, WAITBAR_INTEGER_HANDLE_EXTENSION,
    WAITBAR_INTEGER_PROGRESS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::IntValue;
use runmat_value::Value;

use super::properties::{set_properties_with_runtime, PlotHandle};
use super::state::{
    current_or_any_waitbar_handle, figure_handle_exists, new_figure_handle, set_figure_name,
    set_figure_number_title, set_waitbar_state, waitbar_state_snapshot, FigureHandle,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "waitbar";

#[derive(Debug)]
struct WaitbarRequest {
    progress: f64,
    target: Option<FigureHandle>,
    message: Option<String>,
    figure_properties: crate::call::BuiltinArguments,
    has_figure_properties: bool,
}

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        WAITBAR_ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = WAITBAR_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "waitbar",
    binding_variant = "default",
    builtin_path = "crate::builtins::plotting::waitbar"
)]
pub fn waitbar_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<f64> {
    if args.first().is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WAITBAR_INTEGER_PROGRESS_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if args.get(1).is_some_and(is_typed_integer) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WAITBAR_INTEGER_HANDLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let request = parse_request(args)?;
    let should_update_existing_waitbar =
        request.target.is_none() && request.message.is_none() && !request.has_figure_properties;
    let mut created_new_waitbar = false;
    let handle = match request.target {
        Some(handle) => {
            if !figure_handle_exists(handle) {
                return Err(invalid("waitbar figure handle does not exist"));
            }
            handle
        }
        None if should_update_existing_waitbar => {
            current_or_any_waitbar_handle().unwrap_or_else(|| {
                created_new_waitbar = true;
                new_figure_handle()
            })
        }
        None => {
            created_new_waitbar = true;
            new_figure_handle()
        }
    };

    let previous = waitbar_state_snapshot(handle)
        .map_err(|err| invalid(err.to_string()))?
        .map(|state| state.message);
    let message = request
        .message
        .or(previous)
        .unwrap_or_else(|| "Please wait...".to_string());

    set_waitbar_state(handle, request.progress, message.clone())
        .map_err(|err| invalid(err.to_string()))?;

    if created_new_waitbar {
        set_figure_number_title(handle, false).map_err(|err| invalid(err.to_string()))?;
        set_figure_name(handle, "Waitbar".to_string()).map_err(|err| invalid(err.to_string()))?;
    }

    if request.has_figure_properties {
        for pair in request.figure_properties.chunks_exact(2) {
            let name = text_value(&pair[0])?.expect("validated waitbar property name");
            if is_waitbar_or_figure_property(&name) {
                set_properties_with_runtime(None, PlotHandle::Figure(handle), pair, BUILTIN_NAME)?;
            }
        }
    }

    Ok(handle.as_u32() as f64)
}

fn parse_request(args: crate::call::BuiltinArguments) -> BuiltinResult<WaitbarRequest> {
    let Some(first) = args.first() else {
        return Err(invalid("expected progress fraction"));
    };
    let progress = parse_progress(first)?;
    let mut idx = 1;
    let mut target = None;
    let mut message = None;

    if let Some(value) = args.get(idx) {
        if let Some(text) = message_text(value)? {
            message = Some(text);
            idx += 1;
        } else if let Some(handle) = figure_handle(value) {
            target = Some(handle);
            idx += 1;
            if let Some(value) = args.get(idx) {
                if let Some(text) = message_text(value)? {
                    if !looks_like_property_pair(
                        args.view()
                            .range(idx..)
                            .expect("bounded waitbar option tail"),
                    )? {
                        message = Some(text);
                        idx += 1;
                    }
                }
            }
        }
    }

    let has_figure_properties = parse_property_tail(
        args.view()
            .range(idx..)
            .expect("bounded waitbar property tail"),
    )?;
    let figure_properties = args
        .into_range(idx..)
        .expect("bounded waitbar property tail");
    Ok(WaitbarRequest {
        progress,
        target,
        message,
        figure_properties,
        has_figure_properties,
    })
}

fn parse_progress(value: &Value) -> BuiltinResult<f64> {
    if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
        let progress = exact_integer_f64(integer)
            .ok_or_else(|| invalid("integer progress must be exactly representable as double"))?;
        return Ok(progress.clamp(0.0, 1.0));
    }
    if !matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
        return Err(invalid("progress must be a numeric scalar"));
    }
    let progress = crate::builtins::common::tensor::scalar_f64_from_host_value(value)
        .map_err(|_| invalid("progress must be a numeric scalar"))?
        .ok_or_else(|| invalid("progress must be a numeric scalar"))?;
    if !progress.is_finite() {
        return Err(invalid("progress must be finite"));
    }
    Ok(progress.clamp(0.0, 1.0))
}

fn exact_integer_f64(value: IntValue) -> Option<f64> {
    use num_traits::FromPrimitive;
    let converted = value.to_f64();
    let exact = match value {
        IntValue::I8(value) => num_bigint::BigInt::from(value),
        IntValue::I16(value) => num_bigint::BigInt::from(value),
        IntValue::I32(value) => num_bigint::BigInt::from(value),
        IntValue::I64(value) => num_bigint::BigInt::from(value),
        IntValue::U8(value) => num_bigint::BigInt::from(value),
        IntValue::U16(value) => num_bigint::BigInt::from(value),
        IntValue::U32(value) => num_bigint::BigInt::from(value),
        IntValue::U64(value) => num_bigint::BigInt::from(value),
    };
    (num_bigint::BigInt::from_f64(converted).as_ref() == Some(&exact)).then_some(converted)
}

fn is_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn figure_handle(value: &Value) -> Option<FigureHandle> {
    if !matches!(value, Value::Num(_) | Value::Int(_) | Value::Tensor(_)) {
        return None;
    }
    let raw = crate::builtins::common::tensor::scalar_f64_from_host_value(value)
        .ok()
        .flatten()?;
    if raw.is_finite() && raw >= 1.0 && raw <= u32::MAX as f64 {
        Some(FigureHandle::from(raw.round() as u32))
    } else {
        None
    }
}

fn message_text(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok({
        match value {
            Value::String(_) | Value::CharArray(_) => text_value(value)?,
            Value::StringArray(array) => array
                .data()
                .iter()
                .map(|text| text.as_str())
                .collect::<Option<Vec<_>>>()
                .map(|texts| texts.join("\n")),
            Value::Cell(cell) => {
                let mut parts = Vec::with_capacity(cell.data().len());
                for item in cell.data() {
                    let Some(part) = message_text(item)? else {
                        return Ok(None);
                    };
                    parts.push(part);
                }
                Some(parts.join("\n"))
            }
            _ => None,
        }
    })
}

fn text_value(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok({
        super::style::value_as_string(value).map_err(|source| {
            crate::builtins::plotting::plotting_error_with_source(
                "waitbar",
                source.to_string(),
                source,
            )
        })?
    })
}

fn looks_like_property_pair(tail: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<bool> {
    if tail.len() < 2 || !tail.len().is_multiple_of(2) {
        return Ok(false);
    }
    Ok(text_value(tail.first().expect("nonempty pair tail"))?
        .is_some_and(|name| is_waitbar_or_figure_property(&name)))
}

fn parse_property_tail(tail: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<bool> {
    if tail.is_empty() {
        return Ok(false);
    }
    if !tail.len().is_multiple_of(2) {
        return Err(invalid("property arguments must be name/value pairs"));
    }
    let mut has_figure_properties = false;
    for pair in tail.chunks_exact(2) {
        let name = text_value(&pair[0])?.ok_or_else(|| invalid("property names must be text"))?;
        match name.trim().to_ascii_lowercase().as_str() {
            "createcancelbtn" | "cancelbutton" | "units" | "windowstyle" => {}
            _ if is_waitbar_or_figure_property(&name) => has_figure_properties = true,
            _ => {
                return Err(invalid(format!("unsupported waitbar property '{name}'")));
            }
        }
    }
    Ok(has_figure_properties)
}

fn is_waitbar_or_figure_property(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "name" | "numbertitle" | "visible" | "position" | "color" | "backgroundcolor" | "tag"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::plotting::get::get_builtin;
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, reset_hold_state_for_run};
    use runmat_value::{CellArray, StringArray, Tensor};

    fn waitbar_builtin(args: Vec<Value>) -> BuiltinResult<f64> {
        super::waitbar_builtin(crate::call::BuiltinArguments::from_borrowed(&args, None)?)
    }

    fn setup() -> crate::builtins::plotting::state::PlotTestLockGuard {
        let guard = lock_plot_registry();
        ensure_plot_test_env();
        reset_hold_state_for_run();
        let _ = clear_figure(None);
        guard
    }

    #[test]
    fn waitbar_creates_and_updates_progress_figure() {
        let _guard = setup();
        let handle = waitbar_builtin(vec![
            Value::Num(0.25),
            Value::String("Loading".into()),
            Value::String("Name".into()),
            Value::String("Progress".into()),
        ])
        .unwrap();

        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Name".into())]).unwrap(),
            Value::String("Progress".into())
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarMessage".into())
            ])
            .unwrap(),
            Value::String("Loading".into())
        );

        let nonscalar =
            Value::Tensor(Tensor::new(vec![0.5, 0.75], vec![1, 2]).expect("two progress values"));
        for invalid_args in [vec![nonscalar.clone()], vec![Value::Num(0.5), nonscalar]] {
            let error = waitbar_builtin(invalid_args).expect_err("non-scalar control");
            assert_eq!(error.identifier(), Some("RunMat:waitbar:InvalidArgument"));
        }
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarProgress".into())
            ])
            .unwrap(),
            Value::Num(0.25)
        );

        let same = waitbar_builtin(vec![
            Value::Num(0.75),
            Value::Num(handle),
            Value::String("Almost done".into()),
        ])
        .unwrap();
        assert_eq!(same, handle);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarProgress".into())
            ])
            .unwrap(),
            Value::Num(0.75)
        );
    }

    #[test]
    fn waitbar_progress_without_handle_updates_existing_waitbar() {
        let _guard = setup();
        let before = super::super::state::figure_handles().len();
        let handle = waitbar_builtin(vec![Value::Num(0.25), Value::String("Loading".into())])
            .expect("create waitbar");
        let after_create = super::super::state::figure_handles().len();
        let same = waitbar_builtin(vec![Value::Num(0.75)]).expect("update waitbar");
        assert_eq!(same, handle);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarProgress".into())
            ])
            .unwrap(),
            Value::Num(0.75)
        );
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarMessage".into())
            ])
            .unwrap(),
            Value::String("Loading".into())
        );
        assert_eq!(after_create, before + 1);
        assert_eq!(super::super::state::figure_handles().len(), after_create);
    }

    #[test]
    fn waitbar_accepts_array_messages_and_compatibility_properties() {
        let _guard = setup();
        let message = StringArray::new(vec!["Preparing".into(), "Solving".into()], vec![2, 1])
            .expect("string array");
        let handle = waitbar_builtin(vec![
            Value::Num(0.1),
            Value::StringArray(message),
            Value::String("Units".into()),
            Value::String("normalized".into()),
            Value::String("WindowStyle".into()),
            Value::String("modal".into()),
        ])
        .expect("waitbar with string-array message");
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarMessage".into())
            ])
            .unwrap(),
            Value::String("Preparing\nSolving".into())
        );

        let cell = CellArray::new(
            vec![
                Value::String("Phase 1".into()),
                Value::String("Phase 2".into()),
            ],
            1,
            2,
        )
        .expect("cell message");
        let same = waitbar_builtin(vec![Value::Num(0.2), Value::Num(handle), Value::Cell(cell)])
            .expect("waitbar with cell message");
        assert_eq!(same, handle);
        assert_eq!(
            get_builtin(vec![
                Value::Num(handle),
                Value::String("WaitbarMessage".into())
            ])
            .unwrap(),
            Value::String("Phase 1\nPhase 2".into())
        );
    }

    #[test]
    fn waitbar_accepts_tag_property_and_uses_default_tag() {
        let _guard = setup();
        let handle = waitbar_builtin(vec![Value::Num(0.1)]).expect("default waitbar");
        assert_eq!(
            get_builtin(vec![Value::Num(handle), Value::String("Tag".into())]).unwrap(),
            Value::String("TMWWaitbar".into())
        );

        let tagged = waitbar_builtin(vec![
            Value::Num(0.2),
            Value::String("Tagged".into()),
            Value::String("Tag".into()),
            Value::String("client-progress".into()),
        ])
        .expect("tagged waitbar");
        assert_eq!(
            get_builtin(vec![Value::Num(tagged), Value::String("Tag".into())]).unwrap(),
            Value::String("client-progress".into())
        );
    }

    #[test]
    fn waitbar_integer_roles_are_independently_gated() {
        let _guard = setup();
        let progress = Value::Int(IntValue::U8(1));
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = waitbar_builtin(vec![progress.clone()]).expect_err("progress gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:WaitbarIntegerProgressExtension")
            );
        }
        let handle = {
            let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
            waitbar_builtin(vec![progress, Value::from("Loading")]).expect("integer progress")
        };
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = waitbar_builtin(vec![
                Value::Num(0.5),
                Value::Int(IntValue::U32(handle as u32)),
            ])
            .expect_err("handle gate");
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:WaitbarIntegerHandleExtension")
            );
        }
    }
}
