//! MATLAB-compatible `warning` builtin with state management and formatting support.

use once_cell::sync::Lazy;
use runmat_builtins::{CellArray, StructValue, Value};
use runmat_macros::runtime_builtin;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::sync::Mutex;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_output, ConsoleStream};
use crate::warning_store;
use crate::{build_runtime_error, RuntimeControlFlow};
use tracing;

const DEFAULT_IDENTIFIER: &str = "MATLAB:warning";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "warning",
        builtin_path = "crate::builtins::diagnostics::warning"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "warning"
category: "diagnostics"
keywords: ["warning", "diagnostics", "state", "query", "backtrace"]
summary: "Display formatted warnings, control warning state, and query per-identifier settings."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host. GPU tensors appearing in formatted arguments are gathered before formatting."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::diagnostics::warning::tests"
  integration: null
---

# What does the `warning` function do in MATLAB / RunMat?
`warning` emits diagnostic messages without aborting execution. It also provides a central
interface for enabling, disabling, promoting, or querying warnings by identifier.

RunMat mirrors MATLAB semantics with per-identifier modes (`'on'`, `'off'`, `'once'`, `'error'`),
backtrace control, structured state restoration, and formatted message support. When an identifier
is promoted to `'error'`, future uses of that warning raise an error instead of printing.

## How does the `warning` function behave in MATLAB / RunMat?
- `warning(message)` prints the message using the default identifier `MATLAB:warning`.
- `warning(id, fmt, args...)` associates the warning with the identifier `id`, formats the message
  using MATLAB's `sprintf` rules, and honours per-identifier state.
- `warning(MException_obj)` reissues an existing exception as a warning.
- `warning(struct)` restores warning state previously captured with `warning('query', ...)`.
- `warning(state, id)` changes the mode for `id` (`'on'`, `'off'`, `'once'`, `'error'`, and the aliases `'all'` or `'last'`).
  The call returns a struct describing the prior state so you can restore it later.
- `warning(state, mode)` controls diagnostic verbosity modes such as `'backtrace'` and `'verbose'`, returning
  a struct snapshot of the previous setting.
- `warning('default', id)` resets `id` (or `'all'`, `'backtrace'`, `'verbose'`, `'last'`) to the factory default,
  returning the state that was active before the reset. `warning('reset')` restores every warning, mode, and
  once-tracking flag in one call.
- `warning('query', id)` returns a struct describing the current mode for `id`. Passing `'all'`
  returns a cell vector of structs covering every configured identifier plus the global default and diagnostic modes.
- `warning('status')` prints the global warning table as a formatted summary.
- All state-changing forms return struct snapshots; pass them back to `warning` later to reinstate the captured state.

## GPU execution and residency
`warning` is a control-flow builtin that runs on the host. When formatted arguments include GPU
resident arrays (for example via `%g` specifiers), RunMat gathers the values to host memory before
formatting the message so the diagnostic text matches MATLAB expectations.

## Examples of using the `warning` function in MATLAB / RunMat

### Displaying a simple warning
```matlab
warning("Computation took longer than expected.");
```

### Emitting a warning with a custom identifier
```matlab
warning("runmat:io:deprecatedOption", ...
        "Option '%s' is deprecated and will be removed in %s.", ...
        "VerboseMode", "1.2");
```

### Turning a specific warning off and back on
```matlab
warning("off", "runmat:io:deprecatedOption");
% ... code that triggers the warning ...
warning("on", "runmat:io:deprecatedOption");
```

### Promoting a warning to an error
```matlab
warning("error", "runmat:solver:illConditioned");
% The next call raises an error instead of printing a warning.
warning("runmat:solver:illConditioned", ...
        "Condition number exceeds threshold %.2e.", 1e12);
```

### Querying and restoring warning state
```matlab
state = warning("query", "all");   % capture the current table
warning("off", "all");              % silence all warnings temporarily
warning("Some temporary warning.");
warning(state);                    % restore original configuration
```

### Enabling backtraces for debugging
```matlab
warning("on", "backtrace");   % warning("backtrace","on") is equivalent
warning("runmat:demo:slowPath", "Inspect the call stack above for context.");
warning("off", "backtrace");
```

### Displaying verbose suppression guidance
```matlab
warning("on", "verbose");
warning("runmat:demo:slowPath", "Inspect the call stack above for context.");
warning("default", "verbose");   % restore the default verbosity
```

## FAQ
1. **Does RunMat keep track of the last warning?** Yes. The last identifier and message are stored internally and will be exposed through the MATLAB-compatible `lastwarn` builtin.
2. **What does `'once'` do?** The first occurrence of the warning is shown, and subsequent uses of the same identifier are suppressed until you change the state.
3. **How do I restore defaults after experimentation?** Call `warning('reset')` to revert the global default, per-identifier table, diagnostic modes, and once-tracking.
4. **Do state-changing calls return anything useful?** Yes. Forms such as `warning('off','id')`, `warning('on','backtrace')`, and `warning('default')` return struct snapshots describing the state before the change. Store the struct and pass it back to `warning` later to restore the captured configuration.
5. **What does the `'verbose'` mode do?** When enabled (`warning('on','verbose')`), RunMat prints an additional hint describing how to suppress the warning. Disable it with `warning('default','verbose')`.
6. **What happens when a warning is promoted to `'error'`?** The warning text is reused to raise an error with the same identifier, just like MATLAB.
7. **Does `warning` run on the GPU?** No. Control-flow builtins execute on the host. GPU inputs referenced in formatted messages are gathered automatically before the message is emitted.
8. **Can I provide multiple state structs to `warning(state)`?** Yes. Pass either a struct or a cell array of structs (as returned by `warning('query','all')`); RunMat applies them sequentially.
9. **Does `warning('on','backtrace')` match MATLAB output exactly?** RunMat prints a Rust backtrace for now. Future releases will map this to MATLAB-style stack frames, but the enable/disable semantics match MATLAB.
10. **Is whitespace in identifiers allowed?** Identifiers follow MATLAB rules: they must contain at least one colon and no whitespace. RunMat normalises bare names by prefixing `MATLAB:`.
11. **What if I call `warning` with unsupported arguments?** RunMat issues a MATLAB-compatible usage error so you can correct the call.
12. **Are repeated calls thread-safe?** Internally the warning table is guarded by a mutex, so concurrent invocations are serialised and remain deterministic.

## See Also
[error](./error), [sprintf](./sprintf), [fprintf](./fprintf)

## Source & Feedback
- The full source code for the implementation of the `warning` function is available at: [`crates/runmat-runtime/src/builtins/diagnostics/warning.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/diagnostics/warning.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::diagnostics::warning")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "warning",
    op_kind: GpuOpKind::Custom("control"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Control-flow builtin; GPU backends are never invoked.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::diagnostics::warning")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "warning",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow builtin; excluded from fusion planning.",
};

static MANAGER: Lazy<Mutex<WarningManager>> = Lazy::new(|| Mutex::new(WarningManager::default()));

fn manager() -> &'static Mutex<WarningManager> {
    &MANAGER
}

fn with_manager<F, R>(func: F) -> R
where
    F: FnOnce(&mut WarningManager) -> R,
{
    let mut guard = manager().lock().expect("warning manager mutex poisoned");
    func(&mut guard)
}

fn warning_flow(identifier: &str, message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_builtin("warning")
        .with_identifier(normalize_identifier(identifier))
        .build()
        .into()
}

fn warning_default_error(message: impl Into<String>) -> RuntimeControlFlow {
    warning_flow(DEFAULT_IDENTIFIER, message)
}

fn remap_warning_flow<F>(flow: RuntimeControlFlow, identifier: &str, message: F) -> RuntimeControlFlow
where
    F: FnOnce(&crate::RuntimeError) -> String,
{
    match flow {
        RuntimeControlFlow::Error(err) => build_runtime_error(message(&err))
            .with_builtin("warning")
            .with_identifier(normalize_identifier(identifier))
            .with_source(err)
            .build()
            .into(),
        _ => RuntimeControlFlow::Error(
            build_runtime_error("interaction pending")
                .with_builtin("warning")
                .with_identifier(normalize_identifier(identifier))
                .build(),
        ),
    }
}

#[runtime_builtin(
    name = "warning",
    category = "diagnostics",
    summary = "Display formatted warnings, control warning state, and query per-identifier settings.",
    keywords = "warning,diagnostics,state,query,backtrace",
    accel = "metadata",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::diagnostics::warning"
)]
fn warning_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return handle_query_default();
    }

    let first = &args[0];
    let rest = &args[1..];

    match first {
        Value::Struct(_) | Value::Cell(_) => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: state restoration accepts a single argument",
                ));
            }
            apply_state_value(first)?;
            Ok(Value::Num(0.0))
        }
        Value::MException(mex) => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: additional arguments are not allowed when passing an MException",
                ));
            }
            Ok(reissue_exception(mex)?)
        }
        _ => {
            let first_string = value_to_string("warning", first)?;
            if let Some(command) = parse_command(&first_string) {
                return handle_command(command, rest);
            }
            Ok(handle_message_call(None, first_string, rest)?)
        }
    }
}

fn handle_message_call(
    explicit_identifier: Option<String>,
    first_string: String,
    rest: &[Value],
) -> crate::BuiltinResult<Value> {
    if let Some(identifier) = explicit_identifier {
        return emit_warning(&identifier, &first_string, rest);
    }

    if rest.is_empty() {
        emit_warning(DEFAULT_IDENTIFIER, &first_string, rest)
    } else if is_message_identifier(&first_string) {
        let fmt = value_to_string("warning", &rest[0])?;
        let args = &rest[1..];
        emit_warning(&first_string, &fmt, args)
    } else {
        emit_warning(DEFAULT_IDENTIFIER, &first_string, rest)
    }
}

fn emit_warning(identifier_raw: &str, fmt: &str, args: &[Value]) -> crate::BuiltinResult<Value> {
    let identifier = normalize_identifier(identifier_raw);
    let message = format_variadic(fmt, args).map_err(|flow| {
        remap_warning_flow(flow, DEFAULT_IDENTIFIER, |err| err.message().to_string())
    })?;

    let action = with_manager(|mgr| {
        let action = mgr.action_for(&identifier);
        if matches!(action, WarningAction::Display | WarningAction::AsError) {
            mgr.record_last(&identifier, &message);
        }
        action
    });

    match action {
        WarningAction::Suppress => Ok(Value::Num(0.0)),
        WarningAction::Display => {
            print_warning(&identifier, &message);
            warning_store::push(&identifier, &message);
            Ok(Value::Num(0.0))
        }
        WarningAction::AsError => {
            warning_store::push(&identifier, &message);
            Err(warning_flow(&identifier, message))
        }
    }
}

fn print_warning(identifier: &str, message: &str) {
    let (backtrace_enabled, verbose_enabled) =
        with_manager(|mgr| (mgr.backtrace_enabled, mgr.verbose_enabled));

    emit_stderr_line(format!("Warning: {message}"));
    if identifier != DEFAULT_IDENTIFIER {
        emit_stderr_line(format!("identifier: {identifier}"));
    }

    if verbose_enabled {
        let suppression = if identifier == DEFAULT_IDENTIFIER {
            DEFAULT_IDENTIFIER.to_string()
        } else {
            identifier.to_string()
        };
        emit_stderr_line(format!(
            "(Type \"warning('off','{suppression}')\" to suppress this warning.)"
        ));
    }

    if backtrace_enabled {
        let bt = std::backtrace::Backtrace::force_capture();
        emit_stderr_line(format!("{bt}"));
    }
}

fn emit_stderr_line(line: String) {
    tracing::warn!("{line}");
    record_console_output(ConsoleStream::Stderr, line);
}

fn reissue_exception(mex: &runmat_builtins::MException) -> crate::BuiltinResult<Value> {
    let identifier = normalize_identifier(&mex.identifier);
    emit_warning(&identifier, &mex.message, &[])
}

fn handle_command(command: Command, rest: &[Value]) -> crate::BuiltinResult<Value> {
    match command {
        Command::SetMode(mode) => set_mode_command(mode, rest),
        Command::Default => default_command(rest),
        Command::Reset => {
            if !rest.is_empty() {
                return Err(warning_default_error(
                    "warning: 'reset' does not accept additional arguments",
                ));
            }
            with_manager(WarningManager::reset);
            Ok(Value::Num(0.0))
        }
        Command::Query => query_command(rest),
        Command::Status => status_command(rest),
        Command::Backtrace => backtrace_command(rest),
    }
}

fn set_mode_command(mode: WarningMode, rest: &[Value]) -> crate::BuiltinResult<Value> {
    let identifier = if rest.is_empty() {
        "all".to_string()
    } else if rest.len() == 1 {
        value_to_string("warning", &rest[0])?
    } else {
        return Err(warning_default_error(
            "warning: too many input arguments for state change",
        ));
    };

    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return with_manager(|mgr| {
            let previous = mgr.default_mode;
            let value = Value::Struct(mgr.state_struct_for("all", WarningRule::new(previous)));
            mgr.set_global_mode(mode);
            Ok(value)
        });
    }

    if trimmed.eq_ignore_ascii_case("last") {
        let last_identifier = with_manager(|mgr| mgr.last_warning.clone());
        let Some((identifier, _)) = last_identifier else {
            return Err(warning_default_error(
                "warning: there is no last warning identifier to target",
            ));
        };
        return set_mode_for_identifier(mode, &identifier);
    }

    if trimmed.eq_ignore_ascii_case("backtrace") || trimmed.eq_ignore_ascii_case("verbose") {
        return set_mode_for_special_mode(mode, trimmed);
    }

    let normalized = normalize_identifier(trimmed);
    set_mode_for_identifier(mode, &normalized)
}

fn set_mode_for_identifier(mode: WarningMode, identifier: &str) -> crate::BuiltinResult<Value> {
    with_manager(|mgr| {
        let previous = mgr.lookup_mode(identifier);
        let value = Value::Struct(mgr.state_struct_for(identifier, previous));
        mgr.set_identifier_mode(identifier, mode);
        Ok(value)
    })
}

fn reset_identifier_to_default(identifier: &str) -> crate::BuiltinResult<Value> {
    with_manager(|mgr| {
        let previous = mgr.lookup_mode(identifier);
        let value = Value::Struct(mgr.state_struct_for(identifier, previous));
        mgr.clear_identifier(identifier);
        Ok(value)
    })
}

fn set_mode_for_special_mode(mode: WarningMode, mode_name: &str) -> crate::BuiltinResult<Value> {
    let mode_lower = mode_name.trim().to_ascii_lowercase();
    if !matches!(mode, WarningMode::On | WarningMode::Off) {
        return Err(warning_default_error(format!(
            "warning: only 'on' or 'off' are valid states for '{mode_lower}'"
        )));
    }

    with_manager(|mgr| {
        let previous_enabled = if mode_lower == "backtrace" {
            let prev = mgr.backtrace_enabled;
            mgr.backtrace_enabled = matches!(mode, WarningMode::On);
            prev
        } else if mode_lower == "verbose" {
            let prev = mgr.verbose_enabled;
            mgr.verbose_enabled = matches!(mode, WarningMode::On);
            prev
        } else {
            return Err(warning_default_error(format!(
                "warning: unknown mode '{}'; expected 'backtrace' or 'verbose'",
                mode_name
            )));
        };

        let previous_state = if previous_enabled { "on" } else { "off" };
        let value = state_struct_value(&mode_lower, previous_state);
        Ok(value)
    })
}

fn default_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    match rest.len() {
        0 => {
            let snapshot = with_manager(|mgr| {
                let snapshot = mgr.snapshot();
                mgr.reset_defaults_only();
                snapshot
            });
            structs_to_cell(snapshot)
        }
        1 => {
            let identifier = value_to_string("warning", &rest[0])?;
            let trimmed = identifier.trim();
            if trimmed.eq_ignore_ascii_case("all") {
                let snapshot = with_manager(|mgr| {
                    let snapshot = mgr.snapshot();
                    mgr.reset_defaults_only();
                    snapshot
                });
                return structs_to_cell(snapshot);
            }
            if trimmed.eq_ignore_ascii_case("backtrace") {
                return with_manager(|mgr| {
                    let previous = if mgr.backtrace_enabled { "on" } else { "off" };
                    mgr.backtrace_enabled = false;
                    Ok(state_struct_value("backtrace", previous))
                });
            }
            if trimmed.eq_ignore_ascii_case("verbose") {
                return with_manager(|mgr| {
                    let previous = if mgr.verbose_enabled { "on" } else { "off" };
                    mgr.verbose_enabled = false;
                    Ok(state_struct_value("verbose", previous))
                });
            }
            if trimmed.eq_ignore_ascii_case("last") {
                let last_identifier = with_manager(|mgr| mgr.last_warning.clone());
                let Some((identifier, _)) = last_identifier else {
                    return Err(warning_default_error(
                        "warning: there is no last warning identifier to reset to default",
                    ));
                };
                return reset_identifier_to_default(&identifier);
            }
            let normalized = normalize_identifier(trimmed);
            reset_identifier_to_default(&normalized)
        }
        _ => Err(warning_default_error(
            "warning: 'default' accepts zero or one identifier argument",
        )), 
    }
}

fn query_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(warning_default_error(
            "warning: 'query' accepts at most one identifier argument",
        ));
    }

    let target = if rest.is_empty() {
        "all".to_string()
    } else {
        value_to_string("warning", &rest[0])?
    };

    with_manager(|mgr| {
        if target.trim().eq_ignore_ascii_case("all") {
            let snapshot = mgr.snapshot();
            let rows = snapshot.len();
            let entries: Vec<Value> = snapshot.into_iter().map(Value::Struct).collect();
            let cell = CellArray::new(entries, rows, 1).map_err(|e| {
                warning_default_error(format!("warning: failed to assemble query cell: {e}"))
            })?;
            Ok(Value::Cell(cell))
        } else if target.trim().eq_ignore_ascii_case("last") {
            if let Some((identifier, message)) = mgr.last_warning.clone() {
                let mut st = StructValue::new();
                st.fields
                    .insert("identifier".to_string(), Value::from(identifier));
                st.fields
                    .insert("message".to_string(), Value::from(message));
                st.fields.insert("state".to_string(), Value::from("last"));
                Ok(Value::Struct(st))
            } else {
                let mut st = StructValue::new();
                st.fields.insert("identifier".to_string(), Value::from(""));
                st.fields.insert("message".to_string(), Value::from(""));
                st.fields.insert("state".to_string(), Value::from("none"));
                Ok(Value::Struct(st))
            }
        } else if target.trim().eq_ignore_ascii_case("backtrace") {
            Ok(state_struct_value(
                "backtrace",
                if mgr.backtrace_enabled { "on" } else { "off" },
            ))
        } else if target.trim().eq_ignore_ascii_case("verbose") {
            Ok(state_struct_value(
                "verbose",
                if mgr.verbose_enabled { "on" } else { "off" },
            ))
        } else {
            let normalized = normalize_identifier(&target);
            let state = mgr.lookup_mode(&normalized);
            Ok(Value::Struct(mgr.state_struct_for(&normalized, state)))
        }
    })
}

fn status_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(warning_default_error(
            "warning: 'status' does not accept additional arguments",
        ));
    }
    let value = query_command(&[])?;
    match &value {
        Value::Cell(cell) => {
            emit_stderr_line("Warning status:".to_string());
            for idx in 0..cell.data.len() {
                let entry = (*cell.data[idx]).clone();
                if let Value::Struct(st) = entry {
                    let identifier = st
                        .fields
                        .get("identifier")
                        .and_then(|v| value_to_string("warning", v).ok())
                        .unwrap_or_default();
                    let state = st
                        .fields
                        .get("state")
                        .and_then(|v| value_to_string("warning", v).ok())
                        .unwrap_or_default();
                    emit_stderr_line(format!("  {identifier}: {state}"));
                }
            }
        }
        Value::Struct(st) => {
            let identifier = st
                .fields
                .get("identifier")
                .and_then(|v| value_to_string("warning", v).ok())
                .unwrap_or_default();
            let state = st
                .fields
                .get("state")
                .and_then(|v| value_to_string("warning", v).ok())
                .unwrap_or_default();
            emit_stderr_line(format!("Warning status -> {identifier}: {state}"));
        }
        _ => {}
    }
    Ok(value)
}

fn backtrace_command(rest: &[Value]) -> crate::BuiltinResult<Value> {
    match rest.len() {
        0 => {
            let state = with_manager(|mgr| if mgr.backtrace_enabled { "on" } else { "off" });
            Ok(Value::from(state))
        }
        1 => {
            let setting = value_to_string("warning", &rest[0])?;
            match setting.trim().to_ascii_lowercase().as_str() {
                "on" => with_manager(|mgr| mgr.backtrace_enabled = true),
                "off" => with_manager(|mgr| mgr.backtrace_enabled = false),
                other => {
                    return Err(warning_default_error(format!(
                        "warning: backtrace mode must be 'on' or 'off', got '{other}'"
                    )))
                }
            }
            Ok(Value::Num(0.0))
        }
        _ => Err(warning_default_error(
            "warning: 'backtrace' accepts zero or one argument",
        )), 
    }
}

fn handle_query_default() -> crate::BuiltinResult<Value> {
    query_command(&[])
}

fn apply_state_value(value: &Value) -> crate::BuiltinResult<()> {
    match value {
        Value::Struct(st) => apply_state_struct(st),
        Value::Cell(cell) => {
            for idx in 0..cell.data.len() {
                let entry = (*cell.data[idx]).clone();
                apply_state_value(&entry)?;
            }
            Ok(())
        }
        other => Err(warning_default_error(format!(
            "warning: expected a struct or cell array of structs, got {other:?}"
        ))),
    }
}

fn apply_state_struct(st: &StructValue) -> crate::BuiltinResult<()> {
    let identifier_value = st
        .fields
        .get("identifier")
        .ok_or_else(|| {
            warning_default_error("warning: state struct must contain an 'identifier' field")
        })?;
    let state_value = st
        .fields
        .get("state")
        .ok_or_else(|| warning_default_error("warning: state struct must contain a 'state' field"))?;
    let identifier_raw = value_to_string("warning", identifier_value)?;
    let state_raw = value_to_string("warning", state_value)?;
    let identifier_trimmed = identifier_raw.trim();
    if identifier_trimmed.eq_ignore_ascii_case("all") {
        if let Some(mode) = parse_mode_keyword(&state_raw) {
            with_manager(|mgr| mgr.set_global_mode(mode));
        } else {
            return Err(warning_default_error(format!("warning: unknown state '{}'", state_raw)));
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("backtrace") {
        let state = state_raw.trim().to_ascii_lowercase();
        match state.as_str() {
            "on" => with_manager(|mgr| mgr.backtrace_enabled = true),
            "off" | "default" => with_manager(|mgr| mgr.backtrace_enabled = false),
            other => return Err(warning_default_error(format!("warning: unknown backtrace state '{}'", other))),
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("verbose") {
        let state = state_raw.trim().to_ascii_lowercase();
        match state.as_str() {
            "on" => with_manager(|mgr| mgr.verbose_enabled = true),
            "off" | "default" => with_manager(|mgr| mgr.verbose_enabled = false),
            other => return Err(warning_default_error(format!("warning: unknown verbose state '{}'", other))),
        }
    } else if identifier_trimmed.eq_ignore_ascii_case("last") {
        let last_identifier = with_manager(|mgr| mgr.last_warning.clone());
        let Some((identifier, _)) = last_identifier else {
            return Err(warning_default_error(
                "warning: there is no last warning identifier to apply state",
            ));
        };
        if state_raw.trim().eq_ignore_ascii_case("default") {
            with_manager(|mgr| mgr.clear_identifier(&identifier));
        } else if let Some(mode) = parse_mode_keyword(&state_raw) {
            with_manager(|mgr| mgr.set_identifier_mode(&identifier, mode));
        } else {
            return Err(warning_default_error(format!("warning: unknown state '{}'", state_raw)));
        }
    } else if state_raw.trim().eq_ignore_ascii_case("default") {
        let normalized = normalize_identifier(identifier_trimmed);
        with_manager(|mgr| mgr.clear_identifier(&normalized));
    } else if let Some(mode) = parse_mode_keyword(&state_raw) {
        let normalized = normalize_identifier(identifier_trimmed);
        with_manager(|mgr| mgr.set_identifier_mode(&normalized, mode));
    } else {
        return Err(warning_default_error(format!("warning: unknown state '{}'", state_raw)));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WarningMode {
    On,
    Off,
    Once,
    Error,
}

impl WarningMode {
    fn keyword(self) -> &'static str {
        match self {
            WarningMode::On => "on",
            WarningMode::Off => "off",
            WarningMode::Once => "once",
            WarningMode::Error => "error",
        }
    }
}

#[derive(Clone, Copy)]
enum Command {
    SetMode(WarningMode),
    Default,
    Reset,
    Query,
    Status,
    Backtrace,
}

fn parse_command(text: &str) -> Option<Command> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" => Some(Command::SetMode(WarningMode::On)),
        "off" => Some(Command::SetMode(WarningMode::Off)),
        "once" => Some(Command::SetMode(WarningMode::Once)),
        "error" => Some(Command::SetMode(WarningMode::Error)),
        "default" => Some(Command::Default),
        "reset" => Some(Command::Reset),
        "query" => Some(Command::Query),
        "status" => Some(Command::Status),
        "backtrace" => Some(Command::Backtrace),
        _ => None,
    }
}

fn parse_mode_keyword(text: &str) -> Option<WarningMode> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" => Some(WarningMode::On),
        "off" => Some(WarningMode::Off),
        "once" => Some(WarningMode::Once),
        "error" => Some(WarningMode::Error),
        _ => None,
    }
}

#[derive(Clone, Copy)]
struct WarningRule {
    mode: WarningMode,
    triggered: bool,
}

impl WarningRule {
    fn new(mode: WarningMode) -> Self {
        Self {
            mode,
            triggered: false,
        }
    }
}

enum WarningAction {
    Suppress,
    Display,
    AsError,
}

struct WarningManager {
    default_mode: WarningMode,
    rules: HashMap<String, WarningRule>,
    once_seen_default: HashSet<String>,
    backtrace_enabled: bool,
    verbose_enabled: bool,
    last_warning: Option<(String, String)>,
}

impl Default for WarningManager {
    fn default() -> Self {
        Self {
            default_mode: WarningMode::On,
            rules: HashMap::new(),
            once_seen_default: HashSet::new(),
            backtrace_enabled: false,
            verbose_enabled: false,
            last_warning: None,
        }
    }
}

impl WarningManager {
    fn set_global_mode(&mut self, mode: WarningMode) {
        self.once_seen_default.clear();
        self.default_mode = mode;
    }

    fn set_identifier_mode(&mut self, identifier: &str, mode: WarningMode) {
        if mode == self.default_mode && !matches!(mode, WarningMode::Once) {
            self.rules.remove(identifier);
        } else {
            self.rules
                .insert(identifier.to_string(), WarningRule::new(mode));
        }
        if matches!(mode, WarningMode::Once) {
            self.once_seen_default.remove(identifier);
        }
    }

    fn clear_identifier(&mut self, identifier: &str) {
        self.rules.remove(identifier);
        self.once_seen_default.remove(identifier);
    }

    fn reset(&mut self) {
        self.default_mode = WarningMode::On;
        self.rules.clear();
        self.once_seen_default.clear();
        self.backtrace_enabled = false;
        self.verbose_enabled = false;
        self.last_warning = None;
    }

    fn reset_defaults_only(&mut self) {
        self.default_mode = WarningMode::On;
        self.once_seen_default.clear();
        self.rules.clear();
        self.backtrace_enabled = false;
        self.verbose_enabled = false;
    }

    fn action_for(&mut self, identifier: &str) -> WarningAction {
        if let Some(rule) = self.rules.get_mut(identifier) {
            return match rule.mode {
                WarningMode::On => WarningAction::Display,
                WarningMode::Off => WarningAction::Suppress,
                WarningMode::Error => WarningAction::AsError,
                WarningMode::Once => {
                    if rule.triggered {
                        WarningAction::Suppress
                    } else {
                        rule.triggered = true;
                        WarningAction::Display
                    }
                }
            };
        }

        match self.default_mode {
            WarningMode::On => WarningAction::Display,
            WarningMode::Off => WarningAction::Suppress,
            WarningMode::Error => WarningAction::AsError,
            WarningMode::Once => {
                if self.once_seen_default.contains(identifier) {
                    WarningAction::Suppress
                } else {
                    self.once_seen_default.insert(identifier.to_string());
                    WarningAction::Display
                }
            }
        }
    }

    fn record_last(&mut self, identifier: &str, message: &str) {
        self.last_warning = Some((identifier.to_string(), message.to_string()));
    }

    fn default_state_struct(&self) -> StructValue {
        let mut st = StructValue::new();
        st.fields
            .insert("identifier".to_string(), Value::from("all".to_string()));
        st.fields.insert(
            "state".to_string(),
            Value::from(self.default_mode.keyword()),
        );
        st
    }

    fn state_struct_for(&self, identifier: &str, rule: WarningRule) -> StructValue {
        let mut st = StructValue::new();
        st.fields.insert(
            "identifier".to_string(),
            Value::from(identifier.to_string()),
        );
        st.fields
            .insert("state".to_string(), Value::from(rule.mode.keyword()));
        st
    }

    fn lookup_mode(&self, identifier: &str) -> WarningRule {
        self.rules
            .get(identifier)
            .copied()
            .unwrap_or_else(|| WarningRule::new(self.default_mode))
    }

    fn snapshot(&self) -> Vec<StructValue> {
        let mut entries = Vec::new();
        entries.push(self.default_state_struct());
        for (id, rule) in self.rules.iter() {
            entries.push(self.state_struct_for(id, *rule));
        }
        entries.push(state_struct(
            "backtrace",
            if self.backtrace_enabled { "on" } else { "off" },
        ));
        entries.push(state_struct(
            "verbose",
            if self.verbose_enabled { "on" } else { "off" },
        ));
        entries
    }
}

fn value_to_string(context: &str, value: &Value) -> crate::BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Ok(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Ok(sa.data[0].clone()),
        Value::CharArray(_) => Err(warning_default_error(format!(
            "{context}: expected scalar char array"
        ))),
        Value::StringArray(_) => Err(warning_default_error(format!(
            "{context}: expected scalar string"
        ))),
        other => String::try_from(other).map_err(|_| {
            warning_default_error(format!(
                "{context}: expected string-like argument, got {other:?}"
            ))
        }),
    }
}

fn is_message_identifier(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || !trimmed.contains(':') {
        return false;
    }
    trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ':' | '_' | '.'))
}

fn normalize_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        DEFAULT_IDENTIFIER.to_string()
    } else if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("MATLAB:{trimmed}")
    }
}

fn state_struct(identifier: &str, state: &str) -> StructValue {
    let mut st = StructValue::new();
    st.fields.insert(
        "identifier".to_string(),
        Value::from(identifier.to_string()),
    );
    st.fields
        .insert("state".to_string(), Value::from(state.to_string()));
    st
}

fn state_struct_value(identifier: &str, state: &str) -> Value {
    Value::Struct(state_struct(identifier, state))
}

fn structs_to_cell(structs: Vec<StructValue>) -> crate::BuiltinResult<Value> {
    let rows = structs.len();
    let values: Vec<Value> = structs.into_iter().map(Value::Struct).collect();
    CellArray::new(values, rows, 1)
        .map(Value::Cell)
        .map_err(|e| warning_default_error(format!("warning: failed to assemble state cell: {e}")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    static TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    fn unwrap_error(flow: crate::RuntimeControlFlow) -> crate::RuntimeError {
        match flow {
            crate::RuntimeControlFlow::Error(err) => err,
            other => panic!("expected error, got {other:?}"),
        }
    }

    fn reset_manager() {
        with_manager(WarningManager::reset);
    }

    fn assert_state_struct(value: &Value, identifier: &str, state: &str) {
        match value {
            Value::Struct(st) => {
                let id = st
                    .fields
                    .get("identifier")
                    .and_then(|v| String::try_from(v).ok())
                    .unwrap_or_default();
                let st_state = st
                    .fields
                    .get("state")
                    .and_then(|v| String::try_from(v).ok())
                    .unwrap_or_default();
                assert_eq!(id, identifier);
                assert_eq!(st_state, state);
            }
            other => panic!("expected state struct, got {other:?}"),
        }
    }

    fn structs_from_value(value: Value) -> Vec<StructValue> {
        match value {
            Value::Cell(cell) => cell
                .data
                .iter()
                .map(|ptr| unsafe { &*ptr.as_raw() }.clone())
                .map(|value| match value {
                    Value::Struct(st) => st,
                    other => panic!("expected struct entry, got {other:?}"),
                })
                .collect(),
            Value::Struct(st) => vec![st],
            other => panic!("expected struct array, got {other:?}"),
        }
    }

    fn field_str(struct_value: &StructValue, field: &str) -> Option<String> {
        struct_value
            .fields
            .get(field)
            .and_then(|value| String::try_from(value).ok())
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn emits_basic_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let result = warning_builtin(vec![Value::from("Hello world!")]).expect("warning ok");
        assert!(matches!(result, Value::Num(_)));
        let last = with_manager(|mgr| mgr.last_warning.clone());
        assert_eq!(
            last,
            Some((DEFAULT_IDENTIFIER.to_string(), "Hello world!".to_string()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn emits_warning_with_identifier_and_format() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let args = vec![
            Value::from("runmat:demo:test"),
            Value::from("value is %d"),
            Value::Int(runmat_builtins::IntValue::I32(7)),
        ];
        warning_builtin(args).expect("warning ok");
        let last = with_manager(|mgr| mgr.last_warning.clone());
        assert_eq!(
            last,
            Some(("runmat:demo:test".to_string(), "value is 7".to_string()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn off_suppresses_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let state =
            warning_builtin(vec![Value::from("off"), Value::from("all")]).expect("state change");
        assert_state_struct(&state, "all", "on");
        warning_builtin(vec![Value::from("Should suppress")]).expect("warning ok");
        let last = with_manager(|mgr| mgr.last_warning.clone());
        assert!(last.is_none());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn once_only_emits_first_warning() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let state =
            warning_builtin(vec![Value::from("once"), Value::from("all")]).expect("state change");
        assert_state_struct(&state, "all", "on");
        warning_builtin(vec![Value::from("First")]).expect("warning ok");
        warning_builtin(vec![Value::from("Second")]).expect("warning ok");
        let last = with_manager(|mgr| mgr.last_warning.clone());
        assert_eq!(
            last,
            Some((DEFAULT_IDENTIFIER.to_string(), "First".to_string()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn error_mode_promotes_to_error() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let previous =
            warning_builtin(vec![Value::from("error"), Value::from("all")]).expect("state change");
        assert_state_struct(&previous, "all", "on");
        let err = unwrap_error(
            warning_builtin(vec![Value::from("Promoted")]).expect_err("should error"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
        assert_eq!(err.message(), "Promoted");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_returns_state_struct() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("off"), Value::from("runmat:demo:test")])
            .expect("state change");
        let value = warning_builtin(vec![Value::from("query"), Value::from("runmat:demo:test")])
            .expect("query ok");
        match value {
            Value::Struct(st) => {
                let state = st.fields.get("state").unwrap();
                assert_eq!(String::try_from(state).unwrap(), "off".to_string());
            }
            other => panic!("expected struct, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn state_struct_restores_mode() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let snapshot =
            warning_builtin(vec![Value::from("query"), Value::from("all")]).expect("query all");
        warning_builtin(vec![Value::from("off"), Value::from("all")]).expect("off all");
        warning_builtin(vec![snapshot]).expect("restore");
        let state = warning_builtin(vec![
            Value::from("query"),
            Value::from("runmat:demo:restored"),
        ])
        .expect("query")
        .expect_struct();
        assert_eq!(
            String::try_from(state.fields.get("state").unwrap()).unwrap(),
            "on".to_string()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_backtrace_via_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let prev = warning_builtin(vec![Value::from("on"), Value::from("backtrace")])
            .expect("enable backtrace");
        assert_state_struct(&prev, "backtrace", "off");
        assert!(with_manager(|mgr| mgr.backtrace_enabled));
        let prev = warning_builtin(vec![Value::from("off"), Value::from("backtrace")])
            .expect("disable backtrace");
        assert_state_struct(&prev, "backtrace", "on");
        assert!(!with_manager(|mgr| mgr.backtrace_enabled));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_verbose_via_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let prev = warning_builtin(vec![Value::from("on"), Value::from("verbose")])
            .expect("enable verbose");
        assert_state_struct(&prev, "verbose", "off");
        assert!(with_manager(|mgr| mgr.verbose_enabled));
        let prev = warning_builtin(vec![Value::from("off"), Value::from("verbose")])
            .expect("disable verbose");
        assert_state_struct(&prev, "verbose", "on");
        assert!(!with_manager(|mgr| mgr.verbose_enabled));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn special_mode_rejects_invalid_state() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let err = unwrap_error(
            warning_builtin(vec![Value::from("once"), Value::from("backtrace")])
                .expect_err("invalid state"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
        assert!(
            err.message().contains("only 'on' or 'off'"),
            "unexpected error message: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn set_mode_last_requires_identifier() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let err = unwrap_error(
            warning_builtin(vec![Value::from("off"), Value::from("last")])
                .expect_err("missing last"),
        );
        assert_eq!(err.identifier(), Some(DEFAULT_IDENTIFIER));
        assert!(
            err.message().contains("no last warning identifier"),
            "unexpected error: {}",
            err.message()
        );
        warning_builtin(vec![Value::from("Hello!")]).expect("emit warning");
        let previous =
            warning_builtin(vec![Value::from("off"), Value::from("last")]).expect("disable last");
        assert_state_struct(&previous, DEFAULT_IDENTIFIER, "on");
        let last_mode = with_manager(|mgr| mgr.lookup_mode(DEFAULT_IDENTIFIER).mode);
        assert!(matches!(last_mode, WarningMode::Off));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_returns_snapshot() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![
            Value::from("off"),
            Value::from("runmat:demo:snapshot"),
        ])
        .expect("state change");
        let snapshot = warning_builtin(vec![Value::from("default")]).expect("default");
        let structs = structs_from_value(snapshot);
        assert!(structs.iter().any(|st| {
            field_str(st, "identifier").as_deref() == Some("all")
                && field_str(st, "state").as_deref() == Some("on")
        }));
        assert!(structs.iter().any(|st| {
            field_str(st, "identifier").as_deref() == Some("runmat:demo:snapshot")
                && field_str(st, "state").as_deref() == Some("off")
        }));
        assert!(structs
            .iter()
            .any(|st| { field_str(st, "identifier").as_deref() == Some("backtrace") }));
        assert!(structs
            .iter()
            .any(|st| { field_str(st, "identifier").as_deref() == Some("verbose") }));
        assert!(with_manager(|mgr| mgr.rules.is_empty()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn default_special_modes_reset() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("on"), Value::from("verbose")]).expect("enable verbose");
        warning_builtin(vec![Value::from("on"), Value::from("backtrace")])
            .expect("enable backtrace");
        let verbose_prev =
            warning_builtin(vec![Value::from("default"), Value::from("verbose")]).expect("default");
        assert_state_struct(&verbose_prev, "verbose", "on");
        assert!(!with_manager(|mgr| mgr.verbose_enabled));
        let backtrace_prev =
            warning_builtin(vec![Value::from("default"), Value::from("backtrace")])
                .expect("default");
        assert_state_struct(&backtrace_prev, "backtrace", "on");
        assert!(!with_manager(|mgr| mgr.backtrace_enabled));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn query_backtrace_and_verbose() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        warning_builtin(vec![Value::from("on"), Value::from("verbose")]).expect("enable verbose");
        let verbose = warning_builtin(vec![Value::from("query"), Value::from("verbose")])
            .expect("query verbose");
        assert_state_struct(&verbose, "verbose", "on");
        let backtrace = warning_builtin(vec![Value::from("query"), Value::from("backtrace")])
            .expect("query backtrace");
        assert_state_struct(&backtrace, "backtrace", "off");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn apply_state_struct_special_modes() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_manager();
        let mut backtrace = StructValue::new();
        backtrace
            .fields
            .insert("identifier".to_string(), Value::from("backtrace"));
        backtrace
            .fields
            .insert("state".to_string(), Value::from("on"));
        warning_builtin(vec![Value::Struct(backtrace)]).expect("apply backtrace");
        assert!(with_manager(|mgr| mgr.backtrace_enabled));

        let mut verbose = StructValue::new();
        verbose
            .fields
            .insert("identifier".to_string(), Value::from("verbose"));
        verbose
            .fields
            .insert("state".to_string(), Value::from("default"));
        warning_builtin(vec![Value::Struct(verbose)]).expect("apply verbose");
        assert!(!with_manager(|mgr| mgr.verbose_enabled));
    }

    trait ExpectStruct {
        fn expect_struct(self) -> StructValue;
    }

    impl ExpectStruct for Value {
        fn expect_struct(self) -> StructValue {
            match self {
                Value::Struct(st) => st,
                other => panic!("expected struct, got {other:?}"),
            }
        }
    }

    pub(crate) mod doc_tests {
        use super::*;
        use crate::builtins::common::test_support;

        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
        #[test]
        fn doc_examples_present() {
            let blocks = test_support::doc_examples(DOC_MD);
            assert!(!blocks.is_empty());
        }
    }
}
