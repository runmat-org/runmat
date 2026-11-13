//! Typed ODE options construction; solver admission owns option-value semantics.
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, OdeOptionField, ODESET_DESCRIPTOR, ODESET_ERROR_INVALID_ARGUMENT,
    ODESET_ERROR_INVALID_INPUT, ODESET_ERROR_UNKNOWN_OPTION, ODESET_INTEGER_OPTION_EXTENSION,
    ODESET_MAX_STEPS_EXTENSION, ODESET_RESIDENT_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

mod builder;
#[cfg(test)]
mod tests;

const NAME: &str = "odeset";

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: impl AsRef<str>) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("odeset: {}", detail.as_ref())).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn decode_name(name: &str) -> BuiltinResult<OdeOptionField> {
    if let Some(field) = OdeOptionField::from_public_name(name) {
        return Ok(field);
    }
    let mut matches = OdeOptionField::ALL.into_iter().filter(|field| {
        !name.is_empty()
            && field
                .public_name()
                .get(..name.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(name))
    });
    let Some(field) = matches.next() else {
        return Err(error(
            &ODESET_ERROR_UNKNOWN_OPTION,
            format!("unsupported ODE option '{name}'"),
        ));
    };
    if matches.next().is_some() {
        return Err(error(
            &ODESET_ERROR_UNKNOWN_OPTION,
            format!("ambiguous ODE option '{name}'"),
        ));
    }
    Ok(field)
}

#[runtime_builtin(
    name = "odeset",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::ode::odeset"
)]
async fn odeset_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &ODESET_DESCRIPTOR,
        &ODESET_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let mut values: [Option<&Value>; OdeOptionField::ALL.len()] = [None; OdeOptionField::ALL.len()];
    let mut index = 0;
    // Old/new option structures (or numeric []) merge left-to-right. In a
    // structure merge only nonempty values replace earlier settings.
    while let Some(value) = args.get(index) {
        if let Value::Struct(options) = value {
            for (name, value) in &options.fields {
                let field = decode_name(name)?;
                if !crate::builtins::common::validation::value_is_empty(value) {
                    set(&mut values, field, value);
                }
            }
            index += 1;
        } else if is_empty_options(value) {
            index += 1;
        } else {
            break;
        }
        if index == 2 {
            if args.len() != 2 {
                return Err(error(
                    &ODESET_ERROR_INVALID_ARGUMENT,
                    "the structure merge form requires exactly oldopts and newopts",
                ));
            }
            break;
        }
    }
    if !(args.len() - index).is_multiple_of(2) {
        return Err(error(
            &ODESET_ERROR_INVALID_ARGUMENT,
            "expected options structures followed by complete name/value pairs",
        ));
    }
    while index < args.len() {
        let name = crate::builtins::common::validation::value_text_scalar_string(&args[index])
            .ok_or_else(|| {
                error(
                    &ODESET_ERROR_INVALID_INPUT,
                    "option names must be nonempty text scalars",
                )
            })?;
        let field = decode_name(&name)?;
        // Unlike struct merge, explicit name/value [] resets this field.
        set(&mut values, field, &args[index + 1]);
        index += 2;
    }
    for (field, value) in OdeOptionField::ALL.into_iter().zip(values) {
        let Some(value) = value else { continue };
        if crate::builtins::common::validation::value_is_empty(value) {
            continue;
        }
        if field == OdeOptionField::MaxSteps {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ODESET_MAX_STEPS_EXTENSION,
                NAME,
            )?;
        }
        if crate::builtins::common::validation::value_contains_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ODESET_INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
        if crate::builtins::common::validation::value_contains_explicit_gpu(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &ODESET_RESIDENT_OPTION_EXTENSION,
                NAME,
            )?;
        }
    }
    if args.is_empty() && crate::output_count::current_output_count() == Some(0) {
        crate::console::record_console_line(
            crate::console::ConsoleStream::Stdout,
            "RelTol: positive scalar {1e-3}\nAbsTol: positive scalar {1e-6}\nInitialStep: positive scalar {automatic}\nMaxStep: positive scalar {automatic}\nRefine: positive integer {ode45: 4; ode23/ode15s: 1}\nMaxSteps: positive integer {100000} (RunMat extension)"
        )?;
    }
    // args retains the original roots until final, separately admitted
    // structure publication. No intermediate map or payload gather is needed.
    builder::finish(runtime.as_ref(), values)
}

fn is_empty_options(value: &Value) -> bool {
    matches!(value, Value::Tensor(tensor) if tensor.is_empty())
}

fn set<'a>(
    values: &mut [Option<&'a Value>; OdeOptionField::ALL.len()],
    field: OdeOptionField,
    value: &'a Value,
) {
    let index = OdeOptionField::ALL
        .iter()
        .position(|candidate| *candidate == field)
        .expect("every recognized field has one canonical slot");
    values[index] = Some(value);
}
