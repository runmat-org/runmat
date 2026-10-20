//! MATLAB-compatible `weboptions` builtin for constructing HTTP client options.

use runmat_builtins::{
    BuiltinErrorDescriptor, WebOptionField, WEBOPTIONS_DESCRIPTOR, WEBOPTIONS_ERROR_FLOW,
    WEBOPTIONS_ERROR_INVALID_CREDENTIALS, WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
    WEBOPTIONS_ERROR_INVALID_OPTION_VALUE, WEBOPTIONS_ERROR_MISSING_OPTION_VALUE,
    WEBOPTIONS_ERROR_OUTPUTS, WEBOPTIONS_ERROR_UNKNOWN_OPTION, WEBOPTIONS_EXPLICIT_GPU_EXTENSION,
    WEBOPTIONS_QUERY_PARAMETERS_EXTENSION, WEBOPTIONS_STRUCT_COPY_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const DEFAULT_TIMEOUT_SECONDS: f64 = 5.0;
const MAX_TIMEOUT_SECONDS: f64 = 2147.483647;
const BUILTIN_NAME: &str = "weboptions";

#[runtime_builtin(
    name = "weboptions",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::http::weboptions"
)]
async fn weboptions_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &WEBOPTIONS_DESCRIPTOR,
        &WEBOPTIONS_ERROR_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if rest
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WEBOPTIONS_EXPLICIT_GPU_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(rest.first(), Some(Value::Struct(_))) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &WEBOPTIONS_STRUCT_COPY_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let prepared = PreparedWebOptions::parse(rest.view())?;
    prepared.preflight_host_values(rest.view())?;
    let source = &rest;
    let gathered = rest
        .try_map_preserving_roots_async(rest.len(), runtime.as_ref(), |index| async move {
            gather_if_needed_async(&source[index])
                .await
                .map_err(|flow| {
                    remap_weboptions_flow(&WEBOPTIONS_ERROR_FLOW, flow, |err| {
                        format!("weboptions: {}", err.message())
                    })
                })
        })
        .await?;
    let mut options = default_options_struct();
    if let Some(Value::Struct(source)) = gathered.first() {
        for (name, field) in &prepared.struct_fields {
            let value = source
                .fields
                .get(name)
                .expect("prepared struct field remains present after gather");
            set_option_field(&mut options, *field, value)?;
        }
    }
    for (field, value_index) in prepared.pairs {
        set_option_field(&mut options, field, &gathered[value_index])?;
    }

    let options = options.finish(runtime.as_ref())?;
    let Value::Struct(value) = options.value() else {
        unreachable!("options publishes a structure");
    };
    validate_credentials(value)?;
    Ok(options)
}

struct PreparedWebOptions {
    struct_fields: Vec<(String, WebOptionField)>,
    pairs: Vec<(WebOptionField, usize)>,
}

impl PreparedWebOptions {
    fn parse(rest: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        let mut struct_fields = Vec::new();
        let mut index = 0;
        if let Some(Value::Struct(source)) = rest.first() {
            for name in source.fields.keys() {
                struct_fields.push((name.to_owned(), parse_option_field(name)?));
            }
            index = 1;
        }
        let mut pairs = Vec::new();
        while index < rest.len() {
            let name = expect_string_scalar(
                &rest[index],
                "weboptions: option names must be character vectors or string scalars",
                &WEBOPTIONS_ERROR_INVALID_OPTION_NAME,
            )?;
            let value_index = index + 1;
            if value_index >= rest.len() {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_MISSING_OPTION_VALUE,
                    "weboptions: missing value for name-value argument",
                ));
            }
            pairs.push((parse_option_field(&name)?, value_index));
            index += 2;
        }
        if struct_fields
            .iter()
            .any(|(_, field)| *field == WebOptionField::QueryParameters)
            || pairs
                .iter()
                .any(|(field, _)| *field == WebOptionField::QueryParameters)
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &WEBOPTIONS_QUERY_PARAMETERS_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        Ok(Self {
            struct_fields,
            pairs,
        })
    }

    fn preflight_host_values(
        &self,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<()> {
        let mut preview = default_options_struct();
        if let Some(Value::Struct(source)) = rest.first() {
            for (name, field) in &self.struct_fields {
                let value = &source.fields[name];
                if !crate::dispatcher::value_contains_gpu(value) {
                    set_option_field(&mut preview, *field, value)?;
                }
            }
        }
        for (field, value_index) in &self.pairs {
            let value = &rest[*value_index];
            if !crate::dispatcher::value_contains_gpu(value) {
                set_option_field(&mut preview, *field, value)?;
            }
        }
        Ok(())
    }
}

fn parse_option_field(name: &str) -> BuiltinResult<WebOptionField> {
    WebOptionField::decode(name).ok_or_else(|| {
        weboptions_error_with(
            &WEBOPTIONS_ERROR_UNKNOWN_OPTION,
            format!("weboptions: unknown option '{}'", name),
        )
    })
}

fn weboptions_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_weboptions_flow<F>(
    error: &'static BuiltinErrorDescriptor,
    err: RuntimeError,
    message: F,
) -> RuntimeError
where
    F: FnOnce(&RuntimeError) -> String,
{
    let mut builder = build_runtime_error(message(&err))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

mod options;
mod builder;
use options::*;

#[cfg(test)]
mod tests;
