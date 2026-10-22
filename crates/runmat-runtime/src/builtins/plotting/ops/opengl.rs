//! MATLAB-compatible `opengl` renderer capability builtin.

use runmat_builtins::{
    OPENGL_DESCRIPTOR, OPENGL_ERROR_INVALID_ARGUMENT, OPENGL_ERROR_TOO_MANY_OUTPUTS,
};

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

use runmat_builtins::{OpenGlCommand, OpenGlSaveMode};

const BUILTIN_NAME: &str = "opengl";

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        OPENGL_ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = OPENGL_ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "opengl",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::opengl"
)]
pub fn opengl_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let _scope = runtime.as_ref().map(|runtime| runtime.enter());
    crate::output_count::validate_fixed_outputs(
        &OPENGL_DESCRIPTOR,
        &OPENGL_ERROR_TOO_MANY_OUTPUTS,
        "opengl",
    )
    .map_err(super::coordinate_admission::terminal)?;
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "opengl")?;
    work.finish(opengl_impl(arguments.view(), &work))
}

fn opengl_impl(
    args: crate::call::FunctionArgumentView<'_>,
    _work: &super::coordinate_admission::CoordinateWork<'_>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Ok(Value::String("hardware".into()));
    }

    let command = command_text(&args[0])?;
    let parsed = OpenGlCommand::parse(&command)
        .ok_or_else(|| invalid(format!("unsupported opengl command '{command}'")))?;
    match parsed {
        OpenGlCommand::Report => {
            if args.len() != 1 {
                return Err(invalid(format!(
                    "'{command}' does not accept extra arguments"
                )));
            }
            Ok(renderer_info())
        }
        OpenGlCommand::Hardware
        | OpenGlCommand::HardwareBasic
        | OpenGlCommand::Software
        | OpenGlCommand::AutoSelect
        | OpenGlCommand::NeverSelect => {
            if args.len() != 1 {
                return Err(invalid(format!(
                    "'{command}' does not accept extra arguments"
                )));
            }
            Ok(Value::String(
                parsed.status().expect("status command").into(),
            ))
        }
        OpenGlCommand::Save => {
            validate_save_args(args.range(1..)?)?;
            Ok(Value::String("ok".into()))
        }
    }
}

fn command_text(value: &Value) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .map(|text| text.trim().to_ascii_lowercase())
        .filter(|text| !text.is_empty())
        .ok_or_else(|| invalid("command must be non-empty scalar text"))
}

fn validate_save_args(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.len() > 1 {
        return Err(invalid("save accepts at most one renderer mode"));
    }
    if let Some(value) = args.first() {
        let mode = command_text(value)?;
        if OpenGlSaveMode::parse(&mode).is_none() {
            return Err(invalid(
                "save mode must be hardware, hardwarebasic, software, autoselect, or none",
            ));
        }
    }
    Ok(())
}

fn renderer_info() -> Value {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    st.insert(
        "Version",
        Value::String("RunMat renderer compatibility".into()),
    );
    st.insert("Vendor", Value::String("RunMat".into()));
    st.insert("Renderer", Value::String("runmat-plot".into()));
    st.insert("HardwareSupportLevel", Value::String("full".into()));
    st.insert("SupportsGraphicsSmoothing", Value::Bool(true));
    st.insert("SupportsDepthPeelTransparency", Value::Bool(true));
    st.insert("SupportsAlignVertexCenters", Value::Bool(true));
    st.insert("Software", Value::Bool(false));
    st.insert("MaxTextureSize", Value::Num(16384.0));
    st.insert("Extensions", Value::string(String::new()));
    Value::Struct(st.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opengl_info_reports_renderer_struct() {
        let value = opengl_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "info".into(),
            )]),
        )
        .unwrap();
        let Value::Struct(st) = value else {
            panic!("expected info struct");
        };
        assert_eq!(
            st.fields.get("Renderer"),
            Some(&Value::String("runmat-plot".into()))
        );
        assert_eq!(st.fields.get("Software"), Some(&Value::Bool(false)));
    }

    #[test]
    fn opengl_accepts_compatibility_modes() {
        let scalar = runmat_value::StringArray::new(vec!["info".into()], vec![1, 1]).unwrap();
        assert!(matches!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                    scalar
                )])
            )
            .unwrap(),
            Value::Struct(_)
        ));
        let paged = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "info".chars().collect(),
            vec![1, 2, 2],
        )
        .unwrap();
        let error = opengl_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(paged)]),
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:opengl:InvalidArgument"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new())
            )
            .unwrap(),
            Value::String("hardware".into())
        );
        assert_eq!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "software".into()
                )])
            )
            .unwrap(),
            Value::String("software".into())
        );
        assert_eq!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "hardwarebasic".into()
                )])
            )
            .unwrap(),
            Value::String("hardwarebasic".into())
        );
        assert_eq!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("save".into()),
                    Value::String("hardware".into())
                ])
            )
            .unwrap(),
            Value::String("ok".into())
        );
        assert_eq!(
            opengl_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("save".into()),
                    Value::String("none".into())
                ])
            )
            .unwrap(),
            Value::String("ok".into())
        );
    }
}
