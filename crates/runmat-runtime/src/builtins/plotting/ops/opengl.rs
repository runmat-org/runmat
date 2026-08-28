//! MATLAB-compatible `opengl` renderer capability builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

use crate::builtins::plotting::type_resolvers::get_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "opengl";

const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Renderer status string or information struct.",
}];

const INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const INPUTS_COMMANDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "commands",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Commands such as info, data, hardware, hardwarebasic, software, save, autoselect, or neverselect.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "status = opengl()",
        inputs: &INPUTS_NONE,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "info = opengl('info')",
        inputs: &INPUTS_COMMANDS,
        outputs: &OUTPUT_VALUE,
    },
    BuiltinSignatureDescriptor {
        label: "status = opengl(command, ...)",
        inputs: &INPUTS_COMMANDS,
        outputs: &OUTPUT_VALUE,
    },
];

const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.OPENGL.INVALID_ARGUMENT",
    identifier: Some("RunMat:opengl:InvalidArgument"),
    when: "Command text or command combination is invalid.",
    message: "opengl: invalid argument",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_ARGUMENT];

pub const OPENGL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn invalid(detail: impl AsRef<str>) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {}",
        ERROR_INVALID_ARGUMENT.message,
        detail.as_ref()
    ))
    .with_builtin(BUILTIN_NAME);
    if let Some(identifier) = ERROR_INVALID_ARGUMENT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "opengl",
    category = "plotting",
    summary = "Report and select deterministic RunMat renderer compatibility mode.",
    keywords = "opengl,plotting,renderer,graphics",
    suppress_auto_output = true,
    type_resolver(get_type),
    descriptor(crate::builtins::plotting::opengl::OPENGL_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::opengl"
)]
pub fn opengl_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Ok(Value::String("hardware".into()));
    }

    let command = command_text(&args[0])?;
    match command.as_str() {
        "info" | "data" => {
            if args.len() != 1 {
                return Err(invalid(format!(
                    "'{command}' does not accept extra arguments"
                )));
            }
            Ok(renderer_info())
        }
        "hardware" | "hardwarebasic" | "software" | "autoselect" | "neverselect" => {
            if args.len() != 1 {
                return Err(invalid(format!(
                    "'{command}' does not accept extra arguments"
                )));
            }
            Ok(Value::String(status_for_command(&command).into()))
        }
        "save" => {
            validate_save_args(&args[1..])?;
            Ok(Value::String("ok".into()))
        }
        _ => Err(invalid(format!("unsupported opengl command '{command}'"))),
    }
}

fn command_text(value: &Value) -> BuiltinResult<String> {
    super::style::value_as_string(value)
        .map(|text| text.trim().to_ascii_lowercase())
        .filter(|text| !text.is_empty())
        .ok_or_else(|| invalid("command must be non-empty text"))
}

fn validate_save_args(args: &[Value]) -> BuiltinResult<()> {
    if args.len() > 1 {
        return Err(invalid("save accepts at most one renderer mode"));
    }
    if let Some(value) = args.first() {
        let mode = command_text(value)?;
        if !matches!(
            mode.as_str(),
            "hardware" | "hardwarebasic" | "software" | "autoselect" | "none"
        ) {
            return Err(invalid(
                "save mode must be hardware, hardwarebasic, software, autoselect, or none",
            ));
        }
    }
    Ok(())
}

fn status_for_command(command: &str) -> &'static str {
    match command {
        "hardwarebasic" => "hardwarebasic",
        "software" => "software",
        "neverselect" => "neverselect",
        _ => "hardware",
    }
}

fn renderer_info() -> Value {
    let mut st = StructValue::new();
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
    st.insert("Extensions", Value::String(String::new()));
    Value::Struct(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opengl_info_reports_renderer_struct() {
        let value = opengl_builtin(vec![Value::String("info".into())]).unwrap();
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
        assert_eq!(
            opengl_builtin(Vec::new()).unwrap(),
            Value::String("hardware".into())
        );
        assert_eq!(
            opengl_builtin(vec![Value::String("software".into())]).unwrap(),
            Value::String("software".into())
        );
        assert_eq!(
            opengl_builtin(vec![Value::String("hardwarebasic".into())]).unwrap(),
            Value::String("hardwarebasic".into())
        );
        assert_eq!(
            opengl_builtin(vec![
                Value::String("save".into()),
                Value::String("hardware".into())
            ])
            .unwrap(),
            Value::String("ok".into())
        );
        assert_eq!(
            opengl_builtin(vec![
                Value::String("save".into()),
                Value::String("none".into())
            ])
            .unwrap(),
            Value::String("ok".into())
        );
    }
}
