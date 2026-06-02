use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use std::path::Path;

const MFILENAME_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Current file name.",
}];

const MFILENAME_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "`fullpath` for full source path or `class` for current method class.",
}];

const MFILENAME_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "name = mfilename()",
        inputs: &[],
        outputs: &MFILENAME_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "name = mfilename(option)",
        inputs: &MFILENAME_INPUTS,
        outputs: &MFILENAME_OUTPUT,
    },
];

pub const MFILENAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MFILENAME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "More than one option argument is provided.",
    message: "mfilename: too many input arguments",
};

pub const MFILENAME_ERRORS: [BuiltinErrorDescriptor; 1] = [MFILENAME_ERROR_TOO_MANY_INPUTS];

pub const MFILENAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MFILENAME_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MFILENAME_ERRORS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MfilenameMode {
    Name,
    Fullpath,
    Class,
}

fn text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Some(array.data.iter().collect()),
        _ => None,
    }
}

fn mode_from_args(args: &[Value]) -> crate::BuiltinResult<MfilenameMode> {
    if args.len() > 1 {
        return Err(crate::runtime_descriptor_error(
            "mfilename",
            &MFILENAME_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let Some(option) = args.first().and_then(text_value) else {
        return Ok(MfilenameMode::Name);
    };
    match option.trim().to_ascii_lowercase().as_str() {
        "fullpath" => Ok(MfilenameMode::Fullpath),
        "class" => Ok(MfilenameMode::Class),
        _ => Ok(MfilenameMode::Name),
    }
}

fn path_without_extension(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let mut path = Path::new(name).to_path_buf();
    path.set_extension("");
    path.to_string_lossy().to_string()
}

fn file_stem_without_extension(name: &str) -> String {
    Path::new(name)
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| path_without_extension(name))
}

pub(crate) fn dispatch_mfilename(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let mode = mode_from_args(&args)?;
    let result = match mode {
        MfilenameMode::Name => crate::source_context::current_source_info()
            .map(|source| file_stem_without_extension(&source.name))
            .unwrap_or_default(),
        MfilenameMode::Fullpath => crate::source_context::current_source_info()
            .map(|source| path_without_extension(&source.name))
            .unwrap_or_default(),
        MfilenameMode::Class => crate::class_access_context().unwrap_or_default(),
    };
    Ok(Value::String(result))
}

#[runmat_macros::runtime_builtin(
    name = "mfilename",
    category = "introspection",
    summary = "Return the filename of the currently running source.",
    descriptor(self::MFILENAME_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::mfilename"
)]
pub fn mfilename_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_mfilename(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mfilename_uses_source_context_name() {
        let _guard = crate::source_context::replace_current_source_context(
            Some("/tmp/demo_file.m"),
            Some("name = mfilename();"),
        );
        let value = dispatch_mfilename(Vec::new()).expect("mfilename succeeds");
        assert_eq!(value, Value::String("demo_file".to_string()));
    }

    #[test]
    fn mfilename_fullpath_strips_extension() {
        let _guard = crate::source_context::replace_current_source_context(
            Some("/tmp/demo_file.m"),
            Some("name = mfilename('fullpath');"),
        );
        let value = dispatch_mfilename(vec![Value::String("fullpath".to_string())])
            .expect("mfilename fullpath succeeds");
        assert_eq!(value, Value::String("/tmp/demo_file".to_string()));
    }
}
