use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_value::Value;
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
    path_to_display_string(&path)
}

fn path_to_display_string(path: &Path) -> String {
    let text = path.to_string_lossy().to_string();
    #[cfg(windows)]
    {
        strip_windows_verbatim_prefix(&text)
    }
    #[cfg(not(windows))]
    {
        text
    }
}

#[cfg(windows)]
fn strip_windows_verbatim_prefix(path: &str) -> String {
    if let Some(stripped) = path.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{stripped}")
    } else if let Some(stripped) = path.strip_prefix(r"\\?\") {
        stripped.to_string()
    } else {
        path.to_string()
    }
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
            .map(|source| file_stem_without_extension(source.lexical_file_name()))
            .unwrap_or_default(),
        MfilenameMode::Fullpath => crate::source_context::current_source_info()
            .map(|source| {
                let path = source
                    .lexical_file_fullpath_name()
                    .unwrap_or_else(|| source.lexical_file_name());
                path_without_extension(path)
            })
            .unwrap_or_default(),
        MfilenameMode::Class => crate::class_access_context()
            .map(|identity| identity.display_name().to_owned())
            .unwrap_or_default(),
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

    #[test]
    fn mfilename_uses_lexical_file_while_nested_source_remains_distinct() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _runtime = runtime.enter();
        let outer = crate::source_context::lease_source_context_in_context(
            &runtime,
            "outer_file.m",
            Some(std::sync::Arc::from("/project/outer_file.m")),
            "value = eval('mfilename()');",
        )
        .expect("outer source identity");
        let _outer = crate::source_context::replace_current_source_id_in_context(
            &runtime,
            Some(outer.source_id()),
        );
        let nested = crate::source_context::lease_nested_source_context_in_context(
            &runtime,
            "<eval>",
            "mfilename()",
            Some(outer.source_id()),
        )
        .expect("nested source identity");
        let _nested = crate::source_context::replace_current_source_id_in_context(
            &runtime,
            Some(nested.source_id()),
        );

        let current = crate::source_context::current_source_info().expect("nested source active");
        assert_eq!(current.name.as_ref(), "<eval>");
        assert_eq!(current.text.as_ref(), "mfilename()");
        assert_eq!(
            dispatch_mfilename(Vec::new()).expect("mfilename succeeds"),
            Value::String("outer_file".to_string())
        );
        assert_eq!(
            dispatch_mfilename(vec![Value::String("fullpath".to_string())])
                .expect("mfilename fullpath succeeds"),
            Value::String("/project/outer_file".to_string())
        );
    }
}
