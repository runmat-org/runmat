use runmat_builtins::MFILENAME_ERROR_TOO_MANY_INPUTS;
use runmat_value::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MfilenameMode {
    Name,
    Fullpath,
    Class,
}

fn text_value(value: &Value) -> crate::BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        Value::CharArray(array) if array.rows == 1 => {
            super::text::decode_utf16_with_error(array.data(), || {
                crate::build_runtime_error("mfilename: option is not valid UTF-16 text")
                    .with_builtin("mfilename")
                    .build()
            })
            .map(Some)
        }
        _ => Ok(None),
    }
}

fn mode_from_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<MfilenameMode> {
    if args.len() > 1 {
        return Err(crate::runtime_descriptor_error(
            "mfilename",
            &MFILENAME_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let Some(option) = args.first().map(text_value).transpose()?.flatten() else {
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

#[cfg(test)]
pub(crate) fn dispatch_mfilename(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_mfilename_arguments(crate::call::FunctionArgumentView::from_slice(&args))
}

fn dispatch_mfilename_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    let mode = mode_from_args(args)?;
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
    Ok(Value::string(result))
}

#[runmat_macros::runtime_builtin(
    name = "mfilename",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::mfilename"
)]
pub fn mfilename_builtin_registered(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    dispatch_mfilename_arguments(args.view())
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
        assert_eq!(value, Value::String("demo_file".to_string().into()));
    }

    #[test]
    fn mfilename_fullpath_strips_extension() {
        let _guard = crate::source_context::replace_current_source_context(
            Some("/tmp/demo_file.m"),
            Some("name = mfilename('fullpath');"),
        );
        let value = dispatch_mfilename(vec![Value::String("fullpath".to_string().into())])
            .expect("mfilename fullpath succeeds");
        assert_eq!(value, Value::String("/tmp/demo_file".to_string().into()));
    }

    #[test]
    fn mfilename_class_and_unknown_options_preserve_existing_modes() {
        let _guard = crate::source_context::replace_current_source_context(
            Some("/project/probe.m"),
            Some("name = mfilename();"),
        );
        assert_eq!(
            dispatch_mfilename(vec![Value::from("class")]).expect("class mode"),
            Value::String(String::new().into()),
        );
        assert_eq!(
            dispatch_mfilename(vec![Value::from("unsupported")]).expect("default mode"),
            Value::String("probe".into()),
        );
        assert_eq!(
            dispatch_mfilename(vec![Value::Num(7.0)]).expect("non-text default mode"),
            Value::String("probe".into()),
        );
        let error = dispatch_mfilename(vec![Value::from("class"), Value::from("fullpath")])
            .expect_err("two options");
        assert_eq!(error.identifier(), Some("RunMat:TooManyInputs"));
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
            Value::String("outer_file".to_string().into())
        );
        assert_eq!(
            dispatch_mfilename(vec![Value::String("fullpath".to_string().into())])
                .expect("mfilename fullpath succeeds"),
            Value::String("/project/outer_file".to_string().into())
        );
    }
}
