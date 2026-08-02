use runmat_builtins::{ObjectInstance, ResolveContext, Type, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.TestRunnerPlugin",
    category = "testing/plugins",
    summary = "Construct the base test-runner plugin object.",
    type_resolver(test_runner_plugin_type),
    builtin_path = "crate::builtins::testing::plugins"
)]
fn test_runner_plugin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(
            crate::build_runtime_error("TestRunnerPlugin: too many input arguments")
                .with_identifier("RunMat:Testing:InvalidPlugin")
                .with_builtin("TestRunnerPlugin")
                .build(),
        );
    }
    Ok(Value::Object(ObjectInstance::new(
        "matlab.unittest.plugins.TestRunnerPlugin".into(),
    )))
}

fn test_runner_plugin_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some("matlab.unittest.plugins.TestRunnerPlugin".into()),
        shape: Some(vec![Some(1), Some(1)]),
    }
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.CodeCoveragePlugin.forFolder",
    category = "testing/plugins",
    summary = "Construct a source-folder coverage plugin for a TestRunner.",
    type_resolver(code_coverage_plugin_type),
    builtin_path = "crate::builtins::testing::plugins"
)]
fn code_coverage_plugin_for_folder(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if args.is_empty() {
        return Err(plugin_error(
            "CodeCoveragePlugin.forFolder requires at least one folder",
        ));
    }
    let mut folders = Vec::new();
    let mut including_subfolders = false;
    let mut index = 0;
    while index < args.len() {
        if index + 1 < args.len() {
            if let Value::String(name) = &args[index] {
                if name.eq_ignore_ascii_case("IncludingSubfolders") {
                    including_subfolders = bool_value(&args[index + 1])?;
                    index += 2;
                    continue;
                }
            }
        }
        match &args[index] {
            Value::String(folder) if !folder.trim().is_empty() => {
                folders.push(folder.clone());
            }
            Value::CharArray(folder) => {
                let folder = folder.row_string().ok_or_else(|| {
                    plugin_error("coverage folder character arrays must be row vectors")
                })?;
                if folder.trim().is_empty() {
                    return Err(plugin_error("coverage folders must be non-empty"));
                }
                folders.push(folder);
            }
            Value::StringArray(array) => {
                folders.extend(
                    array
                        .data
                        .iter()
                        .filter(|folder| !folder.trim().is_empty())
                        .cloned(),
                );
            }
            _ => {
                return Err(plugin_error(
                    "CodeCoveragePlugin.forFolder folders must be non-empty text",
                ));
            }
        }
        index += 1;
    }
    if folders.is_empty() {
        return Err(plugin_error(
            "CodeCoveragePlugin.forFolder requires at least one folder",
        ));
    }
    let mut object = ObjectInstance::new("matlab.unittest.plugins.CodeCoveragePlugin".into());
    let count = folders.len();
    object.properties.insert(
        "Folders".into(),
        Value::StringArray(
            runmat_builtins::StringArray::new(folders, vec![1, count])
                .expect("coverage folders form a row"),
        ),
    );
    object.properties.insert(
        "IncludingSubfolders".into(),
        Value::Bool(including_subfolders),
    );
    Ok(Value::Object(object))
}

fn code_coverage_plugin_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some("matlab.unittest.plugins.CodeCoveragePlugin".into()),
        shape: Some(vec![Some(1), Some(1)]),
    }
}

fn bool_value(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        _ => Err(plugin_error("IncludingSubfolders must be scalar logical")),
    }
}

fn plugin_error(message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Testing:InvalidPlugin")
        .with_builtin("CodeCoveragePlugin")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_plugin_preserves_folder_policy() {
        let plugin = code_coverage_plugin_for_folder(vec![
            Value::String("src".into()),
            Value::String("IncludingSubfolders".into()),
            Value::Bool(true),
        ])
        .unwrap();
        let Value::Object(plugin) = plugin else {
            panic!("coverage plugin must be an object");
        };
        assert_eq!(
            plugin.class_name,
            "matlab.unittest.plugins.CodeCoveragePlugin"
        );
        assert_eq!(
            plugin.properties.get("IncludingSubfolders"),
            Some(&Value::Bool(true))
        );
    }
}
