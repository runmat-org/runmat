use runmat_value::Value;

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.CodeCoveragePlugin.forFolder",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::plugins"
)]
fn code_coverage_plugin_for_folder(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
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
        // The first argument is always a folder. Option words are reserved
        // only after that position, so a folder literally named "Filter"
        // remains usable as the required first argument.
        if let Some(option) = (index > 0)
            .then(|| CoverageOption::from_value(&args[index]))
            .flatten()
        {
            match option {
                CoverageOption::IncludingSubfolders => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| plugin_error("IncludingSubfolders requires a value"))?;
                    including_subfolders = bool_value(value)?;
                    index += 2;
                    continue;
                }
                unsupported => return Err(unsupported_option_error(unsupported.name())),
            }
        }
        match &args[index] {
            Value::String(runmat_value::StringElement::Text(folder))
                if !folder.trim().is_empty() =>
            {
                folders.push(folder.as_str().to_owned());
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
                        .data()
                        .iter()
                        .map(|folder| {
                            folder.as_str().ok_or_else(|| {
                                plugin_error(
                                    "CodeCoveragePlugin.forFolder folders must be non-empty text",
                                )
                            })
                        })
                        .collect::<BuiltinResult<Vec<_>>>()?
                        .into_iter()
                        .filter(|folder| !folder.trim().is_empty())
                        .map(str::to_owned),
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
    let count = folders.len();
    let fields = [
        (
            "Folders",
            Value::StringArray(
                runmat_value::StringArray::new(folders, vec![1, count])
                    .expect("coverage folders form a row"),
            ),
        ),
        ("IncludingSubfolders", Value::Bool(including_subfolders)),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::UNIT_TEST_CODE_COVERAGE_PLUGIN.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

#[derive(Clone, Copy)]
enum CoverageOption {
    IncludingSubfolders,
    Producing,
    MetricLevel,
    Filter,
}

impl CoverageOption {
    fn from_value(value: &Value) -> Option<Self> {
        let name = match value {
            Value::String(runmat_value::StringElement::Text(name)) => name.as_str(),
            Value::CharArray(chars) => return chars.row_string().as_deref().and_then(Self::parse),
            _ => return None,
        };
        Self::parse(name)
    }

    fn parse(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("IncludingSubfolders") {
            Some(Self::IncludingSubfolders)
        } else if name.eq_ignore_ascii_case("Producing") {
            Some(Self::Producing)
        } else if name.eq_ignore_ascii_case("MetricLevel") {
            Some(Self::MetricLevel)
        } else if name.eq_ignore_ascii_case("Filter") {
            Some(Self::Filter)
        } else {
            None
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::IncludingSubfolders => "IncludingSubfolders",
            Self::Producing => "Producing",
            Self::MetricLevel => "MetricLevel",
            Self::Filter => "Filter",
        }
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

fn unsupported_option_error(name: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!(
        "CodeCoveragePlugin.forFolder does not support the {name} option"
    ))
    .with_identifier("RunMat:Testing:UnsupportedPluginOption")
    .with_builtin("CodeCoveragePlugin.forFolder")
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code_coverage_plugin_for_folder(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::code_coverage_plugin_for_folder(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }

    #[test]
    fn coverage_plugin_preserves_folder_policy() {
        let plugin = code_coverage_plugin_for_folder(
            None,
            vec![
                Value::String("src".into()),
                Value::String("IncludingSubfolders".into()),
                Value::Bool(true),
            ],
        )
        .unwrap();
        let Value::Object(plugin) = plugin.value() else {
            panic!("coverage plugin must be an object");
        };
        assert!(plugin
            .class_name()
            .is(runmat_types::standard::UNIT_TEST_CODE_COVERAGE_PLUGIN));
        assert_eq!(
            plugin.properties().get("IncludingSubfolders"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn character_row_option_name_is_recognized() {
        let option = runmat_value::CharArray::new_row("IncludingSubfolders");
        let plugin = code_coverage_plugin_for_folder(
            None,
            vec![
                Value::String("src".into()),
                Value::CharArray(option),
                Value::Bool(true),
            ],
        )
        .unwrap();
        let Value::Object(plugin) = plugin.value() else {
            panic!("coverage plugin must be an object");
        };
        assert_eq!(
            plugin.properties().get("IncludingSubfolders"),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn unsupported_options_are_not_silently_recorded_as_folders() {
        for name in ["Producing", "MetricLevel", "Filter"] {
            for value in [
                Value::String(name.into()),
                Value::CharArray(runmat_value::CharArray::new_row(name)),
            ] {
                let error = code_coverage_plugin_for_folder(
                    None,
                    vec![
                        Value::String("src".into()),
                        value,
                        Value::String("ignored".into()),
                    ],
                )
                .unwrap_err();
                assert_eq!(
                    error.identifier(),
                    Some("RunMat:Testing:UnsupportedPluginOption")
                );
            }
        }
    }

    #[test]
    fn dangling_supported_option_is_not_a_folder() {
        let error = code_coverage_plugin_for_folder(
            None,
            vec![
                Value::String("src".into()),
                Value::String("IncludingSubfolders".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidPlugin"));
    }

    #[test]
    fn first_folder_may_have_an_option_spelling_and_later_folder_options_still_work() {
        let plugin = code_coverage_plugin_for_folder(
            None,
            vec![
                Value::String("Filter".into()),
                Value::String("src".into()),
                Value::String("IncludingSubfolders".into()),
                Value::Bool(true),
            ],
        )
        .unwrap();
        let Value::Object(plugin) = plugin.value() else {
            panic!("coverage plugin must be an object");
        };
        let Some(Value::StringArray(folders)) = plugin.properties().get("Folders") else {
            panic!("folder array must be retained");
        };
        assert_eq!(
            folders.data(),
            vec!["Filter".to_string(), "src".to_string()]
        );
        assert_eq!(
            plugin.properties().get("IncludingSubfolders"),
            Some(&Value::Bool(true))
        );
    }
}
