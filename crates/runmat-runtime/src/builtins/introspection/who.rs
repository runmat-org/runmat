//! MATLAB-compatible `who` builtin for RunMat.

#[cfg(test)]
use crate::builtins::introspection::workspace_query::char_array_rows_as_strings;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::introspection::workspace_query::{
    parse_request, WorkspaceQueryOwner, WorkspaceQuerySource,
};
use crate::builtins::io::mat::load::read_mat_file_for_builtin;
#[cfg(test)]
use crate::BuiltinResult;
use crate::{build_runtime_error, make_cell, RuntimeError};

#[runtime_builtin(
    name = "who",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::who"
)]
async fn who_builtin(args: crate::call::BuiltinArguments) -> crate::BuiltinResult<Value> {
    let request = parse_request(args.view(), WorkspaceQueryOwner::Who)?;

    let file_entries = match &request.source {
        WorkspaceQuerySource::File(path) => Some(read_mat_file_for_builtin(path, "who").await?),
        WorkspaceQuerySource::Workspace => None,
    };
    let workspace_entries = if file_entries.is_none() {
        crate::workspace::try_snapshot()
            .map_err(who_flow)?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let entries = file_entries
        .as_ref()
        .map(|entries| entries.value().0.as_slice())
        .unwrap_or(&workspace_entries);

    let global_names: std::collections::HashSet<String> =
        if matches!(request.source, WorkspaceQuerySource::Workspace) {
            crate::workspace::try_global_names()
                .map_err(who_flow)?
                .into_iter()
                .collect()
        } else {
            std::collections::HashSet::new()
        };

    let mut names = Vec::new();
    for (name, _value) in entries {
        if !request.matches(name) {
            continue;
        }
        let is_global = global_names.contains(name);
        if request.only_global && !is_global {
            continue;
        }
        names.push(name.clone());
    }

    names.sort();

    let mut cells = Vec::with_capacity(names.len());
    for name in names.into_iter() {
        cells.push(Value::string(name));
    }
    let rows = cells.len();
    make_cell(cells, rows, 1).map_err(|err| build_runtime_error(err).with_builtin("who").build())
}

fn who_flow(mut err: RuntimeError) -> RuntimeError {
    err.context = err.context.with_builtin("who");
    err
}
#[cfg(test)]
pub(crate) mod tests {
    use super::super::whos::tests::{
        char_array_from_rows as shared_char_array_from_rows,
        ensure_test_resolver as ensure_shared_resolver, set_workspace as shared_set_workspace,
    };
    use super::*;
    use crate::builtins::common::test_support;
    use crate::call_builtin_async;
    use futures::executor::block_on;
    use runmat_value::{CellArray, StringArray, Tensor};
    use tempfile::tempdir;

    fn who_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::who_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn names_from_value(value: Value) -> Vec<String> {
        match value {
            Value::Cell(cell) => cell
                .data()
                .iter()
                .map(|value| match value {
                    Value::String(runmat_value::StringElement::Text(s)) => s.to_string(),
                    Value::CharArray(ca) if ca.rows == 1 => {
                        runmat_value::decode_utf16_code_units(ca.data())
                            .expect("valid UTF-16 fixture")
                            .trim_end_matches([' ', '\0'])
                            .to_string()
                    }
                    other => panic!("expected string entry, got {other:?}"),
                })
                .collect(),
            Value::String(runmat_value::StringElement::Text(s)) => vec![s.to_string()],
            Value::StringArray(sa) => sa
                .data()
                .iter()
                .map(|text| text.as_str().expect("nonmissing name").to_owned())
                .collect(),
            Value::CharArray(ca) => {
                char_array_rows_as_strings(&ca, WorkspaceQueryOwner::Who).unwrap()
            }
            other => panic!("expected cell array result, got {other:?}"),
        }
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_lists_workspace_variables() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        shared_set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Tensor(tensor))],
            &[],
        );

        let value = who_builtin(Vec::new()).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_filters_with_wildcard() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("alpha", Value::Num(1.0)), ("beta", Value::Num(2.0))],
            &[],
        );

        let value = who_builtin(vec![Value::from("a*")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_filters_with_regex() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("foo", Value::Num(1.0)),
                ("bar", Value::Num(2.0)),
                ("baz", Value::Num(3.0)),
            ],
            &[],
        );

        let value = who_builtin(vec![Value::from("-regexp"), Value::from("^ba")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["bar".to_string(), "baz".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_combines_wildcard_and_regex_filters() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Num(2.0)),
                ("gamma", Value::Num(3.0)),
                ("delta", Value::Num(4.0)),
            ],
            &[],
        );

        let value = who_builtin(vec![
            Value::from("a*"),
            Value::from("-regexp"),
            Value::from("ma$"),
        ])
        .expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_filters_global_only() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["shared"],
        );

        let value = who_builtin(vec![Value::from("global")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["shared".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_global_option_is_case_insensitive() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[("Shared", Value::Num(1.0)), ("local", Value::Num(2.0))],
            &["Shared"],
        );

        let value = who_builtin(vec![Value::from("GLOBAL")]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["Shared".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_accepts_char_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let arg = Value::CharArray(shared_char_array_from_rows(&["alpha", "gamma"]));
        let value = who_builtin(vec![arg]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_accepts_string_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let array =
            StringArray::new(vec!["gamma".to_string(), "alpha".to_string()], vec![2, 1]).unwrap();
        let value = who_builtin(vec![Value::StringArray(array)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_accepts_cell_array_arguments() {
        ensure_shared_resolver();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("gamma", Value::Num(3.0)),
                ("omega", Value::Num(4.0)),
            ],
            &[],
        );

        let cell = CellArray::new(vec![Value::from("gamma"), Value::from("alpha")], 2, 1).unwrap();
        let value = who_builtin(vec![Value::Cell(cell)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "gamma".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_rejects_numeric_selection() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::Num(7.0)]).expect_err("who should error");
        let message = error_message(err);
        assert!(
            message.contains("who: selections must"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn who_rejects_resident_numeric_selector_with_owning_error() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        test_support::with_test_provider(|_| {
            let resident = block_on(crate::call_builtin_async("gpuArray", &[Value::Num(7.0)]))
                .expect("resident scalar");
            let error =
                who_builtin(vec![resident.value().clone()]).expect_err("selector is not text");
            assert_eq!(error.context.builtin.as_deref(), Some("who"));
            assert!(error.message().contains("who: selections must"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_rejects_unknown_option() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-bogus")]).expect_err("who should error");
        let message = error_message(err);
        assert!(
            message.contains("unsupported option"),
            "unexpected error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_requires_filename_for_file_option() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-file")]).expect_err("who should error");
        let message = error_message(err);
        assert!(
            message.contains("'-file' requires a filename"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_requires_pattern_for_regexp() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-regexp")]).expect_err("who should error");
        let message = error_message(err);
        assert!(
            message.contains("'-regexp' requires at least one pattern"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_rejects_invalid_regex() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let err = who_builtin(vec![Value::from("-regexp"), Value::from("[")])
            .expect_err("who should error");
        let message = error_message(err);
        assert!(
            message.contains("invalid regular expression"),
            "error: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_returns_empty_column_cell_when_no_match() {
        ensure_shared_resolver();
        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("nothing")]).expect("who");
        match value {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 1);
                assert!(cell.data().is_empty());
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_file_option_reads_mat_file() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("snapshot.mat");
        let path_str = file_path.to_string_lossy().to_string();
        block_on(call_builtin_async(
            "save",
            &[
                Value::from(path_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
            ],
        ))
        .expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("-file"), Value::from(path_str)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_file_option_combines_literal_and_regex_selectors() {
        ensure_shared_resolver();
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        shared_set_workspace(
            &[
                ("alpha", Value::Num(1.0)),
                ("beta", Value::Tensor(tensor.clone())),
                ("gamma", Value::Tensor(tensor)),
            ],
            &[],
        );

        let dir = tempdir().expect("tempdir");
        let stem_path = dir.path().join("snapshot_combo");
        let stem_str = stem_path.to_string_lossy().to_string();
        block_on(call_builtin_async(
            "save",
            &[
                Value::from(stem_str.clone()),
                Value::from("alpha"),
                Value::from("beta"),
                Value::from("gamma"),
            ],
        ))
        .expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![
            Value::from("-file"),
            Value::from(stem_str.clone()),
            Value::from("alpha"),
            Value::from("-regexp"),
            Value::from("^b"),
        ])
        .expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_file_option_adds_mat_extension() {
        ensure_shared_resolver();
        shared_set_workspace(&[("v", Value::Num(2.0))], &[]);

        let dir = tempdir().expect("tempdir");
        let stem_path = dir.path().join("snapshot_no_ext");
        let stem_str = stem_path.to_string_lossy().to_string();
        block_on(call_builtin_async(
            "save",
            &[Value::from(stem_str.clone()), Value::from("v")],
        ))
        .expect("save");

        shared_set_workspace(&[], &[]);
        let value = who_builtin(vec![Value::from("-file"), Value::from(stem_str)]).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["v".to_string()]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_handles_gpu_workspace_entries() {
        ensure_shared_resolver();
        test_support::with_test_provider(|_| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let gpu_value = block_on(crate::call_builtin_async(
                "gpuArray",
                &[Value::Tensor(tensor)],
            ))
            .expect("gpu");
            shared_set_workspace(
                &[
                    ("gpuVar", gpu_value.value().clone()),
                    ("hostVar", Value::Num(5.0)),
                ],
                &[],
            );

            let value = who_builtin(Vec::new()).expect("who");
            let names = names_from_value(value);
            assert_eq!(names, vec!["gpuVar".to_string(), "hostVar".to_string()]);
            shared_set_workspace(&[], &[]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn who_respects_global_filter_with_gpu_variables() {
        ensure_shared_resolver();
        test_support::with_test_provider(|_| {
            shared_set_workspace(&[], &[]);
            let gpu_scalar =
                block_on(crate::call_builtin_async("gpuArray", &[Value::Num(3.0)])).expect("gpu");
            shared_set_workspace(
                &[("shared_gpu", gpu_scalar.value().clone())],
                &["shared_gpu"],
            );

            let value = who_builtin(vec![Value::from("global")]).expect("who");
            let names = names_from_value(value);
            assert_eq!(names, vec!["shared_gpu".to_string()]);
            shared_set_workspace(&[], &[]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn who_handles_workspace_with_wgpu_provider() {
        ensure_shared_resolver();
        let _provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");
        let tensor = Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap();
        let gpu_value = block_on(crate::call_builtin_async(
            "gpuArray",
            &[Value::Tensor(tensor)],
        ))
        .expect("gpuArray");
        shared_set_workspace(&[("wgpuVar", gpu_value.value().clone())], &[]);

        let value = who_builtin(Vec::new()).expect("who");
        let names = names_from_value(value);
        assert_eq!(names, vec!["wgpuVar".to_string()]);
        shared_set_workspace(&[], &[]);
    }
}
