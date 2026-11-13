use super::*;

pub(super) struct UnzipRequest {
    pub(super) archive: runmat_builtins::UnzipSourceRoute,
    pub(super) output_folder: Option<String>,
}

pub(super) fn parse_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<UnzipRequest> {
    match args.len() {
        1 => Ok(UnzipRequest {
            archive: runmat_builtins::UnzipSourceRoute::parse(value_to_string_scalar(
                &args[0],
                &UNZIP_ERROR_FILENAME,
            )?),
            output_folder: None,
        }),
        2 => Ok(UnzipRequest {
            archive: runmat_builtins::UnzipSourceRoute::parse(value_to_string_scalar(
                &args[0],
                &UNZIP_ERROR_FILENAME,
            )?),
            output_folder: Some(value_to_string_scalar(
                &args[1],
                &UNZIP_ERROR_OUTPUT_FOLDER,
            )?),
        }),
        0 => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: ZIP filename is required",
        )),
        _ => Err(unzip_error_with(
            &UNZIP_ERROR_INVALID_ARGUMENT,
            "unzip: expected ZIP filename and optional output folder",
        )),
    }
}

fn value_to_string_scalar(
    value: &Value,
    descriptor: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
        .ok_or_else(|| {
            unzip_error_with(
                descriptor,
                "unzip: expected a string scalar or character vector",
            )
        })
        .and_then(|text| {
            if text.trim().is_empty() {
                Err(unzip_error_with(
                    descriptor,
                    "unzip: path arguments must not be empty",
                ))
            } else {
                Ok(text)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::StringArray;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn canonical_scalar_text_shapes_preserve_supported_routes() {
        let string = Value::String("sample.zip".into());
        let singleton =
            Value::StringArray(StringArray::new(vec!["sample.zip".into()], vec![1, 1, 1]).unwrap());
        let character = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(
                "sample.zip".chars().collect(),
                vec![1, 10, 1],
            )
            .unwrap(),
        );
        for source in [string, singleton, character] {
            let request = parse_arguments(crate::call::FunctionArgumentView::from_slice(&[source]))
                .expect("text scalar ZIP path");
            assert!(matches!(
                request.archive,
                runmat_builtins::UnzipSourceRoute::LocalPath(_)
            ));
        }
        let request = parse_arguments(crate::call::FunctionArgumentView::from_slice(&[
            Value::String("https://example.invalid/sample.zip".into()),
            Value::CharArray(
                CharArray::from_unicode_scalars_with_shape("out".chars().collect(), vec![1, 3, 1])
                    .unwrap(),
            ),
        ]))
        .expect("text scalar destination");
        assert_eq!(request.output_folder.as_deref(), Some("out"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn paged_and_multiscalar_text_reject_with_owning_identifiers() {
        let paged = Value::CharArray(
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap(),
        );
        let error = parse_arguments(crate::call::FunctionArgumentView::from_slice(
            std::slice::from_ref(&paged),
        ))
        .err()
        .expect("paged source");
        assert_eq!(error.identifier(), UNZIP_ERROR_FILENAME.identifier);
        let error = parse_arguments(crate::call::FunctionArgumentView::from_slice(&[
            Value::String("sample.zip".into()),
            paged,
        ]))
        .err()
        .expect("paged destination");
        assert_eq!(error.identifier(), UNZIP_ERROR_OUTPUT_FOLDER.identifier);
        let multiple = Value::StringArray(
            StringArray::new(vec!["a.zip".into(), "b.zip".into()], vec![1, 2]).unwrap(),
        );
        let error = parse_arguments(crate::call::FunctionArgumentView::from_slice(&[multiple]))
            .err()
            .expect("multiple sources");
        assert_eq!(error.identifier(), UNZIP_ERROR_FILENAME.identifier);
    }
}
