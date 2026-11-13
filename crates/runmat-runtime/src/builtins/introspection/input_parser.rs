//! Minimal MATLAB-compatible `inputParser` name-value parsing.

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{HandleRef, StructValue, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const CLASS_IDENTITY: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("inputParser");
const INPUT_PARSER_NAME: &str = "inputParser";
const ADD_PARAMETER_NAME: &str = "addParameter";
const PARSE_NAME: &str = "parse";
const RESULTS_PROPERTY: &str = "Results";
const PARAMETERS_PROPERTY: &str = "__runmat_inputParser_parameters";

use runmat_builtins::{
    INPUT_PARSER_ERROR_DUPLICATE_PARAMETER, INPUT_PARSER_ERROR_INVALID_PARSER,
    INPUT_PARSER_ERROR_NAME_VALUE_PAIRS, INPUT_PARSER_ERROR_PARAMETER_NAME,
    INPUT_PARSER_ERROR_UNKNOWN_PARAMETER,
};

fn input_parser_error(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    input_parser_error_with_detail(builtin, error, "")
}

fn input_parser_error_with_detail(
    builtin: &'static str,
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref().trim();
    let message = if detail.is_empty() {
        error.message.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn text_scalar<'a>(
    value: &'a Value,
    builtin: &'static str,
) -> BuiltinResult<std::borrow::Cow<'a, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(std::borrow::Cow::Borrowed(text.as_str()))
        }
        Value::StringArray(strings)
            if strings.data().len() == 1 && !strings.data()[0].is_missing() =>
        {
            Ok(std::borrow::Cow::Borrowed(
                strings.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            ))
        }
        Value::CharArray(chars) if chars.rows == 1 => Ok(std::borrow::Cow::Owned(
            super::text::decode_utf16_with_error(chars.data(), || {
                input_parser_error(builtin, &INPUT_PARSER_ERROR_PARAMETER_NAME)
            })?,
        )),
        _ => Err(input_parser_error(
            builtin,
            &INPUT_PARSER_ERROR_PARAMETER_NAME,
        )),
    }
}

fn validate_parameter_name(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::String(runmat_value::StringElement::Text(_)) => Ok(()),
        Value::StringArray(strings)
            if strings.data().len() == 1 && !strings.data()[0].is_missing() =>
        {
            Ok(())
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            super::text::decode_utf16_with_error(chars.data(), || {
                input_parser_error(PARSE_NAME, &INPUT_PARSER_ERROR_PARAMETER_NAME)
            })?;
            Ok(())
        }
        _ => Err(input_parser_error(
            PARSE_NAME,
            &INPUT_PARSER_ERROR_PARAMETER_NAME,
        )),
    }
}

fn parameter_name_matches(value: &Value, name: &str) -> bool {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => value.as_str() == name,
        Value::StringArray(strings)
            if strings.data().len() == 1 && !strings.data()[0].is_missing() =>
        {
            strings.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                == name
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            runmat_value::decode_utf16_code_units(chars.data()).is_ok_and(|text| text == name)
        }
        _ => false,
    }
}

fn parser_handle(value: &Value, builtin: &'static str) -> BuiltinResult<HandleRef> {
    match value {
        Value::HandleObject(handle) if handle.class_name.is(CLASS_IDENTITY) => {
            if crate::is_handle_valid(handle) {
                Ok(handle.clone())
            } else {
                Err(input_parser_error(
                    builtin,
                    &INPUT_PARSER_ERROR_INVALID_PARSER,
                ))
            }
        }
        _ => Err(input_parser_error(
            builtin,
            &INPUT_PARSER_ERROR_INVALID_PARSER,
        )),
    }
}

fn with_parser_object_mut<R>(
    parser: &Value,
    builtin: &'static str,
    f: impl FnOnce(&mut runmat_value::ObjectInstance) -> BuiltinResult<R>,
) -> BuiltinResult<R> {
    let handle = parser_handle(parser, builtin)?;
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<R> {
        let Value::Object(obj) = target else {
            return Err(input_parser_error(
                builtin,
                &INPUT_PARSER_ERROR_INVALID_PARSER,
            ));
        };
        if !obj.class_name().is(CLASS_IDENTITY) {
            return Err(input_parser_error(
                builtin,
                &INPUT_PARSER_ERROR_INVALID_PARSER,
            ));
        }
        let result = f(obj);
        if let Some(value) = obj.property(PARAMETERS_PROPERTY) {
            runmat_gc::gc_record_handle_write(&handle.target, value);
        }
        if let Some(value) = obj.property(RESULTS_PROPERTY) {
            runmat_gc::gc_record_handle_write(&handle.target, value);
        }
        result
    })
    .map_err(|err| {
        input_parser_error_with_detail(
            builtin,
            &INPUT_PARSER_ERROR_INVALID_PARSER,
            format!("invalid handle target: {err}"),
        )
    })?
}

fn parameters(
    obj: &runmat_value::ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<StructValue> {
    match obj.property(PARAMETERS_PROPERTY) {
        Some(Value::Struct(parameters)) => Ok(parameters.clone()),
        _ => Err(input_parser_error_with_detail(
            builtin,
            &INPUT_PARSER_ERROR_INVALID_PARSER,
            "parser storage is corrupted",
        )),
    }
}

#[runtime_builtin(
    name = "inputParser",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn input_parser_builtin(
    runtime: Option<crate::context::RuntimeContext>,
) -> BuiltinResult<crate::execution::RootedValue> {
    let fields = [
        (RESULTS_PROPERTY, Value::Struct(StructValue::new())),
        (PARAMETERS_PROPERTY, Value::Struct(StructValue::new())),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        CLASS_IDENTITY.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    let Value::Object(instance) = object.value() else {
        unreachable!("constructed parser object")
    };
    let class_name = instance.class_name_owner().clone();
    object.with_owned(|value| {
        let target = runmat_gc::gc_allocate_rooted(value).map_err(|err| {
            input_parser_error_with_detail(
                INPUT_PARSER_NAME,
                &INPUT_PARSER_ERROR_INVALID_PARSER,
                err.to_string(),
            )
        })?;
        crate::execution::RootedValue::in_optional_context(
            Value::HandleObject(HandleRef {
                class_name,
                target: target.handle(),
                valid: true,
            }),
            runtime.as_ref(),
        )
    })
}

#[runtime_builtin(
    name = "addParameter",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn add_parameter_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    parser: Value,
    name: Value,
    default: Value,
) -> BuiltinResult<Value> {
    let name = text_scalar(&name, ADD_PARAMETER_NAME)?;
    with_parser_object_mut(&parser, ADD_PARAMETER_NAME, |obj| {
        let parameters = parameters(obj, ADD_PARAMETER_NAME)?;
        if parameters.fields.contains_key(name.as_ref()) {
            return Err(input_parser_error_with_detail(
                ADD_PARAMETER_NAME,
                &INPUT_PARSER_ERROR_DUPLICATE_PARAMETER,
                format!("'{name}'"),
            ));
        }
        let prepared =
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                runtime.as_ref(),
                parameters
                    .fields
                    .iter()
                    .chain(std::iter::once((name.as_ref(), &default))),
            )?;
        let fields = [
            (PARAMETERS_PROPERTY, prepared.value()),
            (RESULTS_PROPERTY, prepared.value()),
        ];
        publish_parser_fields(
            runtime.as_ref(),
            obj,
            fields.into_iter(),
            ADD_PARAMETER_NAME,
        )?;
        Ok(())
    })?;
    Ok(parser)
}

#[runtime_builtin(
    name = "parse",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn parse_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    parser: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !rest.len().is_multiple_of(2) {
        return Err(input_parser_error(
            PARSE_NAME,
            &INPUT_PARSER_ERROR_NAME_VALUE_PAIRS,
        ));
    }

    for pair in rest.chunks_exact(2) {
        validate_parameter_name(&pair[0])?;
    }

    with_parser_object_mut(&parser, PARSE_NAME, |obj| {
        let parameters = parameters(obj, PARSE_NAME)?;
        for pair in rest.chunks_exact(2) {
            if !parameters
                .field_names()
                .any(|name| parameter_name_matches(&pair[0], name))
            {
                let name = text_scalar(&pair[0], PARSE_NAME)?;
                return Err(input_parser_error_with_detail(
                    PARSE_NAME,
                    &INPUT_PARSER_ERROR_UNKNOWN_PARAMETER,
                    format!("'{name}'"),
                ));
            }
        }
        let fields = parameters.fields.iter().map(|(name, default)| {
            let value = rest
                .chunks_exact(2)
                .rev()
                .find_map(|pair| {
                    parameter_name_matches(&pair[0], name)
                        .then(|| pair.get(1))
                        .flatten()
                })
                .unwrap_or(default);
            (name, value)
        });
        let results =
            crate::execution::structure::construct_struct_from_fields_in_optional_context(
                runtime.as_ref(),
                fields,
            )?;
        publish_parser_fields(
            runtime.as_ref(),
            obj,
            std::iter::once((RESULTS_PROPERTY, results.value())),
            PARSE_NAME,
        )?;
        Ok(())
    })?;
    Ok(parser)
}

fn publish_parser_fields<'a>(
    runtime: Option<&crate::context::RuntimeContext>,
    object: &mut runmat_value::ObjectInstance,
    fields: impl Iterator<Item = (&'a str, &'a Value)> + Clone,
    builtin: &'static str,
) -> BuiltinResult<()> {
    crate::execution::object::replacement::replace_properties(
        runtime,
        object,
        fields,
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:inputParser:Allocation",
                message,
                source,
            )
        },
        |message| {
            input_parser_error_with_detail(builtin, &INPUT_PARSER_ERROR_INVALID_PARSER, message)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::CharArray;

    fn parser_with_root() -> (Value, runmat_gc::ExplicitRoot) {
        let parser = block_on(input_parser_builtin(None)).expect("inputParser");
        let parser_value = parser.value().clone();
        let Value::HandleObject(handle) = &parser_value else {
            panic!("expected inputParser handle, got {parser:?}");
        };
        let root = runmat_gc::gc_root(handle.target).expect("root inputParser handle");
        (parser_value, root)
    }

    fn property(value: &Value, name: &str) -> Value {
        let Value::HandleObject(handle) = value else {
            panic!("expected inputParser handle, got {value:?}");
        };
        runmat_gc::gc_with_value(&handle.target, |target| match target {
            Value::Object(obj) => obj.property(name).cloned().unwrap(),
            other => panic!("expected object target, got {other:?}"),
        })
        .expect("read target")
    }

    fn result_number(parser: &Value, name: &str) -> f64 {
        let Value::Struct(results) = property(parser, RESULTS_PROPERTY) else {
            panic!("expected Results struct");
        };
        let Some(Value::Num(value)) = results.fields.get(name) else {
            panic!("expected numeric result {name}, got {results:?}");
        };
        *value
    }

    fn run_add_parameter(parser: Value, name: Value, default: Value) -> BuiltinResult<Value> {
        block_on(add_parameter_builtin(None, parser, name, default))
    }

    fn run_parse(parser: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(parse_builtin(
            None,
            parser,
            crate::execution::RootedValues::new(rest)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        ))
    }

    #[test]
    fn input_parser_returns_valid_handle_with_empty_results() {
        let (parser, _root) = parser_with_root();
        let Value::HandleObject(handle) = &parser else {
            panic!("expected handle");
        };
        assert!(handle.class_name.is(CLASS_IDENTITY));
        assert!(crate::is_handle_valid(handle));
        let Value::Struct(results) = property(&parser, RESULTS_PROPERTY) else {
            panic!("expected Results struct");
        };
        assert!(results.fields.is_empty());
    }

    #[test]
    fn add_parameter_stores_defaults_and_rejects_duplicates() {
        let (parser, _root) = parser_with_root();
        run_add_parameter(parser.clone(), Value::from("scale"), Value::Num(2.0))
            .expect("addParameter");
        assert_eq!(result_number(&parser, "scale"), 2.0);

        let err = run_add_parameter(parser, Value::from("scale"), Value::Num(3.0)).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:inputParser:DuplicateParameter")
        );
    }

    #[test]
    fn parse_applies_overrides_preserves_defaults_and_resets_between_parses() {
        let (parser, _root) = parser_with_root();
        run_add_parameter(parser.clone(), Value::from("scale"), Value::Num(2.0))
            .expect("add scale");
        run_add_parameter(parser.clone(), Value::from("offset"), Value::Num(10.0))
            .expect("add offset");

        run_parse(parser.clone(), vec![Value::from("scale"), Value::Num(4.0)])
            .expect("parse override");
        assert_eq!(result_number(&parser, "scale"), 4.0);
        assert_eq!(result_number(&parser, "offset"), 10.0);

        run_parse(parser.clone(), Vec::new()).expect("parse defaults");
        assert_eq!(result_number(&parser, "scale"), 2.0);
        assert_eq!(result_number(&parser, "offset"), 10.0);
    }

    #[test]
    fn parse_accepts_char_name_values() {
        let (parser, _root) = parser_with_root();
        run_add_parameter(
            parser.clone(),
            Value::CharArray(CharArray::new_row("scale")),
            Value::Num(2.0),
        )
        .expect("add char parameter");
        run_parse(
            parser.clone(),
            vec![
                Value::CharArray(CharArray::new_row("scale")),
                Value::Num(4.0),
            ],
        )
        .expect("parse char name");
        assert_eq!(result_number(&parser, "scale"), 4.0);
    }

    #[test]
    fn invalid_parser_name_arity_and_unknown_parameter_have_stable_identifiers() {
        let err =
            run_add_parameter(Value::Num(1.0), Value::from("x"), Value::Num(0.0)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:inputParser:InvalidParser"));

        let (parser, _root) = parser_with_root();
        let err = run_add_parameter(parser.clone(), Value::Num(1.0), Value::Num(0.0)).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:inputParser:InvalidParameterName")
        );

        let err = run_parse(parser.clone(), vec![Value::from("x")]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:inputParser:NameValuePairs"));

        let err = run_parse(parser, vec![Value::from("x"), Value::Num(1.0)]).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:inputParser:UnknownParameter")
        );
    }
}
