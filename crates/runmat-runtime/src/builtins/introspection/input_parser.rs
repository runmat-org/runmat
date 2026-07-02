//! Minimal MATLAB-compatible `inputParser` name-value parsing.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, HandleRef, ObjectInstance, StructValue, Value,
};
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const CLASS_NAME: &str = "inputParser";
const INPUT_PARSER_NAME: &str = "inputParser";
const ADD_PARAMETER_NAME: &str = "addParameter";
const PARSE_NAME: &str = "parse";
const RESULTS_PROPERTY: &str = "Results";
const PARAMETERS_PROPERTY: &str = "__runmat_inputParser_parameters";

const INPUT_PARSER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "inputParser handle object.",
}];

const INPUT_PARSER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = inputParser()",
    inputs: &[],
    outputs: &INPUT_PARSER_OUTPUT,
}];

const ADD_PARAMETER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated inputParser handle object.",
}];

const ADD_PARAMETER_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "inputParser handle object.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Parameter name.",
    },
    BuiltinParamDescriptor {
        name: "default",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Default value for the parameter.",
    },
];

const ADD_PARAMETER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = addParameter(p, name, default)",
    inputs: &ADD_PARAMETER_INPUTS,
    outputs: &ADD_PARAMETER_OUTPUT,
}];

const PARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated inputParser handle object.",
}];

const PARSE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "inputParser handle object.",
    },
    BuiltinParamDescriptor {
        name: "nameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value arguments to parse.",
    },
];

const PARSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = parse(p, Name, Value, ...)",
    inputs: &PARSE_INPUTS,
    outputs: &PARSE_OUTPUT,
}];

const ERROR_INVALID_PARSER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.INVALID_PARSER",
    identifier: Some("RunMat:inputParser:InvalidParser"),
    when: "The parser argument is not a valid inputParser handle.",
    message: "inputParser: expected a valid inputParser handle",
};

const ERROR_PARAMETER_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.PARAMETER_NAME",
    identifier: Some("RunMat:inputParser:InvalidParameterName"),
    when: "A parameter name is not a character vector or string scalar.",
    message: "inputParser: parameter names must be text scalars",
};

const ERROR_DUPLICATE_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.DUPLICATE_PARAMETER",
    identifier: Some("RunMat:inputParser:DuplicateParameter"),
    when: "The same parameter name is registered more than once.",
    message: "inputParser: duplicate parameter name",
};

const ERROR_NAME_VALUE_PAIRS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.NAME_VALUE_PAIRS",
    identifier: Some("RunMat:inputParser:NameValuePairs"),
    when: "The parse argument list does not contain complete name-value pairs.",
    message: "inputParser: expected name-value pairs",
};

const ERROR_UNKNOWN_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INPUTPARSER.UNKNOWN_PARAMETER",
    identifier: Some("RunMat:inputParser:UnknownParameter"),
    when: "A parsed name does not match any registered parameter.",
    message: "inputParser: unknown parameter name",
};

const INPUT_PARSER_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const ADD_PARAMETER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_PARSER,
    ERROR_PARAMETER_NAME,
    ERROR_DUPLICATE_PARAMETER,
];
const PARSE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    ERROR_INVALID_PARSER,
    ERROR_PARAMETER_NAME,
    ERROR_NAME_VALUE_PAIRS,
    ERROR_UNKNOWN_PARAMETER,
];

pub const INPUT_PARSER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INPUT_PARSER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INPUT_PARSER_ERRORS,
};

pub const ADD_PARAMETER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADD_PARAMETER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADD_PARAMETER_ERRORS,
};

pub const PARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PARSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PARSE_ERRORS,
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

fn text_scalar(value: &Value, builtin: &'static str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(strings) if strings.data.len() == 1 => Ok(strings.data[0].clone()),
        Value::CharArray(chars) if chars.rows == 1 => Ok(chars.data.iter().collect()),
        _ => Err(input_parser_error(builtin, &ERROR_PARAMETER_NAME)),
    }
}

fn new_results_struct(parameters: &StructValue) -> StructValue {
    let mut results = StructValue::new();
    for (name, value) in &parameters.fields {
        results.insert(name.clone(), value.clone());
    }
    results
}

fn parser_handle(value: &Value, builtin: &'static str) -> BuiltinResult<HandleRef> {
    match value {
        Value::HandleObject(handle) if handle.class_name == CLASS_NAME => {
            if crate::is_handle_valid(handle) {
                Ok(handle.clone())
            } else {
                Err(input_parser_error(builtin, &ERROR_INVALID_PARSER))
            }
        }
        _ => Err(input_parser_error(builtin, &ERROR_INVALID_PARSER)),
    }
}

fn with_parser_object_mut<R>(
    parser: &Value,
    builtin: &'static str,
    f: impl FnOnce(&mut ObjectInstance) -> BuiltinResult<R>,
) -> BuiltinResult<R> {
    let handle = parser_handle(parser, builtin)?;
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<R> {
        let Value::Object(obj) = target else {
            return Err(input_parser_error(builtin, &ERROR_INVALID_PARSER));
        };
        if obj.class_name != CLASS_NAME {
            return Err(input_parser_error(builtin, &ERROR_INVALID_PARSER));
        }
        let result = f(obj)?;
        if let Some(value) = obj.properties.get(PARAMETERS_PROPERTY) {
            runmat_gc::gc_record_handle_write(&handle.target, value);
        }
        if let Some(value) = obj.properties.get(RESULTS_PROPERTY) {
            runmat_gc::gc_record_handle_write(&handle.target, value);
        }
        Ok(result)
    })
    .map_err(|err| {
        input_parser_error_with_detail(
            builtin,
            &ERROR_INVALID_PARSER,
            format!("invalid handle target: {err}"),
        )
    })?
}

fn parameters_mut<'a>(
    obj: &'a mut ObjectInstance,
    builtin: &'static str,
) -> BuiltinResult<&'a mut StructValue> {
    match obj.properties.get_mut(PARAMETERS_PROPERTY) {
        Some(Value::Struct(parameters)) => Ok(parameters),
        _ => Err(input_parser_error_with_detail(
            builtin,
            &ERROR_INVALID_PARSER,
            "parser storage is corrupted",
        )),
    }
}

#[runtime_builtin(
    name = "inputParser",
    category = "introspection",
    summary = "Create a parser for MATLAB-style name-value arguments.",
    keywords = "inputParser,name-value,arguments,varargin",
    descriptor(crate::builtins::introspection::input_parser::INPUT_PARSER_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn input_parser_builtin() -> BuiltinResult<Value> {
    let mut obj = ObjectInstance::new(CLASS_NAME.to_string());
    obj.properties.insert(
        RESULTS_PROPERTY.to_string(),
        Value::Struct(StructValue::new()),
    );
    obj.properties.insert(
        PARAMETERS_PROPERTY.to_string(),
        Value::Struct(StructValue::new()),
    );
    let target = runmat_gc::gc_allocate(Value::Object(obj)).map_err(|err| {
        input_parser_error_with_detail(INPUT_PARSER_NAME, &ERROR_INVALID_PARSER, err.to_string())
    })?;
    Ok(Value::HandleObject(HandleRef {
        class_name: CLASS_NAME.to_string(),
        target,
        valid: true,
    }))
}

#[runtime_builtin(
    name = "addParameter",
    category = "introspection",
    summary = "Register an inputParser name-value parameter and default.",
    keywords = "inputParser,addParameter,name-value,varargin",
    descriptor(crate::builtins::introspection::input_parser::ADD_PARAMETER_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn add_parameter_builtin(parser: Value, name: Value, default: Value) -> BuiltinResult<Value> {
    let name = text_scalar(&name, ADD_PARAMETER_NAME)?;
    with_parser_object_mut(&parser, ADD_PARAMETER_NAME, |obj| {
        let parameters = parameters_mut(obj, ADD_PARAMETER_NAME)?;
        if parameters.fields.contains_key(&name) {
            return Err(input_parser_error_with_detail(
                ADD_PARAMETER_NAME,
                &ERROR_DUPLICATE_PARAMETER,
                format!("'{name}'"),
            ));
        }
        parameters.insert(name.clone(), default.clone());
        let results = new_results_struct(parameters);
        obj.properties
            .insert(RESULTS_PROPERTY.to_string(), Value::Struct(results));
        Ok(())
    })?;
    Ok(parser)
}

#[runtime_builtin(
    name = "parse",
    category = "introspection",
    summary = "Parse name-value arguments with an inputParser.",
    keywords = "inputParser,parse,name-value,varargin",
    descriptor(crate::builtins::introspection::input_parser::PARSE_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::input_parser"
)]
async fn parse_builtin(parser: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if !rest.len().is_multiple_of(2) {
        return Err(input_parser_error(PARSE_NAME, &ERROR_NAME_VALUE_PAIRS));
    }

    let mut overrides = Vec::with_capacity(rest.len() / 2);
    let mut iter = rest.into_iter();
    while let (Some(name), Some(value)) = (iter.next(), iter.next()) {
        overrides.push((text_scalar(&name, PARSE_NAME)?, value));
    }

    with_parser_object_mut(&parser, PARSE_NAME, |obj| {
        let parameters = parameters_mut(obj, PARSE_NAME)?;
        let mut results = new_results_struct(parameters);
        for (name, value) in overrides {
            if !parameters.fields.contains_key(&name) {
                return Err(input_parser_error_with_detail(
                    PARSE_NAME,
                    &ERROR_UNKNOWN_PARAMETER,
                    format!("'{name}'"),
                ));
            }
            results.insert(name, value);
        }
        obj.properties
            .insert(RESULTS_PROPERTY.to_string(), Value::Struct(results));
        Ok(())
    })?;
    Ok(parser)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn run_gc<T>(f: impl FnOnce() -> T) -> T {
        runmat_gc::gc_test_context(f)
    }

    fn property(value: &Value, name: &str) -> Value {
        let Value::HandleObject(handle) = value else {
            panic!("expected inputParser handle, got {value:?}");
        };
        runmat_gc::gc_with_value(&handle.target, |target| match target {
            Value::Object(obj) => obj.properties.get(name).cloned().unwrap(),
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
        block_on(add_parameter_builtin(parser, name, default))
    }

    fn run_parse(parser: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(parse_builtin(parser, rest))
    }

    #[test]
    fn input_parser_returns_valid_handle_with_empty_results() {
        run_gc(|| {
            let parser = block_on(input_parser_builtin()).expect("inputParser");
            let Value::HandleObject(handle) = &parser else {
                panic!("expected handle");
            };
            assert_eq!(handle.class_name, CLASS_NAME);
            assert!(crate::is_handle_valid(handle));
            let Value::Struct(results) = property(&parser, RESULTS_PROPERTY) else {
                panic!("expected Results struct");
            };
            assert!(results.fields.is_empty());
        });
    }

    #[test]
    fn add_parameter_stores_defaults_and_rejects_duplicates() {
        run_gc(|| {
            let parser = block_on(input_parser_builtin()).expect("inputParser");
            run_add_parameter(parser.clone(), Value::from("scale"), Value::Num(2.0))
                .expect("addParameter");
            assert_eq!(result_number(&parser, "scale"), 2.0);

            let err = run_add_parameter(parser, Value::from("scale"), Value::Num(3.0)).unwrap_err();
            assert_eq!(
                err.identifier(),
                Some("RunMat:inputParser:DuplicateParameter")
            );
        });
    }

    #[test]
    fn parse_applies_overrides_preserves_defaults_and_resets_between_parses() {
        run_gc(|| {
            let parser = block_on(input_parser_builtin()).expect("inputParser");
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
        });
    }

    #[test]
    fn parse_accepts_char_name_values() {
        run_gc(|| {
            let parser = block_on(input_parser_builtin()).expect("inputParser");
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
        });
    }

    #[test]
    fn invalid_parser_name_arity_and_unknown_parameter_have_stable_identifiers() {
        run_gc(|| {
            let err =
                run_add_parameter(Value::Num(1.0), Value::from("x"), Value::Num(0.0)).unwrap_err();
            assert_eq!(err.identifier(), Some("RunMat:inputParser:InvalidParser"));

            let parser = block_on(input_parser_builtin()).expect("inputParser");
            let err =
                run_add_parameter(parser.clone(), Value::Num(1.0), Value::Num(0.0)).unwrap_err();
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
        });
    }
}
