//! Token-detail table helpers for Text Analytics tokenized documents.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    parse_top_level_domains, text_analytics_error, tokenized_document_language,
    top_level_domains_value, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::table::table_from_columns;
use crate::{gather_if_needed_async, BuiltinResult};

const TYPE_DETAILS_PROPERTY: &str = "TypeDetails";

const OUT_DETAILS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tdetails",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Token detail table.",
}];

const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized document object with token type details.",
}];

const IN_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];

const IN_DOCUMENTS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: DiscardKnownValues, TopLevelDomains.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.TOKEN_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:tokenDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "tokenDetails: invalid input",
};

const ERROR_ADD_TYPE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_TYPE_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addTypeDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addTypeDetails: invalid input",
};

const TOKEN_DETAILS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];
const ADD_TYPE_DETAILS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_ADD_TYPE_INVALID_INPUT];

pub const TOKEN_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tdetails = tokenDetails(documents)",
        inputs: &IN_DOCUMENTS,
        outputs: &OUT_DETAILS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TOKEN_DETAILS_ERRORS,
};

pub const ADD_TYPE_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocuments = addTypeDetails(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "newDocuments = addTypeDetails(documents,Name,Value)",
            inputs: &IN_DOCUMENTS_REST,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADD_TYPE_DETAILS_ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "tokenDetails",
    category = "strings/text_analytics",
    summary = "Return token details for tokenizedDocument objects.",
    keywords = "tokenDetails,text analytics,tokenizedDocument,token types",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::details::TOKEN_DETAILS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::details"
)]
async fn token_details_builtin(documents: Value) -> BuiltinResult<Value> {
    let documents = gather_if_needed_async(&documents)
        .await
        .map_err(|err| text_analytics_error("tokenDetails", format!("tokenDetails: {err}")))?;
    let object = tokenized_document_object(documents, "tokenDetails")?;
    token_details_table(&object)
}

#[runtime_builtin(
    name = "addTypeDetails",
    category = "strings/text_analytics",
    summary = "Add token type details to tokenizedDocument objects.",
    keywords = "addTypeDetails,text analytics,tokenizedDocument,token types",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::details::ADD_TYPE_DETAILS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::details"
)]
async fn add_type_details_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "addTypeDetails").await?;
    let (documents, options) = parse_add_type_details_args(gathered)?;
    let mut object = tokenized_document_object(documents, "addTypeDetails")?;
    let mut document_options = options_from_document_object(&object);
    if let Some(top_level_domains) = options.top_level_domains {
        document_options.top_level_domains = top_level_domains;
        document_options.top_level_domains_custom = true;
        object.properties.insert(
            "TopLevelDomains".to_string(),
            top_level_domains_value(&document_options.top_level_domains, "addTypeDetails")?,
        );
        object
            .properties
            .insert("TopLevelDomainsCustom".to_string(), Value::Bool(true));
    }
    let documents = documents_from_object(&object, "addTypeDetails")?;
    let type_details = if options.discard_known_values {
        type_details_cell(&documents, &document_options)?
    } else {
        let stored_types = type_details_from_object(&object, "addTypeDetails")?;
        type_details_cell_preserving_known(&documents, stored_types.as_deref(), &document_options)?
    };
    object
        .properties
        .insert(TYPE_DETAILS_PROPERTY.to_string(), type_details);
    Ok(Value::Object(object))
}

async fn gather_args(args: Vec<Value>, fn_name: &str) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            text_analytics_error(fn_name, format!("{fn_name}: failed to gather input: {err}"))
        })?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct AddTypeDetailsOptions {
    discard_known_values: bool,
    top_level_domains: Option<Vec<String>>,
}

fn parse_add_type_details_args(args: Vec<Value>) -> BuiltinResult<(Value, AddTypeDetailsOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addTypeDetails",
            "addTypeDetails: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "addTypeDetails",
            "addTypeDetails: name-value options must appear in pairs",
        ));
    }
    let mut options = AddTypeDetailsOptions {
        discard_known_values: false,
        top_level_domains: None,
    };
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "addTypeDetails")
            .map_err(|err| text_analytics_error("addTypeDetails", err.to_string()))?;
        if name.eq_ignore_ascii_case("DiscardKnownValues") {
            options.discard_known_values = logical_scalar(&args[idx + 1], "addTypeDetails")?;
        } else if name.eq_ignore_ascii_case("TopLevelDomains") {
            options.top_level_domains =
                Some(parse_top_level_domains(&args[idx + 1], "addTypeDetails")?);
        } else {
            return Err(text_analytics_error(
                "addTypeDetails",
                format!("addTypeDetails: unsupported option '{name}'"),
            ));
        }
        idx += 2;
    }
    Ok((args[0].clone(), options))
}

fn tokenized_document_object(value: Value, fn_name: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(object),
        Value::Object(object) => Err(text_analytics_error(
            fn_name,
            format!(
                "{fn_name}: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: expected tokenizedDocument object, got {other:?}"),
        )),
    }
}

fn token_details_table(object: &ObjectInstance) -> BuiltinResult<Value> {
    let documents = documents_from_object(object, "tokenDetails")?;
    let stored_types = type_details_from_object(object, "tokenDetails")?;
    let include_default_details = has_default_token_details(object);
    let include_type = include_default_details || stored_types.is_some();
    let include_line_language = include_default_details;
    let total = documents.iter().map(Vec::len).sum::<usize>();
    let document_options = options_from_document_object(object);

    let mut tokens = Vec::with_capacity(total);
    let mut document_numbers = Vec::with_capacity(total);
    let mut line_numbers = Vec::with_capacity(total);
    let mut token_types = Vec::with_capacity(total);
    let mut languages = Vec::with_capacity(total);
    let language = tokenized_document_language(object);

    for (doc_idx, doc) in documents.iter().enumerate() {
        for (token_idx, token) in doc.iter().enumerate() {
            tokens.push(token.clone());
            document_numbers.push((doc_idx + 1) as f64);
            if include_line_language {
                line_numbers.push(1.0);
                languages.push(language.clone());
            }
            if include_type {
                let token_type = stored_types
                    .as_ref()
                    .and_then(|types| types.get(doc_idx))
                    .and_then(|types| types.get(token_idx))
                    .cloned()
                    .unwrap_or_else(|| {
                        document_token_type_with_options(token, &document_options)
                            .as_str()
                            .to_string()
                    });
                token_types.push(token_type);
            }
        }
    }

    let mut names = vec!["Token".to_string(), "DocumentNumber".to_string()];
    let mut columns = vec![
        Value::StringArray(
            StringArray::new(tokens, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ),
        Value::Tensor(
            Tensor::new(document_numbers, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ),
    ];
    if include_line_language {
        names.push("LineNumber".to_string());
        columns.push(Value::Tensor(
            Tensor::new(line_numbers, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_type {
        names.push("Type".to_string());
        columns.push(Value::StringArray(
            StringArray::new(token_types, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_line_language {
        names.push("Language".to_string());
        columns.push(Value::StringArray(
            StringArray::new(languages, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    table_from_columns(names, columns)
}

fn has_default_token_details(object: &ObjectInstance) -> bool {
    !matches!(
        object.properties.get("TokenizeMethod"),
        Some(Value::String(value)) if value.eq_ignore_ascii_case("none")
    )
}

fn type_details_cell(
    documents: &[Vec<String>],
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .map(|doc| {
            let types = doc
                .iter()
                .map(|token| {
                    document_token_type_with_options(token, options)
                        .as_str()
                        .to_string()
                })
                .collect::<Vec<_>>();
            StringArray::new(types, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addTypeDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addTypeDetails", err))?,
    ))
}

fn type_details_cell_preserving_known(
    documents: &[Vec<String>],
    stored: Option<&[Vec<String>]>,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .enumerate()
        .map(|(doc_idx, doc)| {
            let types = doc
                .iter()
                .enumerate()
                .map(|(token_idx, token)| {
                    stored
                        .and_then(|types| types.get(doc_idx))
                        .and_then(|types| types.get(token_idx))
                        .filter(|stored_type| is_known_type_detail(stored_type))
                        .cloned()
                        .unwrap_or_else(|| {
                            document_token_type_with_options(token, options)
                                .as_str()
                                .to_string()
                        })
                })
                .collect::<Vec<_>>();
            StringArray::new(types, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addTypeDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addTypeDetails", err))?,
    ))
}

fn is_known_type_detail(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    !normalized.is_empty()
        && normalized != "unknown"
        && !crate::builtins::strings::common::is_missing_string(value)
}

fn type_details_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<String>>>> {
    let Some(value) = object.properties.get(TYPE_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid TypeDetails property"),
        ));
    };
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::StringArray(array) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid TypeDetails entry"),
            ));
        };
        out.push(array.data.clone());
    }
    Ok(Some(out))
}

fn logical_scalar(value: &Value, fn_name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor.data.len() == 1 => match tensor.data[0] {
            0.0 => Ok(false),
            1.0 => Ok(true),
            other => Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: logical scalar option must be true or false, got {other}"),
            )),
        },
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: logical scalar option must be true or false, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::documents::tokenized_document_builtin;
    use crate::builtins::table::{table_variable_names_from_object, table_variables};
    use runmat_builtins::LogicalArray;

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_token_details(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(token_details_builtin(value))
    }

    fn run_add_type(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_type_details_builtin(args))
    }

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        object
    }

    fn table_column(table: &ObjectInstance, name: &str) -> Value {
        table_variables(table)
            .expect("table variables")
            .fields
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("missing table column {name}"))
    }

    fn string_column(table: &ObjectInstance, name: &str) -> Vec<String> {
        match table_column(table, name) {
            Value::StringArray(array) => array.data,
            other => panic!("expected string column {name}, got {other:?}"),
        }
    }

    fn numeric_column(table: &ObjectInstance, name: &str) -> Vec<f64> {
        match table_column(table, name) {
            Value::Tensor(tensor) => tensor.data,
            other => panic!("expected numeric column {name}, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_returns_default_unicode_table() {
        let docs = run_tokenized(vec![Value::StringArray(
            StringArray::new(
                vec![
                    "alpha 123 https://example.com".to_string(),
                    "beta, @user".to_string(),
                ],
                vec![2, 1],
            )
            .unwrap(),
        )])
        .expect("tokenized");
        let table = object(run_token_details(docs).expect("details"));

        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec!["Token", "DocumentNumber", "LineNumber", "Type", "Language"]
        );
        assert_eq!(
            string_column(&table, "Token"),
            vec!["alpha", "123", "https://example.com", "beta", ",", "@user"]
        );
        assert_eq!(
            numeric_column(&table, "DocumentNumber"),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(
            numeric_column(&table, "LineNumber"),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            string_column(&table, "Type"),
            vec![
                "letters",
                "digits",
                "web-address",
                "letters",
                "punctuation",
                "at-mention"
            ]
        );
        assert_eq!(
            string_column(&table, "Language"),
            vec!["en", "en", "en", "en", "en", "en"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pretokenized_documents_have_minimal_details_until_type_details_are_added() {
        let docs = run_tokenized(vec![
            Value::StringArray(
                StringArray::new(vec!["A".into(), "42".into(), "#tag".into()], vec![1, 3]).unwrap(),
            ),
            Value::String("TokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("tokenized");

        let table = object(run_token_details(docs.clone()).expect("details"));
        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec!["Token", "DocumentNumber"]
        );
        assert_eq!(string_column(&table, "Token"), vec!["A", "42", "#tag"]);

        let updated = run_add_type(vec![docs]).expect("add types");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec!["Token", "DocumentNumber", "Type"]
        );
        assert_eq!(
            string_column(&table, "Type"),
            vec!["letters", "digits", "hashtag"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_recomputes_when_discard_known_values_is_true() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut stale = docs.clone();
        stale.properties.insert(
            TYPE_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["other".into(), "other".into()], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let preserved = run_add_type(vec![Value::Object(stale.clone())]).expect("preserve");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(string_column(&table, "Type"), vec!["other", "other"]);

        let recomputed = run_add_type(vec![
            Value::Object(stale),
            Value::String("DiscardKnownValues".into()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ])
        .expect("recompute");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(string_column(&table, "Type"), vec!["letters", "letters"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_uses_bounded_hashtag_and_mention_rules() {
        let docs = run_tokenized(vec![
            Value::StringArray(
                StringArray::new(
                    vec![
                        "#".into(),
                        "@".into(),
                        "#tag".into(),
                        "#1".into(),
                        "@name_1".into(),
                        "@abcdefghijklmnop".into(),
                    ],
                    vec![1, 6],
                )
                .unwrap(),
            ),
            Value::String("TokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("tokenized");
        let updated = run_add_type(vec![docs]).expect("add types");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Type"),
            vec![
                "punctuation",
                "punctuation",
                "hashtag",
                "other",
                "at-mention",
                "other"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_uses_configured_top_level_domains() {
        let docs = run_tokenized(vec![
            Value::StringArray(
                StringArray::new(
                    vec![
                        "example.zz".into(),
                        "example.com".into(),
                        "https://site.zz/path".into(),
                    ],
                    vec![1, 3],
                )
                .unwrap(),
            ),
            Value::String("TokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("tokenized");
        let updated = run_add_type(vec![
            docs,
            Value::String("TopLevelDomains".into()),
            Value::String("zz".into()),
        ])
        .expect("add types");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Type"),
            vec!["web-address", "other", "web-address"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_preserves_existing_types_unless_discarding() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["example.zz".into()], vec![1, 1]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut typed = docs.clone();
        typed.properties.insert(
            TYPE_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["other".into()], vec![1, 1]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let preserved = run_add_type(vec![
            Value::Object(typed.clone()),
            Value::String("TopLevelDomains".into()),
            Value::String("zz".into()),
        ])
        .expect("preserve existing type details");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(string_column(&table, "Type"), vec!["other"]);

        let recomputed = run_add_type(vec![
            Value::Object(typed),
            Value::String("TopLevelDomains".into()),
            Value::String("zz".into()),
            Value::String("DiscardKnownValues".into()),
            Value::Bool(true),
        ])
        .expect("recompute type details");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(string_column(&table, "Type"), vec!["web-address"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_fills_unknown_existing_type_details() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(
                        vec!["known".into(), "example.zz".into(), "missing".into()],
                        vec![1, 3],
                    )
                    .unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut typed = docs.clone();
        typed.properties.insert(
            TYPE_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(
                            vec!["letters".into(), "unknown".into(), "".into()],
                            vec![1, 3],
                        )
                        .unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let updated = run_add_type(vec![
            Value::Object(typed),
            Value::String("TopLevelDomains".into()),
            Value::String("zz".into()),
        ])
        .expect("fill unknown type details");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Type"),
            vec!["letters", "web-address", "letters"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_non_document_input() {
        let err = run_token_details(Value::String("not docs".into())).expect_err("expected error");
        assert!(err.to_string().contains("tokenizedDocument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_type_details_rejects_unknown_option() {
        let docs = run_tokenized(vec![Value::String("alpha".into())]).expect("tokenized");
        let err = run_add_type(vec![
            docs,
            Value::String("Unknown".into()),
            Value::Bool(true),
        ])
        .expect_err("expected error");
        assert!(err.to_string().contains("unsupported option"));
    }
}
