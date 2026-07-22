//! Lemma detail helpers for Text Analytics tokenized documents.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    text_analytics_error, tokenized_document_language, DocumentTokenType, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::normalize::english_lemma;
use crate::{gather_if_needed_async, BuiltinResult};

pub(in crate::builtins::strings::text_analytics) const LEMMA_DETAILS_PROPERTY: &str =
    "LemmaDetails";

const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated tokenized document object.",
}];

const IN_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];

const IN_DOCUMENTS_DISCARD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("DiscardKnownValues"),
        description: "DiscardKnownValues option name.",
    },
    BuiltinParamDescriptor {
        name: "tf",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("false"),
        description: "Whether to recompute existing lemma details.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_LEMMA_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addLemmaDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addLemmaDetails: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ADD_LEMMA_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addLemmaDetails(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addLemmaDetails(documents,'DiscardKnownValues',tf)",
            inputs: &IN_DOCUMENTS_DISCARD,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "addLemmaDetails",
    category = "strings/text_analytics",
    summary = "Add lemma details to tokenizedDocument objects.",
    keywords = "addLemmaDetails,text analytics,tokenizedDocument,lemma,lemmatize",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::lemmas::ADD_LEMMA_DETAILS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::lemmas"
)]
async fn add_lemma_details_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args).await?;
    let (documents, options) = parse_args(gathered)?;
    let mut object = tokenized_document_object(documents)?;
    let documents = documents_from_object(&object, "addLemmaDetails")?;
    let stored = lemma_details_from_object(&object, "addLemmaDetails")?;
    let language = LemmaLanguage::from_document(&tokenized_document_language(&object))?;
    let document_options = options_from_document_object(&object);
    let lemmas = if options.discard_known_values {
        lemma_details_cell(&documents, language, &document_options)?
    } else {
        lemma_details_cell_preserving_known(
            &documents,
            stored.as_deref(),
            language,
            &document_options,
        )?
    };
    object
        .properties
        .insert(LEMMA_DETAILS_PROPERTY.to_string(), lemmas);
    Ok(Value::Object(object))
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            text_analytics_error(
                "addLemmaDetails",
                format!("addLemmaDetails: failed to gather input: {err}"),
            )
        })?);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, Default)]
struct AddLemmaOptions {
    discard_known_values: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LemmaLanguage {
    English,
    Japanese,
    Korean,
}

impl LemmaLanguage {
    fn from_document(language: &str) -> BuiltinResult<Self> {
        match language.trim().to_ascii_lowercase().as_str() {
            "en" => Ok(Self::English),
            "ja" => Ok(Self::Japanese),
            "ko" => Ok(Self::Korean),
            "de" => Err(text_analytics_error(
                "addLemmaDetails",
                "addLemmaDetails: German lemmatization is not supported by MATLAB addLemmaDetails; use normalizeWords with Style='lemma' for RunMat's German identity fallback",
            )),
            other => Err(text_analytics_error(
                "addLemmaDetails",
                format!("addLemmaDetails: unsupported document language '{other}'"),
            )),
        }
    }
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<(Value, AddLemmaOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addLemmaDetails",
            "addLemmaDetails: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "addLemmaDetails",
            "addLemmaDetails: name-value options must appear in pairs",
        ));
    }
    let mut options = AddLemmaOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "addLemmaDetails")
            .map_err(|err| text_analytics_error("addLemmaDetails", err.to_string()))?;
        if name.eq_ignore_ascii_case("DiscardKnownValues") {
            options.discard_known_values = logical_scalar(&args[idx + 1])?;
        } else {
            return Err(text_analytics_error(
                "addLemmaDetails",
                format!("addLemmaDetails: unsupported option '{name}'"),
            ));
        }
        idx += 2;
    }
    Ok((args[0].clone(), options))
}

fn tokenized_document_object(value: Value) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(object),
        Value::Object(object) => Err(text_analytics_error(
            "addLemmaDetails",
            format!(
                "addLemmaDetails: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "addLemmaDetails",
            format!("addLemmaDetails: expected tokenizedDocument object, got {other:?}"),
        )),
    }
}

fn lemma_details_cell(
    documents: &[Vec<String>],
    language: LemmaLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .map(|doc| {
            let lemmas = doc
                .iter()
                .map(|token| lemma_for_token(token, language, options))
                .collect::<Vec<_>>();
            StringArray::new(lemmas, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addLemmaDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addLemmaDetails", err))?,
    ))
}

fn lemma_details_cell_preserving_known(
    documents: &[Vec<String>],
    stored: Option<&[Vec<String>]>,
    language: LemmaLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .enumerate()
        .map(|(doc_idx, doc)| {
            let lemmas = doc
                .iter()
                .enumerate()
                .map(|(token_idx, token)| {
                    stored
                        .and_then(|lemmas| lemmas.get(doc_idx))
                        .and_then(|lemmas| lemmas.get(token_idx))
                        .filter(|lemma| is_known_lemma(lemma))
                        .cloned()
                        .unwrap_or_else(|| lemma_for_token(token, language, options))
                })
                .collect::<Vec<_>>();
            StringArray::new(lemmas, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addLemmaDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addLemmaDetails", err))?,
    ))
}

fn lemma_for_token(
    token: &str,
    language: LemmaLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> String {
    let token_type = document_token_type_with_options(token, options);
    if !matches!(token_type, DocumentTokenType::Letters) {
        return token.to_string();
    }
    match language {
        LemmaLanguage::English => english_lemma(&token.to_ascii_lowercase()),
        LemmaLanguage::Japanese | LemmaLanguage::Korean => token.to_string(),
    }
}

fn is_known_lemma(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && !crate::builtins::strings::common::is_missing_string(trimmed)
}

pub(in crate::builtins::strings::text_analytics) fn lemma_details_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<String>>>> {
    let Some(value) = object.properties.get(LEMMA_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid LemmaDetails property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid LemmaDetails shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::StringArray(array) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid LemmaDetails entry"),
            ));
        };
        if array.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid LemmaDetails entry shape"),
            ));
        }
        out.push(array.data.clone());
    }
    Ok(Some(out))
}

fn logical_scalar(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor.data.len() == 1 => match tensor.data[0] {
            0.0 => Ok(false),
            1.0 => Ok(true),
            other => Err(text_analytics_error(
                "addLemmaDetails",
                format!(
                    "addLemmaDetails: logical scalar option must be true or false, got {other}"
                ),
            )),
        },
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(text_analytics_error(
            "addLemmaDetails",
            format!("addLemmaDetails: logical scalar option must be true or false, got {other:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::details::token_details_builtin;
    use crate::builtins::strings::text_analytics::documents::tokenized_document_builtin;
    use crate::builtins::table::{table_variable_names_from_object, table_variables};
    use runmat_builtins::LogicalArray;

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_add_lemma(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_lemma_details_builtin(args))
    }

    fn run_token_details(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(token_details_builtin(value))
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_lemma_details_adds_lemma_column() {
        let docs = run_tokenized(vec![Value::StringArray(
            StringArray::new(
                vec![
                    "The dogs ran after the cat.".into(),
                    "I am building a house.".into(),
                ],
                vec![2, 1],
            )
            .unwrap(),
        )])
        .expect("tokenized");
        let updated = run_add_lemma(vec![docs]).expect("lemmas");
        let table = object(run_token_details(updated).expect("details"));

        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec![
                "Token",
                "DocumentNumber",
                "LineNumber",
                "Type",
                "Language",
                "Lemma"
            ]
        );
        assert_eq!(
            string_column(&table, "Lemma"),
            vec![
                "the", "dog", "run", "after", "the", "cat", ".", "i", "be", "build", "a", "house",
                "."
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_lemma_details_preserves_existing_known_values_unless_discarding() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["dogs".into(), "ran".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut stale = docs.clone();
        stale.properties.insert(
            LEMMA_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["custom".into(), "".into()], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let preserved = run_add_lemma(vec![Value::Object(stale.clone())]).expect("preserve");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(string_column(&table, "Lemma"), vec!["custom", "run"]);

        let recomputed = run_add_lemma(vec![
            Value::Object(stale),
            Value::String("DiscardKnownValues".into()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ])
        .expect("recompute");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(string_column(&table, "Lemma"), vec!["dog", "run"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_lemma_details_preserves_complex_and_non_word_tokens() {
        let docs = run_tokenized(vec![
            Value::StringArray(
                StringArray::new(
                    vec![
                        "RunMat2S".into(),
                        "https://example.com".into(),
                        "dogs".into(),
                        "42".into(),
                    ],
                    vec![1, 4],
                )
                .unwrap(),
            ),
            Value::String("TokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("tokenized");
        let updated = run_add_lemma(vec![docs]).expect("lemmas");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Lemma"),
            vec!["RunMat2S", "https://example.com", "dog", "42"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_lemma_details_rejects_unsupported_options_and_languages() {
        let docs = run_tokenized(vec![Value::String("dogs ran".into())]).expect("tokenized");
        let err = run_add_lemma(vec![
            docs.clone(),
            Value::String("Unknown".into()),
            Value::Bool(true),
        ])
        .expect_err("expected bad option");
        assert!(err.to_string().contains("unsupported option"));

        let mut german = object(docs);
        german
            .properties
            .insert("Language".into(), Value::String("de".into()));
        let err = run_add_lemma(vec![Value::Object(german)]).expect_err("expected bad language");
        assert!(err.to_string().contains("German lemmatization"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_malformed_lemma_details() {
        let docs =
            object(run_tokenized(vec![Value::String("dogs ran".into())]).expect("tokenized"));
        let mut malformed = docs.clone();
        malformed.properties.insert(
            LEMMA_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["dog".into()], vec![1, 1]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(malformed)).expect_err("expected error");
        assert!(err.to_string().contains("LemmaDetails entry"));

        let docs = object(
            run_tokenized(vec![Value::StringArray(
                StringArray::new(vec!["dogs".into(), "cats".into()], vec![2, 1]).unwrap(),
            )])
            .expect("tokenized"),
        );
        let mut transposed = docs.clone();
        transposed.properties.insert(
            LEMMA_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::StringArray(
                            StringArray::new(vec!["dog".into()], vec![1, 1]).unwrap(),
                        ),
                        Value::StringArray(
                            StringArray::new(vec!["cat".into()], vec![1, 1]).unwrap(),
                        ),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(transposed)).expect_err("expected shape error");
        assert!(err.to_string().contains("LemmaDetails shape"));

        let docs =
            object(run_tokenized(vec![Value::String("dogs ran".into())]).expect("tokenized"));
        let mut column_entry = docs.clone();
        column_entry.properties.insert(
            LEMMA_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["dog".into(), "run".into()], vec![2, 1]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err =
            run_token_details(Value::Object(column_entry)).expect_err("expected entry shape error");
        assert!(err.to_string().contains("LemmaDetails entry shape"));
    }
}
