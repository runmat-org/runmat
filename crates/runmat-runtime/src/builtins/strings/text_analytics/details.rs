//! Token-detail table helpers for Text Analytics tokenized documents.

use std::collections::{HashMap, HashSet};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::dependencies::{
    dependency_details_from_object, dependency_heads_from_object,
};
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    parse_top_level_domains, text_analytics_error, tokenized_document_language,
    top_level_domains_value, words_from_word_vector, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::entities::entity_details_from_object;
use crate::builtins::strings::text_analytics::lemmas::lemma_details_from_object;
use crate::builtins::strings::text_analytics::pos::part_of_speech_details_from_object;
use crate::builtins::strings::text_analytics::stopwords::{
    stop_words_for_language, StopWordsLanguage,
};
use crate::builtins::table::{categorical_labels, table_from_columns, table_variables};
use crate::{gather_if_needed_async, BuiltinResult};

const TYPE_DETAILS_PROPERTY: &str = "TypeDetails";
const SENTENCE_NUMBERS_PROPERTY: &str = "SentenceNumbers";

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
    description: "Updated tokenized document object.",
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

const IN_DOCUMENTS_SENTENCE_REST: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: Abbreviations, Starters, DiscardKnownValues.",
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

const ERROR_ADD_SENTENCE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_SENTENCE_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addSentenceDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addSentenceDetails: invalid input",
};

const TOKEN_DETAILS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];
const ADD_TYPE_DETAILS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_ADD_TYPE_INVALID_INPUT];
const ADD_SENTENCE_DETAILS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_ADD_SENTENCE_INVALID_INPUT];

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

pub const ADD_SENTENCE_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocuments = addSentenceDetails(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "newDocuments = addSentenceDetails(documents,Name,Value)",
            inputs: &IN_DOCUMENTS_SENTENCE_REST,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADD_SENTENCE_DETAILS_ERRORS,
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
pub(in crate::builtins::strings::text_analytics) async fn token_details_builtin(
    documents: Value,
) -> BuiltinResult<Value> {
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

#[runtime_builtin(
    name = "addSentenceDetails",
    category = "strings/text_analytics",
    summary = "Add sentence numbers to tokenizedDocument objects.",
    keywords = "addSentenceDetails,text analytics,tokenizedDocument,sentences",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::details::ADD_SENTENCE_DETAILS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::details"
)]
pub(in crate::builtins::strings::text_analytics) async fn add_sentence_details_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "addSentenceDetails").await?;
    let (documents, options) = parse_add_sentence_details_args(gathered)?;
    let mut object = tokenized_document_object(documents, "addSentenceDetails")?;
    let documents = documents_from_object(&object, "addSentenceDetails")?;
    let sentence_numbers = if options.discard_known_values {
        sentence_numbers_cell(&documents, &options)?
    } else {
        let stored = sentence_numbers_from_object(&object, "addSentenceDetails")?;
        sentence_numbers_cell_preserving_known(&documents, stored.as_deref(), &options)?
    };
    object
        .properties
        .insert(SENTENCE_NUMBERS_PROPERTY.to_string(), sentence_numbers);
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
struct AddSentenceDetailsOptions {
    discard_known_values: bool,
    abbreviations: HashMap<String, AbbreviationUsage>,
    starters: HashSet<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbbreviationUsage {
    Regular,
    Inner,
    Reference,
    Unit,
}

impl Default for AddSentenceDetailsOptions {
    fn default() -> Self {
        Self {
            discard_known_values: false,
            abbreviations: default_abbreviations(),
            starters: default_sentence_starters(),
        }
    }
}

#[derive(Clone, Debug)]
struct AddTypeDetailsOptions {
    discard_known_values: bool,
    top_level_domains: Option<Vec<String>>,
}

fn parse_add_sentence_details_args(
    args: Vec<Value>,
) -> BuiltinResult<(Value, AddSentenceDetailsOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addSentenceDetails",
            "addSentenceDetails: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "addSentenceDetails",
            "addSentenceDetails: name-value options must appear in pairs",
        ));
    }
    let mut options = AddSentenceDetailsOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "addSentenceDetails")
            .map_err(|err| text_analytics_error("addSentenceDetails", err.to_string()))?;
        if name.eq_ignore_ascii_case("DiscardKnownValues") {
            options.discard_known_values = logical_scalar(&args[idx + 1], "addSentenceDetails")?;
        } else if name.eq_ignore_ascii_case("Abbreviations") {
            options.abbreviations = parse_abbreviations(&args[idx + 1])?;
        } else if name.eq_ignore_ascii_case("Starters") {
            options.starters = parse_sentence_starters(&args[idx + 1])?;
        } else {
            return Err(text_analytics_error(
                "addSentenceDetails",
                format!("addSentenceDetails: unsupported option '{name}'"),
            ));
        }
        idx += 2;
    }
    Ok((args[0].clone(), options))
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
    let stored_sentence_numbers = sentence_numbers_from_object(object, "tokenDetails")?;
    let stored_lemmas = lemma_details_from_object(object, "tokenDetails")?;
    let stored_pos = part_of_speech_details_from_object(object, "tokenDetails")?;
    let stored_entities = entity_details_from_object(object, "tokenDetails")?;
    let stored_heads = dependency_heads_from_object(object, "tokenDetails")?;
    let stored_dependencies = dependency_details_from_object(object, "tokenDetails")?;
    validate_sentence_number_shapes(&documents, stored_sentence_numbers.as_deref())?;
    validate_text_detail_shapes(&documents, stored_lemmas.as_deref(), "LemmaDetails")?;
    validate_text_detail_shapes(&documents, stored_pos.as_deref(), "PartOfSpeechDetails")?;
    validate_text_detail_shapes(&documents, stored_entities.as_deref(), "EntityDetails")?;
    validate_head_detail_shapes(&documents, stored_heads.as_deref())?;
    validate_text_detail_shapes(
        &documents,
        stored_dependencies.as_deref(),
        "DependencyDetails",
    )?;
    let include_default_details = has_default_token_details(object);
    let include_type = include_default_details || stored_types.is_some();
    let include_sentence = stored_sentence_numbers.is_some();
    let include_line_language = include_default_details;
    let include_language = include_line_language
        || stored_lemmas.is_some()
        || stored_pos.is_some()
        || stored_entities.is_some()
        || stored_heads.is_some()
        || stored_dependencies.is_some();
    let include_pos = stored_pos.is_some();
    let include_entity = stored_entities.is_some();
    let include_lemma = stored_lemmas.is_some();
    let include_head = stored_heads.is_some();
    let include_dependency = stored_dependencies.is_some();
    let total = documents.iter().map(Vec::len).sum::<usize>();
    let document_options = options_from_document_object(object);

    let mut tokens = Vec::with_capacity(total);
    let mut document_numbers = Vec::with_capacity(total);
    let mut sentence_numbers = Vec::with_capacity(total);
    let mut line_numbers = Vec::with_capacity(total);
    let mut token_types = Vec::with_capacity(total);
    let mut languages = Vec::with_capacity(total);
    let mut part_of_speech = Vec::with_capacity(total);
    let mut entities = Vec::with_capacity(total);
    let mut lemmas = Vec::with_capacity(total);
    let mut heads = Vec::with_capacity(total);
    let mut dependencies = Vec::with_capacity(total);
    let language = tokenized_document_language(object);

    for (doc_idx, doc) in documents.iter().enumerate() {
        for (token_idx, token) in doc.iter().enumerate() {
            tokens.push(token.clone());
            document_numbers.push((doc_idx + 1) as f64);
            if include_sentence {
                sentence_numbers.push(
                    stored_sentence_numbers
                        .as_ref()
                        .and_then(|numbers| numbers.get(doc_idx))
                        .and_then(|numbers| numbers.get(token_idx))
                        .copied()
                        .unwrap_or(1.0),
                );
            }
            if include_line_language {
                line_numbers.push(1.0);
            }
            if include_language {
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
            if include_lemma {
                lemmas.push(
                    stored_lemmas
                        .as_ref()
                        .and_then(|lemmas| lemmas.get(doc_idx))
                        .and_then(|lemmas| lemmas.get(token_idx))
                        .cloned()
                        .unwrap_or_else(|| token.clone()),
                );
            }
            if include_pos {
                part_of_speech.push(
                    stored_pos
                        .as_ref()
                        .and_then(|pos| pos.get(doc_idx))
                        .and_then(|pos| pos.get(token_idx))
                        .cloned()
                        .unwrap_or_else(|| "other".to_string()),
                );
            }
            if include_entity {
                entities.push(
                    stored_entities
                        .as_ref()
                        .and_then(|entities| entities.get(doc_idx))
                        .and_then(|entities| entities.get(token_idx))
                        .cloned()
                        .unwrap_or_else(|| "non-entity".to_string()),
                );
            }
            if include_head {
                heads.push(
                    stored_heads
                        .as_ref()
                        .and_then(|heads| heads.get(doc_idx))
                        .and_then(|heads| heads.get(token_idx))
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            if include_dependency {
                dependencies.push(
                    stored_dependencies
                        .as_ref()
                        .and_then(|dependencies| dependencies.get(doc_idx))
                        .and_then(|dependencies| dependencies.get(token_idx))
                        .cloned()
                        .unwrap_or_else(|| "dep".to_string()),
                );
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
    if include_sentence {
        names.push("SentenceNumber".to_string());
        columns.push(Value::Tensor(
            Tensor::new(sentence_numbers, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
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
    if include_language {
        names.push("Language".to_string());
        columns.push(Value::StringArray(
            StringArray::new(languages, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_pos {
        names.push("PartOfSpeech".to_string());
        columns.push(Value::StringArray(
            StringArray::new(part_of_speech, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_entity {
        names.push("Entity".to_string());
        columns.push(Value::StringArray(
            StringArray::new(entities, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_lemma {
        names.push("Lemma".to_string());
        columns.push(Value::StringArray(
            StringArray::new(lemmas, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_head {
        names.push("Head".to_string());
        columns.push(Value::Tensor(
            Tensor::new(heads, vec![total, 1])
                .map_err(|err| text_analytics_error("tokenDetails", err))?,
        ));
    }
    if include_dependency {
        names.push("Dependency".to_string());
        columns.push(Value::StringArray(
            StringArray::new(dependencies, vec![total, 1])
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

fn sentence_numbers_cell(
    documents: &[Vec<String>],
    options: &AddSentenceDetailsOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .map(|doc| {
            let numbers = sentence_numbers_for_doc(doc, options);
            Tensor::new(numbers, vec![1, doc.len()])
                .map(Value::Tensor)
                .map_err(|err| text_analytics_error("addSentenceDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addSentenceDetails", err))?,
    ))
}

fn sentence_numbers_cell_preserving_known(
    documents: &[Vec<String>],
    stored: Option<&[Vec<f64>]>,
    options: &AddSentenceDetailsOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .enumerate()
        .map(|(doc_idx, doc)| {
            let computed = sentence_numbers_for_doc(doc, options);
            let numbers = if let Some(existing) = stored.and_then(|values| values.get(doc_idx)) {
                if existing.len() == doc.len()
                    && existing
                        .iter()
                        .all(|value| value.is_finite() && *value >= 1.0 && value.fract() == 0.0)
                {
                    existing.clone()
                } else {
                    computed
                }
            } else {
                computed
            };
            Tensor::new(numbers, vec![1, doc.len()])
                .map(Value::Tensor)
                .map_err(|err| text_analytics_error("addSentenceDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addSentenceDetails", err))?,
    ))
}

fn sentence_numbers_for_doc(tokens: &[String], options: &AddSentenceDetailsOptions) -> Vec<f64> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut current = 1.0;
    for idx in 0..tokens.len() {
        out.push(current);
        if is_sentence_terminator_at(tokens, idx, options) {
            current += 1.0;
        }
    }
    out
}

fn is_sentence_terminator_at(
    tokens: &[String],
    idx: usize,
    options: &AddSentenceDetailsOptions,
) -> bool {
    let token = tokens[idx].trim();
    if !matches!(token, "." | "!" | "?" | "。" | "！" | "？") {
        return false;
    }
    if token != "." {
        return true;
    }
    let Some(previous) = previous_word(tokens, idx) else {
        return true;
    };
    if is_single_letter_abbreviation(previous)
        && next_word(tokens, idx).is_some_and(is_single_letter_abbreviation)
    {
        return false;
    }
    let previous_key = normalize_abbreviation(previous);
    let usage = options
        .abbreviations
        .get(&previous_key)
        .copied()
        .or_else(|| is_single_letter_abbreviation(previous).then_some(AbbreviationUsage::Regular));
    let Some(usage) = usage else {
        return true;
    };
    match usage {
        AbbreviationUsage::Inner => false,
        AbbreviationUsage::Regular => next_word_is_sentence_starter(tokens, idx, options),
        AbbreviationUsage::Reference => !next_word(tokens, idx).is_some_and(is_numeric_token),
        AbbreviationUsage::Unit => {
            if previous_word_before(tokens, idx, previous).is_some_and(is_numeric_token) {
                next_word_is_sentence_starter(tokens, idx, options)
            } else {
                true
            }
        }
    }
}

fn previous_word(tokens: &[String], idx: usize) -> Option<&str> {
    tokens[..idx]
        .iter()
        .rev()
        .find(|token| token.chars().any(char::is_alphanumeric))
        .map(String::as_str)
}

fn previous_word_before<'a>(tokens: &'a [String], idx: usize, previous: &str) -> Option<&'a str> {
    let mut seen_previous = false;
    for token in tokens[..idx].iter().rev() {
        if !token.chars().any(char::is_alphanumeric) {
            continue;
        }
        if !seen_previous && token == previous {
            seen_previous = true;
            continue;
        }
        if seen_previous {
            return Some(token);
        }
    }
    None
}

fn next_word(tokens: &[String], idx: usize) -> Option<&str> {
    tokens[idx + 1..]
        .iter()
        .find(|token| token.chars().any(char::is_alphanumeric))
        .map(String::as_str)
}

fn next_word_is_sentence_starter(
    tokens: &[String],
    idx: usize,
    options: &AddSentenceDetailsOptions,
) -> bool {
    let Some(word) = next_word(tokens, idx) else {
        return true;
    };
    word.chars().next().is_some_and(char::is_uppercase)
        && options.starters.contains(&word.to_ascii_lowercase())
}

fn is_numeric_token(token: &str) -> bool {
    token.chars().any(char::is_numeric)
        && token
            .chars()
            .all(|ch| ch.is_numeric() || matches!(ch, '.' | ',' | '_' | '+' | '-'))
}

fn is_single_letter_abbreviation(token: &str) -> bool {
    token.chars().count() == 1 && token.chars().all(char::is_alphabetic)
}

fn parse_abbreviations(value: &Value) -> BuiltinResult<HashMap<String, AbbreviationUsage>> {
    if let Value::Object(object) = value {
        let variables = table_variables(object).map_err(|err| {
            text_analytics_error(
                "addSentenceDetails",
                format!("addSentenceDetails: invalid Abbreviations table: {err}"),
            )
        })?;
        let abbreviations = variables.fields.get("Abbreviation").ok_or_else(|| {
            text_analytics_error(
                "addSentenceDetails",
                "addSentenceDetails: Abbreviations table must contain an Abbreviation variable",
            )
        })?;
        let usages = variables.fields.get("Usage").ok_or_else(|| {
            text_analytics_error(
                "addSentenceDetails",
                "addSentenceDetails: Abbreviations table must contain a Usage variable",
            )
        })?;
        let abbreviations = words_or_categorical_labels(abbreviations)?;
        let usages = words_or_categorical_labels(usages)?;
        if abbreviations.len() != usages.len() {
            return Err(text_analytics_error(
                "addSentenceDetails",
                "addSentenceDetails: Abbreviations and Usage variables must have the same length",
            ));
        }
        let mut out = HashMap::new();
        for (abbreviation, usage) in abbreviations.into_iter().zip(usages) {
            out.insert(
                normalize_abbreviation(&abbreviation),
                parse_abbreviation_usage(&usage)?,
            );
        }
        return Ok(out);
    }

    let mut out = HashMap::new();
    for abbreviation in words_from_word_vector(value, "addSentenceDetails")? {
        out.insert(
            normalize_abbreviation(&abbreviation),
            AbbreviationUsage::Regular,
        );
    }
    Ok(out)
}

fn parse_sentence_starters(value: &Value) -> BuiltinResult<HashSet<String>> {
    Ok(words_or_categorical_labels(value)?
        .into_iter()
        .map(|word| word.trim().to_ascii_lowercase())
        .filter(|word| !word.is_empty())
        .collect())
}

fn words_or_categorical_labels(value: &Value) -> BuiltinResult<Vec<String>> {
    if matches!(value, Value::Object(_)) {
        if let Ok(labels) = categorical_labels(value) {
            return Ok(labels);
        }
    }
    words_from_word_vector(value, "addSentenceDetails")
}

fn parse_abbreviation_usage(value: &str) -> BuiltinResult<AbbreviationUsage> {
    match value.trim().to_ascii_lowercase().as_str() {
        "regular" => Ok(AbbreviationUsage::Regular),
        "inner" => Ok(AbbreviationUsage::Inner),
        "reference" => Ok(AbbreviationUsage::Reference),
        "unit" => Ok(AbbreviationUsage::Unit),
        other => Err(text_analytics_error(
            "addSentenceDetails",
            format!("addSentenceDetails: unsupported abbreviation Usage '{other}'"),
        )),
    }
}

fn normalize_abbreviation(value: &str) -> String {
    value.trim().trim_end_matches('.').to_ascii_lowercase()
}

fn default_abbreviations() -> HashMap<String, AbbreviationUsage> {
    [
        ("mr", AbbreviationUsage::Inner),
        ("mrs", AbbreviationUsage::Inner),
        ("ms", AbbreviationUsage::Inner),
        ("dr", AbbreviationUsage::Inner),
        ("prof", AbbreviationUsage::Inner),
        ("sr", AbbreviationUsage::Inner),
        ("jr", AbbreviationUsage::Inner),
        ("st", AbbreviationUsage::Inner),
        ("vs", AbbreviationUsage::Regular),
        ("etc", AbbreviationUsage::Regular),
        ("appt", AbbreviationUsage::Regular),
        ("fig", AbbreviationUsage::Reference),
        ("eq", AbbreviationUsage::Reference),
        ("sec", AbbreviationUsage::Reference),
        ("cm", AbbreviationUsage::Unit),
        ("mm", AbbreviationUsage::Unit),
        ("in", AbbreviationUsage::Unit),
        ("ft", AbbreviationUsage::Unit),
    ]
    .into_iter()
    .map(|(abbr, usage)| (abbr.to_string(), usage))
    .collect()
}

fn default_sentence_starters() -> HashSet<String> {
    let mut starters = stop_words_for_language(StopWordsLanguage::English)
        .iter()
        .map(|word| (*word).to_string())
        .collect::<HashSet<_>>();
    starters.extend(
        [
            "another",
            "here",
            "let",
            "try",
            "today",
            "tomorrow",
            "yesterday",
        ]
        .into_iter()
        .map(str::to_string),
    );
    starters
}

fn validate_sentence_number_shapes(
    documents: &[Vec<String>],
    stored: Option<&[Vec<f64>]>,
) -> BuiltinResult<()> {
    let Some(stored) = stored else {
        return Ok(());
    };
    if stored.len() != documents.len() {
        return Err(text_analytics_error(
            "tokenDetails",
            format!(
                "tokenDetails: SentenceNumbers has {} documents but Documents has {}",
                stored.len(),
                documents.len()
            ),
        ));
    }
    for (idx, (numbers, doc)) in stored.iter().zip(documents).enumerate() {
        if numbers.len() != doc.len() {
            return Err(text_analytics_error(
                "tokenDetails",
                format!(
                    "tokenDetails: SentenceNumbers entry {} has {} values but document has {} tokens",
                    idx + 1,
                    numbers.len(),
                    doc.len()
                ),
            ));
        }
        if numbers
            .iter()
            .any(|value| !value.is_finite() || *value < 1.0 || value.fract() != 0.0)
        {
            return Err(text_analytics_error(
                "tokenDetails",
                format!(
                    "tokenDetails: SentenceNumbers entry {} contains invalid sentence numbers",
                    idx + 1
                ),
            ));
        }
    }
    Ok(())
}

fn validate_text_detail_shapes(
    documents: &[Vec<String>],
    stored: Option<&[Vec<String>]>,
    property: &str,
) -> BuiltinResult<()> {
    let Some(stored) = stored else {
        return Ok(());
    };
    if stored.len() != documents.len() {
        return Err(text_analytics_error(
            "tokenDetails",
            format!(
                "tokenDetails: {property} has {} documents but Documents has {}",
                stored.len(),
                documents.len()
            ),
        ));
    }
    for (idx, (values, doc)) in stored.iter().zip(documents).enumerate() {
        if values.len() != doc.len() {
            return Err(text_analytics_error(
                "tokenDetails",
                format!(
                    "tokenDetails: {property} entry {} has {} values but document has {} tokens",
                    idx + 1,
                    values.len(),
                    doc.len()
                ),
            ));
        }
    }
    Ok(())
}

fn validate_head_detail_shapes(
    documents: &[Vec<String>],
    stored: Option<&[Vec<f64>]>,
) -> BuiltinResult<()> {
    let Some(stored) = stored else {
        return Ok(());
    };
    if stored.len() != documents.len() {
        return Err(text_analytics_error(
            "tokenDetails",
            format!(
                "tokenDetails: HeadDetails has {} documents but Documents has {}",
                stored.len(),
                documents.len()
            ),
        ));
    }
    for (idx, (values, doc)) in stored.iter().zip(documents).enumerate() {
        if values.len() != doc.len() {
            return Err(text_analytics_error(
                "tokenDetails",
                format!(
                    "tokenDetails: HeadDetails entry {} has {} values but document has {} tokens",
                    idx + 1,
                    values.len(),
                    doc.len()
                ),
            ));
        }
        if values.iter().any(|value| {
            !value.is_finite() || *value < 0.0 || value.fract() != 0.0 || *value > doc.len() as f64
        }) {
            return Err(text_analytics_error(
                "tokenDetails",
                format!(
                    "tokenDetails: HeadDetails entry {} contains invalid head indices",
                    idx + 1
                ),
            ));
        }
    }
    Ok(())
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

fn sentence_numbers_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<f64>>>> {
    let Some(value) = object.properties.get(SENTENCE_NUMBERS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid SentenceNumbers property"),
        ));
    };
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::Tensor(tensor) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument object has invalid SentenceNumbers entry"),
            ));
        };
        out.push(tensor_utils::tensor_values_f64(tensor));
    }
    Ok(Some(out))
}

fn logical_scalar(value: &Value, fn_name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return match value.try_to_u64() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(text_analytics_error(
                        fn_name,
                        format!(
                            "{fn_name}: logical scalar option must be true or false, got {value:?}"
                        ),
                    )),
                };
            }
            match tensor_utils::tensor_value_f64(tensor, 0) {
                0.0 => Ok(false),
                1.0 => Ok(true),
                other => Err(text_analytics_error(
                    fn_name,
                    format!("{fn_name}: logical scalar option must be true or false, got {other}"),
                )),
            }
        }
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
    use crate::builtins::table::{
        categorical_from_args, table_variable_names_from_object, table_variables,
    };
    use runmat_builtins::{IntegerStorage, LogicalArray};

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_token_details(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(token_details_builtin(value))
    }

    fn run_add_type(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_type_details_builtin(args))
    }

    fn run_add_sentence(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_sentence_details_builtin(args))
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
            Value::Tensor(tensor) => tensor.materialize_f64(),
            other => panic!("expected numeric column {name}, got {other:?}"),
        }
    }

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn poisoned_integer_vector(storage: IntegerStorage, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    #[test]
    fn logical_scalar_reads_typed_integer_storage_exactly() {
        assert!(logical_scalar(
            &poisoned_integer_scalar(IntegerStorage::U64(vec![1])),
            "addSentenceDetails"
        )
        .expect("true"));
        assert!(!logical_scalar(
            &poisoned_integer_scalar(IntegerStorage::I16(vec![0])),
            "addSentenceDetails"
        )
        .expect("false"));
    }

    #[test]
    fn sentence_numbers_from_object_reads_typed_integer_storage_exactly() {
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.properties.insert(
            SENTENCE_NUMBERS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![poisoned_integer_vector(IntegerStorage::I16(vec![2, 3]), 2)],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        assert_eq!(
            sentence_numbers_from_object(&object, "tokenDetails")
                .expect("numbers")
                .expect("stored"),
            vec![vec![2.0, 3.0]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_adds_sentence_number_column() {
        let docs = run_tokenized(vec![Value::StringArray(
            StringArray::new(
                vec![
                    "This is an example document. It has two sentences.".to_string(),
                    "This document has one sentence.".to_string(),
                    "Here is another example document. It also has two sentences.".to_string(),
                ],
                vec![3, 1],
            )
            .unwrap(),
        )])
        .expect("tokenized");
        let updated = run_add_sentence(vec![docs]).expect("sentences");
        let table = object(run_token_details(updated).expect("details"));

        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec![
                "Token",
                "DocumentNumber",
                "SentenceNumber",
                "LineNumber",
                "Type",
                "Language"
            ]
        );
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_handles_abbreviations_and_starters() {
        let docs = run_tokenized(vec![Value::String(
            "Dr. Smith measured 30 in. The width is 10 in. wide. Try fig. 3.".into(),
        )])
        .expect("tokenized");
        let updated = run_add_sentence(vec![docs]).expect("sentences");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Token"),
            vec![
                "Dr", ".", "Smith", "measured", "30", "in", ".", "The", "width", "is", "10", "in",
                ".", "wide", ".", "Try", "fig", ".", "3", "."
            ]
        );
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0,
                3.0, 3.0, 3.0, 3.0
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_accepts_custom_abbreviations_and_starters() {
        let docs = run_tokenized(vec![Value::String(
            "Book an appt. We'll meet then. Book an appt. today.".into(),
        )])
        .expect("tokenized");
        let updated = run_add_sentence(vec![
            docs,
            Value::String("Abbreviations".into()),
            Value::String("appt".into()),
            Value::String("Starters".into()),
            Value::StringArray(StringArray::new(vec!["we'll".into()], vec![1, 1]).unwrap()),
        ])
        .expect("sentences");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_accepts_abbreviation_usage_table() {
        let docs = run_tokenized(vec![Value::String(
            "The dept. chair spoke. See ref. 2. Try ref. again.".into(),
        )])
        .expect("tokenized");
        let abbreviations = table_from_columns(
            vec!["Abbreviation".into(), "Usage".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["dept".into(), "ref".into()], vec![2, 1]).unwrap(),
                ),
                Value::StringArray(
                    StringArray::new(vec!["inner".into(), "reference".into()], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .expect("abbreviation table");

        let updated = run_add_sentence(vec![
            docs,
            Value::String("Abbreviations".into()),
            abbreviations,
        ])
        .expect("sentences");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_does_not_split_inside_initialisms() {
        let docs =
            run_tokenized(vec![Value::String("U.S.A. Today wins.".into())]).expect("tokenized");
        let updated = run_add_sentence(vec![docs]).expect("sentences");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Token"),
            vec!["U", ".", "S", ".", "A", ".", "Today", "wins", "."]
        );
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_accepts_categorical_abbreviation_usage() {
        let docs =
            run_tokenized(vec![Value::String("The dept. chair spoke.".into())]).expect("tokenized");
        let usage = categorical_from_args(vec![Value::StringArray(
            StringArray::new(vec!["inner".into()], vec![1, 1]).unwrap(),
        )])
        .expect("categorical usage");
        let abbreviations = table_from_columns(
            vec!["Abbreviation".into(), "Usage".into()],
            vec![
                Value::StringArray(StringArray::new(vec!["dept".into()], vec![1, 1]).unwrap()),
                usage,
            ],
        )
        .expect("abbreviation table");

        let updated = run_add_sentence(vec![
            docs,
            Value::String("Abbreviations".into()),
            abbreviations,
        ])
        .expect("sentences");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_malformed_sentence_numbers() {
        let docs =
            object(run_tokenized(vec![Value::String("One. Two.".into())]).expect("tokenized"));
        let mut malformed = docs.clone();
        malformed.properties.insert(
            SENTENCE_NUMBERS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap())],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(malformed)).expect_err("expected error");
        assert!(err.to_string().contains("SentenceNumbers entry"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_sentence_details_preserves_existing_numbers_unless_discarding() {
        let docs =
            object(run_tokenized(vec![Value::String("One. Two.".into())]).expect("tokenized"));
        let mut stale = docs.clone();
        stale.properties.insert(
            SENTENCE_NUMBERS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::Tensor(
                        Tensor::new(vec![7.0, 7.0, 7.0, 7.0], vec![1, 4]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );

        let preserved = run_add_sentence(vec![Value::Object(stale.clone())]).expect("preserve");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![7.0, 7.0, 7.0, 7.0]
        );

        let recomputed = run_add_sentence(vec![
            Value::Object(stale),
            Value::String("DiscardKnownValues".into()),
            Value::Bool(true),
        ])
        .expect("recompute");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(
            numeric_column(&table, "SentenceNumber"),
            vec![1.0, 1.0, 2.0, 2.0]
        );
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
    fn token_details_uses_custom_token_type_details() {
        let custom_tokens = table_from_columns(
            vec!["Token".into(), "Type".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["Na+".into(), "H2O".into()], vec![2, 1]).unwrap(),
                ),
                Value::StringArray(
                    StringArray::new(vec!["ion".into(), "formula".into()], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .expect("custom token table");
        let docs = run_tokenized(vec![
            Value::String("Na+ in H2O".to_string()),
            Value::String("CustomTokens".to_string()),
            custom_tokens,
        ])
        .expect("tokenized");
        let table = object(run_token_details(docs).expect("details"));

        assert_eq!(string_column(&table, "Token"), vec!["Na+", "in", "H2O"]);
        assert_eq!(
            string_column(&table, "Type"),
            vec!["ion", "letters", "formula"]
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
