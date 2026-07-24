//! Core Text Analytics document and bag-of-words compatibility objects.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};

use once_cell::sync::Lazy;
use regex::Regex;
use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ClassDef, LogicalArray, ObjectInstance, PropertyDef, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::stopwords::{
    stop_words_for_language, StopWordsLanguage,
};
use crate::builtins::table::table_variables;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

pub const TOKENIZED_DOCUMENT_CLASS: &str = "tokenizedDocument";
pub const BAG_OF_WORDS_CLASS: &str = "bagOfWords";
const MAX_DENSE_BAG_COUNT_CELLS: usize = 50_000_000;

thread_local! {
    static TOKENIZED_DOCUMENT_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
    static BAG_OF_WORDS_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const OUT_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized document object.",
}];

const OUT_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "bag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bag-of-words model object.",
}];

const OUT_DOCUMENTS_OR_BAG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocumentsOrBag",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument or bagOfWords object.",
}];

const OUT_DOCUMENTS_FILTERED: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newDocuments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filtered tokenizedDocument object.",
}];

const IN_TEXT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text or pre-tokenized words.",
}];

const IN_TEXT_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text or pre-tokenized words.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Name-value options: TokenizeMethod, Language, DetectPatterns, TopLevelDomains, CustomTokens, RegularExpressions.",
    },
];

const IN_DOCUMENTS_OR_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Tokenized documents, word vector, or unique vocabulary.",
}];

const IN_WORDS_COUNTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "uniqueWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Unique words.",
    },
    BuiltinParamDescriptor {
        name: "counts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Word counts per document.",
    },
];

const IN_REMOVE_SHORT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrBag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or bagOfWords object.",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum word length to remove.",
    },
];

const IN_REMOVE_LONG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrBag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or bagOfWords object.",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum word length to remove.",
    },
];

const IN_REMOVE_WORDS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "documentsOrBag",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument or bagOfWords object.",
    },
    BuiltinParamDescriptor {
        name: "wordsOrIdx",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to remove or indices into the object's Vocabulary.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

const IN_REMOVE_STOP: [BuiltinParamDescriptor; 2] = [
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
        description: "Name-value options: IgnoreCase.",
    },
];

const IN_REMOVE_STOP_DOCUMENTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documents",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object.",
}];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS_DOCUMENTS.INVALID_INPUT",
    identifier: Some("RunMat:textAnalyticsDocuments:InvalidInput"),
    when: "Inputs do not match a supported Text Analytics document or model helper form.",
    message: "Text Analytics document helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

const ERROR_REMOVE_LONG_WORDS_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REMOVELONGWORDS.INVALID_INPUT",
    identifier: Some("RunMat:removeLongWords:InvalidInput"),
    when: "Inputs do not match a supported removeLongWords form.",
    message: "removeLongWords: invalid input",
};

const REMOVE_LONG_WORDS_ERRORS: [BuiltinErrorDescriptor; 1] =
    [ERROR_REMOVE_LONG_WORDS_INVALID_INPUT];

const ERROR_REMOVE_WORDS_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REMOVEWORDS.INVALID_INPUT",
    identifier: Some("RunMat:removeWords:InvalidInput"),
    when: "Inputs do not match a supported removeWords form.",
    message: "removeWords: invalid input",
};

const REMOVE_WORDS_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_REMOVE_WORDS_INVALID_INPUT];

const ERROR_REMOVE_STOP_WORDS_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REMOVESTOPWORDS.INVALID_INPUT",
    identifier: Some("RunMat:removeStopWords:InvalidInput"),
    when: "Inputs do not match a supported removeStopWords form.",
    message: "removeStopWords: invalid input",
};

const REMOVE_STOP_WORDS_ERRORS: [BuiltinErrorDescriptor; 1] =
    [ERROR_REMOVE_STOP_WORDS_INVALID_INPUT];

pub const TOKENIZED_DOCUMENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument",
            inputs: &[],
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument(str)",
            inputs: &IN_TEXT,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "documents = tokenizedDocument(str, Name, Value, ...)",
            inputs: &IN_TEXT_REST,
            outputs: &OUT_DOCUMENTS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const BAG_OF_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords",
            inputs: &[],
            outputs: &OUT_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords(documents)",
            inputs: &IN_DOCUMENTS_OR_WORDS,
            outputs: &OUT_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "bag = bagOfWords(uniqueWords, counts)",
            inputs: &IN_WORDS_COUNTS,
            outputs: &OUT_BAG,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const REMOVE_SHORT_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newDocumentsOrBag = removeShortWords(documentsOrBag, len)",
        inputs: &IN_REMOVE_SHORT,
        outputs: &OUT_DOCUMENTS_OR_BAG,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const REMOVE_LONG_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "newDocumentsOrBag = removeLongWords(documentsOrBag, len)",
        inputs: &IN_REMOVE_LONG,
        outputs: &OUT_DOCUMENTS_OR_BAG,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_LONG_WORDS_ERRORS,
};

pub const REMOVE_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(documentsOrBag, words)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(documentsOrBag, idx)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
        BuiltinSignatureDescriptor {
            label: "newDocumentsOrBag = removeWords(___, 'IgnoreCase', tf)",
            inputs: &IN_REMOVE_WORDS,
            outputs: &OUT_DOCUMENTS_OR_BAG,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_WORDS_ERRORS,
};

pub const REMOVE_STOP_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "newDocuments = removeStopWords(documents)",
            inputs: &IN_REMOVE_STOP_DOCUMENTS,
            outputs: &OUT_DOCUMENTS_FILTERED,
        },
        BuiltinSignatureDescriptor {
            label: "newDocuments = removeStopWords(documents, Name, Value, ...)",
            inputs: &IN_REMOVE_STOP,
            outputs: &OUT_DOCUMENTS_FILTERED,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REMOVE_STOP_WORDS_ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub(in crate::builtins::strings::text_analytics) fn text_analytics_error(
    fn_name: &str,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let identifier = match fn_name {
        "removeStopWords" => "RunMat:removeStopWords:InvalidInput",
        "removeWords" => "RunMat:removeWords:InvalidInput",
        "removeLongWords" => "RunMat:removeLongWords:InvalidInput",
        "tokenDetails" => "RunMat:tokenDetails:InvalidInput",
        "addTypeDetails" => "RunMat:addTypeDetails:InvalidInput",
        "addSentenceDetails" => "RunMat:addSentenceDetails:InvalidInput",
        "addLemmaDetails" => "RunMat:addLemmaDetails:InvalidInput",
        "addPartOfSpeechDetails" => "RunMat:addPartOfSpeechDetails:InvalidInput",
        "addEntityDetails" => "RunMat:addEntityDetails:InvalidInput",
        "addDependencyDetails" => "RunMat:addDependencyDetails:InvalidInput",
        "vaderSentimentScores" => "RunMat:vaderSentimentScores:InvalidInput",
        _ => "RunMat:textAnalyticsDocuments:InvalidInput",
    };
    build_runtime_error(message)
        .with_builtin(fn_name)
        .with_identifier(identifier)
        .build()
}

fn ensure_tokenized_document_class_registered() {
    TOKENIZED_DOCUMENT_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in [
            "Documents",
            "Vocabulary",
            "NumDocuments",
            "DocumentLengths",
            "Shape",
            "TokenizeMethod",
            "Language",
            "DetectPatterns",
            "TopLevelDomains",
            "TopLevelDomainsCustom",
            "TypeDetails",
            "SentenceNumbers",
            "LemmaDetails",
            "PartOfSpeechDetails",
            "EntityDetails",
            "HeadDetails",
            "DependencyDetails",
        ] {
            properties.insert(name.to_string(), property_def(name));
        }
        runmat_builtins::register_class(ClassDef {
            name: TOKENIZED_DOCUMENT_CLASS.to_string(),
            parent: None,
            properties,
            methods: HashMap::new(),
        });
        registered.set(true);
    });
}

fn ensure_bag_of_words_class_registered() {
    BAG_OF_WORDS_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in ["Counts", "Vocabulary", "NumWords", "NumDocuments"] {
            properties.insert(name.to_string(), property_def(name));
        }
        runmat_builtins::register_class(ClassDef {
            name: BAG_OF_WORDS_CLASS.to_string(),
            parent: None,
            properties,
            methods: HashMap::new(),
        });
        registered.set(true);
    });
}

fn property_def(name: &str) -> PropertyDef {
    PropertyDef {
        name: name.to_string(),
        is_static: false,
        is_constant: false,
        is_dependent: false,
        get_access: Access::Public,
        set_access: Access::Public,
        default_value: None,
    }
}

#[runtime_builtin(
    name = "tokenizedDocument",
    category = "strings/text_analytics",
    summary = "Create tokenized document objects for Text Analytics workflows.",
    keywords = "tokenizedDocument,text analytics,tokenize,document",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::TOKENIZED_DOCUMENT_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
pub(in crate::builtins::strings::text_analytics) async fn tokenized_document_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "tokenizedDocument").await?;
    let (input, options) = parse_tokenized_document_args(gathered)?;
    let parsed = match input {
        Some(value) => documents_from_value(value, &options)?,
        None => ParsedDocuments {
            documents: vec![Vec::new()],
            shape: vec![1, 1],
            type_details: None,
        },
    };
    tokenized_document_value(parsed.documents, parsed.shape, options, parsed.type_details)
}

#[runtime_builtin(
    name = "bagOfWords",
    category = "strings/text_analytics",
    summary = "Create bag-of-words model objects.",
    keywords = "bagOfWords,text analytics,word counts,vocabulary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::BAG_OF_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
async fn bag_of_words_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "bagOfWords").await?;
    match gathered.as_slice() {
        [] => bag_from_documents(Vec::new()),
        [value] => match value {
            Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
                bag_from_documents(documents_from_object(object, "bagOfWords")?)
            }
            Value::Object(object) => Err(text_analytics_error(
                "bagOfWords",
                format!(
                    "bagOfWords: expected tokenizedDocument object, got {}",
                    object.class_name
                ),
            )),
            other => bag_from_documents(vec![words_from_word_vector(other, "bagOfWords")?]),
        },
        [words, counts] => bag_from_unique_words_and_counts(words, counts),
        _ => Err(text_analytics_error(
            "bagOfWords",
            "bagOfWords: expected zero, one, or two inputs",
        )),
    }
}

#[runtime_builtin(
    name = "removeShortWords",
    category = "strings/text_analytics",
    summary = "Remove short words from tokenized documents or bag-of-words models.",
    keywords = "removeShortWords,text analytics,tokenizedDocument,bagOfWords",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::REMOVE_SHORT_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
async fn remove_short_words_builtin(value: Value, len: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        text_analytics_error("removeShortWords", format!("removeShortWords: {err}"))
    })?;
    let len = gather_if_needed_async(&len).await.map_err(|err| {
        text_analytics_error("removeShortWords", format!("removeShortWords: {err}"))
    })?;
    let max_len = parse_positive_integer(&len, "removeShortWords")?;
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            let documents = documents_from_object(&object, "removeShortWords")?
                .into_iter()
                .map(|doc| {
                    doc.into_iter()
                        .filter(|token| token.chars().count() > max_len)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let shape = shape_from_object(&object, "removeShortWords")?;
            let options = options_from_document_object(&object);
            tokenized_document_value(documents, shape, options, None)
        }
        Value::Object(object) if object.is_class(BAG_OF_WORDS_CLASS) => {
            remove_short_words_from_bag(object, max_len)
        }
        Value::Object(object) => Err(text_analytics_error(
            "removeShortWords",
            format!(
                "removeShortWords: expected tokenizedDocument or bagOfWords object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "removeShortWords",
            format!(
                "removeShortWords: expected tokenizedDocument or bagOfWords object, got {other:?}"
            ),
        )),
    }
}

#[runtime_builtin(
    name = "removeLongWords",
    category = "strings/text_analytics",
    summary = "Remove long words from tokenized documents or bag-of-words models.",
    keywords = "removeLongWords,text analytics,tokenizedDocument,bagOfWords",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::REMOVE_LONG_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
async fn remove_long_words_builtin(value: Value, len: Value) -> BuiltinResult<Value> {
    let value = gather_if_needed_async(&value).await.map_err(|err| {
        text_analytics_error("removeLongWords", format!("removeLongWords: {err}"))
    })?;
    let len = gather_if_needed_async(&len).await.map_err(|err| {
        text_analytics_error("removeLongWords", format!("removeLongWords: {err}"))
    })?;
    let min_len = parse_positive_integer(&len, "removeLongWords")?;
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            transform_tokenized_document(&object, "removeLongWords", |token, _| {
                Ok((token.chars().count() < min_len).then(|| token.to_string()))
            })
        }
        Value::Object(object) if object.is_class(BAG_OF_WORDS_CLASS) => {
            filter_bag_columns_by_predicate(object, "removeLongWords", |word| {
                word.chars().count() < min_len
            })
        }
        Value::Object(object) => Err(text_analytics_error(
            "removeLongWords",
            format!(
                "removeLongWords: expected tokenizedDocument or bagOfWords object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "removeLongWords",
            format!(
                "removeLongWords: expected tokenizedDocument or bagOfWords object, got {other:?}"
            ),
        )),
    }
}

#[runtime_builtin(
    name = "removeWords",
    category = "strings/text_analytics",
    summary = "Remove selected words from tokenized documents or bag-of-words models.",
    keywords = "removeWords,text analytics,tokenizedDocument,bagOfWords,filter",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::REMOVE_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
async fn remove_words_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "removeWords").await?;
    let (value, selector, options) = parse_remove_words_args(gathered)?;
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            let vocabulary = documents_vocabulary(&object, "removeWords")?;
            let words = selector.words(&vocabulary, "removeWords")?;
            let remove_set = word_set(words, options.ignore_case);
            transform_tokenized_document(&object, "removeWords", |token, _| {
                let key = comparable_word(token, options.ignore_case);
                Ok((!remove_set.contains(&key)).then(|| token.to_string()))
            })
        }
        Value::Object(object) if object.is_class(BAG_OF_WORDS_CLASS) => {
            let vocabulary = vocabulary_from_bag(&object, "removeWords")?;
            let words = selector.words(&vocabulary, "removeWords")?;
            let remove_set = word_set(words, options.ignore_case);
            filter_bag_columns(object, "removeWords", |word| {
                let key = comparable_word(word, options.ignore_case);
                !remove_set.contains(&key)
            })
        }
        Value::Object(object) => Err(text_analytics_error(
            "removeWords",
            format!(
                "removeWords: expected tokenizedDocument or bagOfWords object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "removeWords",
            format!("removeWords: expected tokenizedDocument or bagOfWords object, got {other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "removeStopWords",
    category = "strings/text_analytics",
    summary = "Remove stop words from tokenized documents.",
    keywords = "removeStopWords,stop words,text analytics,tokenizedDocument",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::documents::REMOVE_STOP_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::documents"
)]
async fn remove_stop_words_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "removeStopWords").await?;
    let (value, options) = parse_remove_stop_words_args(gathered)?;
    let Value::Object(object) = value else {
        return Err(text_analytics_error(
            "removeStopWords",
            format!("removeStopWords: expected tokenizedDocument object, got {value:?}"),
        ));
    };
    if !object.is_class(TOKENIZED_DOCUMENT_CLASS) {
        return Err(text_analytics_error(
            "removeStopWords",
            format!(
                "removeStopWords: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        ));
    }
    let language = stop_words_language_from_document_object(&object, "removeStopWords")?;
    let words = stop_words_for_language(language);
    let stop_words = words
        .iter()
        .map(|word| {
            if options.ignore_case {
                word.to_lowercase()
            } else {
                (*word).to_string()
            }
        })
        .collect::<HashSet<_>>();
    transform_tokenized_document(&object, "removeStopWords", |token, token_type| {
        if !matches!(
            token_type,
            DocumentTokenType::Letters | DocumentTokenType::Other
        ) {
            return Ok(Some(token.to_string()));
        }
        let key = if options.ignore_case {
            token.to_lowercase()
        } else {
            token.to_string()
        };
        Ok((!stop_words.contains(&key)).then(|| token.to_string()))
    })
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
pub(in crate::builtins::strings::text_analytics) struct DocumentOptions {
    pub(in crate::builtins::strings::text_analytics) tokenize_method: TokenizeMethod,
    pub(in crate::builtins::strings::text_analytics) language: String,
    pub(in crate::builtins::strings::text_analytics) detect_patterns: DetectPatterns,
    pub(in crate::builtins::strings::text_analytics) top_level_domains: Vec<String>,
    pub(in crate::builtins::strings::text_analytics) top_level_domains_custom: bool,
    custom_tokens: Vec<CustomTokenRule>,
    regular_expressions: Vec<RegexTokenRule>,
}

impl Default for DocumentOptions {
    fn default() -> Self {
        Self {
            tokenize_method: TokenizeMethod::Unicode,
            language: "en".to_string(),
            detect_patterns: DetectPatterns::All,
            top_level_domains: default_top_level_domains(),
            top_level_domains_custom: false,
            custom_tokens: Vec::new(),
            regular_expressions: Vec::new(),
        }
    }
}

impl DocumentOptions {
    fn requires_type_details(&self) -> bool {
        !self.custom_tokens.is_empty() || !self.regular_expressions.is_empty()
    }
}

#[derive(Clone, Debug)]
struct CustomTokenRule {
    token: String,
    token_type: String,
}

#[derive(Clone, Debug)]
struct RegexTokenRule {
    regex: Regex,
    token_type: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum TokenizeMethod {
    Unicode,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum DetectPatterns {
    All,
    None,
    Selected(ComplexPatternSet),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) struct ComplexPatternSet {
    email_address: bool,
    web_address: bool,
    hashtag: bool,
    at_mention: bool,
    emoticon: bool,
}

impl ComplexPatternSet {
    fn empty() -> Self {
        Self {
            email_address: false,
            web_address: false,
            hashtag: false,
            at_mention: false,
            emoticon: false,
        }
    }
}

impl DetectPatterns {
    fn detects_email_address(self) -> bool {
        matches!(
            self,
            Self::All
                | Self::Selected(ComplexPatternSet {
                    email_address: true,
                    ..
                })
        )
    }

    fn detects_web_address(self) -> bool {
        matches!(
            self,
            Self::All
                | Self::Selected(ComplexPatternSet {
                    web_address: true,
                    ..
                })
        )
    }

    fn detects_hashtag(self) -> bool {
        matches!(
            self,
            Self::All | Self::Selected(ComplexPatternSet { hashtag: true, .. })
        )
    }

    fn detects_at_mention(self) -> bool {
        matches!(
            self,
            Self::All
                | Self::Selected(ComplexPatternSet {
                    at_mention: true,
                    ..
                })
        )
    }

    fn detects_emoticon(self) -> bool {
        matches!(
            self,
            Self::All | Self::Selected(ComplexPatternSet { emoticon: true, .. })
        )
    }
}

fn parse_tokenized_document_args(
    args: Vec<Value>,
) -> BuiltinResult<(Option<Value>, DocumentOptions)> {
    if args.is_empty() {
        return Ok((None, DocumentOptions::default()));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "tokenizedDocument",
            "tokenizedDocument: name-value options must appear in pairs",
        ));
    }
    let input = args[0].clone();
    let mut options = DocumentOptions::default();
    let mut idx = 1;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "tokenizedDocument")
            .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "tokenizemethod" => {
                let value = scalar_text(&args[idx + 1], "tokenizedDocument")
                    .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?;
                options.tokenize_method = parse_tokenize_method(&value)?;
            }
            "language" => {
                let value = scalar_text(&args[idx + 1], "tokenizedDocument")
                    .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?;
                options.language = parse_language(&value)?;
            }
            "detectpatterns" => {
                options.detect_patterns =
                    parse_detect_patterns(&args[idx + 1], "tokenizedDocument")?;
            }
            "topleveldomains" => {
                options.top_level_domains =
                    parse_top_level_domains(&args[idx + 1], "tokenizedDocument")?;
                options.top_level_domains_custom = true;
            }
            "customtokens" => {
                options.custom_tokens = parse_custom_tokens(&args[idx + 1])?;
            }
            "regularexpressions" => {
                options.regular_expressions = parse_regular_expressions(&args[idx + 1])?;
            }
            _ => {
                return Err(text_analytics_error(
                    "tokenizedDocument",
                    format!("tokenizedDocument: unsupported option '{name}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((Some(input), options))
}

fn parse_tokenize_method(value: &str) -> BuiltinResult<TokenizeMethod> {
    match value.trim().to_ascii_lowercase().as_str() {
        "unicode" => Ok(TokenizeMethod::Unicode),
        "none" => Ok(TokenizeMethod::None),
        "mecab" => Err(text_analytics_error(
            "tokenizedDocument",
            "tokenizedDocument: TokenizeMethod 'mecab' requires Japanese/Korean tokenizer support and remains tracked",
        )),
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: unsupported TokenizeMethod '{other}'"),
        )),
    }
}

fn parse_language(value: &str) -> BuiltinResult<String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "en" | "de" => Ok(value.trim().to_ascii_lowercase()),
        "ja" | "ko" => Err(text_analytics_error(
            "tokenizedDocument",
            "tokenizedDocument: Japanese/Korean tokenization requires MeCab-compatible support and remains tracked",
        )),
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: Language must be 'en' or 'de' in this slice, got '{other}'"),
        )),
    }
}

fn parse_detect_patterns(value: &Value, fn_name: &str) -> BuiltinResult<DetectPatterns> {
    let patterns = words_from_word_vector(value, fn_name)?
        .into_iter()
        .map(|pattern| pattern.trim().to_ascii_lowercase())
        .filter(|pattern| !pattern.is_empty())
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: DetectPatterns must contain at least one pattern"),
        ));
    }
    if patterns.len() == 1 {
        match patterns[0].as_str() {
            "all" => return Ok(DetectPatterns::All),
            "none" => return Ok(DetectPatterns::None),
            _ => {}
        }
    }

    let mut selected = ComplexPatternSet::empty();
    for pattern in patterns {
        match pattern.as_str() {
            "email-address" => selected.email_address = true,
            "web-address" => selected.web_address = true,
            "hashtag" => selected.hashtag = true,
            "at-mention" => selected.at_mention = true,
            "emoticon" => selected.emoticon = true,
            "all" | "none" => {
                return Err(text_analytics_error(
                    fn_name,
                    format!("{fn_name}: DetectPatterns '{pattern}' must be specified alone"),
                ))
            }
            other => {
                return Err(text_analytics_error(
                    fn_name,
                    format!("{fn_name}: unsupported DetectPatterns value '{other}'"),
                ))
            }
        }
    }
    Ok(DetectPatterns::Selected(selected))
}

pub(in crate::builtins::strings::text_analytics) fn parse_top_level_domains(
    value: &Value,
    fn_name: &str,
) -> BuiltinResult<Vec<String>> {
    let domains = words_from_word_vector_preserving_missing(value, fn_name)?;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in domains {
        if is_missing_string(&raw) {
            continue;
        }
        let domain = raw
            .trim()
            .trim_start_matches('.')
            .trim_end_matches('.')
            .to_ascii_lowercase();
        if domain.is_empty()
            || domain.contains('.')
            || !domain
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
            || domain.starts_with('-')
            || domain.ends_with('-')
        {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: TopLevelDomains contains invalid domain '{raw}'"),
            ));
        }
        if seen.insert(domain.clone()) {
            out.push(domain);
        }
    }
    if out.is_empty() {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: TopLevelDomains must contain at least one domain"),
        ));
    }
    Ok(out)
}

fn parse_custom_tokens(value: &Value) -> BuiltinResult<Vec<CustomTokenRule>> {
    if let Value::Object(object) = value {
        let variables = table_variables(object).map_err(|err| {
            text_analytics_error(
                "tokenizedDocument",
                format!("tokenizedDocument: invalid CustomTokens table: {err}"),
            )
        })?;
        if variables.fields.contains_key("Token") || variables.fields.contains_key("Type") {
            let tokens = text_column(&variables, "Token", "CustomTokens")?;
            let types = optional_text_column(&variables, "Type", "CustomTokens", tokens.len())?;
            return custom_token_rules_from_columns(tokens, types);
        }
    }
    let tokens = words_from_word_vector_preserving_missing(value, "tokenizedDocument")?;
    custom_token_rules_from_columns(tokens, None)
}

fn parse_regular_expressions(value: &Value) -> BuiltinResult<Vec<RegexTokenRule>> {
    if let Value::Object(object) = value {
        let variables = table_variables(object).map_err(|err| {
            text_analytics_error(
                "tokenizedDocument",
                format!("tokenizedDocument: invalid RegularExpressions table: {err}"),
            )
        })?;
        if variables.fields.contains_key("Pattern") || variables.fields.contains_key("Type") {
            let patterns = text_column(&variables, "Pattern", "RegularExpressions")?;
            let types =
                optional_text_column(&variables, "Type", "RegularExpressions", patterns.len())?;
            return regex_token_rules_from_columns(patterns, types);
        }
    }
    let patterns = words_from_word_vector_preserving_missing(value, "tokenizedDocument")?;
    regex_token_rules_from_columns(patterns, None)
}

fn text_column(
    variables: &runmat_builtins::StructValue,
    name: &str,
    option_name: &str,
) -> BuiltinResult<Vec<String>> {
    let Some(value) = variables.fields.get(name) else {
        return Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: {option_name} table must contain a {name} variable"),
        ));
    };
    words_from_word_vector_preserving_missing(value, "tokenizedDocument")
}

fn optional_text_column(
    variables: &runmat_builtins::StructValue,
    name: &str,
    option_name: &str,
    expected_len: usize,
) -> BuiltinResult<Option<Vec<String>>> {
    let Some(value) = variables.fields.get(name) else {
        return Ok(None);
    };
    let values = words_from_word_vector_preserving_missing(value, "tokenizedDocument")?;
    if values.len() != expected_len {
        return Err(text_analytics_error(
            "tokenizedDocument",
            format!(
                "tokenizedDocument: {option_name} table variable {name} has {} rows but expected {expected_len}",
                values.len()
            ),
        ));
    }
    Ok(Some(values))
}

fn custom_token_rules_from_columns(
    tokens: Vec<String>,
    types: Option<Vec<String>>,
) -> BuiltinResult<Vec<CustomTokenRule>> {
    let mut rules = Vec::new();
    let mut seen = HashSet::new();
    for (idx, token) in tokens.into_iter().enumerate() {
        if is_missing_string(&token) {
            continue;
        }
        if token.is_empty() {
            return Err(text_analytics_error(
                "tokenizedDocument",
                "tokenizedDocument: CustomTokens cannot contain empty tokens",
            ));
        }
        let token_type = types
            .as_ref()
            .and_then(|values| values.get(idx))
            .filter(|value| !is_missing_string(value) && !value.is_empty())
            .cloned()
            .unwrap_or_else(|| "custom".to_string());
        if seen.insert(token.clone()) {
            rules.push(CustomTokenRule { token, token_type });
        }
    }
    Ok(rules)
}

fn regex_token_rules_from_columns(
    patterns: Vec<String>,
    types: Option<Vec<String>>,
) -> BuiltinResult<Vec<RegexTokenRule>> {
    let mut rules = Vec::new();
    for (idx, pattern) in patterns.into_iter().enumerate() {
        if is_missing_string(&pattern) {
            continue;
        }
        if pattern.is_empty() {
            return Err(text_analytics_error(
                "tokenizedDocument",
                "tokenizedDocument: RegularExpressions cannot contain empty patterns",
            ));
        }
        let regex = Regex::new(&pattern).map_err(|err| {
            text_analytics_error(
                "tokenizedDocument",
                format!("tokenizedDocument: invalid RegularExpressions pattern '{pattern}': {err}"),
            )
        })?;
        if regex
            .find("")
            .is_some_and(|mat| mat.start() == 0 && mat.end() == 0)
        {
            return Err(text_analytics_error(
                "tokenizedDocument",
                format!(
                    "tokenizedDocument: RegularExpressions pattern '{pattern}' can match empty text"
                ),
            ));
        }
        let token_type = types
            .as_ref()
            .and_then(|values| values.get(idx))
            .filter(|value| !is_missing_string(value) && !value.is_empty())
            .cloned()
            .unwrap_or_else(|| "custom".to_string());
        rules.push(RegexTokenRule { regex, token_type });
    }
    Ok(rules)
}

#[derive(Clone, Copy, Debug)]
struct RemoveStopWordsOptions {
    ignore_case: bool,
}

impl Default for RemoveStopWordsOptions {
    fn default() -> Self {
        Self { ignore_case: true }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RemoveWordsOptions {
    ignore_case: bool,
}

#[derive(Clone, Debug)]
enum RemoveWordsSelector {
    Words(Vec<String>),
    Indices(Vec<usize>),
    LogicalMask { indices: Vec<usize>, len: usize },
}

impl RemoveWordsSelector {
    fn words(self, vocabulary: &[String], fn_name: &str) -> BuiltinResult<Vec<String>> {
        match self {
            Self::Words(words) => Ok(words),
            Self::LogicalMask { indices, len } => {
                if len != vocabulary.len() {
                    return Err(text_analytics_error(
                        fn_name,
                        format!(
                            "{fn_name}: logical index length {len} must match vocabulary length {}",
                            vocabulary.len()
                        ),
                    ));
                }
                indices
                    .into_iter()
                    .map(|idx| {
                        vocabulary.get(idx).cloned().ok_or_else(|| {
                            text_analytics_error(
                                fn_name,
                                format!(
                                    "{fn_name}: vocabulary index {} exceeds vocabulary length {}",
                                    idx + 1,
                                    vocabulary.len()
                                ),
                            )
                        })
                    })
                    .collect()
            }
            Self::Indices(indices) => indices
                .into_iter()
                .map(|idx| {
                    vocabulary.get(idx).cloned().ok_or_else(|| {
                        text_analytics_error(
                            fn_name,
                            format!(
                                "{fn_name}: vocabulary index {} exceeds vocabulary length {}",
                                idx + 1,
                                vocabulary.len()
                            ),
                        )
                    })
                })
                .collect(),
        }
    }
}

fn parse_remove_stop_words_args(
    args: Vec<Value>,
) -> BuiltinResult<(Value, RemoveStopWordsOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "removeStopWords",
            "removeStopWords: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "removeStopWords",
            "removeStopWords: name-value options must appear in pairs",
        ));
    }
    let mut options = RemoveStopWordsOptions::default();
    let mut idx = 1;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "removeStopWords")
            .map_err(|err| text_analytics_error("removeStopWords", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "ignorecase" => {
                options.ignore_case = parse_bool_scalar(&args[idx + 1], "removeStopWords")?;
            }
            other => {
                return Err(text_analytics_error(
                    "removeStopWords",
                    format!("removeStopWords: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((args[0].clone(), options))
}

fn parse_remove_words_args(
    args: Vec<Value>,
) -> BuiltinResult<(Value, RemoveWordsSelector, RemoveWordsOptions)> {
    if args.len() < 2 {
        return Err(text_analytics_error(
            "removeWords",
            "removeWords: expected input object and words or indices",
        ));
    }
    if !(args.len() - 2).is_multiple_of(2) {
        return Err(text_analytics_error(
            "removeWords",
            "removeWords: name-value options must appear in pairs",
        ));
    }
    let mut options = RemoveWordsOptions::default();
    let mut idx = 2;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "removeWords")
            .map_err(|err| text_analytics_error("removeWords", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "ignorecase" => {
                options.ignore_case = parse_bool_scalar(&args[idx + 1], "removeWords")?;
            }
            other => {
                return Err(text_analytics_error(
                    "removeWords",
                    format!("removeWords: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    let selector = parse_remove_words_selector(&args[1])?;
    Ok((args[0].clone(), selector, options))
}

fn parse_remove_words_selector(value: &Value) -> BuiltinResult<RemoveWordsSelector> {
    match value {
        Value::Bool(value) => Ok(RemoveWordsSelector::LogicalMask {
            indices: (*value).then_some(0).into_iter().collect(),
            len: 1,
        }),
        Value::LogicalArray(array) => {
            let indices = logical_indices(array);
            Ok(RemoveWordsSelector::LogicalMask {
                indices,
                len: array.data.len(),
            })
        }
        Value::Num(_) | Value::Tensor(_) => {
            parse_remove_words_indices(value).map(RemoveWordsSelector::Indices)
        }
        _ => words_from_word_vector(value, "removeWords").map(RemoveWordsSelector::Words),
    }
}

fn parse_remove_words_indices(value: &Value) -> BuiltinResult<Vec<usize>> {
    let raw = match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(tensor) => tensor.data.clone(),
        other => {
            return Err(text_analytics_error(
                "removeWords",
                format!("removeWords: expected numeric or logical indices, got {other:?}"),
            ))
        }
    };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut seen = HashSet::new();
    let mut indices = Vec::new();
    for value in raw {
        if !value.is_finite() || value <= 0.0 || value.fract() != 0.0 {
            return Err(text_analytics_error(
                "removeWords",
                format!("removeWords: vocabulary indices must be positive integers, got {value}"),
            ));
        }
        let idx = value as usize - 1;
        if seen.insert(idx) {
            indices.push(idx);
        }
    }
    Ok(indices)
}

fn logical_indices(array: &LogicalArray) -> Vec<usize> {
    array
        .data
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| (*flag != 0).then_some(idx))
        .collect()
}

fn word_set(words: Vec<String>, ignore_case: bool) -> HashSet<String> {
    words
        .into_iter()
        .map(|word| comparable_word(&word, ignore_case))
        .collect()
}

fn comparable_word(word: &str, ignore_case: bool) -> String {
    if ignore_case {
        word.to_lowercase()
    } else {
        word.to_string()
    }
}

struct ParsedDocuments {
    documents: Vec<Vec<String>>,
    shape: Vec<usize>,
    type_details: Option<Vec<Vec<String>>>,
}

struct TokenizedText {
    tokens: Vec<String>,
    types: Option<Vec<String>>,
}

fn documents_from_value(value: Value, options: &DocumentOptions) -> BuiltinResult<ParsedDocuments> {
    match options.tokenize_method {
        TokenizeMethod::Unicode => text_documents(value, options),
        TokenizeMethod::None => pretokenized_documents(value),
    }
}

fn text_documents(value: Value, options: &DocumentOptions) -> BuiltinResult<ParsedDocuments> {
    match value {
        Value::String(text) => {
            let tokenized = tokenize_text(&text, options);
            Ok(ParsedDocuments {
                type_details: tokenized.types.clone().map(|types| vec![types]),
                documents: vec![tokenized.tokens],
                shape: vec![1, 1],
            })
        }
        Value::StringArray(array) => {
            let shape = array.shape.clone();
            let mut docs = Vec::with_capacity(array.data.len());
            let mut type_details = options.requires_type_details().then(Vec::new);
            for text in array.data {
                let tokenized = if is_missing_string(&text) {
                    TokenizedText {
                        tokens: Vec::new(),
                        types: type_details.as_ref().map(|_| Vec::new()),
                    }
                } else {
                    tokenize_text(&text, options)
                };
                if let Some(details) = &mut type_details {
                    details.push(tokenized.types.unwrap_or_default());
                }
                docs.push(tokenized.tokens);
            }
            Ok(ParsedDocuments {
                documents: docs,
                shape,
                type_details,
            })
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            let tokenized = tokenize_text(&text, options);
            Ok(ParsedDocuments {
                type_details: tokenized.types.clone().map(|types| vec![types]),
                documents: vec![tokenized.tokens],
                shape: vec![1, 1],
            })
        }
        Value::CharArray(array) => {
            let mut docs = Vec::with_capacity(array.rows);
            let mut type_details = options.requires_type_details().then(Vec::new);
            for row in 0..array.rows {
                let tokenized = tokenize_text(
                    &char_row_to_string_slice(&array.data, array.cols, row),
                    options,
                );
                if let Some(details) = &mut type_details {
                    details.push(tokenized.types.unwrap_or_default());
                }
                docs.push(tokenized.tokens);
            }
            Ok(ParsedDocuments {
                documents: docs,
                shape: vec![array.rows, 1],
                type_details,
            })
        }
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            let mut docs = Vec::with_capacity(cell.data.len());
            let mut type_details = options.requires_type_details().then(Vec::new);
            for item in cell.data {
                let text = scalar_text(&item, "tokenizedDocument")
                    .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?;
                let tokenized = tokenize_text(&text, options);
                if let Some(details) = &mut type_details {
                    details.push(tokenized.types.unwrap_or_default());
                }
                docs.push(tokenized.tokens);
            }
            Ok(ParsedDocuments {
                documents: docs,
                shape,
                type_details,
            })
        }
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: expected text input, got {other:?}"),
        )),
    }
}

fn pretokenized_documents(value: Value) -> BuiltinResult<ParsedDocuments> {
    match value {
        Value::String(text) => Ok(ParsedDocuments {
            documents: vec![vec![text]],
            shape: vec![1, 1],
            type_details: None,
        }),
        Value::StringArray(array) => Ok(ParsedDocuments {
            documents: vec![array
                .data
                .into_iter()
                .filter(|text| !is_missing_string(text))
                .collect()],
            shape: vec![1, 1],
            type_details: None,
        }),
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            Ok(ParsedDocuments {
                documents: vec![vec![text]],
                shape: vec![1, 1],
                type_details: None,
            })
        }
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            if cell.data.len() == 1 {
                if let Value::StringArray(array) = &cell.data[0] {
                    return Ok(ParsedDocuments {
                        documents: vec![array
                            .data
                            .iter()
                            .filter(|text| !is_missing_string(text))
                            .cloned()
                            .collect()],
                        shape: vec![1, 1],
                        type_details: None,
                    });
                }
            }
            let mut all_string_arrays = true;
            let mut docs = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                match item {
                    Value::StringArray(array) => docs.push(
                        array
                            .data
                            .into_iter()
                            .filter(|text| !is_missing_string(text))
                            .collect(),
                    ),
                    other => {
                        all_string_arrays = false;
                        docs.push(vec![scalar_text(&other, "tokenizedDocument").map_err(
                            |err| text_analytics_error("tokenizedDocument", err.to_string()),
                        )?]);
                    }
                }
            }
            if all_string_arrays {
                Ok(ParsedDocuments {
                    documents: docs,
                    shape,
                    type_details: None,
                })
            } else {
                Ok(ParsedDocuments {
                    documents: vec![docs.into_iter().flatten().collect()],
                    shape: vec![1, 1],
                    type_details: None,
                })
            }
        }
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: expected pre-tokenized word vector, got {other:?}"),
        )),
    }
}

fn tokenize_text(text: &str, options: &DocumentOptions) -> TokenizedText {
    let mut tokens = Vec::new();
    let mut types = options.requires_type_details().then(Vec::new);
    let mut pos = 0;
    while pos < text.len() {
        let rest = &text[pos..];
        if let Some(ch) = rest.chars().next() {
            if ch.is_whitespace() {
                pos += ch.len_utf8();
                continue;
            }
        }
        if let Some((token, token_type, end)) = custom_or_regex_token_at(text, pos, options) {
            if let Some(types) = &mut types {
                types.push(token_type);
            }
            tokens.push(token);
            pos = end;
            continue;
        }
        if let Some((token, end)) = complex_token_at(text, pos, options) {
            if let Some(types) = &mut types {
                types.push(
                    document_token_type_with_options(&token, options)
                        .as_str()
                        .to_string(),
                );
            }
            tokens.push(token);
            pos = end;
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        if ch.is_alphanumeric() || ch == '_' {
            let start = pos;
            pos += ch.len_utf8();
            while pos < text.len() {
                let next = text[pos..].chars().next().unwrap();
                if next.is_alphanumeric() || next == '_' || next == '\'' {
                    pos += next.len_utf8();
                } else {
                    break;
                }
            }
            let token = text[start..pos].to_string();
            if let Some(types) = &mut types {
                types.push(
                    document_token_type_with_options(&token, options)
                        .as_str()
                        .to_string(),
                );
            }
            tokens.push(token);
            continue;
        }
        let token = ch.to_string();
        if let Some(types) = &mut types {
            types.push(
                document_token_type_with_options(&token, options)
                    .as_str()
                    .to_string(),
            );
        }
        tokens.push(token);
        pos += ch.len_utf8();
    }
    TokenizedText { tokens, types }
}

fn custom_or_regex_token_at(
    text: &str,
    pos: usize,
    options: &DocumentOptions,
) -> Option<(String, String, usize)> {
    let rest = &text[pos..];
    let mut regex_match = None;
    for rule in &options.regular_expressions {
        let Some(mat) = rule.regex.find(rest) else {
            continue;
        };
        if mat.start() == 0 && mat.end() > 0 {
            regex_match = Some((
                rest[..mat.end()].to_string(),
                rule.token_type.clone(),
                pos + mat.end(),
            ));
        }
    }
    if regex_match.is_some() {
        return regex_match;
    }

    options
        .custom_tokens
        .iter()
        .filter(|rule| rest.starts_with(&rule.token))
        .max_by_key(|rule| rule.token.len())
        .map(|rule| {
            (
                rule.token.clone(),
                rule.token_type.clone(),
                pos + rule.token.len(),
            )
        })
}

fn complex_token_at(text: &str, pos: usize, options: &DocumentOptions) -> Option<(String, usize)> {
    let rest = &text[pos..];
    if options.detect_patterns.detects_web_address() {
        if let Some((token, end)) = web_token_at(text, pos, options) {
            return Some((token, end));
        }
    }
    if options.detect_patterns.detects_email_address() {
        if let Some((token, end)) = email_token_at(text, pos) {
            return Some((token, end));
        }
    }
    if (options.detect_patterns.detects_hashtag() && rest.starts_with('#'))
        || (options.detect_patterns.detects_at_mention() && rest.starts_with('@'))
    {
        let mut end = pos + 1;
        while end < text.len() {
            let ch = text[end..].chars().next().unwrap();
            if ch.is_ascii_alphanumeric() || ch == '_' {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        if end > pos + 1 {
            return Some((text[pos..end].to_string(), end));
        }
    }
    if options.detect_patterns.detects_emoticon()
        && (rest.starts_with(":-)")
            || rest.starts_with(":-D")
            || rest.starts_with(":)")
            || rest.starts_with(":D"))
    {
        let len = if rest.starts_with(":-)") || rest.starts_with(":-D") {
            3
        } else {
            2
        };
        let end = pos + len;
        return Some((text[pos..end].to_string(), end));
    }
    None
}

fn web_token_at(text: &str, pos: usize, options: &DocumentOptions) -> Option<(String, usize)> {
    let rest = &text[pos..];
    if !(has_web_scheme(rest)
        || starts_with_ascii_ci(rest, "www.")
        || rest
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphanumeric()))
    {
        return None;
    }
    let end = pos + take_while_nonspace(rest);
    let token = text[pos..end].trim_end_matches(is_trailing_punctuation);
    if !is_web_address_token(token, options) {
        return None;
    }
    let token_end = pos + token.len();
    Some((token.to_string(), token_end))
}

fn email_token_at(text: &str, pos: usize) -> Option<(String, usize)> {
    let rest = &text[pos..];
    let at = rest.find('@')?;
    if at == 0 {
        return None;
    }
    let local = &rest[..at];
    if !local
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '%' | '+' | '-'))
    {
        return None;
    }
    let after_at = &rest[at + 1..];
    let domain_len = after_at
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-'))
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()?;
    let raw_end = pos + at + 1 + domain_len;
    let token = text[pos..raw_end].trim_end_matches(is_trailing_punctuation);
    if !token.contains('.') || token.ends_with('@') {
        return None;
    }
    let token_end = pos + token.len();
    Some((token.to_string(), token_end))
}

fn is_web_address_token(token: &str, options: &DocumentOptions) -> bool {
    if !web_address_candidate(token) {
        return false;
    }
    let Some(host) = web_address_host(token) else {
        return false;
    };
    let has_explicit_prefix = has_web_scheme(token) || starts_with_ascii_ci(token, "www.");
    if has_explicit_prefix && !options.top_level_domains_custom {
        return domain_host_has_valid_tld_shape(host);
    }
    domain_host_has_allowed_tld(host, &options.top_level_domains)
}

fn web_address_host(token: &str) -> Option<&str> {
    let stripped = strip_ascii_prefix(token, "http://")
        .or_else(|| strip_ascii_prefix(token, "https://"))
        .unwrap_or(token);
    let host_and_port = stripped
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(stripped)
        .trim_end_matches(is_trailing_punctuation);
    let host = host_and_port.split(':').next().unwrap_or(host_and_port);
    if host.is_empty() || host.contains('@') {
        return None;
    }
    Some(host)
}

fn web_address_candidate(token: &str) -> bool {
    token.contains('.')
        && (has_web_scheme(token)
            || starts_with_ascii_ci(token, "www.")
            || token
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphanumeric()))
}

fn has_web_scheme(text: &str) -> bool {
    starts_with_ascii_ci(text, "http://") || starts_with_ascii_ci(text, "https://")
}

fn starts_with_ascii_ci(text: &str, prefix: &str) -> bool {
    text.as_bytes()
        .get(..prefix.len())
        .is_some_and(|bytes| bytes.eq_ignore_ascii_case(prefix.as_bytes()))
}

fn strip_ascii_prefix<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    starts_with_ascii_ci(text, prefix).then(|| &text[prefix.len()..])
}

fn domain_host_has_valid_tld_shape(host: &str) -> bool {
    domain_host_tld(host).is_some_and(|tld| tld.len() >= 2)
}

fn domain_host_has_allowed_tld(host: &str, top_level_domains: &[String]) -> bool {
    domain_host_tld(host)
        .map(|tld| top_level_domains.iter().any(|domain| domain == &tld))
        .unwrap_or(false)
}

fn domain_host_tld(host: &str) -> Option<String> {
    let host = host.trim_matches('.');
    let mut saw_dot = false;
    let mut tld = None;
    for label in host.split('.') {
        if label.is_empty()
            || label.starts_with('-')
            || label.ends_with('-')
            || !label
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
        {
            return None;
        }
        if tld.is_some() {
            saw_dot = true;
        }
        tld = Some(label);
    }
    saw_dot.then(|| {
        tld.expect("tld present after label loop")
            .to_ascii_lowercase()
    })
}

fn default_top_level_domains() -> Vec<String> {
    [
        "com", "org", "net", "edu", "gov", "mil", "int", "io", "co", "ai", "dev", "app", "info",
        "biz", "name", "pro", "us", "uk", "ca", "au", "de", "fr", "jp", "kr", "cn", "in", "br",
        "mx", "es", "it", "nl", "se", "no", "fi", "dk", "ch", "at", "be", "ie", "nz", "za", "sg",
        "hk", "tw", "ru", "pl", "cz", "me", "tv", "ly", "xyz", "site", "online", "tech",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn take_while_nonspace(text: &str) -> usize {
    text.char_indices()
        .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx))
        .unwrap_or(text.len())
}

fn is_trailing_punctuation(ch: char) -> bool {
    matches!(ch, '.' | ',' | ';' | ':' | '!' | '?')
}

fn tokenized_document_value(
    documents: Vec<Vec<String>>,
    shape: Vec<usize>,
    options: DocumentOptions,
    type_details: Option<Vec<Vec<String>>>,
) -> BuiltinResult<Value> {
    ensure_tokenized_document_class_registered();
    let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
    object
        .properties
        .insert("Documents".to_string(), documents_cell(&documents)?);
    object
        .properties
        .insert("Vocabulary".to_string(), vocabulary_value(&documents)?);
    object.properties.insert(
        "NumDocuments".to_string(),
        Value::Num(documents.len() as f64),
    );
    object.properties.insert(
        "DocumentLengths".to_string(),
        Value::Tensor(
            Tensor::new(
                documents.iter().map(|doc| doc.len() as f64).collect(),
                vec![documents.len(), 1],
            )
            .map_err(|err| text_analytics_error("tokenizedDocument", err))?,
        ),
    );
    object.properties.insert(
        "Shape".to_string(),
        Value::Tensor(
            Tensor::new(
                shape.iter().map(|dim| *dim as f64).collect(),
                vec![1, shape.len()],
            )
            .map_err(|err| text_analytics_error("tokenizedDocument", err))?,
        ),
    );
    object.properties.insert(
        "TokenizeMethod".to_string(),
        Value::String(
            match options.tokenize_method {
                TokenizeMethod::Unicode => "unicode",
                TokenizeMethod::None => "none",
            }
            .to_string(),
        ),
    );
    object
        .properties
        .insert("Language".to_string(), Value::String(options.language));
    object.properties.insert(
        "DetectPatterns".to_string(),
        detect_patterns_value(options.detect_patterns)?,
    );
    object.properties.insert(
        "TopLevelDomains".to_string(),
        top_level_domains_value(&options.top_level_domains, "tokenizedDocument")?,
    );
    object.properties.insert(
        "TopLevelDomainsCustom".to_string(),
        Value::Bool(options.top_level_domains_custom),
    );
    if let Some(type_details) = type_details {
        object
            .properties
            .insert("TypeDetails".to_string(), type_details_cell(&type_details)?);
    }
    Ok(Value::Object(object))
}

fn type_details_cell(type_details: &[Vec<String>]) -> BuiltinResult<Value> {
    let values = type_details
        .iter()
        .map(|types| {
            StringArray::new(types.clone(), vec![1, types.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("tokenizedDocument", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, type_details.len(), 1)
            .map_err(|err| text_analytics_error("tokenizedDocument", err))?,
    ))
}

fn detect_patterns_value(patterns: DetectPatterns) -> BuiltinResult<Value> {
    match patterns {
        DetectPatterns::All => Ok(Value::String("all".to_string())),
        DetectPatterns::None => Ok(Value::String("none".to_string())),
        DetectPatterns::Selected(selected) => {
            let mut values = Vec::new();
            if selected.email_address {
                values.push("email-address".to_string());
            }
            if selected.web_address {
                values.push("web-address".to_string());
            }
            if selected.hashtag {
                values.push("hashtag".to_string());
            }
            if selected.at_mention {
                values.push("at-mention".to_string());
            }
            if selected.emoticon {
                values.push("emoticon".to_string());
            }
            StringArray::new(values.clone(), vec![1, values.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("tokenizedDocument", err))
        }
    }
}

pub(in crate::builtins::strings::text_analytics) fn top_level_domains_value(
    domains: &[String],
    fn_name: &str,
) -> BuiltinResult<Value> {
    StringArray::new(domains.to_vec(), vec![1, domains.len()])
        .map(Value::StringArray)
        .map_err(|err| text_analytics_error(fn_name, err))
}

pub(in crate::builtins::strings::text_analytics) fn tokenized_document_language(
    object: &ObjectInstance,
) -> String {
    match object.properties.get("Language") {
        Some(Value::String(value)) => value.clone(),
        _ => "en".to_string(),
    }
}

pub(in crate::builtins::strings::text_analytics) fn transform_tokenized_document(
    object: &ObjectInstance,
    fn_name: &str,
    mut transform: impl FnMut(&str, DocumentTokenType) -> BuiltinResult<Option<String>>,
) -> BuiltinResult<Value> {
    let options = options_from_document_object(object);
    let documents = documents_from_object(object, fn_name)?
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .filter_map(|token| {
                    match transform(&token, document_token_type_with_options(&token, &options)) {
                        Ok(Some(value)) => Some(Ok(value)),
                        Ok(None) => None,
                        Err(err) => Some(Err(err)),
                    }
                })
                .collect::<BuiltinResult<Vec<_>>>()
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    tokenized_document_value(
        documents,
        shape_from_object(object, fn_name)?,
        options,
        None,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum DocumentTokenType {
    Letters,
    Digits,
    Punctuation,
    Other,
    WebAddress,
    EmailAddress,
    Hashtag,
    AtMention,
    Emoticon,
    Emoji,
}

impl DocumentTokenType {
    pub(in crate::builtins::strings::text_analytics) fn as_str(self) -> &'static str {
        match self {
            Self::Letters => "letters",
            Self::Digits => "digits",
            Self::Punctuation => "punctuation",
            Self::Other => "other",
            Self::WebAddress => "web-address",
            Self::EmailAddress => "email-address",
            Self::Hashtag => "hashtag",
            Self::AtMention => "at-mention",
            Self::Emoticon => "emoticon",
            Self::Emoji => "emoji",
        }
    }
}

pub(in crate::builtins::strings) fn erase_punctuation_tokenized_document(
    object: ObjectInstance,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if !object.is_class(TOKENIZED_DOCUMENT_CLASS) {
        return Err(text_analytics_error(
            "erasePunctuation",
            format!(
                "erasePunctuation: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        ));
    }
    let selected_types = parse_erase_punctuation_args(args)?;
    transform_tokenized_document(&object, "erasePunctuation", |token, token_type| {
        if !selected_types.contains(token_type.as_str()) {
            return Ok(Some(token.to_string()));
        }
        let cleaned = remove_punctuation_and_symbols(token);
        Ok((!cleaned.is_empty()).then_some(cleaned))
    })
}

fn parse_erase_punctuation_args(args: Vec<Value>) -> BuiltinResult<HashSet<String>> {
    match args.as_slice() {
        [] => Ok(["punctuation".to_string(), "other".to_string()]
            .into_iter()
            .collect()),
        [name, types] => {
            let option = scalar_text(name, "erasePunctuation")
                .map_err(|err| text_analytics_error("erasePunctuation", err.to_string()))?
                .to_ascii_lowercase();
            if option != "tokentypes" {
                return Err(text_analytics_error(
                    "erasePunctuation",
                    format!("erasePunctuation: unsupported option '{option}'"),
                ));
            }
            let parsed = words_from_word_vector(types, "erasePunctuation")?
                .into_iter()
                .map(|token_type| token_type.trim().to_ascii_lowercase())
                .filter(|token_type| !token_type.is_empty())
                .collect::<HashSet<_>>();
            if parsed.is_empty() {
                return Err(text_analytics_error(
                    "erasePunctuation",
                    "erasePunctuation: TokenTypes must contain at least one token type",
                ));
            }
            Ok(parsed)
        }
        _ => Err(text_analytics_error(
            "erasePunctuation",
            "erasePunctuation: expected erasePunctuation(documents) or erasePunctuation(documents,'TokenTypes',types)",
        )),
    }
}

pub(in crate::builtins::strings::text_analytics) fn document_token_type(
    token: &str,
) -> DocumentTokenType {
    document_token_type_with_options(token, &DEFAULT_DOCUMENT_OPTIONS)
}

pub(in crate::builtins::strings::text_analytics) fn document_token_type_with_options(
    token: &str,
    options: &DocumentOptions,
) -> DocumentTokenType {
    if is_web_address_token(token, options) {
        return DocumentTokenType::WebAddress;
    }
    if email_token_at(token, 0).is_some_and(|(_, end)| end == token.len()) {
        return DocumentTokenType::EmailAddress;
    }
    if let Some(tag) = token.strip_prefix('#') {
        if !tag.is_empty()
            && tag
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_alphabetic())
            && tag
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return DocumentTokenType::Hashtag;
        }
    }
    if let Some(mention) = token.strip_prefix('@') {
        let len = mention.chars().count();
        if (1..=15).contains(&len)
            && mention
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return DocumentTokenType::AtMention;
        }
    }
    if matches!(token, ":-)" | ":-D" | ":)" | ":D") {
        return DocumentTokenType::Emoticon;
    }
    if !token.is_empty() && token.chars().all(char::is_numeric) {
        return DocumentTokenType::Digits;
    }
    if !token.is_empty() && token.chars().all(is_emoji_char) {
        return DocumentTokenType::Emoji;
    }
    if token
        .chars()
        .all(|ch| !ch.is_alphanumeric() && !ch.is_whitespace())
    {
        return DocumentTokenType::Punctuation;
    }
    if token.chars().any(char::is_alphabetic)
        && token
            .chars()
            .all(|ch| ch.is_alphabetic() || ch == '\'' || ch == '-' || ch == '_')
    {
        return DocumentTokenType::Letters;
    }
    DocumentTokenType::Other
}

fn is_emoji_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x1F300..=0x1FAFF | 0x2600..=0x27BF | 0x2300..=0x23FF | 0xFE0F
    )
}

static DEFAULT_DOCUMENT_OPTIONS: Lazy<DocumentOptions> = Lazy::new(DocumentOptions::default);

fn remove_punctuation_and_symbols(text: &str) -> String {
    static PUNCTUATION_OR_SYMBOL: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"[\p{P}\p{S}]").expect("valid punctuation regex"));
    PUNCTUATION_OR_SYMBOL.replace_all(text, "").to_string()
}

fn stop_words_language_from_document_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<StopWordsLanguage> {
    match tokenized_document_language(object)
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "en" => Ok(StopWordsLanguage::English),
        "de" => Ok(StopWordsLanguage::German),
        "ja" => Ok(StopWordsLanguage::Japanese),
        "ko" => Ok(StopWordsLanguage::Korean),
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: unsupported document language '{other}'"),
        )),
    }
}

fn parse_bool_scalar(value: &Value, fn_name: &str) -> BuiltinResult<bool> {
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
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: logical scalar option must be true or false, got {other:?}"),
        )),
    }
}

fn documents_cell(documents: &[Vec<String>]) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .map(|doc| {
            StringArray::new(doc.clone(), vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("tokenizedDocument", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("tokenizedDocument", err))?,
    ))
}

fn vocabulary_value(documents: &[Vec<String>]) -> BuiltinResult<Value> {
    let mut seen = HashSet::new();
    let mut words = Vec::new();
    for token in documents.iter().flatten() {
        if seen.insert(token.clone()) {
            words.push(token.clone());
        }
    }
    StringArray::new(words.clone(), vec![1, words.len()])
        .map(Value::StringArray)
        .map_err(|err| text_analytics_error("tokenizedDocument", err))
}

pub(in crate::builtins::strings::text_analytics) fn documents_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Vec<Vec<String>>> {
    let Some(Value::Cell(cell)) = object.properties.get("Documents") else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object missing Documents property"),
        ));
    };
    let mut documents = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        documents.push(words_from_word_vector(item, fn_name)?);
    }
    Ok(documents)
}

pub(in crate::builtins::strings::text_analytics) fn replace_tokenized_document_documents(
    object: &mut ObjectInstance,
    documents: Vec<Vec<String>>,
    fn_name: &str,
) -> BuiltinResult<()> {
    let shape = document_shape_from_object(object, fn_name)?;
    let expected = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument Shape property overflows element count"),
            )
        })
    })?;
    if expected != documents.len() {
        return Err(text_analytics_error(
            fn_name,
            format!(
                "{fn_name}: replacement documents has {} documents but Shape requires {expected}",
                documents.len()
            ),
        ));
    }
    object
        .properties
        .insert("Documents".to_string(), documents_cell(&documents)?);
    object
        .properties
        .insert("Vocabulary".to_string(), vocabulary_value(&documents)?);
    object.properties.insert(
        "NumDocuments".to_string(),
        Value::Num(documents.len() as f64),
    );
    object.properties.insert(
        "DocumentLengths".to_string(),
        Value::Tensor(
            Tensor::new(
                documents.iter().map(|doc| doc.len() as f64).collect(),
                vec![documents.len(), 1],
            )
            .map_err(|err| text_analytics_error(fn_name, err))?,
        ),
    );
    Ok(())
}

pub(in crate::builtins::strings::text_analytics) fn document_shape_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Vec<usize>> {
    let expected = object
        .properties
        .get("Documents")
        .and_then(|value| match value {
            Value::Cell(cell) => Some(cell.data.len()),
            _ => None,
        })
        .or_else(|| {
            object
                .properties
                .get("NumDocuments")
                .and_then(nonnegative_dimension)
        })
        .unwrap_or(1);
    let shape = shape_from_object(object, fn_name)?;
    let cells = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            text_analytics_error(
                fn_name,
                format!("{fn_name}: tokenizedDocument Shape property overflows element count"),
            )
        })
    })?;
    if cells != expected {
        return Err(text_analytics_error(
            fn_name,
            format!(
                "{fn_name}: tokenizedDocument Shape property has {cells} elements but Documents has {expected}"
            ),
        ));
    }
    Ok(shape)
}

fn shape_from_object(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<Vec<usize>> {
    if let Some(Value::Tensor(tensor)) = object.properties.get("Shape") {
        tensor
            .data
            .iter()
            .map(|value| {
                nonnegative_platform_usize(*value).ok_or_else(|| {
                    text_analytics_error(
                        fn_name,
                        format!(
                            "{fn_name}: tokenizedDocument Shape property has invalid dimension {value}"
                        ),
                    )
                })
            })
            .collect()
    } else {
        let num_documents = match object.properties.get("NumDocuments") {
            Some(value) => nonnegative_dimension(value).ok_or_else(|| {
                text_analytics_error(
                    fn_name,
                    format!(
                        "{fn_name}: tokenizedDocument NumDocuments property is invalid: {value:?}"
                    ),
                )
            })?,
            None => 1,
        };
        Ok(vec![num_documents, 1])
    }
}

fn nonnegative_dimension(value: &Value) -> Option<usize> {
    match value {
        Value::Num(value) => nonnegative_platform_usize(*value),
        Value::Int(value) => value.try_to_usize(),
        _ => None,
    }
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

pub(in crate::builtins::strings::text_analytics) fn options_from_document_object(
    object: &ObjectInstance,
) -> DocumentOptions {
    let tokenize_method = match object.properties.get("TokenizeMethod") {
        Some(Value::String(value)) if value == "none" => TokenizeMethod::None,
        _ => TokenizeMethod::Unicode,
    };
    let language = match object.properties.get("Language") {
        Some(Value::String(value)) => value.clone(),
        _ => "en".to_string(),
    };
    let detect_patterns = object
        .properties
        .get("DetectPatterns")
        .and_then(|value| parse_detect_patterns(value, "tokenizedDocument").ok())
        .unwrap_or(DetectPatterns::All);
    let top_level_domains = object
        .properties
        .get("TopLevelDomains")
        .and_then(|value| parse_top_level_domains(value, "tokenizedDocument").ok())
        .unwrap_or_else(default_top_level_domains);
    let top_level_domains_custom = match object.properties.get("TopLevelDomainsCustom") {
        Some(Value::Bool(value)) => *value,
        _ => false,
    };
    DocumentOptions {
        tokenize_method,
        language,
        detect_patterns,
        top_level_domains,
        top_level_domains_custom,
        custom_tokens: Vec::new(),
        regular_expressions: Vec::new(),
    }
}

fn bag_from_documents(documents: Vec<Vec<String>>) -> BuiltinResult<Value> {
    let mut vocabulary = Vec::new();
    let mut positions = BTreeMap::new();
    for token in documents.iter().flatten() {
        if !positions.contains_key(token) {
            positions.insert(token.clone(), vocabulary.len());
            vocabulary.push(token.clone());
        }
    }
    let rows = documents.len();
    let cols = vocabulary.len();
    let mut counts = vec![0.0; checked_count_len(rows, cols, "bagOfWords")?];
    for (doc_idx, doc) in documents.iter().enumerate() {
        for token in doc {
            if let Some(col) = positions.get(token) {
                counts[doc_idx + col * rows] += 1.0;
            }
        }
    }
    bag_object(vocabulary, counts, rows)
}

fn bag_from_unique_words_and_counts(words: &Value, counts: &Value) -> BuiltinResult<Value> {
    let raw_words = words_from_word_vector_preserving_missing(words, "bagOfWords")?;
    let Value::Tensor(tensor) = counts else {
        return Err(text_analytics_error(
            "bagOfWords",
            format!("bagOfWords: counts must be a numeric matrix, got {counts:?}"),
        ));
    };
    if tensor.cols != raw_words.len() {
        return Err(text_analytics_error(
            "bagOfWords",
            format!(
                "bagOfWords: counts columns ({}) must match uniqueWords length ({})",
                tensor.cols,
                raw_words.len()
            ),
        ));
    }
    if tensor
        .data
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0 || value.fract() != 0.0)
    {
        return Err(text_analytics_error(
            "bagOfWords",
            "bagOfWords: counts must be nonnegative integers",
        ));
    }
    let mut seen = HashSet::new();
    let mut vocabulary = Vec::new();
    let mut keep_cols = Vec::new();
    for (col, word) in raw_words.iter().enumerate() {
        if is_missing_string(word) {
            continue;
        }
        if !seen.insert(word.clone()) {
            return Err(text_analytics_error(
                "bagOfWords",
                format!("bagOfWords: uniqueWords contains duplicate word '{word}'"),
            ));
        }
        vocabulary.push(word.clone());
        keep_cols.push(col);
    }
    let mut filtered_counts = Vec::with_capacity(checked_count_len(
        tensor.rows,
        keep_cols.len(),
        "bagOfWords",
    )?);
    for col in keep_cols {
        for row in 0..tensor.rows {
            filtered_counts.push(tensor.data[row + col * tensor.rows]);
        }
    }
    bag_object(vocabulary, filtered_counts, tensor.rows)
}

fn bag_object(vocabulary: Vec<String>, counts: Vec<f64>, rows: usize) -> BuiltinResult<Value> {
    ensure_bag_of_words_class_registered();
    let cols = vocabulary.len();
    let expected = checked_count_len(rows, cols, "bagOfWords")?;
    if counts.len() != expected {
        return Err(text_analytics_error(
            "bagOfWords",
            format!(
                "bagOfWords: count storage has {} values but expected {} for a {}x{} model",
                counts.len(),
                expected,
                rows,
                cols
            ),
        ));
    }
    let mut object = ObjectInstance::new(BAG_OF_WORDS_CLASS.to_string());
    object.properties.insert(
        "Vocabulary".to_string(),
        Value::StringArray(
            StringArray::new(vocabulary.clone(), vec![1, vocabulary.len()])
                .map_err(|err| text_analytics_error("bagOfWords", err))?,
        ),
    );
    object.properties.insert(
        "Counts".to_string(),
        Value::Tensor(
            Tensor::new(counts, vec![rows, cols])
                .map_err(|err| text_analytics_error("bagOfWords", err))?,
        ),
    );
    object
        .properties
        .insert("NumWords".to_string(), Value::Num(cols as f64));
    object
        .properties
        .insert("NumDocuments".to_string(), Value::Num(rows as f64));
    Ok(Value::Object(object))
}

pub(in crate::builtins::strings::text_analytics) fn checked_count_len(
    rows: usize,
    cols: usize,
    fn_name: &str,
) -> BuiltinResult<usize> {
    let len = rows.checked_mul(cols).ok_or_else(|| {
        text_analytics_error(
            fn_name,
            format!("{fn_name}: bag count matrix dimensions overflow"),
        )
    })?;
    if len > MAX_DENSE_BAG_COUNT_CELLS {
        return Err(text_analytics_error(
            fn_name,
            format!(
                "{fn_name}: dense bag count matrix would require {len} values; sparse bag storage remains tracked"
            ),
        ));
    }
    Ok(len)
}

fn remove_short_words_from_bag(object: ObjectInstance, max_len: usize) -> BuiltinResult<Value> {
    filter_bag_columns_by_predicate(object, "removeShortWords", |word| {
        word.chars().count() > max_len
    })
}

fn filter_bag_columns_by_predicate(
    object: ObjectInstance,
    fn_name: &str,
    mut keep_word: impl FnMut(&str) -> bool,
) -> BuiltinResult<Value> {
    filter_bag_columns(object, fn_name, |word| keep_word(word))
}

fn filter_bag_columns(
    object: ObjectInstance,
    fn_name: &str,
    mut keep_word: impl FnMut(&str) -> bool,
) -> BuiltinResult<Value> {
    let vocabulary = vocabulary_from_bag(&object, fn_name)?;
    let counts = counts_from_bag(&object, fn_name)?;
    let keep = vocabulary
        .iter()
        .enumerate()
        .filter_map(|(idx, word)| keep_word(word).then_some(idx))
        .collect::<Vec<_>>();
    let mut new_vocab = Vec::with_capacity(keep.len());
    let mut new_counts = Vec::with_capacity(counts.rows * keep.len());
    for col in keep {
        new_vocab.push(vocabulary[col].clone());
        for row in 0..counts.rows {
            new_counts.push(counts.data[row + col * counts.rows]);
        }
    }
    bag_object(new_vocab, new_counts, counts.rows)
}

fn documents_vocabulary(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties.get("Vocabulary") {
        Some(value) => words_from_word_vector(value, fn_name),
        None => Ok(vocabulary_from_documents(&documents_from_object(
            object, fn_name,
        )?)),
    }
}

fn vocabulary_from_documents(documents: &[Vec<String>]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut vocabulary = Vec::new();
    for token in documents.iter().flatten() {
        if seen.insert(token.clone()) {
            vocabulary.push(token.clone());
        }
    }
    vocabulary
}

pub(in crate::builtins::strings::text_analytics) fn vocabulary_from_bag(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Vec<String>> {
    match object.properties.get("Vocabulary") {
        Some(value) => words_from_word_vector(value, fn_name),
        None => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: bagOfWords object missing Vocabulary property"),
        )),
    }
}

pub(in crate::builtins::strings::text_analytics) fn counts_from_bag(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Tensor> {
    match object.properties.get("Counts") {
        Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
        _ => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: bagOfWords object missing Counts property"),
        )),
    }
}

pub(in crate::builtins::strings::text_analytics) fn words_from_word_vector(
    value: &Value,
    fn_name: &str,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array
            .data
            .iter()
            .filter(|text| !is_missing_string(text))
            .cloned()
            .collect()),
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            Ok(vec![text])
        }
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                words.push(char_row_to_string_slice(&array.data, array.cols, row));
            }
            Ok(words)
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                scalar_text(item, fn_name)
                    .map_err(|err| text_analytics_error(fn_name, err.to_string()))
            })
            .collect(),
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: expected word vector, got {other:?}"),
        )),
    }
}

pub(in crate::builtins::strings::text_analytics) fn words_from_word_vector_preserving_missing(
    value: &Value,
    fn_name: &str,
) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            Ok(vec![text])
        }
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                words.push(char_row_to_string_slice(&array.data, array.cols, row));
            }
            Ok(words)
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| {
                scalar_text(item, fn_name)
                    .map_err(|err| text_analytics_error(fn_name, err.to_string()))
            })
            .collect(),
        other => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: expected word vector, got {other:?}"),
        )),
    }
}

fn parse_positive_integer(value: &Value, fn_name: &str) -> BuiltinResult<usize> {
    let n = match value {
        Value::Num(n) => *n,
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: length must be a positive integer scalar, got {other:?}"),
            ))
        }
    };
    if !n.is_finite() || n <= 0.0 || n.fract() != 0.0 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: length must be a positive integer, got {n}"),
        ));
    }
    Ok(n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::table::table_from_columns;

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_bag(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(bag_of_words_builtin(args))
    }

    fn run_remove_short(value: Value, len: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(remove_short_words_builtin(value, len))
    }

    fn run_remove_long(value: Value, len: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(remove_long_words_builtin(value, len))
    }

    fn run_remove_words(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(remove_words_builtin(args))
    }

    fn run_remove_stop(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(remove_stop_words_builtin(args))
    }

    fn object(value: Value) -> ObjectInstance {
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        object
    }

    fn string_array_property(object: &ObjectInstance, name: &str) -> Vec<String> {
        let Some(Value::StringArray(array)) = object.properties.get(name) else {
            panic!("expected string array property {name}");
        };
        array.data.clone()
    }

    fn tensor_property(object: &ObjectInstance, name: &str) -> Tensor {
        let Some(Value::Tensor(tensor)) = object.properties.get(name) else {
            panic!("expected tensor property {name}");
        };
        tensor.clone()
    }

    fn documents_property(object: &ObjectInstance) -> Vec<Vec<String>> {
        documents_from_object(object, "test").expect("documents property")
    }

    fn type_details_property(object: &ObjectInstance) -> Vec<Vec<String>> {
        let Some(Value::Cell(cell)) = object.properties.get("TypeDetails") else {
            panic!("expected TypeDetails property");
        };
        cell.data
            .iter()
            .map(|value| {
                let Value::StringArray(array) = value else {
                    panic!("expected TypeDetails string array");
                };
                array.data.clone()
            })
            .collect()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_tokenizes_string_array_and_vocabulary() {
        let input = StringArray::new(
            vec![
                "an example of a short sentence".to_string(),
                "a second short sentence".to_string(),
            ],
            vec![2, 1],
        )
        .unwrap();
        let doc = object(run_tokenized(vec![Value::StringArray(input)]).expect("tokenized"));
        assert_eq!(doc.class_name, TOKENIZED_DOCUMENT_CLASS);
        assert_eq!(doc.properties.get("NumDocuments"), Some(&Value::Num(2.0)));
        assert_eq!(
            string_array_property(&doc, "Vocabulary"),
            vec!["an", "example", "of", "a", "short", "sentence", "second"]
        );
        let lengths = tensor_property(&doc, "DocumentLengths");
        assert_eq!(lengths.data, vec![6.0, 4.0]);
        let shape = tensor_property(&doc, "Shape");
        assert_eq!(shape.data, vec![2.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_none_accepts_pretokenized_words() {
        let input = StringArray::new(
            vec![
                "For".to_string(),
                "more".to_string(),
                "information".to_string(),
            ],
            vec![1, 3],
        )
        .unwrap();
        let doc = object(
            run_tokenized(vec![
                Value::StringArray(input),
                Value::String("TokenizeMethod".to_string()),
                Value::String("none".to_string()),
            ])
            .expect("tokenized"),
        );
        assert_eq!(
            string_array_property(&doc, "Vocabulary"),
            vec!["For", "more", "information"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_no_input_is_scalar_empty_document() {
        let doc = object(run_tokenized(Vec::new()).expect("tokenized"));
        assert_eq!(doc.properties.get("NumDocuments"), Some(&Value::Num(1.0)));
        assert_eq!(
            string_array_property(&doc, "Vocabulary"),
            Vec::<String>::new()
        );
        let lengths = tensor_property(&doc, "DocumentLengths");
        assert_eq!(lengths.shape, vec![1, 1]);
        assert_eq!(lengths.data, vec![0.0]);
        let shape = tensor_property(&doc, "Shape");
        assert_eq!(shape.data, vec![1.0, 1.0]);

        let bag = object(run_bag(vec![Value::Object(doc)]).expect("bag"));
        assert_eq!(bag.properties.get("NumDocuments"), Some(&Value::Num(1.0)));
        assert_eq!(bag.properties.get("NumWords"), Some(&Value::Num(0.0)));
        let counts = tensor_property(&bag, "Counts");
        assert_eq!(counts.shape, vec![1, 0]);
        assert!(counts.data.is_empty());
    }

    #[test]
    fn tokenized_document_shape_metadata_requires_exact_platform_dimensions() {
        let mut document = object(
            run_tokenized(vec![Value::StringArray(
                StringArray::new(vec!["first".into(), "second".into()], vec![2, 1]).unwrap(),
            )])
            .expect("tokenized"),
        );

        document.properties.remove("Shape");
        document.properties.insert(
            "NumDocuments".to_string(),
            Value::Int(runmat_builtins::IntValue::U16(2)),
        );
        assert_eq!(
            document_shape_from_object(&document, "test").unwrap(),
            vec![2, 1]
        );

        document
            .properties
            .insert("NumDocuments".to_string(), Value::Num(1.5));
        assert!(document_shape_from_object(&document, "test").is_err());

        document.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![1.5, 1.0], vec![1, 2]).unwrap()),
        );
        assert!(document_shape_from_object(&document, "test").is_err());

        document.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![usize::MAX as f64 + 1.0, 1.0], vec![1, 2]).unwrap()),
        );
        assert!(document_shape_from_object(&document, "test").is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_detects_complex_tokens_by_default() {
        let doc = object(
            run_tokenized(vec![Value::String(
                "Analyze #MATLAB :-) at help@example.com or https://www.mathworks.com/help/"
                    .to_string(),
            )])
            .expect("tokenized"),
        );
        let vocabulary = string_array_property(&doc, "Vocabulary");
        assert!(vocabulary.contains(&"#MATLAB".to_string()));
        assert!(vocabulary.contains(&":-)".to_string()));
        assert!(vocabulary.contains(&"help@example.com".to_string()));
        assert!(vocabulary.contains(&"https://www.mathworks.com/help/".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_preserves_trailing_punctuation_after_url_token() {
        let doc = object(
            run_tokenized(vec![Value::String(
                "Visit https://example.com.".to_string(),
            )])
            .expect("tokenized"),
        );
        let vocabulary = string_array_property(&doc, "Vocabulary");
        assert!(vocabulary.contains(&"https://example.com".to_string()));
        assert!(vocabulary.contains(&".".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_preserves_explicit_urls_with_uncommon_default_tlds() {
        let doc = object(
            run_tokenized(vec![Value::String(
                "Visit HTTPS://example.software and www.example.community.".to_string(),
            )])
            .expect("tokenized"),
        );
        assert_eq!(
            documents_property(&doc),
            vec![vec![
                "Visit",
                "HTTPS://example.software",
                "and",
                "www.example.community",
                "."
            ]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_detects_bare_domains_with_configured_tlds() {
        let tlds = StringArray::new(vec!["zz".into(), "dev".into()], vec![1, 2]).unwrap();
        let doc = object(
            run_tokenized(vec![
                Value::String("See example.zz, docs.example.dev/path and example.com".to_string()),
                Value::String("TopLevelDomains".to_string()),
                Value::StringArray(tlds),
            ])
            .expect("tokenized"),
        );
        assert_eq!(
            documents_property(&doc),
            vec![vec![
                "See",
                "example.zz",
                ",",
                "docs.example.dev/path",
                "and",
                "example",
                ".",
                "com"
            ]]
        );
        assert_eq!(
            string_array_property(&doc, "TopLevelDomains"),
            vec!["zz", "dev"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_respects_detect_patterns_subset() {
        let patterns =
            StringArray::new(vec!["hashtag".into(), "email-address".into()], vec![1, 2]).unwrap();
        let doc = object(
            run_tokenized(vec![
                Value::String("Mail a@example.com #MATLAB at https://example.com :-D".to_string()),
                Value::String("DetectPatterns".to_string()),
                Value::StringArray(patterns),
            ])
            .expect("tokenized"),
        );
        assert_eq!(
            documents_property(&doc),
            vec![vec![
                "Mail",
                "a@example.com",
                "#MATLAB",
                "at",
                "https",
                ":",
                "/",
                "/",
                "example",
                ".",
                "com",
                ":",
                "-",
                "D"
            ]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_supports_custom_tokens_and_longest_conflict() {
        let custom = StringArray::new(vec!["C++".into(), "C++17".into()], vec![1, 2]).unwrap();
        let doc = object(
            run_tokenized(vec![
                Value::String("Use C++17 and C++.".to_string()),
                Value::String("CustomTokens".to_string()),
                Value::StringArray(custom),
            ])
            .expect("tokenized"),
        );
        assert_eq!(
            documents_property(&doc),
            vec![vec!["Use", "C++17", "and", "C++", "."]]
        );
        assert_eq!(
            type_details_property(&doc),
            vec![vec![
                "letters",
                "custom",
                "letters",
                "custom",
                "punctuation"
            ]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_supports_custom_token_table_types() {
        let table = table_from_columns(
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
        .expect("custom table");
        let doc = object(
            run_tokenized(vec![
                Value::String("Na+ in H2O".to_string()),
                Value::String("CustomTokens".to_string()),
                table,
            ])
            .expect("tokenized"),
        );
        assert_eq!(documents_property(&doc), vec![vec!["Na+", "in", "H2O"]]);
        assert_eq!(
            type_details_property(&doc),
            vec![vec!["ion", "letters", "formula"]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_supports_regex_table_and_regex_precedence() {
        let custom = StringArray::new(vec!["abc123".into()], vec![1, 1]).unwrap();
        let regex_table = table_from_columns(
            vec!["Pattern".into(), "Type".into()],
            vec![
                Value::StringArray(
                    StringArray::new(vec![r"[a-z]+\d+".into(), r"abc\d+".into()], vec![2, 1])
                        .unwrap(),
                ),
                Value::StringArray(
                    StringArray::new(vec!["alnum".into(), "code".into()], vec![2, 1]).unwrap(),
                ),
            ],
        )
        .expect("regex table");
        let doc = object(
            run_tokenized(vec![
                Value::String("abc123 xyz9".to_string()),
                Value::String("CustomTokens".to_string()),
                Value::StringArray(custom),
                Value::String("RegularExpressions".to_string()),
                regex_table,
            ])
            .expect("tokenized"),
        );
        assert_eq!(documents_property(&doc), vec![vec!["abc123", "xyz9"]]);
        assert_eq!(type_details_property(&doc), vec![vec!["code", "alnum"]]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_supports_non_table_custom_and_regex_vectors() {
        let custom = CellArray::new(
            vec![
                Value::String("R&D".to_string()),
                Value::String("C#".to_string()),
            ],
            1,
            2,
        )
        .unwrap();
        let regexes = StringArray::new(vec![r"\d{4}-\d{2}-\d{2}".into()], vec![1, 1]).unwrap();
        let doc = object(
            run_tokenized(vec![
                Value::String("R&D shipped C# on 2026-07-17.".to_string()),
                Value::String("CustomTokens".to_string()),
                Value::Cell(custom),
                Value::String("RegularExpressions".to_string()),
                Value::StringArray(regexes),
            ])
            .expect("tokenized"),
        );

        assert_eq!(
            documents_property(&doc),
            vec![vec!["R&D", "shipped", "C#", "on", "2026-07-17", "."]]
        );
        assert_eq!(
            type_details_property(&doc),
            vec![vec![
                "custom",
                "letters",
                "custom",
                "letters",
                "custom",
                "punctuation"
            ]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_rejects_invalid_custom_and_regex_options() {
        let err = run_tokenized(vec![
            Value::String("abc".to_string()),
            Value::String("RegularExpressions".to_string()),
            Value::String("(".to_string()),
        ])
        .expect_err("invalid regex");
        assert!(err
            .to_string()
            .contains("invalid RegularExpressions pattern"));

        let err = run_tokenized(vec![
            Value::String("abc".to_string()),
            Value::String("RegularExpressions".to_string()),
            Value::String(".*".to_string()),
        ])
        .expect_err("empty regex");
        assert!(err.to_string().contains("can match empty text"));

        let table = table_from_columns(
            vec!["Type".into()],
            vec![Value::StringArray(
                StringArray::new(vec!["custom".into()], vec![1, 1]).unwrap(),
            )],
        )
        .expect("bad custom table");
        let err = run_tokenized(vec![
            Value::String("abc".to_string()),
            Value::String("CustomTokens".to_string()),
            table,
        ])
        .expect_err("missing token variable");
        assert!(err.to_string().contains("must contain a Token variable"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bag_of_words_counts_tokenized_documents() {
        let input = StringArray::new(
            vec![
                "an example of a short sentence".to_string(),
                "a second short sentence".to_string(),
            ],
            vec![2, 1],
        )
        .unwrap();
        let docs = run_tokenized(vec![Value::StringArray(input)]).expect("tokenized");
        let bag = object(run_bag(vec![docs]).expect("bag"));
        assert_eq!(bag.class_name, BAG_OF_WORDS_CLASS);
        assert_eq!(
            string_array_property(&bag, "Vocabulary"),
            vec!["an", "example", "of", "a", "short", "sentence", "second"]
        );
        let Some(Value::Tensor(counts)) = bag.properties.get("Counts") else {
            panic!("expected Counts");
        };
        assert_eq!(counts.shape, vec![2, 7]);
        assert_eq!(counts.data[0], 1.0);
        assert_eq!(counts.data[1], 0.0);
        assert_eq!(counts.data[3 * 2], 1.0);
        assert_eq!(counts.data[3 * 2 + 1], 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bag_of_words_accepts_unique_words_and_counts() {
        let words = StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap();
        let counts = Tensor::new(vec![2.0, 0.0, 1.0, 3.0], vec![2, 2]).unwrap();
        let bag =
            object(run_bag(vec![Value::StringArray(words), Value::Tensor(counts)]).expect("bag"));
        assert_eq!(bag.properties.get("NumDocuments"), Some(&Value::Num(2.0)));
        assert_eq!(bag.properties.get("NumWords"), Some(&Value::Num(2.0)));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bag_of_words_rejects_duplicate_unique_words() {
        let words = StringArray::new(vec!["alpha".into(), "alpha".into()], vec![1, 2]).unwrap();
        let counts = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err = run_bag(vec![Value::StringArray(words), Value::Tensor(counts)])
            .expect_err("expected duplicate rejection");
        assert!(err.to_string().contains("duplicate"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bag_of_words_drops_missing_unique_word_and_count_column() {
        let words = StringArray::new(
            vec!["alpha".into(), "<missing>".into(), "beta".into()],
            vec![1, 3],
        )
        .unwrap();
        let counts = Tensor::new(vec![1.0, 0.0, 9.0, 9.0, 2.0, 3.0], vec![2, 3]).unwrap();
        let bag =
            object(run_bag(vec![Value::StringArray(words), Value::Tensor(counts)]).expect("bag"));
        assert_eq!(
            string_array_property(&bag, "Vocabulary"),
            vec!["alpha", "beta"]
        );
        let counts = tensor_property(&bag, "Counts");
        assert_eq!(counts.shape, vec![2, 2]);
        assert_eq!(counts.data, vec![1.0, 0.0, 2.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn bag_of_words_checks_dense_count_size_before_allocation() {
        let err = checked_count_len(usize::MAX, 2, "bagOfWords")
            .expect_err("expected overflow rejection");
        assert!(err.to_string().contains("overflow"));

        let err = checked_count_len(MAX_DENSE_BAG_COUNT_CELLS + 1, 1, "bagOfWords")
            .expect_err("expected dense size rejection");
        assert!(err.to_string().contains("sparse bag storage"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_short_words_filters_documents_and_bag() {
        let docs = run_tokenized(vec![Value::String(
            "an example of a short sentence".to_string(),
        )])
        .expect("tokenized");
        let filtered_docs =
            object(run_remove_short(docs.clone(), Value::Num(2.0)).expect("remove docs"));
        assert_eq!(
            string_array_property(&filtered_docs, "Vocabulary"),
            vec!["example", "short", "sentence"]
        );

        let bag = run_bag(vec![docs]).expect("bag");
        let filtered_bag = object(run_remove_short(bag, Value::Num(2.0)).expect("remove bag"));
        assert_eq!(
            string_array_property(&filtered_bag, "Vocabulary"),
            vec!["example", "short", "sentence"]
        );
        let Some(Value::Tensor(counts)) = filtered_bag.properties.get("Counts") else {
            panic!("expected Counts");
        };
        assert_eq!(counts.shape, vec![1, 3]);
        assert_eq!(counts.data, vec![1.0, 1.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_long_words_filters_documents_and_bag() {
        let input = StringArray::new(
            vec![
                "An example of a short sentence".to_string(),
                "A second compact note".to_string(),
            ],
            vec![2, 1],
        )
        .unwrap();
        let docs = run_tokenized(vec![Value::StringArray(input)]).expect("tokenized");
        let filtered_docs =
            object(run_remove_long(docs.clone(), Value::Num(7.0)).expect("remove docs"));
        assert_eq!(
            documents_property(&filtered_docs),
            vec![vec!["An", "of", "a", "short"], vec!["A", "second", "note"]]
        );
        assert_eq!(
            string_array_property(&filtered_docs, "Vocabulary"),
            vec!["An", "of", "a", "short", "A", "second", "note"]
        );

        let bag = run_bag(vec![docs]).expect("bag");
        let filtered_bag = object(run_remove_long(bag, Value::Num(7.0)).expect("remove bag"));
        assert_eq!(
            string_array_property(&filtered_bag, "Vocabulary"),
            vec!["An", "of", "a", "short", "A", "second", "note"]
        );
        let counts = tensor_property(&filtered_bag, "Counts");
        assert_eq!(counts.shape, vec![2, 7]);
        assert_eq!(
            counts.data,
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_words_filters_by_word_list_indices_and_ignore_case() {
        let input = StringArray::new(
            vec![
                "Short second sentence".to_string(),
                "short example sentence".to_string(),
            ],
            vec![2, 1],
        )
        .unwrap();
        let docs = run_tokenized(vec![Value::StringArray(input)]).expect("tokenized");
        let words = StringArray::new(vec!["short".into(), "example".into()], vec![1, 2]).unwrap();
        let filtered = object(
            run_remove_words(vec![
                docs.clone(),
                Value::StringArray(words),
                Value::String("IgnoreCase".to_string()),
                Value::Bool(true),
            ])
            .expect("remove words"),
        );
        assert_eq!(
            documents_property(&filtered),
            vec![vec!["second", "sentence"], vec!["sentence"]]
        );

        let filtered_by_index = object(
            run_remove_words(vec![
                docs.clone(),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            ])
            .expect("remove indexed words"),
        );
        assert_eq!(
            documents_property(&filtered_by_index),
            vec![vec!["second"], vec!["short", "example"]]
        );

        let bag = run_bag(vec![docs]).expect("bag");
        let mask = LogicalArray::new(vec![0, 1, 0, 1, 0], vec![1, 5]).unwrap();
        let filtered_bag =
            object(run_remove_words(vec![bag, Value::LogicalArray(mask)]).expect("remove mask"));
        assert_eq!(
            string_array_property(&filtered_bag, "Vocabulary"),
            vec!["Short", "sentence", "example"]
        );
        let counts = tensor_property(&filtered_bag, "Counts");
        assert_eq!(counts.shape, vec![2, 3]);
        assert_eq!(counts.data, vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_words_rejects_bad_indices_and_mask_lengths() {
        let docs = run_tokenized(vec![Value::String("alpha beta".to_string())]).expect("tokenized");
        let err = run_remove_words(vec![docs.clone(), Value::Num(0.0)])
            .expect_err("expected bad numeric index");
        assert!(err.to_string().contains("positive integers"));

        let mask = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let err = run_remove_words(vec![docs, Value::LogicalArray(mask)])
            .expect_err("expected mask length mismatch");
        assert!(err.to_string().contains("logical index length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_stop_words_filters_tokenized_documents_with_case_option() {
        let input = StringArray::new(
            vec![
                "an example of a short sentence".to_string(),
                "The second short sentence".to_string(),
            ],
            vec![2, 1],
        )
        .unwrap();
        let docs = run_tokenized(vec![Value::StringArray(input)]).expect("tokenized");
        let filtered = object(run_remove_stop(vec![docs]).expect("remove stop words"));
        assert_eq!(
            documents_property(&filtered),
            vec![
                vec!["example", "short", "sentence"],
                vec!["second", "short", "sentence"],
            ]
        );

        let words = StringArray::new(
            vec!["The".to_string(), "the".to_string(), "word".to_string()],
            vec![1, 3],
        )
        .unwrap();
        let docs = run_tokenized(vec![
            Value::StringArray(words),
            Value::String("TokenizeMethod".to_string()),
            Value::String("none".to_string()),
        ])
        .expect("tokenized");
        let filtered = object(
            run_remove_stop(vec![
                docs,
                Value::String("IgnoreCase".to_string()),
                Value::Bool(false),
            ])
            .expect("case-sensitive remove stop words"),
        );
        assert_eq!(documents_property(&filtered), vec![vec!["The", "word"]]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_punctuation_filters_document_tokens_by_token_type() {
        let docs = object(
            run_tokenized(vec![Value::String(
                "An example: email help@example.com, visit https://example.com.".to_string(),
            )])
            .expect("tokenized"),
        );
        let filtered =
            object(erase_punctuation_tokenized_document(docs, vec![]).expect("erase punctuation"));
        assert_eq!(
            documents_property(&filtered),
            vec![vec![
                "An",
                "example",
                "email",
                "help@example.com",
                "visit",
                "https://example.com"
            ]]
        );

        let words = StringArray::new(
            vec![
                "it's".to_string(),
                "alpha-beta".to_string(),
                "help@example.com".to_string(),
                "https://example.com".to_string(),
            ],
            vec![1, 4],
        )
        .unwrap();
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(words),
                Value::String("TokenizeMethod".to_string()),
                Value::String("none".to_string()),
            ])
            .expect("tokenized"),
        );
        let filtered = object(
            erase_punctuation_tokenized_document(
                docs,
                vec![
                    Value::String("TokenTypes".to_string()),
                    Value::String("letters".to_string()),
                ],
            )
            .expect("erase letters punctuation"),
        );
        assert_eq!(
            documents_property(&filtered),
            vec![vec![
                "its",
                "alphabeta",
                "help@example.com",
                "https://example.com"
            ]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unimplemented_japanese_tokenization() {
        let err = run_tokenized(vec![
            Value::String("東京に行きます".to_string()),
            Value::String("Language".to_string()),
            Value::String("ja".to_string()),
        ])
        .expect_err("expected unsupported ja");
        assert!(err.to_string().contains("MeCab"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_rejects_invalid_detect_patterns() {
        let err = run_tokenized(vec![
            Value::String("email me at a@example.com".to_string()),
            Value::String("DetectPatterns".to_string()),
            Value::StringArray(
                StringArray::new(vec!["all".into(), "web-address".into()], vec![1, 2]).unwrap(),
            ),
        ])
        .expect_err("expected mixed all rejection");
        assert!(err.to_string().contains("specified alone"));

        let err = run_tokenized(vec![
            Value::String("email me at a@example.com".to_string()),
            Value::String("DetectPatterns".to_string()),
            Value::String("url".to_string()),
        ])
        .expect_err("expected unsupported pattern");
        assert!(err.to_string().contains("unsupported DetectPatterns"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_rejects_invalid_top_level_domains() {
        let err = run_tokenized(vec![
            Value::String("visit example.com".to_string()),
            Value::String("TopLevelDomains".to_string()),
            Value::String("co.uk".to_string()),
        ])
        .expect_err("expected invalid tld");
        assert!(err.to_string().contains("TopLevelDomains"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn remove_short_words_requires_positive_integer_length() {
        let docs =
            run_tokenized(vec![Value::String("a short document".to_string())]).expect("tokenized");
        let zero =
            run_remove_short(docs.clone(), Value::Num(0.0)).expect_err("expected zero rejection");
        assert!(zero.to_string().contains("positive integer"));

        let fractional =
            run_remove_short(docs, Value::Num(1.5)).expect_err("expected fractional rejection");
        assert!(fractional.to_string().contains("positive integer"));
    }
}
