//! Core Text Analytics document and bag-of-words compatibility objects.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ClassDef, ObjectInstance, PropertyDef, ResolveContext, StringArray, Tensor, Type,
    Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::core::compat::scalar_text;
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
        description: "Name-value options: TokenizeMethod, Language, DetectPatterns ('all' or 'none' in this slice).",
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

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS_DOCUMENTS.INVALID_INPUT",
    identifier: Some("RunMat:textAnalyticsDocuments:InvalidInput"),
    when:
        "Inputs do not match a supported tokenizedDocument, bagOfWords, or removeShortWords form.",
    message: "Text Analytics document helper received invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

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

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub(in crate::builtins::strings::text_analytics) fn text_analytics_error(
    fn_name: &str,
    message: impl Into<String>,
) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin(fn_name)
        .with_identifier("RunMat:textAnalyticsDocuments:InvalidInput")
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
async fn tokenized_document_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "tokenizedDocument").await?;
    let (input, options) = parse_tokenized_document_args(gathered)?;
    let parsed = match input {
        Some(value) => documents_from_value(value, &options)?,
        None => ParsedDocuments {
            documents: vec![Vec::new()],
            shape: vec![1, 1],
        },
    };
    tokenized_document_value(parsed.documents, parsed.shape, options)
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
            let shape = shape_from_object(&object);
            let options = options_from_document_object(&object);
            tokenized_document_value(documents, shape, options)
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
struct DocumentOptions {
    tokenize_method: TokenizeMethod,
    language: String,
    detect_patterns: DetectPatterns,
}

impl Default for DocumentOptions {
    fn default() -> Self {
        Self {
            tokenize_method: TokenizeMethod::Unicode,
            language: "en".to_string(),
            detect_patterns: DetectPatterns::All,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenizeMethod {
    Unicode,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetectPatterns {
    All,
    None,
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
                let value = scalar_text(&args[idx + 1], "tokenizedDocument")
                    .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?;
                options.detect_patterns = parse_detect_patterns(&value)?;
            }
            "customtokens" | "regularexpressions" | "topleveldomains" => {
                return Err(text_analytics_error(
                    "tokenizedDocument",
                    format!(
                        "tokenizedDocument: option '{name}' requires custom-token/table token infrastructure and remains tracked by the Text Analytics umbrella"
                    ),
                ));
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

fn parse_detect_patterns(value: &str) -> BuiltinResult<DetectPatterns> {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" => Ok(DetectPatterns::All),
        "none" => Ok(DetectPatterns::None),
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!(
                "tokenizedDocument: DetectPatterns currently supports 'all' or 'none', got '{other}'"
            ),
        )),
    }
}

struct ParsedDocuments {
    documents: Vec<Vec<String>>,
    shape: Vec<usize>,
}

fn documents_from_value(value: Value, options: &DocumentOptions) -> BuiltinResult<ParsedDocuments> {
    match options.tokenize_method {
        TokenizeMethod::Unicode => text_documents(value, options),
        TokenizeMethod::None => pretokenized_documents(value),
    }
}

fn text_documents(value: Value, options: &DocumentOptions) -> BuiltinResult<ParsedDocuments> {
    match value {
        Value::String(text) => Ok(ParsedDocuments {
            documents: vec![tokenize_text(&text, options)],
            shape: vec![1, 1],
        }),
        Value::StringArray(array) => {
            let shape = array.shape.clone();
            Ok(ParsedDocuments {
                documents: array
                    .data
                    .into_iter()
                    .map(|text| {
                        if is_missing_string(&text) {
                            Vec::new()
                        } else {
                            tokenize_text(&text, options)
                        }
                    })
                    .collect(),
                shape,
            })
        }
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            Ok(ParsedDocuments {
                documents: vec![tokenize_text(&text, options)],
                shape: vec![1, 1],
            })
        }
        Value::CharArray(array) => {
            let mut docs = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                docs.push(tokenize_text(
                    &char_row_to_string_slice(&array.data, array.cols, row),
                    options,
                ));
            }
            Ok(ParsedDocuments {
                documents: docs,
                shape: vec![array.rows, 1],
            })
        }
        Value::Cell(cell) => {
            let shape = cell.shape.clone();
            let mut docs = Vec::with_capacity(cell.data.len());
            for item in cell.data {
                let text = scalar_text(&item, "tokenizedDocument")
                    .map_err(|err| text_analytics_error("tokenizedDocument", err.to_string()))?;
                docs.push(tokenize_text(&text, options));
            }
            Ok(ParsedDocuments {
                documents: docs,
                shape,
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
        }),
        Value::StringArray(array) => Ok(ParsedDocuments {
            documents: vec![array
                .data
                .into_iter()
                .filter(|text| !is_missing_string(text))
                .collect()],
            shape: vec![1, 1],
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
                })
            } else {
                Ok(ParsedDocuments {
                    documents: vec![docs.into_iter().flatten().collect()],
                    shape: vec![1, 1],
                })
            }
        }
        other => Err(text_analytics_error(
            "tokenizedDocument",
            format!("tokenizedDocument: expected pre-tokenized word vector, got {other:?}"),
        )),
    }
}

fn tokenize_text(text: &str, options: &DocumentOptions) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut pos = 0;
    while pos < text.len() {
        let rest = &text[pos..];
        if let Some(ch) = rest.chars().next() {
            if ch.is_whitespace() {
                pos += ch.len_utf8();
                continue;
            }
        }
        if options.detect_patterns == DetectPatterns::All {
            if let Some((token, end)) = complex_token_at(text, pos) {
                tokens.push(token);
                pos = end;
                continue;
            }
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
            tokens.push(text[start..pos].to_string());
            continue;
        }
        tokens.push(ch.to_string());
        pos += ch.len_utf8();
    }
    tokens
}

fn complex_token_at(text: &str, pos: usize) -> Option<(String, usize)> {
    let rest = &text[pos..];
    if rest.starts_with("http://") || rest.starts_with("https://") || rest.starts_with("www.") {
        let end = pos + take_while_nonspace(rest);
        let token = text[pos..end].trim_end_matches(is_trailing_punctuation);
        let token_end = pos + token.len();
        return Some((token.to_string(), token_end));
    }
    if rest.starts_with('#') || rest.starts_with('@') {
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
    if rest.starts_with(":-)")
        || rest.starts_with(":-D")
        || rest.starts_with(":)")
        || rest.starts_with(":D")
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
    Ok(Value::Object(object))
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

fn shape_from_object(object: &ObjectInstance) -> Vec<usize> {
    if let Some(Value::Tensor(tensor)) = object.properties.get("Shape") {
        tensor.data.iter().map(|value| *value as usize).collect()
    } else {
        vec![
            object
                .properties
                .get("NumDocuments")
                .and_then(|value| match value {
                    Value::Num(n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(1),
            1,
        ]
    }
}

fn options_from_document_object(object: &ObjectInstance) -> DocumentOptions {
    let tokenize_method = match object.properties.get("TokenizeMethod") {
        Some(Value::String(value)) if value == "none" => TokenizeMethod::None,
        _ => TokenizeMethod::Unicode,
    };
    let language = match object.properties.get("Language") {
        Some(Value::String(value)) => value.clone(),
        _ => "en".to_string(),
    };
    DocumentOptions {
        tokenize_method,
        language,
        detect_patterns: DetectPatterns::All,
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
    let vocabulary = vocabulary_from_bag(&object, "removeShortWords")?;
    let counts = counts_from_bag(&object, "removeShortWords")?;
    let keep = vocabulary
        .iter()
        .enumerate()
        .filter_map(|(idx, word)| (word.chars().count() > max_len).then_some(idx))
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

fn vocabulary_from_bag(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<Vec<String>> {
    match object.properties.get("Vocabulary") {
        Some(value) => words_from_word_vector(value, fn_name),
        None => Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: bagOfWords object missing Vocabulary property"),
        )),
    }
}

fn counts_from_bag(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<Tensor> {
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

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_bag(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(bag_of_words_builtin(args))
    }

    fn run_remove_short(value: Value, len: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(remove_short_words_builtin(value, len))
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn tokenized_document_detects_complex_tokens_by_default() {
        let doc = object(
            run_tokenized(vec![Value::String(
                "Analyze #MATLAB :-) at https://www.mathworks.com/help/".to_string(),
            )])
            .expect("tokenized"),
        );
        let vocabulary = string_array_property(&doc, "Vocabulary");
        assert!(vocabulary.contains(&"#MATLAB".to_string()));
        assert!(vocabulary.contains(&":-)".to_string()));
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
    fn rejects_unimplemented_specific_detect_patterns() {
        let err = run_tokenized(vec![
            Value::String("email me at a@example.com".to_string()),
            Value::String("DetectPatterns".to_string()),
            Value::String("email-address".to_string()),
        ])
        .expect_err("expected unsupported specific DetectPatterns");
        assert!(err.to_string().contains("DetectPatterns"));
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
