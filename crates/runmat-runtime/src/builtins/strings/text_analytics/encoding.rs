//! Word encoding compatibility objects and word/index lookup helpers.

use std::cell::Cell;
use std::collections::HashMap;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ClassDef, LogicalArray, ObjectInstance, PropertyDef, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    documents_from_object, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::embeddings::{
    build_word_lookup, word_embedding_vocabulary_from_object, WORD_EMBEDDING_CLASS,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

pub const WORD_ENCODING_CLASS: &str = "wordEncoding";

thread_local! {
    static WORD_ENCODING_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const OUT_ENCODING: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "enc",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word encoding compatibility object.",
}];

const OUT_INDICES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word encoding indices, with NaN for words outside the vocabulary.",
}];

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Words mapped from encoding indices.",
}];

const OUT_LOGICAL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical membership mask.",
}];

const IN_DOCUMENTS_OR_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "documentsOrWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tokenizedDocument object or word vector.",
}];

const IN_DOCUMENTS_OR_WORDS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "documentsOrWords",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object or word vector.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Order, MaxNumWords.",
    },
];

const IN_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to indices.",
    },
];

const IN_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to indices.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

const IN_INDICES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "enc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive integer word encoding indices.",
    },
];

const IN_VOCABULARY_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to test.",
    },
];

const IN_VOCABULARY_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to test.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

const ERROR_ENCODING_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORDENCODING.INVALID_INPUT",
    identifier: Some("RunMat:wordEncoding:InvalidInput"),
    when: "Inputs do not match a supported wordEncoding form.",
    message: "wordEncoding received invalid input",
};

const ERROR_WORD2IND_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2IND.INVALID_INPUT",
    identifier: Some("RunMat:word2ind:InvalidInput"),
    when: "Inputs do not match a supported word2ind form.",
    message: "word2ind received invalid input",
};

const ERROR_IND2WORD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IND2WORD.INVALID_INPUT",
    identifier: Some("RunMat:ind2word:InvalidInput"),
    when: "Inputs do not match a supported ind2word form.",
    message: "ind2word received invalid input",
};

const ERROR_IS_VOCABULARY_WORD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVOCABULARYWORD.INVALID_INPUT",
    identifier: Some("RunMat:isVocabularyWord:InvalidInput"),
    when: "Inputs do not match a supported isVocabularyWord form.",
    message: "isVocabularyWord received invalid input",
};

const WORD_ENCODING_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_ENCODING_INVALID_INPUT];
const WORD2IND_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_WORD2IND_INVALID_INPUT];
const IND2WORD_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_IND2WORD_INVALID_INPUT];
const IS_VOCABULARY_WORD_ERRORS: [BuiltinErrorDescriptor; 1] =
    [ERROR_IS_VOCABULARY_WORD_INVALID_INPUT];

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub const WORD_ENCODING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "enc = wordEncoding(documents)",
            inputs: &IN_DOCUMENTS_OR_WORDS,
            outputs: &OUT_ENCODING,
        },
        BuiltinSignatureDescriptor {
            label: "enc = wordEncoding(words)",
            inputs: &IN_DOCUMENTS_OR_WORDS,
            outputs: &OUT_ENCODING,
        },
        BuiltinSignatureDescriptor {
            label: "enc = wordEncoding(documents, Name, Value)",
            inputs: &IN_DOCUMENTS_OR_WORDS_REST,
            outputs: &OUT_ENCODING,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WORD_ENCODING_ERRORS,
};

pub const WORD2IND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "M = word2ind(enc, words)",
            inputs: &IN_WORDS,
            outputs: &OUT_INDICES,
        },
        BuiltinSignatureDescriptor {
            label: "M = word2ind(enc, words, 'IgnoreCase', true)",
            inputs: &IN_WORDS_REST,
            outputs: &OUT_INDICES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WORD2IND_ERRORS,
};

pub const IND2WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "words = ind2word(enc, M)",
        inputs: &IN_INDICES,
        outputs: &OUT_WORDS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IND2WORD_ERRORS,
};

pub const IS_VOCABULARY_WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(emb, words)",
            inputs: &IN_VOCABULARY_WORDS,
            outputs: &OUT_LOGICAL,
        },
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(enc, words)",
            inputs: &IN_VOCABULARY_WORDS,
            outputs: &OUT_LOGICAL,
        },
        BuiltinSignatureDescriptor {
            label: "tf = isVocabularyWord(___, 'IgnoreCase', true)",
            inputs: &IN_VOCABULARY_WORDS_REST,
            outputs: &OUT_LOGICAL,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IS_VOCABULARY_WORD_ERRORS,
};

#[runtime_builtin(
    name = "wordEncoding",
    category = "strings/text_analytics",
    summary = "Create a word encoding object that maps words to indices and back.",
    keywords = "wordEncoding,text analytics,words,indices,vocabulary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::encoding::WORD_ENCODING_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn word_encoding_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "wordEncoding").await?;
    let (source, options) = parse_word_encoding_args(gathered)?;
    word_encoding_object(build_word_encoding(source, options)?)
}

#[runtime_builtin(
    name = "word2ind",
    category = "strings/text_analytics",
    summary = "Map words to indices in a wordEncoding object.",
    keywords = "word2ind,wordEncoding,text analytics,indices,vocabulary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::encoding::WORD2IND_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn word2ind_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "word2ind").await?;
    let (object, words, options) = parse_word2ind_args(gathered)?;
    let encoding = word_encoding_from_object(&object, "word2ind")?;
    let lookup = build_word_lookup(&encoding.vocabulary, options.ignore_case);
    let indices = words
        .words
        .into_iter()
        .map(|word| {
            let key = if options.ignore_case {
                word.to_lowercase()
            } else {
                word
            };
            lookup
                .get(&key)
                .map(|idx| (*idx + 1) as f64)
                .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    Tensor::new(indices, words.shape)
        .map(Value::Tensor)
        .map_err(|err| encoding_error("word2ind", err))
}

#[runtime_builtin(
    name = "ind2word",
    category = "strings/text_analytics",
    summary = "Map wordEncoding indices back to words.",
    keywords = "ind2word,wordEncoding,text analytics,indices,vocabulary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::encoding::IND2WORD_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn ind2word_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "ind2word").await?;
    let (object, indices) = parse_ind2word_args(gathered)?;
    let encoding = word_encoding_from_object(&object, "ind2word")?;
    let words = indices
        .values
        .into_iter()
        .map(|idx| {
            let word_idx = positive_index(idx, encoding.vocabulary.len(), "ind2word")?;
            Ok(encoding.vocabulary[word_idx].clone())
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    StringArray::new(words, indices.shape)
        .map(Value::StringArray)
        .map_err(|err| encoding_error("ind2word", err))
}

#[runtime_builtin(
    name = "isVocabularyWord",
    category = "strings/text_analytics",
    summary = "Test whether words are in a wordEmbedding or wordEncoding vocabulary.",
    keywords = "isVocabularyWord,wordEmbedding,wordEncoding,text analytics,vocabulary",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::encoding::IS_VOCABULARY_WORD_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn is_vocabulary_word_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "isVocabularyWord").await?;
    let (object, words, options) = parse_is_vocabulary_word_args(gathered)?;
    let vocabulary = if object.is_class(WORD_ENCODING_CLASS) {
        word_encoding_from_object(&object, "isVocabularyWord")?.vocabulary
    } else if object.is_class(WORD_EMBEDDING_CLASS) {
        word_embedding_vocabulary_from_object(&object, "isVocabularyWord")?
    } else {
        return Err(encoding_error(
            "isVocabularyWord",
            format!(
                "isVocabularyWord: expected wordEmbedding or wordEncoding object, got {}",
                object.class_name
            ),
        ));
    };
    let lookup = build_word_lookup(&vocabulary, options.ignore_case);
    let flags = words
        .words
        .into_iter()
        .map(|word| {
            let key = if options.ignore_case {
                word.to_lowercase()
            } else {
                word
            };
            u8::from(lookup.contains_key(&key))
        })
        .collect::<Vec<_>>();
    LogicalArray::new(flags, words.shape)
        .map(Value::LogicalArray)
        .map_err(|err| encoding_error("isVocabularyWord", err))
}

async fn gather_args(args: Vec<Value>, fn_name: &str) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            encoding_error(fn_name, format!("{fn_name}: failed to gather input: {err}"))
        })?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub(in crate::builtins::strings::text_analytics) struct WordEncodingModel {
    pub vocabulary: Vec<String>,
}

pub(in crate::builtins::strings::text_analytics) fn word_encoding_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<WordEncodingModel> {
    if !object.is_class(WORD_ENCODING_CLASS) {
        return Err(encoding_error(
            fn_name,
            format!(
                "{fn_name}: expected wordEncoding object, got {}",
                object.class_name
            ),
        ));
    }
    let vocabulary = match object.properties.get("Vocabulary") {
        Some(Value::StringArray(array)) => array.data.clone(),
        other => {
            return Err(encoding_error(
                fn_name,
                format!(
                    "{fn_name}: wordEncoding object has invalid Vocabulary property: {other:?}"
                ),
            ));
        }
    };
    match object.properties.get("NumWords") {
        Some(Value::Num(value)) if *value == vocabulary.len() as f64 => {}
        other => {
            return Err(encoding_error(
                fn_name,
                format!("{fn_name}: wordEncoding object has invalid NumWords property: {other:?}"),
            ));
        }
    }
    Ok(WordEncodingModel { vocabulary })
}

fn word_encoding_object(model: WordEncodingModel) -> BuiltinResult<Value> {
    ensure_word_encoding_class_registered();
    let mut object = ObjectInstance::new(WORD_ENCODING_CLASS.to_string());
    object.properties.insert(
        "NumWords".to_string(),
        Value::Num(model.vocabulary.len() as f64),
    );
    object.properties.insert(
        "Vocabulary".to_string(),
        Value::StringArray(
            StringArray::new(model.vocabulary.clone(), vec![1, model.vocabulary.len()])
                .map_err(|err| encoding_error("wordEncoding", err))?,
        ),
    );
    Ok(Value::Object(object))
}

fn ensure_word_encoding_class_registered() {
    WORD_ENCODING_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in ["NumWords", "Vocabulary"] {
            properties.insert(name.to_string(), property_def(name));
        }
        runmat_builtins::register_class(ClassDef {
            name: WORD_ENCODING_CLASS.to_string(),
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

enum EncodingSource {
    Documents(Vec<Vec<String>>),
    Words(Vec<String>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodingOrder {
    FirstSeen,
    Frequency,
}

#[derive(Clone, Copy, Debug)]
struct WordEncodingOptions {
    order: EncodingOrder,
    max_num_words: Option<usize>,
}

impl Default for WordEncodingOptions {
    fn default() -> Self {
        Self {
            order: EncodingOrder::FirstSeen,
            max_num_words: None,
        }
    }
}

fn parse_word_encoding_args(
    args: Vec<Value>,
) -> BuiltinResult<(EncodingSource, WordEncodingOptions)> {
    if args.is_empty() {
        return Err(encoding_error(
            "wordEncoding",
            "wordEncoding: expected tokenizedDocument object or word vector",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(encoding_error(
            "wordEncoding",
            "wordEncoding: name-value options must be paired",
        ));
    }
    let source = match &args[0] {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            EncodingSource::Documents(documents_from_object(object, "wordEncoding")?)
        }
        Value::Object(object) => {
            return Err(encoding_error(
                "wordEncoding",
                format!(
                    "wordEncoding: expected tokenizedDocument object or word vector, got {}",
                    object.class_name
                ),
            ));
        }
        value => EncodingSource::Words(word_input_from_value(value, "wordEncoding")?.words),
    };
    if matches!(source, EncodingSource::Words(_)) && args.len() > 1 {
        return Err(encoding_error(
            "wordEncoding",
            "wordEncoding: Order and MaxNumWords options are only supported for tokenizedDocument input",
        ));
    }
    let mut options = WordEncodingOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "wordEncoding")
            .map_err(|err| encoding_error("wordEncoding", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "order" => {
                let value = scalar_text(&args[idx + 1], "wordEncoding")
                    .map_err(|err| encoding_error("wordEncoding", err.to_string()))?
                    .to_ascii_lowercase();
                options.order = match value.as_str() {
                    "first-seen" => EncodingOrder::FirstSeen,
                    "frequency" => EncodingOrder::Frequency,
                    other => {
                        return Err(encoding_error(
                            "wordEncoding",
                            format!(
                                "wordEncoding: Order must be 'first-seen' or 'frequency', got '{other}'"
                            ),
                        ));
                    }
                };
            }
            "maxnumwords" => {
                options.max_num_words = parse_max_num_words(&args[idx + 1])?;
            }
            other => {
                return Err(encoding_error(
                    "wordEncoding",
                    format!("wordEncoding: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((source, options))
}

fn build_word_encoding(
    source: EncodingSource,
    options: WordEncodingOptions,
) -> BuiltinResult<WordEncodingModel> {
    let words = match source {
        EncodingSource::Documents(documents) => documents.into_iter().flatten().collect::<Vec<_>>(),
        EncodingSource::Words(words) => words,
    };
    let mut counts = HashMap::<String, (usize, usize)>::new();
    for (pos, word) in words.into_iter().enumerate() {
        let entry = counts.entry(word).or_insert((0, pos));
        entry.0 += 1;
    }
    let mut ranked = counts
        .into_iter()
        .map(|(word, (count, first_pos))| (word, count, first_pos))
        .collect::<Vec<_>>();
    match options.order {
        EncodingOrder::FirstSeen => ranked.sort_by(|left, right| left.2.cmp(&right.2)),
        EncodingOrder::Frequency => {
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then(left.2.cmp(&right.2)))
        }
    }
    if let Some(max) = options.max_num_words {
        ranked.truncate(max);
    }
    Ok(WordEncodingModel {
        vocabulary: ranked.into_iter().map(|(word, _, _)| word).collect(),
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct LookupOptions {
    ignore_case: bool,
}

fn parse_word2ind_args(
    args: Vec<Value>,
) -> BuiltinResult<(ObjectInstance, WordInput, LookupOptions)> {
    if args.len() < 2 {
        return Err(encoding_error(
            "word2ind",
            "word2ind: expected word2ind(enc, words)",
        ));
    }
    let object = object_arg(&args[0], "word2ind", "wordEncoding")?;
    let words = word_input_from_value(&args[1], "word2ind")?;
    let options = parse_lookup_options(&args[2..], "word2ind")?;
    Ok((object, words, options))
}

fn parse_is_vocabulary_word_args(
    args: Vec<Value>,
) -> BuiltinResult<(ObjectInstance, WordInput, LookupOptions)> {
    if args.len() < 2 {
        return Err(encoding_error(
            "isVocabularyWord",
            "isVocabularyWord: expected isVocabularyWord(embOrEnc, words)",
        ));
    }
    let object = object_arg(
        &args[0],
        "isVocabularyWord",
        "wordEmbedding or wordEncoding",
    )?;
    let words = word_input_from_value(&args[1], "isVocabularyWord")?;
    let options = parse_lookup_options(&args[2..], "isVocabularyWord")?;
    Ok((object, words, options))
}

fn parse_ind2word_args(args: Vec<Value>) -> BuiltinResult<(ObjectInstance, NumericInput)> {
    if args.len() != 2 {
        return Err(encoding_error(
            "ind2word",
            "ind2word: expected ind2word(enc, M)",
        ));
    }
    let object = object_arg(&args[0], "ind2word", "wordEncoding")?;
    let indices = numeric_input_from_value(&args[1], "ind2word")?;
    Ok((object, indices))
}

fn object_arg(value: &Value, fn_name: &str, expected: &str) -> BuiltinResult<ObjectInstance> {
    match value {
        Value::Object(object) => Ok(object.clone()),
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: expected {expected} object, got {other:?}"),
        )),
    }
}

fn parse_lookup_options(args: &[Value], fn_name: &str) -> BuiltinResult<LookupOptions> {
    if !args.len().is_multiple_of(2) {
        return Err(encoding_error(
            fn_name,
            format!("{fn_name}: name-value options must be paired"),
        ));
    }
    let mut options = LookupOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], fn_name)
            .map_err(|err| encoding_error(fn_name, err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "ignorecase" => options.ignore_case = parse_bool_scalar(&args[idx + 1], fn_name)?,
            other => {
                return Err(encoding_error(
                    fn_name,
                    format!("{fn_name}: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok(options)
}

struct WordInput {
    words: Vec<String>,
    shape: Vec<usize>,
}

fn word_input_from_value(value: &Value, fn_name: &str) -> BuiltinResult<WordInput> {
    match value {
        Value::String(text) => Ok(WordInput {
            words: vec![text.clone()],
            shape: vec![1, 1],
        }),
        Value::StringArray(array) => Ok(WordInput {
            words: array.data.clone(),
            shape: array.shape.clone(),
        }),
        Value::CharArray(array) if array.rows <= 1 => Ok(WordInput {
            words: vec![char_row_to_string(array)],
            shape: vec![1, 1],
        }),
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let mut text = String::with_capacity(array.cols);
                for col in 0..array.cols {
                    text.push(array.data[row + col * array.rows]);
                }
                words.push(text.trim_end().to_string());
            }
            Ok(WordInput {
                words,
                shape: vec![array.rows, 1],
            })
        }
        Value::Cell(cell) => {
            let words = cell
                .data
                .iter()
                .map(|item| {
                    scalar_text(item, fn_name)
                        .map_err(|err| encoding_error(fn_name, err.to_string()))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(WordInput {
                words,
                shape: cell.shape.clone(),
            })
        }
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: expected string, character vector, or cell array of words, got {other:?}"),
        )),
    }
}

struct NumericInput {
    values: Vec<f64>,
    shape: Vec<usize>,
}

fn numeric_input_from_value(value: &Value, fn_name: &str) -> BuiltinResult<NumericInput> {
    match value {
        Value::Num(value) => Ok(NumericInput {
            values: vec![*value],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(NumericInput {
            values: vec![int_value_to_f64(value)],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => Ok(NumericInput {
            values: tensor_utils::tensor_values_f64(tensor),
            shape: tensor.shape.clone(),
        }),
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: expected numeric positive integer indices, got {other:?}"),
        )),
    }
}

fn positive_index(value: f64, len: usize, fn_name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(encoding_error(
            fn_name,
            format!("{fn_name}: indices must be positive integers, got {value}"),
        ));
    }
    let idx = value as usize;
    if idx > len {
        return Err(encoding_error(
            fn_name,
            format!("{fn_name}: index {idx} exceeds vocabulary size {len}"),
        ));
    }
    Ok(idx - 1)
}

fn parse_max_num_words(value: &Value) -> BuiltinResult<Option<usize>> {
    let n = numeric_scalar(value, "wordEncoding", "MaxNumWords")?;
    if n.is_infinite() && n.is_sign_positive() {
        return Ok(None);
    }
    if !n.is_finite() || n < 1.0 || n.fract() != 0.0 {
        return Err(encoding_error(
            "wordEncoding",
            format!("wordEncoding: MaxNumWords must be a positive integer or Inf, got {n}"),
        ));
    }
    Ok(Some(n as usize))
}

fn numeric_scalar(value: &Value, fn_name: &str, option: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(int_value_to_f64(value)),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: {option} must be a numeric scalar, got {other:?}"),
        )),
    }
}

fn parse_bool_scalar(value: &Value, fn_name: &str) -> BuiltinResult<bool> {
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
                    _ => Err(encoding_error(
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
                other => Err(encoding_error(
                    fn_name,
                    format!("{fn_name}: logical scalar option must be true or false, got {other}"),
                )),
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: logical scalar option must be true or false, got {other:?}"),
        )),
    }
}

fn int_value_to_f64(value: &runmat_builtins::IntValue) -> f64 {
    match value {
        runmat_builtins::IntValue::I8(value) => *value as f64,
        runmat_builtins::IntValue::I16(value) => *value as f64,
        runmat_builtins::IntValue::I32(value) => *value as f64,
        runmat_builtins::IntValue::I64(value) => *value as f64,
        runmat_builtins::IntValue::U8(value) => *value as f64,
        runmat_builtins::IntValue::U16(value) => *value as f64,
        runmat_builtins::IntValue::U32(value) => *value as f64,
        runmat_builtins::IntValue::U64(value) => *value as f64,
    }
}

fn char_row_to_string(array: &CharArray) -> String {
    array.data.iter().collect()
}

fn encoding_error(fn_name: &str, message: impl Into<String>) -> crate::RuntimeError {
    let descriptor = match fn_name {
        "word2ind" => ERROR_WORD2IND_INVALID_INPUT,
        "ind2word" => ERROR_IND2WORD_INVALID_INPUT,
        "isVocabularyWord" => ERROR_IS_VOCABULARY_WORD_INVALID_INPUT,
        _ => ERROR_ENCODING_INVALID_INPUT,
    };
    let builder = build_runtime_error(message.into()).with_builtin(fn_name);
    match descriptor.identifier {
        Some(identifier) => builder.with_identifier(identifier).build(),
        None => builder.build(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{CellArray, IntegerStorage};

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn poisoned_integer_vector(storage: IntegerStorage, cols: usize) -> Value {
        let tensor = Tensor::new_integer(storage, vec![1, cols]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn tokenized_document_object(rows: Vec<Vec<&str>>) -> ObjectInstance {
        let values = rows
            .into_iter()
            .map(|row| {
                let len = row.len();
                Value::StringArray(
                    StringArray::new(
                        row.into_iter().map(str::to_string).collect::<Vec<_>>(),
                        vec![1, len],
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let rows = values.len();
        let documents = CellArray::new(values, rows, 1).unwrap();
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object
            .properties
            .insert("Documents".to_string(), Value::Cell(documents));
        object
    }

    #[test]
    fn scalar_option_parsers_read_typed_integer_storage_exactly() {
        assert_eq!(
            numeric_scalar(
                &poisoned_integer_scalar(IntegerStorage::U16(vec![12])),
                "wordEncoding",
                "MaxNumWords"
            )
            .expect("numeric"),
            12.0
        );
        assert!(parse_bool_scalar(
            &poisoned_integer_scalar(IntegerStorage::U8(vec![1])),
            "wordEncoding"
        )
        .expect("bool"));
        assert!(!parse_bool_scalar(
            &poisoned_integer_scalar(IntegerStorage::I16(vec![0])),
            "wordEncoding"
        )
        .expect("bool"));
    }

    #[test]
    fn numeric_input_reads_typed_integer_storage_exactly() {
        let input = numeric_input_from_value(
            &poisoned_integer_vector(IntegerStorage::I16(vec![2, 3]), 2),
            "ind2word",
        )
        .expect("numeric");

        assert_eq!(input.values, vec![2.0, 3.0]);
        assert_eq!(input.shape, vec![1, 2]);
    }

    #[tokio::test]
    async fn word_encoding_builds_first_seen_and_frequency_vocabularies() {
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["beta", "alpha", "beta"],
            vec!["gamma", "alpha", "beta"],
        ]));
        let first_seen = word_encoding_builtin(vec![documents.clone()])
            .await
            .unwrap();
        let Value::Object(first_seen) = first_seen else {
            panic!("expected object");
        };
        let model = word_encoding_from_object(&first_seen, "test").unwrap();
        assert_eq!(model.vocabulary, vec!["beta", "alpha", "gamma"]);

        let frequency = word_encoding_builtin(vec![
            documents,
            Value::String("Order".into()),
            Value::String("frequency".into()),
            Value::String("MaxNumWords".into()),
            Value::Num(2.0),
        ])
        .await
        .unwrap();
        let Value::Object(frequency) = frequency else {
            panic!("expected object");
        };
        let model = word_encoding_from_object(&frequency, "test").unwrap();
        assert_eq!(model.vocabulary, vec!["beta", "alpha"]);
    }

    #[tokio::test]
    async fn word_encoding_accepts_word_arrays_and_validates_options() {
        let words = Value::StringArray(
            StringArray::new(vec!["red".into(), "blue".into(), "red".into()], vec![1, 3]).unwrap(),
        );
        let enc = word_encoding_builtin(vec![words]).await.unwrap();
        let Value::Object(enc) = enc else {
            panic!("expected object");
        };
        assert_eq!(enc.properties.get("NumWords"), Some(&Value::Num(2.0)));

        let err = word_encoding_builtin(vec![
            Value::String("x".into()),
            Value::String("Order".into()),
            Value::String("frequency".into()),
        ])
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("only supported for tokenizedDocument input"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn word2ind_preserves_shape_and_supports_ignore_case() {
        let enc = word_encoding_builtin(vec![Value::StringArray(
            StringArray::new(vec!["Alpha".into(), "beta".into()], vec![1, 2]).unwrap(),
        )])
        .await
        .unwrap();
        let words = Value::StringArray(
            StringArray::new(
                vec![
                    "beta".into(),
                    "missing".into(),
                    "alpha".into(),
                    "Alpha".into(),
                ],
                vec![2, 2],
            )
            .unwrap(),
        );
        let out = word2ind_builtin(vec![
            enc,
            words,
            Value::String("IgnoreCase".into()),
            Value::Bool(true),
        ])
        .await
        .unwrap();
        let Value::Tensor(indices) = out else {
            panic!("expected tensor");
        };
        assert_eq!(indices.shape, vec![2, 2]);
        assert_eq!(indices.materialize_f64()[0], 2.0);
        assert!(indices.materialize_f64()[1].is_nan());
        assert_eq!(indices.materialize_f64()[2], 1.0);
        assert_eq!(indices.materialize_f64()[3], 1.0);
    }

    #[tokio::test]
    async fn ind2word_preserves_numeric_shape_and_rejects_bad_indices() {
        let enc = word_encoding_builtin(vec![Value::StringArray(
            StringArray::new(
                vec!["red".into(), "blue".into(), "green".into()],
                vec![1, 3],
            )
            .unwrap(),
        )])
        .await
        .unwrap();
        let out = ind2word_builtin(vec![
            enc.clone(),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
        ])
        .await
        .unwrap();
        let Value::StringArray(words) = out else {
            panic!("expected string array");
        };
        assert_eq!(words.shape, vec![1, 2]);
        assert_eq!(words.data, vec!["red", "green"]);

        let err = ind2word_builtin(vec![enc, Value::Num(4.0)])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exceeds vocabulary"), "{err}");
    }

    #[tokio::test]
    async fn is_vocabulary_word_supports_word_encoding() {
        let enc = word_encoding_builtin(vec![Value::StringArray(
            StringArray::new(vec!["RunMat".into(), "GPU".into()], vec![1, 2]).unwrap(),
        )])
        .await
        .unwrap();
        let words = Value::StringArray(
            StringArray::new(vec!["runmat".into(), "cpu".into()], vec![1, 2]).unwrap(),
        );
        let out = is_vocabulary_word_builtin(vec![
            enc,
            words,
            Value::String("IgnoreCase".into()),
            Value::Bool(true),
        ])
        .await
        .unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical array");
        };
        assert_eq!(mask.shape, vec![1, 2]);
        assert_eq!(mask.data, vec![1, 0]);
    }
}
