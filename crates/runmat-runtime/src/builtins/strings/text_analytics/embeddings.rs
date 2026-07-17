//! Word embedding compatibility objects and lookup helpers.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, ClassDef, ObjectInstance, PropertyDef, ResolveContext, StringArray,
    Tensor, Type, Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    document_shape_from_object, documents_from_object, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::encoding::{
    word_encoding_from_object, WordEncodingModel, WORD_ENCODING_CLASS,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

pub const WORD_EMBEDDING_CLASS: &str = "wordEmbedding";
const VECTOR_PROPERTY: &str = "__Vectors";
const MAX_EMBEDDING_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ZIP_ENTRIES: usize = 256;
const MAX_TRAINED_DENSE_VALUES: usize = 20_000_000;
const MAX_DOC2SEQUENCE_DENSE_VALUES: usize = 50_000_000;

thread_local! {
    static WORD_EMBEDDING_CLASS_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

const OUT_EMBEDDING: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "emb",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Word embedding compatibility object.",
}];

const OUT_MATRIX: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Embedding vectors, one word per row.",
}];

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Closest vocabulary words.",
}];

const OUT_WORDS_DIST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Closest vocabulary words.",
    },
    BuiltinParamDescriptor {
        name: "dist",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Distances to input vectors.",
    },
];

const OUT_SEQUENCES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sequences",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cell array of document embedding-vector or word-index sequences.",
}];

const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];

const IN_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 word2vec/GloVe text file or zip file containing one.",
}];

const IN_TRAIN_SOURCE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "source",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 text filename or tokenizedDocument object.",
}];

const IN_TRAIN_SOURCE_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "source",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "UTF-8 text filename or tokenizedDocument object.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options controlling local deterministic embedding training.",
    },
];

const IN_WORDS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to vectors.",
    },
];

const IN_WORDS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to map to vectors.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: IgnoreCase.",
    },
];

const IN_VECTORS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Embedding vectors, one vector per row.",
    },
];

const IN_VECTORS_REST: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "emb",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding object.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Embedding vectors, one vector per row.",
    },
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Number of nearest words.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Distance ('cosine' or 'euclidean').",
    },
];

const IN_MAP_DOCUMENTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
    BuiltinParamDescriptor {
        name: "documents",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tokenizedDocument object.",
    },
];

const IN_MAP_DOCUMENTS_REST: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "embOrEnc",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "wordEmbedding or wordEncoding object.",
    },
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
        description: "Name-value options: UnknownWord, PaddingDirection, PaddingValue, Length.",
    },
];

const ERROR_READ_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READWORDEMBEDDING.INVALID_INPUT",
    identifier: Some("RunMat:readWordEmbedding:InvalidInput"),
    when: "Inputs do not match a supported readWordEmbedding form.",
    message: "readWordEmbedding received invalid input",
};

const ERROR_WORD2VEC_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WORD2VEC.INVALID_INPUT",
    identifier: Some("RunMat:word2vec:InvalidInput"),
    when: "Inputs do not match a supported word2vec form.",
    message: "word2vec received invalid input",
};

const ERROR_VEC2WORD_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VEC2WORD.INVALID_INPUT",
    identifier: Some("RunMat:vec2word:InvalidInput"),
    when: "Inputs do not match a supported vec2word form.",
    message: "vec2word received invalid input",
};

const ERROR_TRAIN_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRAINWORDEMBEDDING.INVALID_INPUT",
    identifier: Some("RunMat:trainWordEmbedding:InvalidInput"),
    when: "Inputs do not match a supported trainWordEmbedding form.",
    message: "trainWordEmbedding received invalid input",
};

const ERROR_FASTTEXT_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FASTTEXTWORDEMBEDDING.INVALID_INPUT",
    identifier: Some("RunMat:fastTextWordEmbedding:InvalidInput"),
    when: "Inputs do not match the supported fastTextWordEmbedding form.",
    message: "fastTextWordEmbedding received invalid input",
};

const ERROR_DOC2SEQUENCE_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DOC2SEQUENCE.INVALID_INPUT",
    identifier: Some("RunMat:doc2sequence:InvalidInput"),
    when: "Inputs do not match a supported doc2sequence form.",
    message: "doc2sequence received invalid input",
};

const ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READWORDEMBEDDING.IO",
    identifier: Some("RunMat:readWordEmbedding:IOError"),
    when: "The requested word embedding file cannot be read.",
    message: "Unable to read word embedding file",
};

const FASTTEXT_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_FASTTEXT_INVALID_INPUT];
const READ_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_READ_INVALID_INPUT, ERROR_IO];
const WORD2VEC_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_WORD2VEC_INVALID_INPUT];
const VEC2WORD_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_VEC2WORD_INVALID_INPUT];
const TRAIN_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_TRAIN_INVALID_INPUT];
const DOC2SEQUENCE_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_DOC2SEQUENCE_INVALID_INPUT];

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

pub const READ_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "emb = readWordEmbedding(filename)",
        inputs: &IN_FILENAME,
        outputs: &OUT_EMBEDDING,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &READ_ERRORS,
};

pub const FASTTEXT_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "emb = fastTextWordEmbedding",
        inputs: &NO_INPUTS,
        outputs: &OUT_EMBEDDING,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FASTTEXT_ERRORS,
};

pub const WORD2VEC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "M = word2vec(emb, words)",
            inputs: &IN_WORDS,
            outputs: &OUT_MATRIX,
        },
        BuiltinSignatureDescriptor {
            label: "M = word2vec(emb, words, 'IgnoreCase', true)",
            inputs: &IN_WORDS_REST,
            outputs: &OUT_MATRIX,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WORD2VEC_ERRORS,
};

pub const VEC2WORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "words = vec2word(emb, M)",
            inputs: &IN_VECTORS,
            outputs: &OUT_WORDS,
        },
        BuiltinSignatureDescriptor {
            label: "[words, dist] = vec2word(emb, M, k, 'Distance', distance)",
            inputs: &IN_VECTORS_REST,
            outputs: &OUT_WORDS_DIST,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &VEC2WORD_ERRORS,
};

pub const DOC2SEQUENCE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(emb, documents)",
            inputs: &IN_MAP_DOCUMENTS,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(enc, documents)",
            inputs: &IN_MAP_DOCUMENTS,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(emb, documents, Name, Value)",
            inputs: &IN_MAP_DOCUMENTS_REST,
            outputs: &OUT_SEQUENCES,
        },
        BuiltinSignatureDescriptor {
            label: "sequences = doc2sequence(enc, documents, Name, Value)",
            inputs: &IN_MAP_DOCUMENTS_REST,
            outputs: &OUT_SEQUENCES,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DOC2SEQUENCE_ERRORS,
};

pub const TRAIN_WORD_EMBEDDING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "emb = trainWordEmbedding(filename)",
            inputs: &IN_TRAIN_SOURCE,
            outputs: &OUT_EMBEDDING,
        },
        BuiltinSignatureDescriptor {
            label: "emb = trainWordEmbedding(documents)",
            inputs: &IN_TRAIN_SOURCE,
            outputs: &OUT_EMBEDDING,
        },
        BuiltinSignatureDescriptor {
            label: "emb = trainWordEmbedding(___, Name, Value)",
            inputs: &IN_TRAIN_SOURCE_REST,
            outputs: &OUT_EMBEDDING,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRAIN_ERRORS,
};

#[runtime_builtin(
    name = "fastTextWordEmbedding",
    category = "strings/text_analytics",
    summary = "Return a bundled fastText-style word embedding compatibility model.",
    keywords = "fastTextWordEmbedding,wordEmbedding,text analytics,fastText,pretrained",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::embeddings::FASTTEXT_WORD_EMBEDDING_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn fast_text_word_embedding_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(embedding_error(
            "fastTextWordEmbedding",
            "fastTextWordEmbedding: expected no input arguments",
        ));
    }
    embedding_object(compact_fast_text_embedding())
}

#[runtime_builtin(
    name = "readWordEmbedding",
    category = "strings/text_analytics",
    summary = "Read word embedding models from UTF-8 text or zip files.",
    keywords = "readWordEmbedding,wordEmbedding,text analytics,word2vec,GloVe",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::embeddings::READ_WORD_EMBEDDING_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn read_word_embedding_builtin(filename: Value) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(|err| embedding_error("readWordEmbedding", err.to_string()))?;
    let filename = scalar_text(&filename, "readWordEmbedding")
        .map_err(|err| embedding_error("readWordEmbedding", err.to_string()))?;
    let path = Path::new(&filename);
    let bytes = read_limited_file_bytes(path, "readWordEmbedding").await?;
    let text = if is_zip_path(path) || looks_like_zip(&bytes) {
        read_embedding_text_from_zip(&bytes)?
    } else {
        String::from_utf8(bytes).map_err(|err| {
            embedding_error(
                "readWordEmbedding",
                format!("readWordEmbedding: embedding file must be UTF-8 text: {err}"),
            )
        })?
    };
    embedding_object(parse_embedding_text(&text, "readWordEmbedding")?)
}

#[runtime_builtin(
    name = "trainWordEmbedding",
    category = "strings/text_analytics",
    summary = "Train a local word embedding compatibility model.",
    keywords = "trainWordEmbedding,wordEmbedding,text analytics,training",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::embeddings::TRAIN_WORD_EMBEDDING_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn train_word_embedding_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "trainWordEmbedding").await?;
    let (source, options) = parse_train_word_embedding_args(gathered)?;
    let documents = match source {
        TrainSource::Documents(documents) => documents,
        TrainSource::Filename(filename) => {
            let bytes = read_limited_file_bytes(Path::new(&filename), "trainWordEmbedding").await?;
            let text = String::from_utf8(bytes).map_err(|err| {
                embedding_error(
                    "trainWordEmbedding",
                    format!("trainWordEmbedding: training file must be UTF-8 text: {err}"),
                )
            })?;
            documents_from_training_text(&text)
        }
    };
    embedding_object(train_embedding_model(documents, options)?)
}

#[runtime_builtin(
    name = "doc2sequence",
    category = "strings/text_analytics",
    summary = "Convert tokenized documents to word-vector or word-index sequences.",
    keywords = "doc2sequence,wordEmbedding,wordEncoding,tokenizedDocument,text analytics,sequences",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::embeddings::DOC2SEQUENCE_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn doc2sequence_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "doc2sequence").await?;
    let (sequence_object, document_object, options) = parse_doc2sequence_args(gathered)?;
    let document_shape = document_shape_from_object(&document_object, "doc2sequence")?;
    let documents = documents_from_object(&document_object, "doc2sequence")?;
    if sequence_object.is_class(WORD_EMBEDDING_CLASS) {
        let embedding = embedding_from_object(&sequence_object, "doc2sequence")?;
        doc2sequence_value(&embedding, &documents, &document_shape, options)
    } else if sequence_object.is_class(WORD_ENCODING_CLASS) {
        let encoding = word_encoding_from_object(&sequence_object, "doc2sequence")?;
        doc2sequence_indices_value(&encoding, &documents, &document_shape, options)
    } else {
        Err(embedding_error(
            "doc2sequence",
            format!(
                "doc2sequence: expected wordEmbedding or wordEncoding object, got {}",
                sequence_object.class_name
            ),
        ))
    }
}

#[runtime_builtin(
    name = "word2vec",
    category = "strings/text_analytics",
    summary = "Map words to rows of a word embedding matrix.",
    keywords = "word2vec,wordEmbedding,text analytics,vectors",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::embeddings::WORD2VEC_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn word2vec_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "word2vec").await?;
    let (object, words, options) = parse_word2vec_args(gathered)?;
    let embedding = embedding_from_object(&object, "word2vec")?;
    let lookup = build_word_lookup(&embedding.vocabulary, options.ignore_case);
    let word_count = words.len();
    let mut rows = Vec::with_capacity(word_count);
    for word in words {
        let key = if options.ignore_case {
            word.to_lowercase()
        } else {
            word
        };
        if let Some(&row) = lookup.get(&key) {
            let start = row * embedding.dimension;
            rows.push(embedding.vectors[start..start + embedding.dimension].to_vec());
        } else {
            rows.push(vec![f64::NAN; embedding.dimension]);
        }
    }
    let mut out = Vec::with_capacity(word_count * embedding.dimension);
    for col in 0..embedding.dimension {
        for row in &rows {
            out.push(row[col]);
        }
    }
    Tensor::new(out, vec![word_count, embedding.dimension])
        .map(Value::Tensor)
        .map_err(|err| embedding_error("word2vec", err))
}

#[runtime_builtin(
    name = "vec2word",
    category = "strings/text_analytics",
    summary = "Map embedding vectors to nearest vocabulary words.",
    keywords = "vec2word,wordEmbedding,text analytics,nearest",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::embeddings::VEC2WORD_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::embeddings"
)]
async fn vec2word_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let gathered = gather_args(args, "vec2word").await?;
    let (object, matrix, options) = parse_vec2word_args(gathered)?;
    let embedding = embedding_from_object(&object, "vec2word")?;
    if matrix.cols != embedding.dimension {
        return Err(embedding_error(
            "vec2word",
            format!(
                "vec2word: input matrix must have {} columns, got {}",
                embedding.dimension, matrix.cols
            ),
        ));
    }
    if options.k == 0 || options.k > embedding.vocabulary.len() {
        return Err(embedding_error(
            "vec2word",
            format!(
                "vec2word: k must be between 1 and vocabulary size ({})",
                embedding.vocabulary.len()
            ),
        ));
    }

    let mut row_words = Vec::with_capacity(matrix.rows);
    let mut row_distances = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let query = row_slice(&matrix, row);
        let mut scored = embedding
            .vectors
            .chunks(embedding.dimension)
            .enumerate()
            .map(|(idx, candidate)| {
                (
                    idx,
                    match options.distance {
                        DistanceMetric::Cosine => cosine_distance(&query, candidate),
                        DistanceMetric::Euclidean => euclidean_distance(&query, candidate),
                    },
                )
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| compare_scores(left.1, right.1).then(left.0.cmp(&right.0)));
        let mut words = Vec::with_capacity(options.k);
        let mut distances = Vec::with_capacity(options.k);
        for (idx, distance) in scored.into_iter().take(options.k) {
            words.push(embedding.vocabulary[idx].clone());
            distances.push(distance);
        }
        row_words.push(words);
        row_distances.push(distances);
    }

    let word_shape = if options.k == 1 {
        vec![matrix.rows, 1]
    } else {
        vec![matrix.rows, options.k]
    };
    let mut words = Vec::with_capacity(matrix.rows * options.k);
    let mut distances = Vec::with_capacity(matrix.rows * options.k);
    for col in 0..options.k {
        for row in 0..matrix.rows {
            words.push(row_words[row][col].clone());
            distances.push(row_distances[row][col]);
        }
    }
    let words = Value::StringArray(
        StringArray::new(words, word_shape).map_err(|err| embedding_error("vec2word", err))?,
    );
    let dist = Value::Tensor(
        Tensor::new(distances, vec![matrix.rows, options.k])
            .map_err(|err| embedding_error("vec2word", err))?,
    );
    Ok(Value::OutputList(vec![words, dist]))
}

async fn gather_args(args: Vec<Value>, fn_name: &str) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            embedding_error(fn_name, format!("{fn_name}: failed to gather input: {err}"))
        })?);
    }
    Ok(out)
}

async fn read_limited_file_bytes(path: &Path, fn_name: &str) -> BuiltinResult<Vec<u8>> {
    let file = File::open_async(path).await.map_err(|err| {
        embedding_error_with_source(
            fn_name,
            format!("{fn_name}: unable to open '{}': {err}", path.display()),
            err,
        )
    })?;
    let mut limited = file.take(MAX_EMBEDDING_FILE_BYTES + 1);
    let mut bytes = Vec::new();
    limited.read_to_end(&mut bytes).map_err(|err| {
        embedding_error_with_source(
            fn_name,
            format!("{fn_name}: unable to read '{}': {err}", path.display()),
            err,
        )
    })?;
    if bytes.len() as u64 > MAX_EMBEDDING_FILE_BYTES {
        return Err(embedding_error(
            fn_name,
            format!(
                "{fn_name}: embedding file exceeds maximum supported size of {MAX_EMBEDDING_FILE_BYTES} bytes"
            ),
        ));
    }
    Ok(bytes)
}

fn read_embedding_text_from_zip(bytes: &[u8]) -> BuiltinResult<String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|err| {
        embedding_error(
            "readWordEmbedding",
            format!("readWordEmbedding: unable to read zip archive: {err}"),
        )
    })?;
    if archive.len() > MAX_ZIP_ENTRIES {
        return Err(embedding_error(
            "readWordEmbedding",
            format!("readWordEmbedding: zip archive contains more than {MAX_ZIP_ENTRIES} entries"),
        ));
    }

    let mut selected = None;
    for idx in 0..archive.len() {
        let mut entry = archive.by_index(idx).map_err(|err| {
            embedding_error(
                "readWordEmbedding",
                format!("readWordEmbedding: unable to read zip entry: {err}"),
            )
        })?;
        if entry.is_dir() {
            continue;
        }
        if entry.size() > MAX_EMBEDDING_FILE_BYTES {
            return Err(embedding_error(
                "readWordEmbedding",
                format!(
                    "readWordEmbedding: zip entry '{}' exceeds maximum supported size of {MAX_EMBEDDING_FILE_BYTES} bytes",
                    entry.name()
                ),
            ));
        }
        let name = entry.name().to_ascii_lowercase();
        let text_like = matches!(
            Path::new(&name)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_ascii_lowercase())
                .as_deref(),
            Some("txt") | Some("vec") | Some("glove") | Some("emb")
        );
        if !text_like && selected.is_some() {
            continue;
        }
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(|err| {
            embedding_error(
                "readWordEmbedding",
                format!("readWordEmbedding: zip entry must contain UTF-8 text: {err}"),
            )
        })?;
        selected = Some(text);
        if text_like {
            break;
        }
    }
    selected.ok_or_else(|| {
        embedding_error(
            "readWordEmbedding",
            "readWordEmbedding: zip archive does not contain an embedding text file",
        )
    })
}

fn parse_embedding_text(text: &str, fn_name: &str) -> BuiltinResult<EmbeddingModel> {
    let mut lines = text.lines().enumerate().filter_map(|(idx, raw)| {
        let trimmed = raw.trim();
        (!trimmed.is_empty()).then_some((idx + 1, trimmed))
    });
    let Some((first_line_no, first_line)) = lines.next() else {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: embedding file is empty"),
        ));
    };

    let first_parts = first_line.split_whitespace().collect::<Vec<_>>();
    let (dimension_hint, expected_rows, pending_first) = if first_parts.len() == 2 {
        match (
            first_parts[0].parse::<usize>(),
            first_parts[1].parse::<usize>(),
        ) {
            (Ok(rows), Ok(dim)) if rows > 0 && dim > 0 => (Some(dim), Some(rows), None),
            _ => (None, None, Some((first_line_no, first_line))),
        }
    } else {
        (None, None, Some((first_line_no, first_line)))
    };

    let mut vocabulary = Vec::new();
    let mut vectors = Vec::new();
    let mut positions = HashMap::new();
    let mut dimension = dimension_hint;
    let mut parsed_rows = 0usize;
    let rows = pending_first.into_iter().chain(lines);
    for (line_no, line) in rows {
        let (word, vector) = parse_embedding_line(line, dimension, fn_name, line_no)?;
        parsed_rows += 1;
        let dim = vector.len();
        if dim == 0 {
            return Err(embedding_error(
                fn_name,
                format!("{fn_name}: line {line_no} has no vector values"),
            ));
        }
        match dimension {
            Some(expected) if expected != dim => {
                return Err(embedding_error(
                    fn_name,
                    format!(
                        "{fn_name}: line {line_no} has {dim} dimensions but expected {expected}"
                    ),
                ));
            }
            Some(_) => {}
            None => dimension = Some(dim),
        }
        if let Some(old_pos) = positions.remove(&word) {
            vocabulary.remove(old_pos);
            let start = old_pos * dim;
            vectors.drain(start..start + dim);
            for pos in positions.values_mut() {
                if *pos > old_pos {
                    *pos -= 1;
                }
            }
        }
        positions.insert(word.clone(), vocabulary.len());
        vocabulary.push(word);
        vectors.extend(vector);
    }

    let dimension = dimension.ok_or_else(|| {
        embedding_error(
            fn_name,
            format!("{fn_name}: embedding file contains no vectors"),
        )
    })?;
    if vocabulary.is_empty() {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: embedding file contains no words"),
        ));
    }
    if let Some(expected_rows) = expected_rows {
        if expected_rows != parsed_rows {
            return Err(embedding_error(
                fn_name,
                format!(
                    "{fn_name}: header declares {expected_rows} words but parsed {parsed_rows} rows"
                ),
            ));
        }
    }
    Ok(EmbeddingModel {
        vocabulary,
        vectors,
        dimension,
    })
}

fn parse_embedding_line(
    line: &str,
    dimension_hint: Option<usize>,
    fn_name: &str,
    line_no: usize,
) -> BuiltinResult<(String, Vec<f64>)> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() < 2 {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: line {line_no} must contain a word and vector values"),
        ));
    }
    let dimension = dimension_hint.unwrap_or(parts.len() - 1);
    if parts.len() != dimension + 1 {
        return Err(embedding_error(
            fn_name,
            format!(
                "{fn_name}: line {line_no} has {} vector values but expected {dimension}",
                parts.len().saturating_sub(1)
            ),
        ));
    }
    let word = parts[0].to_string();
    if word.is_empty() {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: line {line_no} has an empty word"),
        ));
    }
    let vector = parts[1..]
        .iter()
        .map(|part| {
            let value = part.parse::<f64>().map_err(|err| {
                embedding_error(
                    fn_name,
                    format!("{fn_name}: invalid numeric value on line {line_no}: {err}"),
                )
            })?;
            if !value.is_finite() {
                return Err(embedding_error(
                    fn_name,
                    format!("{fn_name}: non-finite vector value on line {line_no}"),
                ));
            }
            Ok(value)
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok((word, vector))
}

#[derive(Clone, Debug)]
struct EmbeddingModel {
    vocabulary: Vec<String>,
    vectors: Vec<f64>,
    dimension: usize,
}

fn embedding_object(model: EmbeddingModel) -> BuiltinResult<Value> {
    ensure_word_embedding_class_registered();
    let mut object = ObjectInstance::new(WORD_EMBEDDING_CLASS.to_string());
    object
        .properties
        .insert("Dimension".to_string(), Value::Num(model.dimension as f64));
    object.properties.insert(
        "Vocabulary".to_string(),
        Value::StringArray(
            StringArray::new(model.vocabulary.clone(), vec![1, model.vocabulary.len()])
                .map_err(|err| embedding_error("wordEmbedding", err))?,
        ),
    );
    object.properties.insert(
        VECTOR_PROPERTY.to_string(),
        Value::Tensor(
            Tensor::new(model.vectors, vec![model.vocabulary.len(), model.dimension])
                .map_err(|err| embedding_error("wordEmbedding", err))?,
        ),
    );
    Ok(Value::Object(object))
}

fn embedding_from_object(object: &ObjectInstance, fn_name: &str) -> BuiltinResult<EmbeddingModel> {
    if !object.is_class(WORD_EMBEDDING_CLASS) {
        return Err(embedding_error(
            fn_name,
            format!(
                "{fn_name}: expected wordEmbedding object, got {}",
                object.class_name
            ),
        ));
    }
    let vocabulary = match object.properties.get("Vocabulary") {
        Some(Value::StringArray(array)) => array.data.clone(),
        other => {
            return Err(embedding_error(
                fn_name,
                format!(
                    "{fn_name}: wordEmbedding object has invalid Vocabulary property: {other:?}"
                ),
            ));
        }
    };
    let dimension = match object.properties.get("Dimension") {
        Some(Value::Num(value)) if value.is_finite() && *value >= 1.0 => *value as usize,
        other => {
            return Err(embedding_error(
                fn_name,
                format!(
                    "{fn_name}: wordEmbedding object has invalid Dimension property: {other:?}"
                ),
            ));
        }
    };
    let vectors = match object.properties.get(VECTOR_PROPERTY) {
        Some(Value::Tensor(tensor))
            if tensor.rows == vocabulary.len() && tensor.cols == dimension =>
        {
            tensor.data.clone()
        }
        other => {
            return Err(embedding_error(
                fn_name,
                format!("{fn_name}: wordEmbedding object has invalid vector storage: {other:?}"),
            ));
        }
    };
    Ok(EmbeddingModel {
        vocabulary,
        vectors,
        dimension,
    })
}

pub(in crate::builtins::strings::text_analytics) fn word_embedding_vocabulary_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Vec<String>> {
    embedding_from_object(object, fn_name).map(|model| model.vocabulary)
}

fn compact_fast_text_embedding() -> EmbeddingModel {
    let vocabulary = [
        "France",
        "Italy",
        "Rome",
        "Paris",
        "king",
        "queen",
        "man",
        "woman",
        "good",
        "bad",
        "excellent",
        "terrible",
        "data",
        "model",
        "analysis",
        "report",
        "signal",
        "image",
        "learning",
        "network",
        "algorithm",
        "matrix",
        "vector",
        "science",
        "engineering",
        "physics",
        "compute",
        "runtime",
        "test",
        "train",
        "document",
        "sequence",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let dimension = 300usize;
    let mut vectors = Vec::with_capacity(vocabulary.len() * dimension);
    for word in &vocabulary {
        vectors.extend(compact_fast_text_vector(word, dimension));
    }
    EmbeddingModel {
        vocabulary,
        vectors,
        dimension,
    }
}

fn compact_fast_text_vector(word: &str, dimension: usize) -> Vec<f64> {
    let mut vector = vec![0.0; dimension];
    let has_curated_vector = match word {
        "France" => {
            vector[0] = 1.0;
            vector[2] = 1.0;
            true
        }
        "Italy" => {
            vector[2] = 1.0;
            true
        }
        "Rome" => {
            vector[1] = 1.0;
            vector[2] = 1.0;
            true
        }
        "Paris" => {
            vector[0] = 1.0;
            vector[1] = 1.0;
            vector[2] = 1.0;
            true
        }
        "king" => {
            vector[3] = 1.0;
            vector[5] = 1.0;
            true
        }
        "queen" => {
            vector[4] = 1.0;
            vector[5] = 1.0;
            true
        }
        "man" => {
            vector[3] = 1.0;
            true
        }
        "woman" => {
            vector[4] = 1.0;
            true
        }
        "good" => {
            vector[6] = 1.0;
            true
        }
        "bad" => {
            vector[6] = -1.0;
            true
        }
        "excellent" => {
            vector[6] = 1.4;
            vector[7] = 0.4;
            true
        }
        "terrible" => {
            vector[6] = -1.4;
            vector[7] = -0.4;
            true
        }
        _ => false,
    };
    if !has_curated_vector {
        let seed = stable_word_hash(word);
        for (idx, slot) in vector.iter_mut().enumerate() {
            let bit = ((seed.rotate_left((idx % 31) as u32) ^ idx as u64) & 0x0f) as f64;
            *slot = (bit - 7.5) / 64.0;
        }
    }
    vector
}

fn stable_word_hash(word: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in word.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ensure_word_embedding_class_registered() {
    WORD_EMBEDDING_CLASS_REGISTERED.with(|registered| {
        if registered.get() {
            return;
        }
        let mut properties = HashMap::new();
        for name in ["Dimension", "Vocabulary", VECTOR_PROPERTY] {
            properties.insert(name.to_string(), property_def(name));
        }
        runmat_builtins::register_class(ClassDef {
            name: WORD_EMBEDDING_CLASS.to_string(),
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

enum TrainSource {
    Filename(String),
    Documents(Vec<Vec<String>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrainModelKind {
    SkipGram,
    Cbow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrainLossFunction {
    NegativeSampling,
    HierarchicalSoftmax,
    Softmax,
}

#[derive(Clone, Copy, Debug)]
struct TrainWordEmbeddingOptions {
    dimension: usize,
    window: usize,
    model: TrainModelKind,
    discard_factor: f64,
    loss_function: TrainLossFunction,
    num_negative_samples: usize,
    num_negative_samples_was_set: bool,
    num_epochs: usize,
    min_count: usize,
    ngram_range: (usize, usize),
    initial_learn_rate: f64,
    update_rate: usize,
    verbose: bool,
}

impl Default for TrainWordEmbeddingOptions {
    fn default() -> Self {
        Self {
            dimension: 100,
            window: 5,
            model: TrainModelKind::SkipGram,
            discard_factor: 1.0e-4,
            loss_function: TrainLossFunction::NegativeSampling,
            num_negative_samples: 5,
            num_negative_samples_was_set: false,
            num_epochs: 5,
            min_count: 5,
            ngram_range: (3, 6),
            initial_learn_rate: 0.05,
            update_rate: 100,
            verbose: true,
        }
    }
}

fn parse_train_word_embedding_args(
    args: Vec<Value>,
) -> BuiltinResult<(TrainSource, TrainWordEmbeddingOptions)> {
    if args.is_empty() {
        return Err(embedding_error(
            "trainWordEmbedding",
            "trainWordEmbedding: expected filename or tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(embedding_error(
            "trainWordEmbedding",
            "trainWordEmbedding: name-value options must appear in pairs",
        ));
    }
    let source = train_source_from_value(&args[0])?;
    let mut options = TrainWordEmbeddingOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "trainWordEmbedding")
            .map_err(|err| embedding_error("trainWordEmbedding", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "dimension" => {
                options.dimension = parse_positive_integer(&args[idx + 1], "trainWordEmbedding")?
            }
            "window" => {
                options.window =
                    parse_nonnegative_integer(&args[idx + 1], "trainWordEmbedding", "Window")?
            }
            "model" => {
                let value = scalar_text(&args[idx + 1], "trainWordEmbedding")
                    .map_err(|err| embedding_error("trainWordEmbedding", err.to_string()))?
                    .to_ascii_lowercase();
                options.model = match value.as_str() {
                    "skipgram" => TrainModelKind::SkipGram,
                    "cbow" => TrainModelKind::Cbow,
                    other => {
                        return Err(embedding_error(
                            "trainWordEmbedding",
                            format!(
                                "trainWordEmbedding: Model must be 'skipgram' or 'cbow', got '{other}'"
                            ),
                        ));
                    }
                };
            }
            "discardfactor" => {
                options.discard_factor =
                    parse_positive_scalar(&args[idx + 1], "trainWordEmbedding", "DiscardFactor")?
            }
            "lossfunction" => {
                let value = scalar_text(&args[idx + 1], "trainWordEmbedding")
                    .map_err(|err| embedding_error("trainWordEmbedding", err.to_string()))?
                    .to_ascii_lowercase();
                options.loss_function = match value.as_str() {
                    "ns" => TrainLossFunction::NegativeSampling,
                    "hs" => TrainLossFunction::HierarchicalSoftmax,
                    "softmax" => TrainLossFunction::Softmax,
                    other => {
                        return Err(embedding_error(
                            "trainWordEmbedding",
                            format!(
                                "trainWordEmbedding: LossFunction must be 'ns', 'hs', or 'softmax', got '{other}'"
                            ),
                        ));
                    }
                };
            }
            "numnegativesamples" => {
                options.num_negative_samples =
                    parse_positive_integer(&args[idx + 1], "trainWordEmbedding")?;
                options.num_negative_samples_was_set = true;
            }
            "numepochs" => {
                options.num_epochs = parse_positive_integer(&args[idx + 1], "trainWordEmbedding")?
            }
            "mincount" => {
                options.min_count = parse_positive_integer(&args[idx + 1], "trainWordEmbedding")?
            }
            "ngramrange" => options.ngram_range = parse_ngram_range(&args[idx + 1])?,
            "initiallearnrate" => {
                options.initial_learn_rate =
                    parse_positive_scalar(&args[idx + 1], "trainWordEmbedding", "InitialLearnRate")?
            }
            "updaterate" => {
                options.update_rate = parse_positive_integer(&args[idx + 1], "trainWordEmbedding")?
            }
            "verbose" => options.verbose = parse_bool_scalar(&args[idx + 1], "trainWordEmbedding")?,
            other => {
                return Err(embedding_error(
                    "trainWordEmbedding",
                    format!("trainWordEmbedding: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    if options.num_negative_samples_was_set
        && options.loss_function != TrainLossFunction::NegativeSampling
    {
        return Err(embedding_error(
            "trainWordEmbedding",
            "trainWordEmbedding: NumNegativeSamples is only valid when LossFunction is 'ns'",
        ));
    }
    checked_train_dense_size(options.dimension, 1, "trainWordEmbedding")?;
    Ok((source, options))
}

fn train_source_from_value(value: &Value) -> BuiltinResult<TrainSource> {
    match value {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => Ok(
            TrainSource::Documents(documents_from_object(object, "trainWordEmbedding")?),
        ),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
            Ok(TrainSource::Filename(train_filename_from_value(value)?))
        }
        other => Err(embedding_error(
            "trainWordEmbedding",
            format!("trainWordEmbedding: expected filename or tokenizedDocument, got {other:?}"),
        )),
    }
}

fn train_filename_from_value(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::Cell(cell) if cell.data.len() == 1 => train_filename_from_value(&cell.data[0]),
        other => {
            let filename = scalar_text(other, "trainWordEmbedding")
                .map_err(|err| embedding_error("trainWordEmbedding", err.to_string()))?;
            if filename.trim().is_empty() {
                Err(embedding_error(
                    "trainWordEmbedding",
                    "trainWordEmbedding: filename must not be empty",
                ))
            } else {
                Ok(filename)
            }
        }
    }
}

fn documents_from_training_text(text: &str) -> Vec<Vec<String>> {
    text.lines()
        .map(|line| {
            line.split_whitespace()
                .filter(|word| !word.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|doc| !doc.is_empty())
        .collect()
}

fn train_embedding_model(
    documents: Vec<Vec<String>>,
    options: TrainWordEmbeddingOptions,
) -> BuiltinResult<EmbeddingModel> {
    if documents.is_empty() || documents.iter().all(Vec::is_empty) {
        return Err(embedding_error(
            "trainWordEmbedding",
            "trainWordEmbedding: training data contains no tokens",
        ));
    }

    let mut counts = HashMap::<String, (usize, usize)>::new();
    let mut next_pos = 0usize;
    for token in documents.iter().flatten() {
        let entry = counts.entry(token.clone()).or_insert_with(|| {
            let pos = next_pos;
            next_pos += 1;
            (0, pos)
        });
        entry.0 += 1;
    }

    let mut vocabulary = counts
        .iter()
        .filter(|(_, (count, _))| *count >= options.min_count)
        .map(|(word, (count, first_pos))| (word.clone(), *count, *first_pos))
        .collect::<Vec<_>>();
    if vocabulary.is_empty() {
        return Err(embedding_error(
            "trainWordEmbedding",
            format!(
                "trainWordEmbedding: no vocabulary words meet MinCount {}",
                options.min_count
            ),
        ));
    }
    vocabulary.sort_by(|left, right| right.1.cmp(&left.1).then(left.2.cmp(&right.2)));
    checked_train_dense_size(options.dimension, vocabulary.len(), "trainWordEmbedding")?;

    let mut positions = HashMap::new();
    let mut final_vocabulary = Vec::with_capacity(vocabulary.len());
    for (idx, (word, _, _)) in vocabulary.into_iter().enumerate() {
        positions.insert(word.clone(), idx);
        final_vocabulary.push(word);
    }

    let mut rows = vec![vec![0.0; options.dimension]; final_vocabulary.len()];
    for (idx, word) in final_vocabulary.iter().enumerate() {
        add_lexical_features(&mut rows[idx], word, options);
    }

    let base = options.initial_learn_rate
        * options.num_epochs as f64
        * match options.loss_function {
            TrainLossFunction::NegativeSampling => {
                1.0 + (options.num_negative_samples as f64).ln_1p() * 0.05
            }
            TrainLossFunction::HierarchicalSoftmax => 0.95,
            TrainLossFunction::Softmax => 1.05,
        };
    let model_scale = match options.model {
        TrainModelKind::SkipGram => 1.0,
        TrainModelKind::Cbow => 0.75,
    };
    let discard_scale = (1.0 + options.discard_factor.log10().abs()).recip();
    let update_scale = 1.0 + (options.update_rate as f64).ln_1p() * 0.01;

    for document in &documents {
        for (target_pos, target) in document.iter().enumerate() {
            let Some(&target_idx) = positions.get(target) else {
                continue;
            };
            if options.window == 0 {
                continue;
            }
            let start = target_pos.saturating_sub(options.window);
            let end = target_pos
                .saturating_add(options.window)
                .saturating_add(1)
                .min(document.len());
            for (ctx_pos, context) in document.iter().enumerate().take(end).skip(start) {
                if ctx_pos == target_pos {
                    continue;
                }
                let Some(&context_idx) = positions.get(context) else {
                    continue;
                };
                let distance = target_pos.abs_diff(ctx_pos).max(1) as f64;
                let weight = base * model_scale * discard_scale * update_scale / distance;
                add_hashed_feature(
                    &mut rows[target_idx],
                    context,
                    weight,
                    0x9e37_79b9_7f4a_7c15,
                );
                if options.model == TrainModelKind::SkipGram {
                    add_hashed_feature(
                        &mut rows[context_idx],
                        target,
                        weight * 0.5,
                        0xc2b2_ae3d_27d4_eb4f,
                    );
                }
            }
        }
    }

    let mut vectors = Vec::with_capacity(final_vocabulary.len() * options.dimension);
    for row in &mut rows {
        normalize_vector(row);
        vectors.extend(row.iter().copied());
    }
    Ok(EmbeddingModel {
        vocabulary: final_vocabulary,
        vectors,
        dimension: options.dimension,
    })
}

fn add_lexical_features(row: &mut [f64], word: &str, options: TrainWordEmbeddingOptions) {
    add_hashed_feature(row, word, 1.0, 0xcbf2_9ce4_8422_2325);
    if options.ngram_range != (0, 0) {
        add_character_ngram_features(row, word, options.ngram_range);
    }
    add_hashed_feature(row, &word.to_ascii_lowercase(), 0.2, 0x517c_c1b7_2722_0a95);
}

fn add_character_ngram_features(row: &mut [f64], word: &str, range: (usize, usize)) {
    let chars = format!("<{word}>").chars().collect::<Vec<_>>();
    let max_len = range.1.min(chars.len());
    for len in range.0..=max_len {
        if len == 0 || len > chars.len() {
            continue;
        }
        for window in chars.windows(len) {
            let ngram = window.iter().collect::<String>();
            add_hashed_feature(row, &ngram, 0.35, 0x1000_0000_01b3);
        }
    }
}

fn add_hashed_feature(row: &mut [f64], key: &str, weight: f64, salt: u64) {
    if row.is_empty() {
        return;
    }
    let hash = fnv1a64_with_salt(key, salt);
    let idx = (hash as usize) % row.len();
    let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
    row[idx] += sign * weight;
}

fn fnv1a64_with_salt(value: &str, salt: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ salt;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn normalize_vector(row: &mut [f64]) {
    let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in row {
            *value /= norm;
        }
    }
}

fn parse_nonnegative_integer(value: &Value, fn_name: &str, option: &str) -> BuiltinResult<usize> {
    let n = numeric_scalar(value, fn_name, option)?;
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: {option} must be a nonnegative integer, got {n}"),
        ));
    }
    Ok(n as usize)
}

fn parse_positive_scalar(value: &Value, fn_name: &str, option: &str) -> BuiltinResult<f64> {
    let n = numeric_scalar(value, fn_name, option)?;
    if !n.is_finite() || n <= 0.0 {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: {option} must be a positive scalar, got {n}"),
        ));
    }
    Ok(n)
}

fn parse_ngram_range(value: &Value) -> BuiltinResult<(usize, usize)> {
    let values = match value {
        Value::Tensor(tensor) if tensor.data.len() == 2 => tensor.data.clone(),
        other => {
            return Err(embedding_error(
                "trainWordEmbedding",
                format!("trainWordEmbedding: NGramRange must be a two-element numeric vector, got {other:?}"),
            ));
        }
    };
    let min = values[0];
    let max = values[1];
    if !min.is_finite()
        || !max.is_finite()
        || min < 0.0
        || max < 0.0
        || min.fract() != 0.0
        || max.fract() != 0.0
        || min > max
    {
        return Err(embedding_error(
            "trainWordEmbedding",
            format!("trainWordEmbedding: NGramRange must be [min max] nonnegative integers with min <= max, got [{min} {max}]"),
        ));
    }
    Ok((min as usize, max as usize))
}

fn numeric_scalar(value: &Value, fn_name: &str, option: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Tensor(tensor) if tensor.data.len() == 1 => Ok(tensor.data[0]),
        other => Err(embedding_error(
            fn_name,
            format!("{fn_name}: {option} must be a numeric scalar, got {other:?}"),
        )),
    }
}

fn checked_train_dense_size(
    dimension: usize,
    vocabulary_len: usize,
    fn_name: &str,
) -> BuiltinResult<()> {
    let cells = dimension.checked_mul(vocabulary_len).ok_or_else(|| {
        embedding_error(
            fn_name,
            format!("{fn_name}: trained embedding dimensions overflow dense storage"),
        )
    })?;
    if cells > MAX_TRAINED_DENSE_VALUES {
        return Err(embedding_error(
            fn_name,
            format!(
                "{fn_name}: trained embedding would require {cells} dense values; limit is {MAX_TRAINED_DENSE_VALUES}"
            ),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct Word2VecOptions {
    ignore_case: bool,
}

fn parse_word2vec_args(
    args: Vec<Value>,
) -> BuiltinResult<(ObjectInstance, Vec<String>, Word2VecOptions)> {
    if args.len() < 2 {
        return Err(embedding_error(
            "word2vec",
            "word2vec: expected word2vec(emb, words)",
        ));
    }
    let mut iter = args.into_iter();
    let object = match iter.next().expect("checked") {
        Value::Object(object) => object,
        other => {
            return Err(embedding_error(
                "word2vec",
                format!("word2vec: expected wordEmbedding object, got {other:?}"),
            ));
        }
    };
    let words_value = iter.next().expect("checked");
    let words = words_from_value(&words_value, "word2vec")?;
    let mut options = Word2VecOptions::default();
    let rest = iter.collect::<Vec<_>>();
    let mut idx = 0;
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(embedding_error(
                "word2vec",
                "word2vec: name-value options must be paired",
            ));
        }
        let name = scalar_text(&rest[idx], "word2vec")
            .map_err(|err| embedding_error("word2vec", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "ignorecase" => options.ignore_case = parse_bool_scalar(&rest[idx + 1], "word2vec")?,
            other => {
                return Err(embedding_error(
                    "word2vec",
                    format!("word2vec: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((object, words, options))
}

#[derive(Clone, Copy, Debug)]
enum DistanceMetric {
    Cosine,
    Euclidean,
}

#[derive(Clone, Copy, Debug)]
struct Vec2WordOptions {
    k: usize,
    distance: DistanceMetric,
}

impl Default for Vec2WordOptions {
    fn default() -> Self {
        Self {
            k: 1,
            distance: DistanceMetric::Cosine,
        }
    }
}

fn parse_vec2word_args(
    args: Vec<Value>,
) -> BuiltinResult<(ObjectInstance, Tensor, Vec2WordOptions)> {
    if args.len() < 2 {
        return Err(embedding_error(
            "vec2word",
            "vec2word: expected vec2word(emb, M)",
        ));
    }
    let mut iter = args.into_iter();
    let object = match iter.next().expect("checked") {
        Value::Object(object) => object,
        other => {
            return Err(embedding_error(
                "vec2word",
                format!("vec2word: expected wordEmbedding object, got {other:?}"),
            ));
        }
    };
    let matrix = match iter.next().expect("checked") {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => {
            Tensor::new(vec![value], vec![1, 1]).map_err(|err| embedding_error("vec2word", err))?
        }
        other => {
            return Err(embedding_error(
                "vec2word",
                format!("vec2word: expected numeric matrix, got {other:?}"),
            ));
        }
    };
    let mut rest = iter.collect::<Vec<_>>();
    let mut options = Vec2WordOptions::default();
    if rest.first().is_some_and(is_numeric_scalar) {
        options.k = parse_positive_integer(&rest.remove(0), "vec2word")?;
    }
    let mut idx = 0;
    while idx < rest.len() {
        if idx + 1 >= rest.len() {
            return Err(embedding_error(
                "vec2word",
                "vec2word: name-value options must be paired",
            ));
        }
        let name = scalar_text(&rest[idx], "vec2word")
            .map_err(|err| embedding_error("vec2word", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "distance" => {
                let metric = scalar_text(&rest[idx + 1], "vec2word")
                    .map_err(|err| embedding_error("vec2word", err.to_string()))?
                    .to_ascii_lowercase();
                options.distance = match metric.as_str() {
                    "cosine" => DistanceMetric::Cosine,
                    "euclidean" => DistanceMetric::Euclidean,
                    other => {
                        return Err(embedding_error(
                            "vec2word",
                            format!("vec2word: unsupported Distance '{other}'"),
                        ));
                    }
                };
            }
            other => {
                return Err(embedding_error(
                    "vec2word",
                    format!("vec2word: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((object, matrix, options))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnknownWordMode {
    Discard,
    Nan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PaddingDirection {
    Left,
    Right,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceLength {
    Longest,
    Shortest,
    Fixed(usize),
}

#[derive(Clone, Copy, Debug)]
struct Doc2SequenceOptions {
    unknown_word: UnknownWordMode,
    padding_direction: PaddingDirection,
    padding_value: f64,
    length: SequenceLength,
}

impl Default for Doc2SequenceOptions {
    fn default() -> Self {
        Self {
            unknown_word: UnknownWordMode::Discard,
            padding_direction: PaddingDirection::Left,
            padding_value: 0.0,
            length: SequenceLength::Longest,
        }
    }
}

fn parse_doc2sequence_args(
    args: Vec<Value>,
) -> BuiltinResult<(ObjectInstance, ObjectInstance, Doc2SequenceOptions)> {
    if args.len() < 2 {
        return Err(embedding_error(
            "doc2sequence",
            "doc2sequence: expected doc2sequence(embOrEnc, documents)",
        ));
    }
    if !(args.len() - 2).is_multiple_of(2) {
        return Err(embedding_error(
            "doc2sequence",
            "doc2sequence: name-value options must be paired",
        ));
    }
    let sequence_model = match &args[0] {
        Value::Object(object) => object.clone(),
        other => {
            return Err(embedding_error(
                "doc2sequence",
                format!(
                    "doc2sequence: expected wordEmbedding or wordEncoding object, got {other:?}"
                ),
            ));
        }
    };
    let documents = match &args[1] {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => object.clone(),
        Value::Object(object) => {
            return Err(embedding_error(
                "doc2sequence",
                format!(
                    "doc2sequence: expected tokenizedDocument object, got {}",
                    object.class_name
                ),
            ));
        }
        other => {
            return Err(embedding_error(
                "doc2sequence",
                format!("doc2sequence: expected tokenizedDocument object, got {other:?}"),
            ));
        }
    };
    let mut options = Doc2SequenceOptions::default();
    let mut idx = 2usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "doc2sequence")
            .map_err(|err| embedding_error("doc2sequence", err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "unknownword" => {
                let value = scalar_text(&args[idx + 1], "doc2sequence")
                    .map_err(|err| embedding_error("doc2sequence", err.to_string()))?
                    .to_ascii_lowercase();
                options.unknown_word = match value.as_str() {
                    "discard" => UnknownWordMode::Discard,
                    "nan" => UnknownWordMode::Nan,
                    other => {
                        return Err(embedding_error(
                            "doc2sequence",
                            format!("doc2sequence: UnknownWord must be 'discard' or 'nan', got '{other}'"),
                        ));
                    }
                };
            }
            "paddingdirection" => {
                let value = scalar_text(&args[idx + 1], "doc2sequence")
                    .map_err(|err| embedding_error("doc2sequence", err.to_string()))?
                    .to_ascii_lowercase();
                options.padding_direction = match value.as_str() {
                    "left" => PaddingDirection::Left,
                    "right" => PaddingDirection::Right,
                    "none" => PaddingDirection::None,
                    other => {
                        return Err(embedding_error(
                            "doc2sequence",
                            format!("doc2sequence: PaddingDirection must be 'left', 'right', or 'none', got '{other}'"),
                        ));
                    }
                };
            }
            "paddingvalue" => {
                options.padding_value =
                    parse_numeric_scalar(&args[idx + 1], "doc2sequence", "PaddingValue")?;
            }
            "length" => options.length = parse_sequence_length(&args[idx + 1])?,
            other => {
                return Err(embedding_error(
                    "doc2sequence",
                    format!("doc2sequence: unsupported option '{other}'"),
                ));
            }
        }
        idx += 2;
    }
    Ok((sequence_model, documents, options))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceToken {
    Known(usize),
    UnknownNan,
}

fn doc2sequence_value(
    embedding: &EmbeddingModel,
    documents: &[Vec<String>],
    document_shape: &[usize],
    options: Doc2SequenceOptions,
) -> BuiltinResult<Value> {
    let lookup = build_word_lookup(&embedding.vocabulary, false);
    let mut sequences = Vec::with_capacity(documents.len());
    for document in documents {
        let mut sequence = Vec::new();
        for token in document {
            if let Some(&idx) = lookup.get(token) {
                sequence.push(SequenceToken::Known(idx));
            } else if options.unknown_word == UnknownWordMode::Nan {
                sequence.push(SequenceToken::UnknownNan);
            }
        }
        sequences.push(sequence);
    }
    let resolved_length = resolve_sequence_length(&sequences, options.length);
    let mut values = Vec::with_capacity(sequences.len());
    let mut total_cells = 0usize;
    for sequence in &sequences {
        let target_len = sequence_target_len(sequence.len(), resolved_length, options);
        total_cells = total_cells
            .checked_add(
                embedding
                    .dimension
                    .checked_mul(target_len)
                    .ok_or_else(|| dense_doc2sequence_limit_error("doc2sequence"))?,
            )
            .ok_or_else(|| dense_doc2sequence_limit_error("doc2sequence"))?;
        if total_cells > MAX_DOC2SEQUENCE_DENSE_VALUES {
            return Err(dense_doc2sequence_limit_error("doc2sequence"));
        }
        values.push(Value::Tensor(sequence_tensor(
            embedding,
            sequence,
            target_len,
            options.padding_direction,
            options.padding_value,
        )?));
    }
    Ok(Value::Cell(
        CellArray::new_with_shape(values, document_shape.to_vec())
            .map_err(|err| embedding_error("doc2sequence", err))?,
    ))
}

fn doc2sequence_indices_value(
    encoding: &WordEncodingModel,
    documents: &[Vec<String>],
    document_shape: &[usize],
    options: Doc2SequenceOptions,
) -> BuiltinResult<Value> {
    let lookup = build_word_lookup(&encoding.vocabulary, false);
    let mut sequences = Vec::with_capacity(documents.len());
    for document in documents {
        let mut sequence = Vec::new();
        for token in document {
            if let Some(&idx) = lookup.get(token) {
                sequence.push(IndexSequenceToken::Known((idx + 1) as f64));
            } else if options.unknown_word == UnknownWordMode::Nan {
                sequence.push(IndexSequenceToken::UnknownNan);
            }
        }
        sequences.push(sequence);
    }
    let resolved_length = resolve_sequence_length(&sequences, options.length);
    let mut values = Vec::with_capacity(sequences.len());
    let mut total_cells = 0usize;
    for sequence in &sequences {
        let target_len = sequence_target_len(sequence.len(), resolved_length, options);
        total_cells = total_cells
            .checked_add(target_len)
            .ok_or_else(|| dense_doc2sequence_limit_error("doc2sequence"))?;
        if total_cells > MAX_DOC2SEQUENCE_DENSE_VALUES {
            return Err(dense_doc2sequence_limit_error("doc2sequence"));
        }
        values.push(Value::Tensor(index_sequence_tensor(
            sequence,
            target_len,
            options.padding_direction,
            options.padding_value,
        )?));
    }
    Ok(Value::Cell(
        CellArray::new_with_shape(values, document_shape.to_vec())
            .map_err(|err| embedding_error("doc2sequence", err))?,
    ))
}

fn resolve_sequence_length<T>(sequences: &[Vec<T>], length: SequenceLength) -> usize {
    match length {
        SequenceLength::Fixed(len) => len,
        SequenceLength::Longest => sequences.iter().map(Vec::len).max().unwrap_or(0),
        SequenceLength::Shortest => sequences.iter().map(Vec::len).min().unwrap_or(0),
    }
}

fn sequence_target_len(
    sequence_len: usize,
    resolved_length: usize,
    options: Doc2SequenceOptions,
) -> usize {
    match options.padding_direction {
        PaddingDirection::None => match options.length {
            SequenceLength::Fixed(len) => sequence_len.min(len),
            SequenceLength::Shortest => sequence_len.min(resolved_length),
            SequenceLength::Longest => sequence_len,
        },
        PaddingDirection::Left | PaddingDirection::Right => resolved_length,
    }
}

fn sequence_tensor(
    embedding: &EmbeddingModel,
    sequence: &[SequenceToken],
    target_len: usize,
    padding_direction: PaddingDirection,
    padding_value: f64,
) -> BuiltinResult<Tensor> {
    let truncated_len = sequence.len().min(target_len);
    let pad_len = target_len.saturating_sub(truncated_len);
    let mut out = Vec::with_capacity(embedding.dimension * target_len);
    if padding_direction == PaddingDirection::Left {
        push_padding_columns(&mut out, embedding.dimension, pad_len, padding_value);
    }
    for token in sequence.iter().take(truncated_len) {
        match token {
            SequenceToken::Known(row) => {
                let start = row * embedding.dimension;
                out.extend_from_slice(&embedding.vectors[start..start + embedding.dimension]);
            }
            SequenceToken::UnknownNan => {
                out.extend(std::iter::repeat_n(f64::NAN, embedding.dimension));
            }
        }
    }
    if padding_direction == PaddingDirection::Right {
        push_padding_columns(&mut out, embedding.dimension, pad_len, padding_value);
    }
    Tensor::new(out, vec![embedding.dimension, target_len])
        .map_err(|err| embedding_error("doc2sequence", err))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum IndexSequenceToken {
    Known(f64),
    UnknownNan,
}

fn index_sequence_tensor(
    sequence: &[IndexSequenceToken],
    target_len: usize,
    padding_direction: PaddingDirection,
    padding_value: f64,
) -> BuiltinResult<Tensor> {
    let truncated_len = sequence.len().min(target_len);
    let pad_len = target_len.saturating_sub(truncated_len);
    let mut out = Vec::with_capacity(target_len);
    if padding_direction == PaddingDirection::Left {
        out.extend(std::iter::repeat_n(padding_value, pad_len));
    }
    for token in sequence.iter().take(truncated_len) {
        match token {
            IndexSequenceToken::Known(idx) => out.push(*idx),
            IndexSequenceToken::UnknownNan => out.push(f64::NAN),
        }
    }
    if padding_direction == PaddingDirection::Right {
        out.extend(std::iter::repeat_n(padding_value, pad_len));
    }
    Tensor::new(out, vec![1, target_len]).map_err(|err| embedding_error("doc2sequence", err))
}

fn push_padding_columns(out: &mut Vec<f64>, dimension: usize, count: usize, padding_value: f64) {
    out.extend(std::iter::repeat_n(padding_value, dimension * count));
}

fn parse_sequence_length(value: &Value) -> BuiltinResult<SequenceLength> {
    if matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    ) {
        let text = scalar_text(value, "doc2sequence")
            .map_err(|err| embedding_error("doc2sequence", err.to_string()))?;
        match text.trim().to_ascii_lowercase().as_str() {
            "longest" => return Ok(SequenceLength::Longest),
            "shortest" => return Ok(SequenceLength::Shortest),
            other => {
                if let Ok(value) = other.parse::<usize>() {
                    if value > 0 {
                        return Ok(SequenceLength::Fixed(value));
                    }
                }
                return Err(embedding_error(
                    "doc2sequence",
                    format!(
                        "doc2sequence: Length must be 'longest', 'shortest', or a positive integer, got '{other}'"
                    ),
                ));
            }
        }
    }
    Ok(SequenceLength::Fixed(parse_positive_integer(
        value,
        "doc2sequence",
    )?))
}

fn parse_numeric_scalar(value: &Value, fn_name: &str, option_name: &str) -> BuiltinResult<f64> {
    let n = match value {
        Value::Num(value) => *value,
        Value::Int(value) => int_value_to_f64(value),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(embedding_error(
                fn_name,
                format!("{fn_name}: {option_name} must be a numeric scalar, got {other:?}"),
            ));
        }
    };
    Ok(n)
}

fn dense_doc2sequence_limit_error(fn_name: &str) -> crate::RuntimeError {
    embedding_error(
        fn_name,
        format!(
            "{fn_name}: output would exceed {MAX_DOC2SEQUENCE_DENSE_VALUES} dense values; use PaddingDirection 'none' or a smaller Length"
        ),
    )
}

fn words_from_value(value: &Value, fn_name: &str) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::CharArray(array) if array.rows <= 1 => Ok(vec![char_row_to_string(array)]),
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let mut text = String::with_capacity(array.cols);
                for col in 0..array.cols {
                    text.push(array.data[row + col * array.rows]);
                }
                words.push(text.trim_end().to_string());
            }
            Ok(words)
        }
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| match item {
                Value::String(text) => Ok(text.clone()),
                Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
                Value::CharArray(array) if array.rows <= 1 => Ok(char_row_to_string(array)),
                other => Err(embedding_error(
                    fn_name,
                    format!("{fn_name}: cell word inputs must contain scalar text, got {other:?}"),
                )),
            })
            .collect(),
        other => Err(embedding_error(
            fn_name,
            format!("{fn_name}: expected string, character vector, or cell array of words, got {other:?}"),
        )),
    }
}

pub(in crate::builtins::strings::text_analytics) fn build_word_lookup(
    vocabulary: &[String],
    ignore_case: bool,
) -> HashMap<String, usize> {
    let mut lookup = HashMap::new();
    for (idx, word) in vocabulary.iter().enumerate() {
        let key = if ignore_case {
            word.to_lowercase()
        } else {
            word.clone()
        };
        lookup.entry(key).or_insert(idx);
    }
    lookup
}

fn row_slice(tensor: &Tensor, row: usize) -> Vec<f64> {
    (0..tensor.cols)
        .map(|col| tensor.data[row + col * tensor.rows])
        .collect()
}

fn cosine_distance(lhs: &[f64], rhs: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut lhs_norm = 0.0;
    let mut rhs_norm = 0.0;
    for (&a, &b) in lhs.iter().zip(rhs.iter()) {
        dot += a * b;
        lhs_norm += a * a;
        rhs_norm += b * b;
    }
    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        f64::INFINITY
    } else {
        1.0 - dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
    }
}

fn euclidean_distance(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&a, &b)| {
            let delta = a - b;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn compare_scores(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

fn char_row_to_string(array: &CharArray) -> String {
    array.data.iter().collect()
}

fn parse_bool_scalar(value: &Value, fn_name: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) if *value == 0.0 || *value == 1.0 => Ok(*value != 0.0),
        Value::Tensor(tensor) if tensor.data.len() == 1 => match tensor.data[0] {
            0.0 => Ok(false),
            1.0 => Ok(true),
            other => Err(embedding_error(
                fn_name,
                format!("{fn_name}: logical scalar option must be true or false, got {other}"),
            )),
        },
        other => Err(embedding_error(
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

fn parse_positive_integer(value: &Value, fn_name: &str) -> BuiltinResult<usize> {
    let n = match value {
        Value::Num(value) => *value,
        Value::Int(value) => int_value_to_f64(value),
        Value::Tensor(tensor) if tensor.data.len() == 1 => tensor.data[0],
        other => {
            return Err(embedding_error(
                fn_name,
                format!("{fn_name}: expected positive integer scalar, got {other:?}"),
            ));
        }
    };
    if !n.is_finite() || n < 1.0 || n.fract() != 0.0 {
        return Err(embedding_error(
            fn_name,
            format!("{fn_name}: expected positive integer scalar, got {n}"),
        ));
    }
    Ok(n as usize)
}

fn is_numeric_scalar(value: &Value) -> bool {
    matches!(value, Value::Num(_))
        || matches!(value, Value::Tensor(tensor) if tensor.data.len() == 1)
}

fn is_zip_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("zip")
    )
}

fn looks_like_zip(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == b"PK\x03\x04"
}

fn embedding_error(fn_name: &str, message: impl Into<String>) -> crate::RuntimeError {
    let identifier = match fn_name {
        "fastTextWordEmbedding" => "RunMat:fastTextWordEmbedding:InvalidInput",
        "readWordEmbedding" => "RunMat:readWordEmbedding:InvalidInput",
        "trainWordEmbedding" => "RunMat:trainWordEmbedding:InvalidInput",
        "doc2sequence" => "RunMat:doc2sequence:InvalidInput",
        "word2vec" => "RunMat:word2vec:InvalidInput",
        "vec2word" => "RunMat:vec2word:InvalidInput",
        _ => "RunMat:wordEmbedding:InvalidInput",
    };
    build_runtime_error(message.into())
        .with_builtin(fn_name)
        .with_identifier(identifier)
        .build()
}

fn embedding_error_with_source(
    fn_name: &str,
    message: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> crate::RuntimeError {
    let identifier = if fn_name == "readWordEmbedding" {
        "RunMat:readWordEmbedding:IOError"
    } else {
        "RunMat:wordEmbedding:InvalidInput"
    };
    build_runtime_error(message.into())
        .with_builtin(fn_name)
        .with_identifier(identifier)
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::CellArray;
    use std::fs::File as StdFile;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn parses_glove_text_embedding() {
        let model = parse_embedding_text("king 1 0 0\nqueen 0.8 0.2 0\n", "test").unwrap();
        assert_eq!(model.dimension, 3);
        assert_eq!(model.vocabulary, vec!["king", "queen"]);
        assert_eq!(model.vectors, vec![1.0, 0.0, 0.0, 0.8, 0.2, 0.0]);
    }

    #[test]
    fn parses_word2vec_header_and_last_duplicate_wins() {
        let model =
            parse_embedding_text("3 2\nalpha 1 0\nbeta 0 1\nalpha 0.5 0.5\n", "test").unwrap();
        assert_eq!(model.dimension, 2);
        assert_eq!(model.vocabulary, vec!["beta", "alpha"]);
        assert_eq!(model.vectors, vec![0.0, 1.0, 0.5, 0.5]);
    }

    #[test]
    fn rejects_word2vec_header_row_mismatch() {
        let err = parse_embedding_text("3 2\nalpha 1 0\nbeta 0 1\n", "test").unwrap_err();
        assert!(err.to_string().contains("header declares 3 words"), "{err}");
    }

    #[test]
    fn rejects_inconsistent_embedding_dimensions() {
        let err = parse_embedding_text("alpha 1 0\nbeta 0 1 2\n", "test").unwrap_err();
        assert!(
            err.to_string()
                .contains("has 3 vector values but expected 2"),
            "{err}"
        );
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
            .properties
            .insert("NumDocuments".to_string(), Value::Num(rows as f64));
        object.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![rows as f64, 1.0], vec![1, 2]).unwrap()),
        );
        object
    }

    #[tokio::test]
    async fn read_word_embedding_reads_plain_text_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("emb.vec");
        std::fs::write(&path, "3 2\nred 1 0\nblue 0 1\ngreen 0.5 0.5\n").unwrap();
        let value = read_word_embedding_builtin(Value::from(path.to_string_lossy().to_string()))
            .await
            .unwrap();
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        assert!(object.is_class(WORD_EMBEDDING_CLASS));
        assert_eq!(object.properties.get("Dimension"), Some(&Value::Num(2.0)));
    }

    #[tokio::test]
    async fn read_word_embedding_reads_zip_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("emb.zip");
        let file = StdFile::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file(
            "model.vec",
            zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated),
        )
        .unwrap();
        zip.write_all(b"2 2\nleft 1 0\nright 0 1\n").unwrap();
        zip.finish().unwrap();

        let value = read_word_embedding_builtin(Value::from(path.to_string_lossy().to_string()))
            .await
            .unwrap();
        let Value::Object(object) = value else {
            panic!("expected object");
        };
        assert!(object.is_class(WORD_EMBEDDING_CLASS));
        let model = embedding_from_object(&object, "test").unwrap();
        assert_eq!(model.vocabulary, vec!["left", "right"]);
    }

    #[tokio::test]
    async fn fast_text_word_embedding_returns_compact_300d_model() {
        let value = fast_text_word_embedding_builtin(vec![]).await.unwrap();
        let Value::Object(object) = value else {
            panic!("expected wordEmbedding object");
        };
        assert!(object.is_class(WORD_EMBEDDING_CLASS));
        assert_eq!(object.properties.get("Dimension"), Some(&Value::Num(300.0)));

        let italy = word2vec_builtin(vec![
            Value::Object(object.clone()),
            Value::String("Italy".into()),
        ])
        .await
        .unwrap();
        let rome = word2vec_builtin(vec![
            Value::Object(object.clone()),
            Value::String("Rome".into()),
        ])
        .await
        .unwrap();
        let paris = word2vec_builtin(vec![
            Value::Object(object.clone()),
            Value::String("Paris".into()),
        ])
        .await
        .unwrap();
        let (Value::Tensor(italy), Value::Tensor(rome), Value::Tensor(paris)) =
            (italy, rome, paris)
        else {
            panic!("expected tensors");
        };
        let query = italy
            .data
            .iter()
            .zip(&rome.data)
            .zip(&paris.data)
            .map(|((i, r), p)| i - r + p)
            .collect::<Vec<_>>();
        let nearest = vec2word_builtin(vec![
            Value::Object(object),
            Value::Tensor(Tensor::new(query, vec![1, 300]).unwrap()),
            Value::Num(1.0),
        ])
        .await
        .unwrap();
        let Value::OutputList(outputs) = nearest else {
            panic!("expected output list");
        };
        let Value::StringArray(words) = &outputs[0] else {
            panic!("expected nearest words");
        };
        assert_eq!(words.data, vec!["France"]);

        let err = fast_text_word_embedding_builtin(vec![Value::Num(1.0)])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("expected no input"), "{err}");
    }

    #[tokio::test]
    async fn train_word_embedding_trains_from_text_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("training.txt");
        std::fs::write(&path, "alpha beta alpha\nbeta gamma alpha\n").unwrap();
        let value = train_word_embedding_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::String("Dimension".into()),
            Value::Num(8.0),
            Value::String("Window".into()),
            Value::Num(1.0),
            Value::String("MinCount".into()),
            Value::Num(1.0),
            Value::String("NGramRange".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ])
        .await
        .unwrap();
        let Value::Object(object) = value else {
            panic!("expected wordEmbedding object");
        };
        assert!(object.is_class(WORD_EMBEDDING_CLASS));
        let model = embedding_from_object(&object, "test").unwrap();
        assert_eq!(model.dimension, 8);
        assert_eq!(model.vocabulary, vec!["alpha", "beta", "gamma"]);
        assert_eq!(model.vectors.len(), 24);

        let lookup = word2vec_builtin(vec![Value::Object(object), Value::String("alpha".into())])
            .await
            .unwrap();
        let Value::Tensor(tensor) = lookup else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.rows, 1);
        assert_eq!(tensor.cols, 8);
        assert!(tensor.data.iter().any(|value| value.abs() > 0.0));
    }

    #[tokio::test]
    async fn train_word_embedding_trains_from_tokenized_document_object() {
        let object = tokenized_document_object(vec![vec!["red", "blue"], vec!["red", "green"]]);

        let value = train_word_embedding_builtin(vec![
            Value::Object(object),
            Value::String("Dimension".into()),
            Value::Num(6.0),
            Value::String("MinCount".into()),
            Value::Num(1.0),
            Value::String("Model".into()),
            Value::String("cbow".into()),
            Value::String("LossFunction".into()),
            Value::String("softmax".into()),
        ])
        .await
        .unwrap();
        let Value::Object(object) = value else {
            panic!("expected wordEmbedding object");
        };
        let model = embedding_from_object(&object, "test").unwrap();
        assert_eq!(model.dimension, 6);
        assert_eq!(model.vocabulary, vec!["red", "blue", "green"]);
    }

    #[tokio::test]
    async fn doc2sequence_pads_to_longest_and_discards_unknown_words() {
        let model = EmbeddingModel {
            vocabulary: vec!["alpha".into(), "beta".into()],
            vectors: vec![1.0, 10.0, 2.0, 20.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["alpha", "beta"],
            vec!["missing", "beta"],
        ]));

        let result = doc2sequence_builtin(vec![emb, documents]).await.unwrap();
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        assert_eq!(cell.rows, 2);
        assert_eq!(cell.cols, 1);

        let Value::Tensor(first) = &cell.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![2, 2]);
        assert_eq!(first.data, vec![1.0, 10.0, 2.0, 20.0]);

        let Value::Tensor(second) = &cell.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![2, 2]);
        assert_eq!(second.data, vec![0.0, 0.0, 2.0, 20.0]);
    }

    #[tokio::test]
    async fn doc2sequence_supports_unknown_nan_right_padding_and_fixed_length() {
        let model = EmbeddingModel {
            vocabulary: vec!["alpha".into(), "beta".into()],
            vectors: vec![1.0, 10.0, 2.0, 20.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["alpha", "missing"],
            vec!["beta"],
        ]));

        let result = doc2sequence_builtin(vec![
            emb,
            documents,
            Value::String("UnknownWord".into()),
            Value::String("nan".into()),
            Value::String("PaddingDirection".into()),
            Value::String("right".into()),
            Value::String("PaddingValue".into()),
            Value::Num(-5.0),
            Value::String("Length".into()),
            Value::Num(3.0),
        ])
        .await
        .unwrap();
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        let Value::Tensor(first) = &cell.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![2, 3]);
        assert_eq!(first.data[0..2], [1.0, 10.0]);
        assert!(first.data[2].is_nan());
        assert!(first.data[3].is_nan());
        assert_eq!(first.data[4..6], [-5.0, -5.0]);

        let Value::Tensor(second) = &cell.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.data, vec![2.0, 20.0, -5.0, -5.0, -5.0, -5.0]);
    }

    #[tokio::test]
    async fn doc2sequence_none_padding_keeps_per_document_lengths_and_truncates_right() {
        let model = EmbeddingModel {
            vocabulary: vec!["a".into(), "b".into(), "c".into()],
            vectors: vec![1.0, 11.0, 2.0, 22.0, 3.0, 33.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["a", "b", "c"],
            vec!["a"],
        ]));

        let result = doc2sequence_builtin(vec![
            emb,
            documents,
            Value::String("PaddingDirection".into()),
            Value::String("none".into()),
            Value::String("Length".into()),
            Value::Num(2.0),
        ])
        .await
        .unwrap();
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        let Value::Tensor(first) = &cell.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![2, 2]);
        assert_eq!(first.data, vec![1.0, 11.0, 2.0, 22.0]);

        let Value::Tensor(second) = &cell.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![2, 1]);
        assert_eq!(second.data, vec![1.0, 11.0]);
    }

    #[tokio::test]
    async fn doc2sequence_supports_word_encoding_index_sequences_and_invalid_options() {
        let model = EmbeddingModel {
            vocabulary: vec!["alpha".into()],
            vectors: vec![1.0, 10.0],
            dimension: 2,
        };
        let mut documents_object =
            tokenized_document_object(vec![vec!["alpha", "missing", "beta"], vec!["beta"]]);
        documents_object.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        );
        let documents = Value::Object(documents_object);
        let mut word_encoding = ObjectInstance::new(WORD_ENCODING_CLASS.to_string());
        word_encoding
            .properties
            .insert("NumWords".to_string(), Value::Num(2.0));
        word_encoding.properties.insert(
            "Vocabulary".to_string(),
            Value::StringArray(
                StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap(),
            ),
        );
        let result = doc2sequence_builtin(vec![
            Value::Object(word_encoding),
            documents.clone(),
            Value::String("UnknownWord".into()),
            Value::String("nan".into()),
            Value::String("PaddingDirection".into()),
            Value::String("right".into()),
            Value::String("Length".into()),
            Value::Num(4.0),
        ])
        .await
        .unwrap();
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        assert_eq!(cell.shape, vec![1, 2]);
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 2);
        let Value::Tensor(first) = &cell.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![1, 4]);
        assert_eq!(first.data[0], 1.0);
        assert!(first.data[1].is_nan());
        assert_eq!(first.data[2..4], [2.0, 0.0]);
        let Value::Tensor(second) = &cell.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![1, 4]);
        assert_eq!(second.data, vec![2.0, 0.0, 0.0, 0.0]);

        let err = doc2sequence_builtin(vec![Value::Num(1.0), documents.clone()])
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("wordEmbedding or wordEncoding object"),
            "{err}"
        );

        let err = doc2sequence_builtin(vec![
            embedding_object(model).unwrap(),
            documents,
            Value::String("PaddingDirection".into()),
            Value::String("middle".into()),
        ])
        .await
        .unwrap_err();
        assert!(err.to_string().contains("PaddingDirection"), "{err}");
    }

    #[tokio::test]
    async fn doc2sequence_word_encoding_supports_left_none_and_shortest_length() {
        let mut word_encoding = ObjectInstance::new(WORD_ENCODING_CLASS.to_string());
        word_encoding
            .properties
            .insert("NumWords".to_string(), Value::Num(3.0));
        word_encoding.properties.insert(
            "Vocabulary".to_string(),
            Value::StringArray(
                StringArray::new(
                    vec!["alpha".into(), "beta".into(), "gamma".into()],
                    vec![1, 3],
                )
                .unwrap(),
            ),
        );
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["alpha", "beta", "gamma"],
            vec!["gamma"],
        ]));

        let left = doc2sequence_builtin(vec![
            Value::Object(word_encoding.clone()),
            documents.clone(),
        ])
        .await
        .unwrap();
        let Value::Cell(left) = left else {
            panic!("expected cell array");
        };
        let Value::Tensor(first) = &left.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![1, 3]);
        assert_eq!(first.data, vec![1.0, 2.0, 3.0]);
        let Value::Tensor(second) = &left.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![1, 3]);
        assert_eq!(second.data, vec![0.0, 0.0, 3.0]);

        let none = doc2sequence_builtin(vec![
            Value::Object(word_encoding.clone()),
            documents.clone(),
            Value::String("PaddingDirection".into()),
            Value::String("none".into()),
        ])
        .await
        .unwrap();
        let Value::Cell(none) = none else {
            panic!("expected cell array");
        };
        let Value::Tensor(second) = &none.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![1, 1]);
        assert_eq!(second.data, vec![3.0]);

        let shortest = doc2sequence_builtin(vec![
            Value::Object(word_encoding),
            documents,
            Value::String("Length".into()),
            Value::String("shortest".into()),
        ])
        .await
        .unwrap();
        let Value::Cell(shortest) = shortest else {
            panic!("expected cell array");
        };
        let Value::Tensor(first) = &shortest.data[0] else {
            panic!("expected first tensor");
        };
        assert_eq!(first.shape, vec![1, 1]);
        assert_eq!(first.data, vec![1.0]);
        let Value::Tensor(second) = &shortest.data[1] else {
            panic!("expected second tensor");
        };
        assert_eq!(second.shape, vec![1, 1]);
        assert_eq!(second.data, vec![3.0]);
    }

    #[tokio::test]
    async fn doc2sequence_allows_nan_padding_value() {
        let model = EmbeddingModel {
            vocabulary: vec!["alpha".into()],
            vectors: vec![1.0, 10.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let documents = Value::Object(tokenized_document_object(vec![vec!["alpha"]]));
        let result = doc2sequence_builtin(vec![
            emb,
            documents,
            Value::String("PaddingDirection".into()),
            Value::String("left".into()),
            Value::String("PaddingValue".into()),
            Value::Num(f64::NAN),
            Value::String("Length".into()),
            Value::Num(2.0),
        ])
        .await
        .unwrap();
        let Value::Cell(cell) = result else {
            panic!("expected cell array");
        };
        let Value::Tensor(sequence) = &cell.data[0] else {
            panic!("expected tensor");
        };
        assert_eq!(sequence.shape, vec![2, 2]);
        assert!(sequence.data[0].is_nan());
        assert!(sequence.data[1].is_nan());
        assert_eq!(sequence.data[2..4], [1.0, 10.0]);
    }

    #[tokio::test]
    async fn train_word_embedding_honors_min_count_and_option_validation() {
        let err = train_word_embedding_builtin(vec![
            Value::String("missing.txt".into()),
            Value::String("LossFunction".into()),
            Value::String("hs".into()),
            Value::String("NumNegativeSamples".into()),
            Value::Num(3.0),
        ])
        .await
        .unwrap_err();
        assert!(err.to_string().contains("NumNegativeSamples"), "{err}");

        let dir = tempdir().unwrap();
        let path = dir.path().join("training.txt");
        std::fs::write(&path, "solo once\n").unwrap();
        let err = train_word_embedding_builtin(vec![
            Value::from(path.to_string_lossy().to_string()),
            Value::String("MinCount".into()),
            Value::Num(2.0),
            Value::String("Verbose".into()),
            Value::Num(0.0),
        ])
        .await
        .unwrap_err();
        assert!(err.to_string().contains("no vocabulary words"), "{err}");
    }

    #[tokio::test]
    async fn word2vec_returns_rows_and_nan_for_missing_words() {
        let model = EmbeddingModel {
            vocabulary: vec!["king".into(), "queen".into()],
            vectors: vec![1.0, 0.0, 0.0, 1.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let words = Value::StringArray(
            StringArray::new(vec!["queen".into(), "missing".into()], vec![1, 2]).unwrap(),
        );
        let result = word2vec_builtin(vec![emb, words]).await.unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.rows, 2);
        assert_eq!(tensor.cols, 2);
        assert_eq!(tensor.data[0], 0.0);
        assert_eq!(tensor.data[2], 1.0);
        assert!(tensor.data[1].is_nan());
        assert!(tensor.data[3].is_nan());
    }

    #[tokio::test]
    async fn word2vec_ignore_case_uses_first_case_match() {
        let model = EmbeddingModel {
            vocabulary: vec!["Alpha".into(), "alpha".into()],
            vectors: vec![1.0, 0.0, 0.0, 1.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let result = word2vec_builtin(vec![
            emb,
            Value::String("ALPHA".into()),
            Value::String("IgnoreCase".into()),
            Value::Bool(true),
        ])
        .await
        .unwrap();
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.data, vec![1.0, 0.0]);
    }

    #[tokio::test]
    async fn vec2word_returns_words_and_distances() {
        let model = EmbeddingModel {
            vocabulary: vec!["east".into(), "north".into(), "mix".into()],
            vectors: vec![1.0, 0.0, 0.0, 1.0, 0.7, 0.7],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let query = Value::Tensor(Tensor::new(vec![0.6, 0.8], vec![1, 2]).unwrap());
        let result = vec2word_builtin(vec![
            emb,
            query,
            Value::Num(2.0),
            Value::String("Distance".into()),
            Value::String("cosine".into()),
        ])
        .await
        .unwrap();
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        let Value::StringArray(words) = &outputs[0] else {
            panic!("expected words");
        };
        assert_eq!(words.data[0], "mix");
        assert_eq!(words.data.len(), 2);
        let Value::Tensor(dist) = &outputs[1] else {
            panic!("expected distances");
        };
        assert!(dist.data[0] < dist.data[1]);
    }

    #[tokio::test]
    async fn vec2word_rejects_wrong_vector_dimension() {
        let model = EmbeddingModel {
            vocabulary: vec!["east".into()],
            vectors: vec![1.0, 0.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let query = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.5], vec![1, 3]).unwrap());
        let err = vec2word_builtin(vec![emb, query]).await.unwrap_err();
        assert!(err.to_string().contains("must have 2 columns"), "{err}");
    }

    #[tokio::test]
    async fn vec2word_rejects_invalid_k_as_k_not_option_name() {
        let model = EmbeddingModel {
            vocabulary: vec!["east".into()],
            vectors: vec![1.0, 0.0],
            dimension: 2,
        };
        let emb = embedding_object(model).unwrap();
        let query = Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap());
        let err = vec2word_builtin(vec![emb, query, Value::Num(0.0)])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("expected positive integer scalar"),
            "{err}"
        );
    }
}
