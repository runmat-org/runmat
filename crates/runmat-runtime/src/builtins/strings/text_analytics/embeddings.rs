//! Word embedding compatibility objects and lookup helpers.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use runmat_builtins::{
    Access, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ClassDef, ObjectInstance, PropertyDef, ResolveContext, StringArray, Tensor, Type,
    Value,
};
use runmat_filesystem::File;
use runmat_macros::runtime_builtin;

use crate::builtins::strings::core::compat::scalar_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

pub const WORD_EMBEDDING_CLASS: &str = "wordEmbedding";
const VECTOR_PROPERTY: &str = "__Vectors";
const MAX_EMBEDDING_FILE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ZIP_ENTRIES: usize = 256;

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

const IN_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "UTF-8 word2vec/GloVe text file or zip file containing one.",
}];

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

const ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READWORDEMBEDDING.IO",
    identifier: Some("RunMat:readWordEmbedding:IOError"),
    when: "The requested word embedding file cannot be read.",
    message: "Unable to read word embedding file",
};

const READ_ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_READ_INVALID_INPUT, ERROR_IO];
const WORD2VEC_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_WORD2VEC_INVALID_INPUT];
const VEC2WORD_ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_VEC2WORD_INVALID_INPUT];

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

fn build_word_lookup(vocabulary: &[String], ignore_case: bool) -> HashMap<String, usize> {
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

fn parse_positive_integer(value: &Value, fn_name: &str) -> BuiltinResult<usize> {
    let n = match value {
        Value::Num(value) => *value,
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
        "readWordEmbedding" => "RunMat:readWordEmbedding:InvalidInput",
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
