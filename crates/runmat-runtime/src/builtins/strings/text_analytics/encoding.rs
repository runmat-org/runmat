//! Word encoding compatibility objects and word/index lookup helpers.
pub use runmat_builtins::{
    IND2WORD_DESCRIPTOR, IND2WORD_ERROR_IND2WORD_INVALID_INPUT as ERROR_IND2WORD_INVALID_INPUT,
    IND2WORD_EXTENSIONS, IND2WORD_INTEGER_CAPABILITIES, IND2WORD_NONVECTOR_EXTENSION,
    IND2WORD_RESIDENT_EXTENSION, IND2WORD_TYPED_INTEGER_EXTENSION, IS_VOCABULARY_WORD_DESCRIPTOR,
    IS_VOCABULARY_WORD_ERROR_IS_VOCABULARY_WORD_INVALID_INPUT as ERROR_IS_VOCABULARY_WORD_INVALID_INPUT,
    IS_VOCABULARY_WORD_INTEGER_AUDIT, WORD2IND_DESCRIPTOR,
    WORD2IND_ERROR_WORD2IND_INVALID_INPUT as ERROR_WORD2IND_INVALID_INPUT, WORD2IND_INTEGER_AUDIT,
};
use runmat_types::MemberAccess;

use runmat_value::IntValue;
use std::collections::HashMap;

use runmat_builtins::WORD_ENCODING_ERROR_INVALID_INPUT;
pub use runmat_builtins::{
    WORD_ENCODING_DESCRIPTOR, WORD_ENCODING_EXTENSIONS, WORD_ENCODING_INTEGER_CAPABILITIES,
    WORD_ENCODING_INTEGER_MAX_WORDS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, LogicalArray, ObjectInstance, StringArray, Tensor, Value};

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    documents_from_object, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::strings::text_analytics::embeddings::{
    build_word_lookup, word_embedding_vocabulary_from_object, WORD_EMBEDDING_CLASS,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

pub const WORD_ENCODING_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("wordEncoding");

static WORD_ENCODING_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(WORD_ENCODING_CLASS);

#[runtime_builtin(
    name = "wordEncoding",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn word_encoding_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    for index in 0..args.len().saturating_sub(1) {
        if scalar_text(&args[index], "wordEncoding")
            .is_ok_and(|name| name.eq_ignore_ascii_case("MaxNumWords"))
            && is_typed_integer_value(&args[index + 1])
        {
            crate::compatibility::ensure_builtin_extension_enabled(
                &WORD_ENCODING_INTEGER_MAX_WORDS_EXTENSION,
                "wordEncoding",
            )?;
        }
    }
    let gathered = args
        .try_map_preserving_roots_async(args.len(), runtime.as_ref(), async |index| {
            gather_encoding_value(&args[index], "wordEncoding").await
        })
        .await?;
    let (source, options) = parse_word_encoding_args(gathered.view())?;
    word_encoding_object_in_context(runtime.as_ref(), build_word_encoding(source, options)?)
}

#[runtime_builtin(
    name = "word2ind",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn word2ind_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::WORD2IND_DESCRIPTOR,
        &runmat_builtins::WORD2IND_ERROR_TOO_MANY_OUTPUTS,
        "word2ind",
    )?;
    let work = super::model_admission::UnboundedModelWork::admit(runtime.as_ref(), "word2ind")?;
    work.finish(
        async {
            if args.iter().skip(1).any(|value| {
                crate::builtins::common::validation::value_contains_native_integer_class(value)
                    || value_contains_resident(value)
            }) {
                return Err(encoding_error(
            "word2ind",
            "word2ind: words and option names must be host text and IgnoreCase must be logical",
        ));
            }
            let gathered = gather_args(&args, runtime.as_ref(), "word2ind").await?;
            work.check()?;
            let (object, words, options) = parse_word2ind_args(gathered.view())?;
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
        .await,
    )
}

#[runtime_builtin(
    name = "ind2word",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn ind2word_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::IND2WORD_DESCRIPTOR,
        &runmat_builtins::IND2WORD_ERROR_TOO_MANY_OUTPUTS,
        "ind2word",
    )?;
    let work = super::model_admission::UnboundedModelWork::admit(runtime.as_ref(), "ind2word")?;
    work.finish(
        async {
            ensure_ind2word_extensions(args.view())?;
            let gathered = gather_args(&args, runtime.as_ref(), "ind2word").await?;
            work.check()?;
            let (object, indices) = parse_ind2word_args(gathered.view())?;
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
        .await,
    )
}

fn ensure_ind2word_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    if args.len() != 2 {
        return Ok(());
    }
    let indices = &args[1];
    if is_typed_integer_value(indices) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IND2WORD_TYPED_INTEGER_EXTENSION,
            "ind2word",
        )?;
    }
    if crate::dispatcher::value_contains_gpu(indices) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IND2WORD_RESIDENT_EXTENSION,
            "ind2word",
        )?;
    }
    if value_shape(indices).is_some_and(|shape| !is_vector_shape(shape)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IND2WORD_NONVECTOR_EXTENSION,
            "ind2word",
        )?;
    }
    Ok(())
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn value_shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) => Some(&[1, 1]),
        _ => None,
    }
}

fn is_vector_shape(shape: &[usize]) -> bool {
    shape.len() <= 2 && shape.iter().filter(|extent| **extent > 1).count() <= 1
}

#[runtime_builtin(
    name = "isVocabularyWord",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::strings::text_analytics::encoding"
)]
async fn is_vocabulary_word_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::IS_VOCABULARY_WORD_DESCRIPTOR,
        &runmat_builtins::IS_VOCABULARY_WORD_ERROR_TOO_MANY_OUTPUTS,
        "isVocabularyWord",
    )?;
    let work =
        super::model_admission::UnboundedModelWork::admit(runtime.as_ref(), "isVocabularyWord")?;
    work.finish(async {

    if args.iter().any(value_contains_resident) {
        return Err(encoding_error(
            "isVocabularyWord",
            "isVocabularyWord: provider-resident numeric inputs are not vocabulary objects, words, or controls",
        ));
    }
    let gathered = gather_args(&args, runtime.as_ref(), "isVocabularyWord").await?;
 work.check()?;
    let (object, words, options) = parse_is_vocabulary_word_args(gathered.view())?;
    let vocabulary = if object.is_class(WORD_ENCODING_CLASS) {
        word_encoding_from_object(&object, "isVocabularyWord")?.vocabulary
    } else if object.is_class(WORD_EMBEDDING_CLASS) {
        word_embedding_vocabulary_from_object(&object, "isVocabularyWord")?
    } else {
        return Err(encoding_error(
            "isVocabularyWord",
            format!(
                "isVocabularyWord: expected wordEmbedding or wordEncoding object, got {}",
                object.class_name()
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
 }.await)
}

fn value_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(value) => value.data().iter().any(value_contains_resident),
        Value::Struct(value) => value.fields.values().any(value_contains_resident),
        Value::StructArray(array) => array.any_value(value_contains_resident),
        Value::Object(value) => value.properties().values().any(value_contains_resident),
        Value::Closure(value) => value.captures.iter().any(value_contains_resident),
        Value::OutputList(values) => values.iter().any(value_contains_resident),
        _ => false,
    }
}

async fn gather_args(
    args: &crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    fn_name: &str,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let work = super::model_admission::UnboundedModelWork::admit(runtime, fn_name)?;
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        work.wait(gather_encoding_value(&args[index], fn_name))
            .await
    })
    .await
}

async fn gather_encoding_value(value: &Value, _fn_name: &str) -> BuiltinResult<Value> {
    gather_if_needed_async(value).await
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
                object.class_name()
            ),
        ));
    }
    let vocabulary: Vec<String> = match object.properties().get("Vocabulary") {
        Some(Value::StringArray(array)) => array
            .data()
            .iter()
            .map(|text| required_word_text(text, fn_name).map(str::to_owned))
            .collect::<BuiltinResult<Vec<_>>>()?,
        other => {
            return Err(encoding_error(
                fn_name,
                format!(
                    "{fn_name}: wordEncoding object has invalid Vocabulary property: {other:?}"
                ),
            ));
        }
    };
    match object.properties().get("NumWords") {
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

fn word_encoding_object_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    model: WordEncodingModel,
) -> BuiltinResult<Value> {
    ensure_word_encoding_class_registered();
    let fields = [
        ("NumWords", Value::Num(model.vocabulary.len() as f64)),
        (
            "Vocabulary",
            Value::StringArray(
                StringArray::new(model.vocabulary.clone(), vec![1, model.vocabulary.len()])
                    .map_err(|err| encoding_error("wordEncoding", err))?,
            ),
        ),
    ];
    let object = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        WORD_ENCODING_CLASS.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )?;
    Ok(object.into_value_and_lease().0)
}

fn ensure_word_encoding_class_registered() {
    WORD_ENCODING_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        for name in ["NumWords", "Vocabulary"] {
            properties.insert(name.into(), property_def(name));
        }
        crate::class_registry::register_builtin_class(
            WORD_ENCODING_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: WORD_ENCODING_CLASS.into(),
                parent: None,
                properties,
                methods: HashMap::new(),
            },
        );
    });
}

fn property_def(name: &str) -> crate::class_registry::RuntimeProperty {
    crate::class_registry::RuntimeProperty {
        name: name.into(),
        is_static: false,
        is_constant: false,
        is_dependent: false,
        get_access: MemberAccess::Public,
        set_access: MemberAccess::Public,
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
    args: crate::call::FunctionArgumentView<'_>,
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
                    object.class_name()
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
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(ObjectInstance, WordInput, LookupOptions)> {
    if args.len() < 2 {
        return Err(encoding_error(
            "word2ind",
            "word2ind: expected word2ind(enc, words)",
        ));
    }
    let object = object_arg(&args[0], "word2ind", "wordEncoding")?;
    let words = word_input_from_value(&args[1], "word2ind")?;
    let options = parse_lookup_options(args.range(2..)?, "word2ind")?;
    Ok((object, words, options))
}

fn parse_is_vocabulary_word_args(
    args: crate::call::FunctionArgumentView<'_>,
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
    let options = parse_lookup_options(args.range(2..)?, "isVocabularyWord")?;
    Ok((object, words, options))
}

fn parse_ind2word_args(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(ObjectInstance, NumericInput)> {
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

fn parse_lookup_options(
    args: crate::call::FunctionArgumentView<'_>,
    fn_name: &str,
) -> BuiltinResult<LookupOptions> {
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
            words: vec![required_word_text(text, fn_name)?.to_owned()],
            shape: vec![1, 1],
        }),
        Value::StringArray(array) => Ok(WordInput {
            words: array.data().iter().map(|text| required_word_text(text, fn_name).map(str::to_owned)).collect::<BuiltinResult<Vec<_>>>()?,
            shape: array.shape.to_vec(),
        }),
        Value::CharArray(array) if array.rows <= 1 => Ok(WordInput {
            words: vec![char_row_to_string(array, 0, fn_name)?],
            shape: vec![1, 1],
        }),
        Value::CharArray(array) => {
            let mut words = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let text = char_row_to_string(array, row, fn_name)?;
                words.push(text.trim_end().to_string());
            }
            Ok(WordInput {
                words,
                shape: vec![array.rows, 1],
            })
        }
        Value::Cell(cell) => {
            let words = cell
                .data()
                .iter()
                .map(|item| {
                    scalar_text(item, fn_name)
                        .map_err(|err| encoding_error(fn_name, err.to_string()))
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            Ok(WordInput {
                words,
                shape: cell.shape.to_vec(),
            })
        }
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: expected string, character vector, or cell array of words, got {other:?}"),
        )),
    }
}

struct NumericInput {
    values: Vec<NumericIndex>,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
enum NumericIndex {
    Float(f64),
    Integer(IntValue),
}

fn numeric_input_from_value(value: &Value, fn_name: &str) -> BuiltinResult<NumericInput> {
    match value {
        Value::Num(value) => Ok(NumericInput {
            values: vec![NumericIndex::Float(*value)],
            shape: vec![1, 1],
        }),
        Value::Int(value) => Ok(NumericInput {
            values: vec![NumericIndex::Integer(value.clone())],
            shape: vec![1, 1],
        }),
        Value::Tensor(tensor) => {
            let values = if let Some(storage) = tensor.integer_storage() {
                (0..storage.len())
                    .map(|index| {
                        storage
                            .value_at(index)
                            .map(NumericIndex::Integer)
                            .expect("integer index is within storage bounds")
                    })
                    .collect()
            } else {
                tensor_utils::tensor_values_f64(tensor)
                    .into_iter()
                    .map(NumericIndex::Float)
                    .collect()
            };
            Ok(NumericInput {
                values,
                shape: tensor.shape().to_vec(),
            })
        }
        other => Err(encoding_error(
            fn_name,
            format!("{fn_name}: expected numeric positive integer indices, got {other:?}"),
        )),
    }
}

fn positive_index(value: NumericIndex, len: usize, fn_name: &str) -> BuiltinResult<usize> {
    let idx = match value {
        NumericIndex::Float(value) => {
            if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
                return Err(encoding_error(
                    fn_name,
                    format!("{fn_name}: indices must be positive integers, got {value}"),
                ));
            }
            if value > usize::MAX as f64 {
                return Err(encoding_error(
                    fn_name,
                    format!("{fn_name}: index exceeds platform limits"),
                ));
            }
            value as usize
        }
        NumericIndex::Integer(value) => value.try_to_usize().ok_or_else(|| {
            encoding_error(
                fn_name,
                format!("{fn_name}: indices must be positive integers"),
            )
        })?,
    };
    if idx == 0 {
        return Err(encoding_error(
            fn_name,
            format!("{fn_name}: indices must be positive integers"),
        ));
    }
    if idx > len {
        return Err(encoding_error(
            fn_name,
            format!("{fn_name}: index {idx} exceeds vocabulary size {len}"),
        ));
    }
    Ok(idx - 1)
}

fn parse_max_num_words(value: &Value) -> BuiltinResult<Option<usize>> {
    if let Value::Int(value) = value {
        return value
            .try_to_usize()
            .filter(|value| *value >= 1)
            .map(Some)
            .ok_or_else(|| {
                encoding_error(
                    "wordEncoding",
                    "wordEncoding: MaxNumWords must be a positive integer or Inf",
                )
            });
    }
    if let Value::Tensor(tensor) = value {
        if tensor_utils::is_scalar_tensor(tensor) {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return value
                    .try_to_usize()
                    .filter(|value| *value >= 1)
                    .map(Some)
                    .ok_or_else(|| {
                        encoding_error(
                            "wordEncoding",
                            "wordEncoding: MaxNumWords must be a positive integer or Inf",
                        )
                    });
            }
        }
    }
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

fn int_value_to_f64(value: &runmat_value::IntValue) -> f64 {
    match value {
        runmat_value::IntValue::I8(value) => *value as f64,
        runmat_value::IntValue::I16(value) => *value as f64,
        runmat_value::IntValue::I32(value) => *value as f64,
        runmat_value::IntValue::I64(value) => *value as f64,
        runmat_value::IntValue::U8(value) => *value as f64,
        runmat_value::IntValue::U16(value) => *value as f64,
        runmat_value::IntValue::U32(value) => *value as f64,
        runmat_value::IntValue::U64(value) => *value as f64,
    }
}

fn char_row_to_string(array: &CharArray, row: usize, fn_name: &str) -> BuiltinResult<String> {
    if array.rows == 0 {
        return Ok(String::new());
    }
    crate::builtins::strings::common::char_row_to_string_slice(array.data(), array.cols, row)
        .map_err(|source| {
            let mut diagnostic = encoding_error(
                fn_name,
                format!("{fn_name}: malformed UTF-16 character input"),
            );
            diagnostic.source = Some(Box::new(source));
            diagnostic
        })
}

fn encoding_error(fn_name: &str, message: impl Into<String>) -> crate::RuntimeError {
    let descriptor = match fn_name {
        "word2ind" => ERROR_WORD2IND_INVALID_INPUT,
        "ind2word" => ERROR_IND2WORD_INVALID_INPUT,
        "isVocabularyWord" => ERROR_IS_VOCABULARY_WORD_INVALID_INPUT,
        _ => WORD_ENCODING_ERROR_INVALID_INPUT,
    };
    let builder = build_runtime_error(message.into()).with_builtin(fn_name);
    match descriptor.identifier {
        Some(identifier) => builder.with_identifier(identifier).build(),
        None => builder.build(),
    }
}

fn required_word_text<'a>(
    text: &'a runmat_value::StringElement,
    fn_name: &str,
) -> BuiltinResult<&'a str> {
    text.as_str()
        .ok_or_else(|| encoding_error(fn_name, format!("{fn_name}: word text must not be missing")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CellArray, IntegerStorage, Utf16CodeUnit};

    #[test]
    fn word_input_reads_row_major_utf16_without_cross_row_surrogates() {
        let rows = CharArray::from_code_units(
            vec![
                Utf16CodeUnit::from_raw('A' as u16),
                Utf16CodeUnit::from_raw('B' as u16),
                Utf16CodeUnit::from_raw('C' as u16),
                Utf16CodeUnit::from_raw('D' as u16),
            ],
            2,
            2,
        )
        .expect("character rows");
        let words = word_input_from_value(&Value::CharArray(rows), "wordEncoding")
            .expect("row-major words");
        assert_eq!(words.words, vec!["AB".to_string(), "CD".to_string()]);

        let empty = CharArray::from_code_units(Vec::new(), 0, 3).expect("0-by-3 character array");
        let empty_words = word_input_from_value(&Value::CharArray(empty), "wordEncoding")
            .expect("empty geometry");
        assert_eq!(empty_words.words, vec![String::new()]);

        let split_pair = CharArray::from_code_units(
            vec![
                Utf16CodeUnit::from_raw(0xD83D),
                Utf16CodeUnit::from_raw(0xDE42),
            ],
            2,
            1,
        )
        .expect("raw units");
        let error = word_input_from_value(&Value::CharArray(split_pair), "wordEncoding")
            .err()
            .expect("surrogates cannot join across rows");
        assert_eq!(
            error.identifier(),
            WORD_ENCODING_ERROR_INVALID_INPUT.identifier
        );
    }

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
        object.replace_property_unbounded("Documents".to_string(), Value::Cell(documents));
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

        assert_eq!(
            input.values,
            vec![
                NumericIndex::Integer(IntValue::I16(2)),
                NumericIndex::Integer(IntValue::I16(3))
            ]
        );
        assert_eq!(input.shape, vec![1, 2]);
    }

    #[tokio::test]
    async fn word_encoding_builds_first_seen_and_frequency_vocabularies() {
        let documents = Value::Object(tokenized_document_object(vec![
            vec!["beta", "alpha", "beta"],
            vec!["gamma", "alpha", "beta"],
        ]));
        let first_seen = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![documents.clone()]),
        )
        .await
        .unwrap();
        let Value::Object(first_seen) = first_seen else {
            panic!("expected object");
        };
        let model = word_encoding_from_object(&first_seen, "test").unwrap();
        assert_eq!(model.vocabulary, vec!["beta", "alpha", "gamma"]);

        let frequency = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                documents,
                Value::String("Order".into()),
                Value::String("frequency".into()),
                Value::String("MaxNumWords".into()),
                Value::Num(2.0),
            ]),
        )
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
        let enc = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![words]),
        )
        .await
        .unwrap();
        let Value::Object(enc) = enc else {
            panic!("expected object");
        };
        assert_eq!(enc.properties().get("NumWords"), Some(&Value::Num(2.0)));

        let err = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("x".into()),
                Value::String("Order".into()),
                Value::String("frequency".into()),
            ]),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("only supported for tokenizedDocument input"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn word_encoding_typed_maximum_is_a_gated_exact_control() {
        let documents = Value::Object(tokenized_document_object(vec![vec!["alpha", "beta"]]));
        let args = vec![
            documents,
            Value::String("MaxNumWords".into()),
            Value::Int(IntValue::U64(1)),
        ];
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args.clone()),
        )
        .await
        .expect_err("typed MaxNumWords is gated in strict mode");
        assert_eq!(
            error.identifier(),
            WORD_ENCODING_INTEGER_MAX_WORDS_EXTENSION.error_identifier
        );
        drop(_strict);

        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let value = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
        .expect("RunMat mode accepts typed MaxNumWords");
        let Value::Object(object) = value else {
            panic!("expected wordEncoding object");
        };
        assert_eq!(
            word_encoding_from_object(&object, "test")
                .unwrap()
                .vocabulary,
            vec!["alpha"]
        );
    }

    #[tokio::test]
    async fn word2ind_preserves_shape_and_supports_ignore_case() {
        let enc = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                StringArray::new(vec!["Alpha".into(), "beta".into()], vec![1, 2]).unwrap(),
            )]),
        )
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
        let out = word2ind_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                enc,
                words,
                Value::String("IgnoreCase".into()),
                Value::Bool(true),
            ]),
        )
        .await
        .unwrap();
        let Value::Tensor(indices) = out else {
            panic!("expected tensor");
        };
        assert_eq!(indices.shape(), &[2, 2]);
        assert_eq!(indices.materialize_f64()[0], 2.0);
        assert!(indices.materialize_f64()[1].is_nan());
        assert_eq!(indices.materialize_f64()[2], 1.0);
        assert_eq!(indices.materialize_f64()[3], 1.0);
    }

    #[tokio::test]
    async fn word2ind_rejects_integer_words_before_object_validation() {
        let error = word2ind_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("not an object".into()),
                Value::Int(IntValue::U8(1)),
            ]),
        )
        .await
        .expect_err("integer words are outside the text-only surface");
        assert!(error.message().contains("must be host text"));
    }

    #[tokio::test]
    async fn ind2word_preserves_numeric_shape_and_rejects_bad_indices() {
        let enc = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                StringArray::new(
                    vec!["red".into(), "blue".into(), "green".into()],
                    vec![1, 3],
                )
                .unwrap(),
            )]),
        )
        .await
        .unwrap();
        let out = ind2word_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                enc.clone(),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![1, 2]).unwrap()),
            ]),
        )
        .await
        .unwrap();
        let Value::StringArray(words) = out else {
            panic!("expected string array");
        };
        assert_eq!(words.shape, vec![1, 2]);
        assert_eq!(words.data(), vec!["red", "green"]);

        let err = ind2word_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![enc, Value::Num(4.0)]),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("exceeds vocabulary"), "{err}");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn ind2word_preserves_rank_zero_and_rank_one_numeric_shapes() {
        let enc = futures::executor::block_on(word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap(),
            )]),
        ))
        .unwrap();

        for (indices, expected_shape, expected_words) in [
            (
                Tensor::new(vec![2.0], vec![]).unwrap(),
                Vec::new(),
                vec!["blue".to_string()],
            ),
            (
                Tensor::new(vec![1.0, 2.0], vec![2]).unwrap(),
                vec![2],
                vec!["red".to_string(), "blue".to_string()],
            ),
        ] {
            let value = futures::executor::block_on(ind2word_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    enc.clone(),
                    Value::Tensor(indices),
                ]),
            ))
            .expect("ind2word low-rank input");
            let Value::StringArray(words) = value else {
                panic!("expected string array")
            };
            assert_eq!(words.shape, expected_shape);
            assert_eq!(words.data(), expected_words);
        }
    }

    #[test]
    fn ind2word_extensions_gate_before_gather_and_integer_indices_stay_exact() {
        let enc = futures::executor::block_on(word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap(),
            )]),
        ))
        .unwrap();

        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let integer_error = futures::executor::block_on(ind2word_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    enc.clone(),
                    poisoned_integer_vector(IntegerStorage::U64(vec![1]), 1),
                ]),
            ))
            .unwrap_err();
            assert_eq!(
                integer_error.identifier(),
                Some("RunMat:compatibility:Ind2wordTypedIntegerExtension")
            );
            let matrix_error = futures::executor::block_on(ind2word_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    enc.clone(),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).unwrap()),
                ]),
            ))
            .unwrap_err();
            assert_eq!(
                matrix_error.identifier(),
                Some("RunMat:compatibility:Ind2wordNonvectorExtension")
            );
        }

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let wide = futures::executor::block_on(ind2word_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                enc.clone(),
                poisoned_integer_vector(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1),
            ]),
        ))
        .unwrap_err();
        assert!(wide.message().contains("exceeds vocabulary"));

        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &tensor)
                .expect("resident indices");
            runmat_accelerate::fusion_residency::mark(&handle);
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = futures::executor::block_on(ind2word_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    enc.clone(),
                    Value::GpuTensor(handle.clone()),
                ]),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:Ind2wordResidentExtension")
            );
            assert!(runmat_accelerate::fusion_residency::is_resident(&handle));
            handle.allocation_lease().validate_live().unwrap();
            handle.release().expect("release owned resident indices");
        });
    }

    #[tokio::test]
    async fn is_vocabulary_word_supports_word_encoding() {
        let enc = word_encoding_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::StringArray(
                StringArray::new(vec!["RunMat".into(), "GPU".into()], vec![1, 2]).unwrap(),
            )]),
        )
        .await
        .unwrap();
        let words = Value::StringArray(
            StringArray::new(vec!["runmat".into(), "cpu".into()], vec![1, 2]).unwrap(),
        );
        let out = is_vocabulary_word_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                enc,
                words,
                Value::String("IgnoreCase".into()),
                Value::Bool(true),
            ]),
        )
        .await
        .unwrap();
        let Value::LogicalArray(mask) = out else {
            panic!("expected logical array");
        };
        assert_eq!(mask.shape, vec![1, 2]);
        assert_eq!(mask.data, vec![1, 0]);
    }
}
