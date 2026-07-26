//! Sentiment scoring helpers for Text Analytics compatibility objects.

use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ObjectInstance, ResolveContext, StringArray, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type, documents_from_object, text_analytics_error, tokenized_document_language,
    words_from_word_vector, TOKENIZED_DOCUMENT_CLASS,
};
use crate::builtins::table::{table_variables, TABLE_CLASS};
use crate::{gather_if_needed_async, BuiltinResult};

const BOOSTER_AMOUNT: f64 = 0.293;
const CAPS_AMOUNT: f64 = 0.733;
const NEGATION_SCALAR: f64 = -0.74;
const NORMALIZATION_ALPHA: f64 = 15.0;

const OUT_COMPOUND: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "compoundScores",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Compound VADER-style sentiment score for each document.",
}];

const OUT_ALL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "compoundScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Compound VADER-style sentiment score for each document.",
    },
    BuiltinParamDescriptor {
        name: "positiveScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive sentiment proportions.",
    },
    BuiltinParamDescriptor {
        name: "negativeScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Negative sentiment proportions.",
    },
    BuiltinParamDescriptor {
        name: "neutralScores",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Neutral token proportions.",
    },
];

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
        description: "Name-value options: SentimentLexicon, Boosters, Dampeners, Negations.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.VADER_SENTIMENT_SCORES.INVALID_INPUT",
    identifier: Some("RunMat:vaderSentimentScores:InvalidInput"),
    when: "Inputs do not match supported vaderSentimentScores forms.",
    message: "vaderSentimentScores: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const VADER_SENTIMENT_SCORES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "compoundScores = vaderSentimentScores(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_COMPOUND,
        },
        BuiltinSignatureDescriptor {
            label: "compoundScores = vaderSentimentScores(documents,Name,Value)",
            inputs: &IN_DOCUMENTS_REST,
            outputs: &OUT_COMPOUND,
        },
        BuiltinSignatureDescriptor {
            label:
                "[compoundScores,positiveScores,negativeScores,neutralScores] = vaderSentimentScores(___)",
            inputs: &IN_DOCUMENTS_REST,
            outputs: &OUT_ALL,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

#[runtime_builtin(
    name = "vaderSentimentScores",
    category = "strings/text_analytics",
    summary = "Score tokenized documents using VADER-style sentiment rules.",
    keywords = "vaderSentimentScores,VADER,sentiment,text analytics,tokenizedDocument",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::sentiment::VADER_SENTIMENT_SCORES_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::text_analytics::sentiment"
)]
async fn vader_sentiment_scores_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args(args).await?;
    let (documents_object, options) = parse_args(args)?;
    if !tokenized_document_language(&documents_object)
        .trim()
        .eq_ignore_ascii_case("en")
    {
        return Err(sentiment_error(
            "vaderSentimentScores: only English tokenizedDocument inputs are supported",
        ));
    }
    let documents = documents_from_object(&documents_object, "vaderSentimentScores")?;
    let scores = score_documents(&documents, &options)?;
    scores.into_value()
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            sentiment_error(format!(
                "vaderSentimentScores: failed to gather input: {err}"
            ))
        })?);
    }
    Ok(out)
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<(ObjectInstance, SentimentOptions)> {
    if args.is_empty() {
        return Err(sentiment_error(
            "vaderSentimentScores: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(sentiment_error(
            "vaderSentimentScores: name-value options must appear in pairs",
        ));
    }
    let documents = match &args[0] {
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => object.clone(),
        Value::Object(object) => {
            return Err(sentiment_error(format!(
                "vaderSentimentScores: expected tokenizedDocument object, got {}",
                object.class_name
            )));
        }
        other => {
            return Err(sentiment_error(format!(
                "vaderSentimentScores: expected tokenizedDocument object, got {other:?}"
            )));
        }
    };

    let mut options = SentimentOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "vaderSentimentScores")
            .map_err(|err| sentiment_error(err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "sentimentlexicon" => {
                options.lexicon = sentiment_lexicon_from_table(&args[idx + 1])?;
            }
            "boosters" => {
                options.boosters = modifier_ngrams_from_value(&args[idx + 1], "Boosters")?;
            }
            "dampeners" => {
                options.dampeners = modifier_ngrams_from_value(&args[idx + 1], "Dampeners")?;
            }
            "negations" => {
                let negations: HashSet<String> =
                    words_from_word_vector(&args[idx + 1], "vaderSentimentScores")?
                        .into_iter()
                        .map(|word| word.trim().to_ascii_lowercase())
                        .filter(|word| !word.is_empty())
                        .collect();
                if negations.is_empty() {
                    return Err(sentiment_error(
                        "vaderSentimentScores: Negations must contain at least one word",
                    ));
                }
                options.negations = negations;
            }
            other => {
                return Err(sentiment_error(format!(
                    "vaderSentimentScores: unsupported option '{other}'"
                )));
            }
        }
        idx += 2;
    }
    Ok((documents, options))
}

#[derive(Clone, Debug)]
struct SentimentOptions {
    lexicon: HashMap<String, f64>,
    boosters: Vec<Vec<String>>,
    dampeners: Vec<Vec<String>>,
    negations: HashSet<String>,
}

impl Default for SentimentOptions {
    fn default() -> Self {
        Self {
            lexicon: DEFAULT_LEXICON.clone(),
            boosters: DEFAULT_BOOSTERS.iter().map(|item| phrase(item)).collect(),
            dampeners: DEFAULT_DAMPENERS.iter().map(|item| phrase(item)).collect(),
            negations: DEFAULT_NEGATIONS
                .iter()
                .map(|word| (*word).to_string())
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct SentimentOutputs {
    compound: Vec<f64>,
    positive: Vec<f64>,
    negative: Vec<f64>,
    neutral: Vec<f64>,
}

impl SentimentOutputs {
    fn into_value(self) -> BuiltinResult<Value> {
        let shape = vec![self.compound.len(), 1];
        let compound = tensor_value(self.compound, shape.clone())?;
        if let Some(out_count) = crate::output_count::current_output_count() {
            if out_count == 0 {
                return Ok(Value::OutputList(Vec::new()));
            }
            if out_count == 1 {
                return Ok(Value::OutputList(vec![compound]));
            }
            return Ok(crate::output_count::output_list_with_padding(
                out_count,
                vec![
                    compound,
                    tensor_value(self.positive, shape.clone())?,
                    tensor_value(self.negative, shape.clone())?,
                    tensor_value(self.neutral, shape)?,
                ],
            ));
        }
        Ok(compound)
    }
}

fn score_documents(
    documents: &[Vec<String>],
    options: &SentimentOptions,
) -> BuiltinResult<SentimentOutputs> {
    let mut compound = Vec::with_capacity(documents.len());
    let mut positive = Vec::with_capacity(documents.len());
    let mut negative = Vec::with_capacity(documents.len());
    let mut neutral = Vec::with_capacity(documents.len());
    for document in documents {
        let score = score_document(document, options);
        compound.push(score.compound);
        positive.push(score.positive);
        negative.push(score.negative);
        neutral.push(score.neutral);
    }
    Ok(SentimentOutputs {
        compound,
        positive,
        negative,
        neutral,
    })
}

#[derive(Clone, Copy, Debug)]
struct DocumentScore {
    compound: f64,
    positive: f64,
    negative: f64,
    neutral: f64,
}

fn score_document(tokens: &[String], options: &SentimentOptions) -> DocumentScore {
    let lower_tokens = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let has_caps_emphasis = tokens.iter().any(|token| is_all_caps_word(token))
        && tokens.iter().any(|token| is_mixed_or_lower_word(token));

    let mut valences = Vec::with_capacity(tokens.len());
    let mut neutral_count = 0.0f64;
    for (idx, token) in tokens.iter().enumerate() {
        let Some(lexeme) = sentiment_lexeme(token, &lower_tokens[idx], options) else {
            if is_neutral_counted_token(token) {
                neutral_count += 1.0;
            }
            valences.push(0.0);
            continue;
        };
        let mut valence = *options.lexicon.get(lexeme).unwrap_or(&0.0);
        if valence != 0.0 {
            valence = apply_modifiers(valence, idx, &lower_tokens, options);
            if has_caps_emphasis && is_all_caps_word(token) {
                valence += if valence >= 0.0 {
                    CAPS_AMOUNT
                } else {
                    -CAPS_AMOUNT
                };
            }
        } else {
            neutral_count += 1.0;
        }
        valences.push(valence);
    }

    let mut sum = valences.iter().sum::<f64>();
    sum += punctuation_emphasis(tokens, sum);
    let compound = normalize_score(sum);
    let positive_sum = valences.iter().filter(|value| **value > 0.0).sum::<f64>();
    let negative_sum = valences
        .iter()
        .filter(|value| **value < 0.0)
        .map(|value| value.abs())
        .sum::<f64>();
    let total = positive_sum + negative_sum + neutral_count;
    let (positive, negative, neutral) = if total > 0.0 {
        (
            positive_sum / total,
            negative_sum / total,
            neutral_count / total,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    DocumentScore {
        compound,
        positive,
        negative,
        neutral,
    }
}

fn sentiment_lexeme<'a>(
    token: &'a str,
    lower_token: &'a str,
    options: &'a SentimentOptions,
) -> Option<&'a str> {
    if options.lexicon.contains_key(lower_token) {
        return Some(lower_token);
    }
    let stripped =
        lower_token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '\'' && ch != '-');
    if stripped != lower_token && options.lexicon.contains_key(stripped) {
        return Some(stripped);
    }
    if token.chars().count() == 1 {
        return None;
    }
    None
}

fn apply_modifiers(
    mut valence: f64,
    idx: usize,
    lower_tokens: &[String],
    options: &SentimentOptions,
) -> f64 {
    for distance in 1..=3 {
        if idx < distance {
            continue;
        }
        let prev = &lower_tokens[idx - distance];
        let decay = 0.95f64.powi((distance - 1) as i32);
        if is_single_word_modifier(prev, &options.boosters) {
            valence += signed_modifier(valence, BOOSTER_AMOUNT * decay);
        }
        if is_single_word_modifier(prev, &options.dampeners) {
            valence -= signed_modifier(valence, BOOSTER_AMOUNT * decay);
        }
        if options.negations.contains(prev) || prev.ends_with("n't") {
            valence *= NEGATION_SCALAR;
        }
    }
    if let Some(amount) = phrase_modifier_ending_at(idx, lower_tokens, &options.boosters) {
        valence += signed_modifier(valence, amount);
    }
    if let Some(amount) = phrase_modifier_ending_at(idx, lower_tokens, &options.dampeners) {
        valence -= signed_modifier(valence, amount);
    }
    valence
}

fn signed_modifier(valence: f64, amount: f64) -> f64 {
    if valence < 0.0 {
        -amount
    } else {
        amount
    }
}

fn phrase_modifier_ending_at(
    idx: usize,
    lower_tokens: &[String],
    phrases: &[Vec<String>],
) -> Option<f64> {
    let mut best_len = 0usize;
    for phrase in phrases {
        if phrase.len() <= 1 || phrase.len() > idx || phrase.len() <= best_len {
            continue;
        }
        let start = idx - phrase.len();
        if lower_tokens[start..idx]
            .iter()
            .map(String::as_str)
            .eq(phrase.iter().map(String::as_str))
        {
            best_len = phrase.len();
        }
    }
    (best_len > 0).then_some(BOOSTER_AMOUNT * best_len as f64)
}

fn is_single_word_modifier(word: &str, phrases: &[Vec<String>]) -> bool {
    phrases
        .iter()
        .any(|phrase| phrase.len() == 1 && phrase[0] == word)
}

fn punctuation_emphasis(tokens: &[String], sum: f64) -> f64 {
    if sum == 0.0 {
        return 0.0;
    }
    let exclamations = tokens
        .iter()
        .map(|token| token.chars().filter(|ch| *ch == '!').count())
        .sum::<usize>()
        .min(4) as f64;
    let question_marks = tokens
        .iter()
        .map(|token| token.chars().filter(|ch| *ch == '?').count())
        .sum::<usize>();
    let mut emphasis = exclamations * 0.292;
    emphasis += match question_marks {
        0 | 1 => 0.0,
        2 | 3 => question_marks as f64 * 0.18,
        _ => 0.96,
    };
    if sum < 0.0 {
        -emphasis
    } else {
        emphasis
    }
}

fn normalize_score(sum: f64) -> f64 {
    if sum == 0.0 {
        0.0
    } else {
        sum / (sum.mul_add(sum, NORMALIZATION_ALPHA)).sqrt()
    }
}

fn is_neutral_counted_token(token: &str) -> bool {
    matches!(
        document_token_type(token),
        crate::builtins::strings::text_analytics::documents::DocumentTokenType::Letters
            | crate::builtins::strings::text_analytics::documents::DocumentTokenType::Digits
            | crate::builtins::strings::text_analytics::documents::DocumentTokenType::Hashtag
            | crate::builtins::strings::text_analytics::documents::DocumentTokenType::AtMention
            | crate::builtins::strings::text_analytics::documents::DocumentTokenType::Other
    )
}

fn is_all_caps_word(token: &str) -> bool {
    let mut saw_alpha = false;
    let mut saw_lower = false;
    for ch in token.chars().filter(|ch| ch.is_alphabetic()) {
        saw_alpha = true;
        if ch.is_lowercase() {
            saw_lower = true;
            break;
        }
    }
    saw_alpha && !saw_lower
}

fn is_mixed_or_lower_word(token: &str) -> bool {
    token
        .chars()
        .any(|ch| ch.is_alphabetic() && ch.is_lowercase())
}

fn sentiment_lexicon_from_table(value: &Value) -> BuiltinResult<HashMap<String, f64>> {
    let object = match value {
        Value::Object(object) if object.is_class(TABLE_CLASS) => object,
        Value::Object(object) => {
            return Err(sentiment_error(format!(
                "vaderSentimentScores: SentimentLexicon must be a table, got {}",
                object.class_name
            )));
        }
        other => {
            return Err(sentiment_error(format!(
                "vaderSentimentScores: SentimentLexicon must be a table, got {other:?}"
            )));
        }
    };
    let variables = table_variables(object)
        .map_err(|err| sentiment_error(format!("vaderSentimentScores: {err}")))?;
    let token_column = variables.fields.get("Token").ok_or_else(|| {
        sentiment_error("vaderSentimentScores: SentimentLexicon missing Token column")
    })?;
    let score_column = variables.fields.get("SentimentScore").ok_or_else(|| {
        sentiment_error("vaderSentimentScores: SentimentLexicon missing SentimentScore column")
    })?;
    let tokens = words_from_word_vector(token_column, "vaderSentimentScores")?;
    let scores = numeric_vector(score_column, "SentimentScore")?;
    if tokens.len() != scores.len() {
        return Err(sentiment_error(
            "vaderSentimentScores: SentimentLexicon Token and SentimentScore columns must have equal height",
        ));
    }
    let mut lexicon = HashMap::with_capacity(tokens.len());
    for (token, score) in tokens.into_iter().zip(scores) {
        let token = token.trim().to_string();
        if token.is_empty() || token != token.to_ascii_lowercase() {
            return Err(sentiment_error(
                "vaderSentimentScores: SentimentLexicon tokens must be nonempty lowercase strings",
            ));
        }
        if !score.is_finite() || !(-4.0..=4.0).contains(&score) {
            return Err(sentiment_error(
                "vaderSentimentScores: SentimentScore values must be finite scalars in [-4, 4]",
            ));
        }
        lexicon.insert(token, score);
    }
    Ok(lexicon)
}

fn numeric_vector(value: &Value, column_name: &str) -> BuiltinResult<Vec<f64>> {
    match value {
        Value::Num(value) => Ok(vec![*value]),
        Value::Int(value) => Ok(vec![int_value_to_f64(value)]),
        Value::Tensor(tensor) => Ok(tensor.data.clone()),
        Value::Cell(cell) => cell
            .data
            .iter()
            .map(|item| numeric_scalar(item, column_name))
            .collect(),
        other => Err(sentiment_error(format!(
            "vaderSentimentScores: {column_name} must be numeric, got {other:?}"
        ))),
    }
}

fn numeric_scalar(value: &Value, column_name: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) => Ok(*value),
        Value::Int(value) => Ok(int_value_to_f64(value)),
        Value::Tensor(tensor) if tensor.data.len() == 1 => {
            Ok(tensor_utils::tensor_value_f64(tensor, 0))
        }
        other => Err(sentiment_error(format!(
            "vaderSentimentScores: {column_name} entries must be numeric scalars, got {other:?}"
        ))),
    }
}

fn modifier_ngrams_from_value(value: &Value, option_name: &str) -> BuiltinResult<Vec<Vec<String>>> {
    let mut phrases = match value {
        Value::String(_) | Value::CharArray(_) | Value::Cell(_) => {
            words_from_word_vector(value, "vaderSentimentScores")?
                .into_iter()
                .map(|word| vec![word])
                .collect()
        }
        Value::StringArray(array) => phrases_from_string_array(array),
        other => {
            return Err(sentiment_error(format!(
                "vaderSentimentScores: {option_name} must be a string array, character array, or cell array, got {other:?}"
            )));
        }
    };
    for phrase in &mut phrases {
        for token in phrase.iter_mut() {
            *token = token.trim().to_ascii_lowercase();
        }
        phrase.retain(|token| !token.is_empty());
    }
    phrases.retain(|phrase| !phrase.is_empty());
    if phrases.is_empty() {
        return Err(sentiment_error(format!(
            "vaderSentimentScores: {option_name} must contain at least one word or n-gram"
        )));
    }
    Ok(phrases)
}

fn phrases_from_string_array(array: &StringArray) -> Vec<Vec<String>> {
    if array.data.is_empty() {
        return Vec::new();
    }
    if array.rows <= 1 || array.cols <= 1 {
        return array.data.iter().map(|word| vec![word.clone()]).collect();
    }
    let mut phrases = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let mut phrase = Vec::new();
        for col in 0..array.cols {
            let idx = row + col * array.rows;
            if let Some(token) = array.data.get(idx) {
                phrase.push(token.clone());
            }
        }
        phrases.push(phrase);
    }
    phrases
}

fn phrase(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| word.to_ascii_lowercase())
        .collect()
}

fn tensor_value(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape)
        .map(Value::Tensor)
        .map_err(|err| sentiment_error(format!("vaderSentimentScores: {err}")))
}

fn sentiment_error(message: impl Into<String>) -> crate::RuntimeError {
    text_analytics_error("vaderSentimentScores", message)
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

static DEFAULT_LEXICON: Lazy<HashMap<String, f64>> = Lazy::new(|| {
    [
        ("abandon", -1.9),
        ("abandoned", -2.0),
        ("abuse", -3.2),
        ("accused", -1.6),
        ("amazing", 2.8),
        ("awesome", 3.1),
        ("awful", -2.9),
        ("bad", -2.5),
        ("best", 3.2),
        ("better", 2.0),
        ("boring", -1.3),
        ("creative", 2.0),
        ("delicious", 2.7),
        ("disappointing", -2.4),
        ("efficiency", 1.8),
        ("excellent", 3.0),
        ("fail", -2.5),
        ("failed", -2.4),
        ("failure", -2.3),
        ("fantastic", 3.2),
        ("good", 1.9),
        ("great", 3.1),
        ("growth", 1.7),
        ("happy", 2.7),
        ("hate", -2.7),
        ("horrible", -2.9),
        ("improved", 2.1),
        ("innovative", 2.2),
        ("joy", 2.5),
        ("like", 1.5),
        ("love", 3.2),
        ("misleading", -2.1),
        ("negative", -1.4),
        ("poor", -2.1),
        ("positive", 1.8),
        ("risk", -1.5),
        ("sad", -2.1),
        ("strong", 1.6),
        ("strengthen", 2.1),
        ("terrible", -3.0),
        ("weak", -1.8),
        ("worst", -3.1),
        ("wrong", -2.1),
    ]
    .into_iter()
    .map(|(token, score)| (token.to_string(), score))
    .collect()
});

static DEFAULT_BOOSTERS: &[&str] = &[
    "absolutely",
    "amazingly",
    "awfully",
    "completely",
    "considerably",
    "decidedly",
    "deeply",
    "especially",
    "exceptionally",
    "extremely",
    "highly",
    "incredibly",
    "majorly",
    "really",
    "remarkably",
    "so",
    "substantially",
    "thoroughly",
    "totally",
    "utterly",
    "very",
];

static DEFAULT_DAMPENERS: &[&str] = &[
    "almost",
    "barely",
    "hardly",
    "kind of",
    "kinda",
    "less",
    "little",
    "marginally",
    "partly",
    "scarcely",
    "slightly",
    "somewhat",
    "sort of",
];

static DEFAULT_NEGATIONS: &[&str] = &[
    "aint", "aren't", "cannot", "can't", "couldn't", "didn't", "doesn't", "don't", "hardly",
    "isn't", "lack", "lacks", "neither", "never", "no", "none", "nor", "not", "nothing", "nowhere",
    "rarely", "scarcely", "wasn't", "weren't", "without", "won't", "wouldn't",
];

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{CellArray, IntegerStorage, StructValue};

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        tensor.data.fill(f64::NAN);
        Value::Tensor(tensor)
    }

    fn run(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(vader_sentiment_scores_builtin(args))
    }

    #[test]
    fn numeric_scalar_reads_typed_integer_storage_exactly() {
        assert_eq!(
            numeric_scalar(
                &poisoned_integer_scalar(IntegerStorage::I16(vec![-3])),
                "Positive"
            )
            .expect("numeric"),
            -3.0
        );
    }

    fn documents(docs: Vec<Vec<&str>>) -> Value {
        let values = docs
            .iter()
            .map(|doc| {
                Value::StringArray(
                    StringArray::new(
                        doc.iter().map(|token| (*token).to_string()).collect(),
                        vec![1, doc.len()],
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        object.properties.insert(
            "Documents".to_string(),
            Value::Cell(CellArray::new(values, docs.len(), 1).unwrap()),
        );
        object
            .properties
            .insert("NumDocuments".to_string(), Value::Num(docs.len() as f64));
        object.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![docs.len() as f64, 1.0], vec![1, 2]).unwrap()),
        );
        object
            .properties
            .insert("Language".to_string(), Value::String("en".to_string()));
        Value::Object(object)
    }

    fn compound(value: Value) -> Vec<f64> {
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        tensor.data
    }

    fn outputs(value: Value) -> Vec<Value> {
        let Value::OutputList(values) = value else {
            panic!("expected output list");
        };
        values
    }

    #[test]
    fn vader_scores_positive_negative_and_neutral_documents() {
        let out = compound(
            run(vec![documents(vec![
                vec!["The", "book", "was", "VERY", "good", "!", "!", "!", "!"],
                vec!["The", "book", "was", "not", "very", "good", "."],
                vec!["plain", "reference", "text"],
            ])])
            .unwrap(),
        );
        assert_eq!(out.len(), 3);
        assert!(out[0] > 0.60, "positive score was {}", out[0]);
        assert!(out[1] < -0.20, "negated score was {}", out[1]);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn vader_returns_four_requested_score_vectors() {
        let _guard = crate::output_count::push_output_count(Some(4));
        let values = outputs(run(vec![documents(vec![vec!["good"], vec!["bad"]])]).unwrap());
        assert_eq!(values.len(), 4);
        let compound_scores = compound(values[0].clone());
        let positive_scores = compound(values[1].clone());
        let negative_scores = compound(values[2].clone());
        let neutral_scores = compound(values[3].clone());
        assert!(compound_scores[0] > 0.0);
        assert!(compound_scores[1] < 0.0);
        assert!(positive_scores[0] > negative_scores[0]);
        assert!(negative_scores[1] > positive_scores[1]);
        assert_eq!(neutral_scores, vec![0.0, 0.0]);
    }

    #[test]
    fn vader_accepts_custom_sentiment_lexicon_table() {
        let table = table_value(
            vec!["Token".to_string(), "SentimentScore".to_string()],
            vec![
                Value::StringArray(
                    StringArray::new(vec!["innovative".into(), "risk".into()], vec![2, 1]).unwrap(),
                ),
                Value::Tensor(Tensor::new(vec![4.0, -3.0], vec![2, 1]).unwrap()),
            ],
        );
        let out = compound(
            run(vec![
                documents(vec![vec!["innovative"], vec!["risk"]]),
                Value::String("SentimentLexicon".to_string()),
                table,
            ])
            .unwrap(),
        );
        assert!(out[0] > 0.70);
        assert!(out[1] < -0.60);
    }

    #[test]
    fn vader_accepts_custom_boosters_dampeners_and_negations() {
        let out = compound(
            run(vec![
                documents(vec![
                    vec!["exceptionally", "good"],
                    vec!["kinda", "good"],
                    vec!["never", "good"],
                ]),
                Value::String("Boosters".to_string()),
                Value::StringArray(
                    StringArray::new(vec!["exceptionally".into()], vec![1, 1]).unwrap(),
                ),
                Value::String("Dampeners".to_string()),
                Value::StringArray(StringArray::new(vec!["kinda".into()], vec![1, 1]).unwrap()),
                Value::String("Negations".to_string()),
                Value::StringArray(StringArray::new(vec!["never".into()], vec![1, 1]).unwrap()),
            ])
            .unwrap(),
        );
        assert!(out[0] > out[1]);
        assert!(out[2] < 0.0);
    }

    #[test]
    fn vader_accepts_modifier_ngram_rows() {
        let out = compound(
            run(vec![
                documents(vec![vec!["kind", "of", "good"], vec!["good"]]),
                Value::String("Dampeners".to_string()),
                Value::StringArray(
                    StringArray::new(
                        vec!["kind".into(), "unused".into(), "of".into(), "".into()],
                        vec![2, 2],
                    )
                    .unwrap(),
                ),
            ])
            .unwrap(),
        );
        assert!(out[0] < out[1]);
    }

    #[test]
    fn vader_rejects_invalid_custom_lexicon_scores() {
        let table = table_value(
            vec!["Token".to_string(), "SentimentScore".to_string()],
            vec![
                Value::StringArray(StringArray::new(vec!["strong".into()], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![5.0], vec![1, 1]).unwrap()),
            ],
        );
        let err = run(vec![
            documents(vec![vec!["strong"]]),
            Value::String("SentimentLexicon".to_string()),
            table,
        ])
        .unwrap_err();
        assert!(err.to_string().contains("[-4, 4]"));
    }

    #[test]
    fn vader_rejects_empty_custom_negations() {
        let err = run(vec![
            documents(vec![vec!["good"]]),
            Value::String("Negations".to_string()),
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("Negations"));
    }

    fn table_value(names: Vec<String>, columns: Vec<Value>) -> Value {
        let mut variables = StructValue::new();
        for (name, column) in names.into_iter().zip(columns) {
            variables.insert(name, column);
        }
        let mut object = ObjectInstance::new(TABLE_CLASS.to_string());
        object
            .properties
            .insert("__table_variables".to_string(), Value::Struct(variables));
        Value::Object(object)
    }
}
