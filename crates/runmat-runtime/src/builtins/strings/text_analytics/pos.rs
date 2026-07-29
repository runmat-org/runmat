//! Part-of-speech detail helpers for Text Analytics tokenized documents.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, ObjectInstance, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::tensor as tensor_utils;
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::details::add_sentence_details_builtin;
use crate::builtins::strings::text_analytics::documents::{
    document_token_type_with_options, documents_from_object, options_from_document_object,
    replace_tokenized_document_documents, text_analytics_error, tokenized_document_language,
    DocumentTokenType, TOKENIZED_DOCUMENT_CLASS,
};
use crate::{gather_if_needed_async, BuiltinResult};

pub(in crate::builtins::strings::text_analytics) const POS_DETAILS_PROPERTY: &str =
    "PartOfSpeechDetails";

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
        description: "Name-value options: RetokenizeMethod, Abbreviations, DiscardKnownValues.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEXT_ANALYTICS.ADD_PART_OF_SPEECH_DETAILS.INVALID_INPUT",
    identifier: Some("RunMat:addPartOfSpeechDetails:InvalidInput"),
    when: "Input is not a supported tokenizedDocument object or option form.",
    message: "addPartOfSpeechDetails: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const ADD_PART_OF_SPEECH_DETAILS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addPartOfSpeechDetails(documents)",
            inputs: &IN_DOCUMENTS,
            outputs: &OUT_DOCUMENTS,
        },
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = addPartOfSpeechDetails(documents,Name,Value)",
            inputs: &IN_DOCUMENTS_REST,
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
    name = "addPartOfSpeechDetails",
    category = "strings/text_analytics",
    summary = "Add part-of-speech details to tokenizedDocument objects.",
    keywords = "addPartOfSpeechDetails,text analytics,tokenizedDocument,part of speech,pos",
    accel = "sink",
    type_resolver(any_type),
    descriptor(
        crate::builtins::strings::text_analytics::pos::ADD_PART_OF_SPEECH_DETAILS_DESCRIPTOR
    ),
    builtin_path = "crate::builtins::strings::text_analytics::pos"
)]
pub(in crate::builtins::strings::text_analytics) async fn add_part_of_speech_details_builtin(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let gathered = gather_args(args).await?;
    let (documents, options) = parse_args(gathered)?;
    let mut object = tokenized_document_object(documents)?;
    let language = PosLanguage::from_document(&tokenized_document_language(&object))?;

    if options.retokenize_method == RetokenizeMethod::PartOfSpeech {
        let documents = documents_from_object(&object, "addPartOfSpeechDetails")?;
        let retokenized = documents
            .iter()
            .map(|doc| retokenize_for_pos(doc))
            .collect::<Vec<_>>();
        if retokenized != documents {
            replace_tokenized_document_documents(
                &mut object,
                retokenized,
                "addPartOfSpeechDetails",
            )?;
            clear_token_aligned_details(&mut object);
        }
    }

    if !object.properties.contains_key("SentenceNumbers") {
        object = add_sentence_details_for_pos(object, &options).await?;
    }

    let documents = documents_from_object(&object, "addPartOfSpeechDetails")?;
    let stored = if options.discard_known_values {
        None
    } else {
        part_of_speech_details_from_object(&object, "addPartOfSpeechDetails")?
    };
    let document_options = options_from_document_object(&object);
    let pos = if options.discard_known_values {
        part_of_speech_details_cell(&documents, language, &document_options)?
    } else {
        part_of_speech_details_cell_preserving_known(
            &documents,
            stored.as_deref(),
            language,
            &document_options,
        )?
    };
    object
        .properties
        .insert(POS_DETAILS_PROPERTY.to_string(), pos);
    Ok(Value::Object(object))
}

async fn gather_args(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(gather_if_needed_async(&arg).await.map_err(|err| {
            text_analytics_error(
                "addPartOfSpeechDetails",
                format!("addPartOfSpeechDetails: failed to gather input: {err}"),
            )
        })?);
    }
    Ok(out)
}

#[derive(Clone, Debug)]
struct AddPosOptions {
    discard_known_values: bool,
    retokenize_method: RetokenizeMethod,
    abbreviations: Option<Value>,
}

impl Default for AddPosOptions {
    fn default() -> Self {
        Self {
            discard_known_values: false,
            retokenize_method: RetokenizeMethod::PartOfSpeech,
            abbreviations: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetokenizeMethod {
    PartOfSpeech,
    None,
}

impl RetokenizeMethod {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "part-of-speech" | "partofspeech" => Ok(Self::PartOfSpeech),
            "none" => Ok(Self::None),
            other => Err(text_analytics_error(
                "addPartOfSpeechDetails",
                format!("addPartOfSpeechDetails: unsupported RetokenizeMethod '{other}'"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum PosLanguage {
    English,
    Japanese,
    German,
    Korean,
}

impl PosLanguage {
    fn from_document(language: &str) -> BuiltinResult<Self> {
        Self::from_document_for(language, "addPartOfSpeechDetails")
    }

    pub(in crate::builtins::strings::text_analytics) fn from_document_for(
        language: &str,
        fn_name: &str,
    ) -> BuiltinResult<Self> {
        match language.trim().to_ascii_lowercase().as_str() {
            "en" => Ok(Self::English),
            "ja" => Ok(Self::Japanese),
            "de" => Ok(Self::German),
            "ko" => Ok(Self::Korean),
            other => Err(text_analytics_error(
                fn_name,
                format!("{fn_name}: unsupported document language '{other}'"),
            )),
        }
    }
}

fn parse_args(args: Vec<Value>) -> BuiltinResult<(Value, AddPosOptions)> {
    if args.is_empty() {
        return Err(text_analytics_error(
            "addPartOfSpeechDetails",
            "addPartOfSpeechDetails: expected tokenizedDocument input",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(text_analytics_error(
            "addPartOfSpeechDetails",
            "addPartOfSpeechDetails: name-value options must appear in pairs",
        ));
    }
    let mut options = AddPosOptions::default();
    let mut idx = 1usize;
    while idx < args.len() {
        let name = scalar_text(&args[idx], "addPartOfSpeechDetails")
            .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err.to_string()))?;
        if name.eq_ignore_ascii_case("DiscardKnownValues") {
            options.discard_known_values = logical_scalar(&args[idx + 1])?;
        } else if name.eq_ignore_ascii_case("RetokenizeMethod") {
            let value = scalar_text(&args[idx + 1], "addPartOfSpeechDetails")
                .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err.to_string()))?;
            options.retokenize_method = RetokenizeMethod::parse(&value)?;
        } else if name.eq_ignore_ascii_case("Abbreviations") {
            options.abbreviations = Some(args[idx + 1].clone());
        } else {
            return Err(text_analytics_error(
                "addPartOfSpeechDetails",
                format!("addPartOfSpeechDetails: unsupported option '{name}'"),
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
            "addPartOfSpeechDetails",
            format!(
                "addPartOfSpeechDetails: expected tokenizedDocument object, got {}",
                object.class_name
            ),
        )),
        other => Err(text_analytics_error(
            "addPartOfSpeechDetails",
            format!("addPartOfSpeechDetails: expected tokenizedDocument object, got {other:?}"),
        )),
    }
}

fn clear_token_aligned_details(object: &mut ObjectInstance) {
    for property in [
        "TypeDetails",
        "SentenceNumbers",
        "LemmaDetails",
        POS_DETAILS_PROPERTY,
        "EntityDetails",
        "HeadDetails",
        "DependencyDetails",
    ] {
        object.properties.remove(property);
    }
}

async fn add_sentence_details_for_pos(
    object: ObjectInstance,
    options: &AddPosOptions,
) -> BuiltinResult<ObjectInstance> {
    let mut args = vec![Value::Object(object)];
    if let Some(abbreviations) = &options.abbreviations {
        args.push(Value::String("Abbreviations".to_string()));
        args.push(abbreviations.clone());
    }
    let Value::Object(object) = add_sentence_details_builtin(args).await? else {
        return Err(text_analytics_error(
            "addPartOfSpeechDetails",
            "addPartOfSpeechDetails: addSentenceDetails did not return tokenizedDocument",
        ));
    };
    Ok(object)
}

fn part_of_speech_details_cell(
    documents: &[Vec<String>],
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .map(|doc| {
            let tags = doc
                .iter()
                .map(|token| part_of_speech_for_token(token, language, options).to_string())
                .collect::<Vec<_>>();
            StringArray::new(tags, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err))?,
    ))
}

fn part_of_speech_details_cell_preserving_known(
    documents: &[Vec<String>],
    stored: Option<&[Vec<String>]>,
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> BuiltinResult<Value> {
    let values = documents
        .iter()
        .enumerate()
        .map(|(doc_idx, doc)| {
            let tags = doc
                .iter()
                .enumerate()
                .map(|(token_idx, token)| {
                    stored
                        .and_then(|values| values.get(doc_idx))
                        .and_then(|values| values.get(token_idx))
                        .filter(|tag| is_known_part_of_speech(tag))
                        .cloned()
                        .unwrap_or_else(|| {
                            part_of_speech_for_token(token, language, options).to_string()
                        })
                })
                .collect::<Vec<_>>();
            StringArray::new(tags, vec![1, doc.len()])
                .map(Value::StringArray)
                .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err))
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(Value::Cell(
        CellArray::new(values, documents.len(), 1)
            .map_err(|err| text_analytics_error("addPartOfSpeechDetails", err))?,
    ))
}

pub(in crate::builtins::strings::text_analytics) fn part_of_speech_for_token(
    token: &str,
    language: PosLanguage,
    options: &crate::builtins::strings::text_analytics::documents::DocumentOptions,
) -> &'static str {
    match document_token_type_with_options(token, options) {
        DocumentTokenType::Punctuation => "punctuation",
        DocumentTokenType::Digits => "numeral",
        DocumentTokenType::WebAddress
        | DocumentTokenType::EmailAddress
        | DocumentTokenType::Hashtag
        | DocumentTokenType::AtMention
        | DocumentTokenType::Emoticon
        | DocumentTokenType::Emoji
        | DocumentTokenType::Other => "other",
        DocumentTokenType::Letters => match language {
            PosLanguage::English => english_part_of_speech(token),
            PosLanguage::German => german_part_of_speech(token),
            PosLanguage::Japanese | PosLanguage::Korean => "noun",
        },
    }
}

fn english_part_of_speech(token: &str) -> &'static str {
    let lower = token.to_ascii_lowercase();
    let word = lower.trim_matches('\'');
    if word.is_empty() {
        return "other";
    }
    if matches!(
        word,
        "i" | "me"
            | "my"
            | "mine"
            | "myself"
            | "we"
            | "us"
            | "our"
            | "ours"
            | "ourselves"
            | "you"
            | "your"
            | "yours"
            | "yourself"
            | "yourselves"
            | "he"
            | "him"
            | "his"
            | "himself"
            | "she"
            | "her"
            | "hers"
            | "herself"
            | "it"
            | "its"
            | "itself"
            | "they"
            | "them"
            | "their"
            | "theirs"
            | "themselves"
            | "who"
            | "whom"
            | "whose"
            | "which"
            | "that"
            | "this"
            | "these"
            | "those"
    ) {
        return "pronoun";
    }
    if matches!(
        word,
        "a" | "an" | "the" | "some" | "any" | "each" | "every" | "no"
    ) {
        return "determiner";
    }
    if matches!(word, "and" | "or" | "but" | "nor" | "yet" | "so" | "for") {
        return "coord-conjunction";
    }
    if matches!(
        word,
        "because" | "although" | "while" | "if" | "when" | "since" | "unless"
    ) {
        return "subord-conjunction";
    }
    if matches!(
        word,
        "in" | "on"
            | "at"
            | "by"
            | "from"
            | "with"
            | "without"
            | "under"
            | "over"
            | "into"
            | "onto"
            | "through"
            | "during"
            | "before"
            | "after"
            | "of"
            | "as"
            | "than"
    ) {
        return "adposition";
    }
    if matches!(word, "to" | "n't" | "not") {
        return "particle";
    }
    if matches!(
        word,
        "am" | "are"
            | "is"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "have"
            | "has"
            | "had"
            | "do"
            | "does"
            | "did"
            | "can"
            | "could"
            | "may"
            | "might"
            | "must"
            | "shall"
            | "should"
            | "will"
            | "would"
            | "'m"
            | "'re"
            | "'ve"
            | "'ll"
            | "'d"
            | "'s"
    ) {
        return "auxiliary-verb";
    }
    if matches!(
        word,
        "run"
            | "runs"
            | "ran"
            | "walk"
            | "walks"
            | "go"
            | "goes"
            | "went"
            | "make"
            | "makes"
            | "made"
            | "build"
            | "builds"
            | "built"
            | "read"
            | "write"
            | "writes"
            | "wrote"
            | "see"
            | "sees"
            | "saw"
            | "want"
            | "wants"
            | "chase"
            | "chases"
            | "chased"
            | "attend"
            | "attends"
            | "attended"
            | "going"
            | "sleep"
            | "sleeps"
            | "fly"
            | "flies"
            | "using"
            | "use"
            | "uses"
    ) || word.ends_with("ing")
        || word.ends_with("ed")
    {
        return "verb";
    }
    if word.ends_with("ly") {
        return "adverb";
    }
    if word.ends_with("ive")
        || word.ends_with("ous")
        || word.ends_with("ful")
        || word.ends_with("less")
        || word.ends_with("able")
        || word.ends_with("ible")
        || matches!(word, "good" | "bad" | "new" | "old" | "large" | "small")
    {
        return "adjective";
    }
    "noun"
}

fn german_part_of_speech(token: &str) -> &'static str {
    let lower = token.to_ascii_lowercase();
    let word = lower.trim_matches('\'');
    if word.is_empty() {
        return "other";
    }
    if matches!(
        word,
        "ich"
            | "mich"
            | "mir"
            | "du"
            | "dich"
            | "dir"
            | "er"
            | "ihn"
            | "ihm"
            | "sie"
            | "ihr"
            | "es"
            | "wir"
            | "uns"
            | "euch"
            | "mein"
            | "dein"
            | "sein"
            | "dieser"
            | "diese"
            | "dieses"
            | "wer"
            | "was"
    ) {
        return "pronoun";
    }
    if matches!(
        word,
        "der"
            | "die"
            | "das"
            | "den"
            | "dem"
            | "des"
            | "ein"
            | "eine"
            | "einen"
            | "einem"
            | "einer"
            | "eines"
            | "kein"
            | "keine"
            | "jeder"
            | "jede"
            | "jedes"
    ) {
        return "determiner";
    }
    if matches!(word, "und" | "oder" | "aber" | "denn") {
        return "coord-conjunction";
    }
    if matches!(word, "weil" | "wenn" | "dass" | "obwohl" | "während") {
        return "subord-conjunction";
    }
    if matches!(
        word,
        "in" | "auf"
            | "an"
            | "bei"
            | "von"
            | "mit"
            | "ohne"
            | "unter"
            | "über"
            | "durch"
            | "vor"
            | "nach"
            | "zu"
            | "aus"
            | "für"
            | "gegen"
            | "als"
    ) {
        return "adposition";
    }
    if word.ends_with("lich")
        || word.ends_with("ig")
        || word.ends_with("isch")
        || matches!(
            word,
            "gut" | "gute" | "guter" | "guten" | "neu" | "alte" | "groß" | "klein"
        )
    {
        return "adjective";
    }
    if token.chars().next().is_some_and(char::is_uppercase) {
        return "noun";
    }
    if word.ends_with("weise") || matches!(word, "heute" | "morgen" | "wie" | "gern") {
        return "adverb";
    }
    if matches!(
        word,
        "bin"
            | "bist"
            | "ist"
            | "sind"
            | "seid"
            | "war"
            | "waren"
            | "sein"
            | "gewesen"
            | "habe"
            | "hast"
            | "hat"
            | "haben"
            | "hatte"
            | "hatten"
            | "werde"
            | "wird"
            | "werden"
            | "wurde"
            | "kann"
            | "können"
            | "muss"
            | "müssen"
            | "soll"
            | "sollen"
    ) {
        return "auxiliary-verb";
    }
    if matches!(
        word,
        "geht"
            | "gehen"
            | "ging"
            | "laufen"
            | "läuft"
            | "machen"
            | "macht"
            | "sagen"
            | "sagt"
            | "kommt"
            | "kommen"
    ) || word.ends_with("en")
        || word.ends_with("st")
        || word.ends_with("te")
    {
        return "verb";
    }
    "noun"
}

fn retokenize_for_pos(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < tokens.len() {
        if let Some((merged, consumed)) = dotted_initialism_at(tokens, idx) {
            out.push(merged);
            idx += consumed;
            continue;
        }
        if tokens[idx] == "." && tokens.get(idx + 1).is_some_and(|token| token == ".") {
            let mut consumed = 1usize;
            while tokens.get(idx + consumed).is_some_and(|token| token == ".") {
                consumed += 1;
            }
            out.push(".".repeat(consumed));
            idx += consumed;
            continue;
        }
        if let Some(split) = split_pos_token(&tokens[idx]) {
            out.extend(split);
        } else {
            out.push(tokens[idx].clone());
        }
        idx += 1;
    }
    out
}

fn dotted_initialism_at(tokens: &[String], idx: usize) -> Option<(String, usize)> {
    let mut cursor = idx;
    let mut letters = Vec::new();
    while cursor + 1 < tokens.len()
        && is_single_letter(&tokens[cursor])
        && tokens[cursor + 1] == "."
    {
        letters.push(tokens[cursor].clone());
        cursor += 2;
    }
    if letters.len() < 2 {
        return None;
    }
    Some((format!("{}.", letters.join(".")), cursor - idx))
}

fn is_single_letter(token: &str) -> bool {
    token.chars().count() == 1 && token.chars().all(char::is_alphabetic)
}

fn split_pos_token(token: &str) -> Option<Vec<String>> {
    let lower = token.to_ascii_lowercase();
    match lower.as_str() {
        "can't" => Some(vec!["can".to_string(), "n't".to_string()]),
        "cannot" => Some(vec!["can".to_string(), "not".to_string()]),
        "won't" => Some(vec!["will".to_string(), "not".to_string()]),
        "wanna" => Some(vec!["want".to_string(), "to".to_string()]),
        "gonna" => Some(vec!["going".to_string(), "to".to_string()]),
        _ => {
            for suffix in ["n't", "'re", "'ve", "'ll", "'d", "'m", "'s"] {
                if lower.ends_with(suffix) && lower.len() > suffix.len() {
                    let stem = token[..token.len() - suffix.len()].to_string();
                    return Some(vec![stem, suffix.to_string()]);
                }
            }
            None
        }
    }
}

fn is_known_part_of_speech(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed != "unknown"
        && !crate::builtins::strings::common::is_missing_string(trimmed)
}

pub(in crate::builtins::strings::text_analytics) fn part_of_speech_details_from_object(
    object: &ObjectInstance,
    fn_name: &str,
) -> BuiltinResult<Option<Vec<Vec<String>>>> {
    let Some(value) = object.properties.get(POS_DETAILS_PROPERTY) else {
        return Ok(None);
    };
    let Value::Cell(cell) = value else {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid PartOfSpeechDetails property"),
        ));
    };
    if cell.cols != 1 {
        return Err(text_analytics_error(
            fn_name,
            format!("{fn_name}: tokenizedDocument object has invalid PartOfSpeechDetails shape"),
        ));
    }
    let mut out = Vec::with_capacity(cell.data.len());
    for item in &cell.data {
        let Value::StringArray(array) = item else {
            return Err(text_analytics_error(
                fn_name,
                format!(
                    "{fn_name}: tokenizedDocument object has invalid PartOfSpeechDetails entry"
                ),
            ));
        };
        if array.rows != 1 {
            return Err(text_analytics_error(
                fn_name,
                format!(
                    "{fn_name}: tokenizedDocument object has invalid PartOfSpeechDetails entry shape"
                ),
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
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return match value.try_to_u64() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(text_analytics_error("addPartOfSpeechDetails", format!("addPartOfSpeechDetails: logical scalar option must be true or false, got {value:?}"))),
                };
            }
            match tensor_utils::tensor_value_f64(tensor, 0) {
            0.0 => Ok(false),
            1.0 => Ok(true),
            other => Err(text_analytics_error(
                "addPartOfSpeechDetails",
                format!(
                    "addPartOfSpeechDetails: logical scalar option must be true or false, got {other}"
                ),
            )),
            }
        }
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        other => Err(text_analytics_error(
            "addPartOfSpeechDetails",
            format!(
                "addPartOfSpeechDetails: logical scalar option must be true or false, got {other:?}"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::strings::text_analytics::details::token_details_builtin;
    use crate::builtins::strings::text_analytics::documents::tokenized_document_builtin;
    use crate::builtins::table::{table_variable_names_from_object, table_variables};
    use runmat_builtins::{IntegerStorage, LogicalArray, Tensor};

    fn run_tokenized(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(tokenized_document_builtin(args))
    }

    fn run_add_pos(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(add_part_of_speech_details_builtin(args))
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

    fn poisoned_integer_scalar(storage: IntegerStorage) -> Value {
        let mut tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer tensor");
        tensor.data.clear();
        Value::Tensor(tensor)
    }

    #[test]
    fn logical_scalar_reads_typed_integer_storage_exactly() {
        assert!(
            logical_scalar(&poisoned_integer_scalar(IntegerStorage::U8(vec![1]))).expect("true")
        );
        assert!(
            !logical_scalar(&poisoned_integer_scalar(IntegerStorage::I16(vec![0]))).expect("false")
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_adds_sentence_and_pos_columns() {
        let docs =
            run_tokenized(vec![Value::String("The dogs are running.".into())]).expect("tokenized");
        let updated = run_add_pos(vec![docs]).expect("pos");
        let table = object(run_token_details(updated).expect("details"));

        assert_eq!(
            table_variable_names_from_object(&table).expect("names"),
            vec![
                "Token",
                "DocumentNumber",
                "SentenceNumber",
                "LineNumber",
                "Type",
                "Language",
                "PartOfSpeech"
            ]
        );
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec![
                "determiner",
                "noun",
                "auxiliary-verb",
                "verb",
                "punctuation"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_retokenizes_common_pos_forms() {
        let docs = run_tokenized(vec![Value::String(
            "I can't attend the U.S.A. event.".into(),
        )])
        .expect("tokenized");
        let updated = run_add_pos(vec![docs]).expect("pos");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "Token"),
            vec!["I", "can", "n't", "attend", "the", "U.S.A.", "event", "."]
        );
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec![
                "pronoun",
                "auxiliary-verb",
                "particle",
                "verb",
                "determiner",
                "other",
                "noun",
                "punctuation"
            ]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_preserves_existing_known_values_unless_discarding() {
        let docs = object(
            run_tokenized(vec![
                Value::StringArray(
                    StringArray::new(vec!["dogs".into(), "run".into()], vec![1, 2]).unwrap(),
                ),
                Value::String("TokenizeMethod".into()),
                Value::String("none".into()),
            ])
            .expect("tokenized"),
        );
        let mut stale = docs.clone();
        stale.properties.insert(
            "SentenceNumbers".to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::Tensor(
                        Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        stale.properties.insert(
            POS_DETAILS_PROPERTY.to_string(),
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

        let preserved = run_add_pos(vec![
            Value::Object(stale.clone()),
            Value::String("RetokenizeMethod".into()),
            Value::String("none".into()),
        ])
        .expect("preserve");
        let table = object(run_token_details(preserved).expect("details"));
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec!["custom", "verb"]
        );

        let recomputed = run_add_pos(vec![
            Value::Object(stale),
            Value::String("RetokenizeMethod".into()),
            Value::String("none".into()),
            Value::String("DiscardKnownValues".into()),
            Value::LogicalArray(LogicalArray::new(vec![1], vec![1, 1]).unwrap()),
        ])
        .expect("recompute");
        let table = object(run_token_details(recomputed).expect("details"));
        assert_eq!(string_column(&table, "PartOfSpeech"), vec!["noun", "verb"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_rejects_bad_options_and_languages() {
        let docs = run_tokenized(vec![Value::String("dogs run".into())]).expect("tokenized");
        let err = run_add_pos(vec![
            docs.clone(),
            Value::String("Unknown".into()),
            Value::Bool(true),
        ])
        .expect_err("expected bad option");
        assert!(err.to_string().contains("unsupported option"));

        let err = run_add_pos(vec![
            docs.clone(),
            Value::String("RetokenizeMethod".into()),
            Value::String("bogus".into()),
        ])
        .expect_err("expected bad retokenizer");
        assert!(err.to_string().contains("unsupported RetokenizeMethod"));

        let mut unsupported = object(docs);
        unsupported
            .properties
            .insert("Language".into(), Value::String("fr".into()));
        let err = run_add_pos(vec![Value::Object(unsupported)]).expect_err("expected bad language");
        assert!(err.to_string().contains("unsupported document language"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_supports_documented_tags_and_german() {
        assert_eq!(english_part_of_speech("and"), "coord-conjunction");
        assert_eq!(english_part_of_speech("because"), "subord-conjunction");

        let docs = object(run_tokenized(vec![Value::String("Guten Morgen.".into())]).unwrap());
        let mut german = docs;
        german
            .properties
            .insert("Language".into(), Value::String("de".into()));
        let updated = run_add_pos(vec![Value::Object(german)]).expect("german pos");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(
            string_column(&table, "PartOfSpeech"),
            vec!["adjective", "noun", "punctuation"]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn add_part_of_speech_details_discard_known_values_ignores_malformed_stored_tags() {
        let docs =
            object(run_tokenized(vec![Value::String("dogs run".into())]).expect("tokenized"));
        let mut malformed = docs.clone();
        malformed.properties.insert(
            POS_DETAILS_PROPERTY.to_string(),
            Value::String("bad".into()),
        );

        let updated = run_add_pos(vec![
            Value::Object(malformed),
            Value::String("DiscardKnownValues".into()),
            Value::Bool(true),
        ])
        .expect("discard malformed stored tags");
        let table = object(run_token_details(updated).expect("details"));
        assert_eq!(string_column(&table, "PartOfSpeech"), vec!["noun", "verb"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn token_details_rejects_malformed_part_of_speech_details() {
        let docs =
            object(run_tokenized(vec![Value::String("dogs run".into())]).expect("tokenized"));
        let mut malformed = docs.clone();
        malformed.properties.insert(
            POS_DETAILS_PROPERTY.to_string(),
            Value::Cell(
                CellArray::new(
                    vec![Value::StringArray(
                        StringArray::new(vec!["noun".into()], vec![1, 1]).unwrap(),
                    )],
                    1,
                    1,
                )
                .unwrap(),
            ),
        );
        let err = run_token_details(Value::Object(malformed)).expect_err("expected error");
        assert!(err.to_string().contains("PartOfSpeechDetails entry"));
    }
}
