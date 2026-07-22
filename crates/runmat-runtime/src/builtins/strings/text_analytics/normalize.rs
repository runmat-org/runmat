//! MATLAB-compatible standalone `normalizeWords` helper.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, ResolveContext, StringArray, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::builtins::strings::core::compat::scalar_text;
use crate::builtins::strings::text_analytics::documents::{
    tokenized_document_language, transform_tokenized_document, DocumentTokenType,
    TOKENIZED_DOCUMENT_CLASS,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "updatedWords",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Normalized words or tokenizedDocument object.",
}];

const IN_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Words to stem or lemmatize.",
}];

const IN_WORDS_REST: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "words",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Words to stem or lemmatize.",
    },
    BuiltinParamDescriptor {
        name: "NameValue",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name-value options: Language and Style.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NORMALIZEWORDS.INVALID_INPUT",
    identifier: Some("RunMat:normalizeWords:InvalidInput"),
    when: "Inputs are not a supported normalizeWords form.",
    message: "normalizeWords: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const NORMALIZE_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "updatedDocuments = normalizeWords(documents)",
            inputs: &IN_WORDS,
            outputs: &OUT_WORDS,
        },
        BuiltinSignatureDescriptor {
            label: "updatedWords = normalizeWords(words)",
            inputs: &IN_WORDS,
            outputs: &OUT_WORDS,
        },
        BuiltinSignatureDescriptor {
            label: "updatedWords = normalizeWords(words, Name, Value, ...)",
            inputs: &IN_WORDS_REST,
            outputs: &OUT_WORDS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn any_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::Unknown
}

fn normalize_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("normalizeWords")
        .with_identifier("RunMat:normalizeWords:InvalidInput")
        .build()
}

#[runtime_builtin(
    name = "normalizeWords",
    category = "strings/text_analytics",
    summary = "Stem or lemmatize standalone word arrays.",
    keywords = "normalizeWords,stem,lemma,text analytics,words",
    accel = "sink",
    type_resolver(any_type),
    descriptor(crate::builtins::strings::text_analytics::normalize::NORMALIZE_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::normalize"
)]
async fn normalize_words_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (words, options) = parse_args(args).await?;
    normalize_words_value(words, options)
}

async fn parse_args(args: Vec<Value>) -> BuiltinResult<(Value, NormalizeOptions)> {
    if args.is_empty() {
        return Err(normalize_error(
            "normalizeWords: expected words input and optional name-value pairs",
        ));
    }
    if !(args.len() - 1).is_multiple_of(2) {
        return Err(normalize_error(
            "normalizeWords: name-value options must appear in pairs",
        ));
    }

    let words = gather_if_needed_async(&args[0]).await.map_err(|err| {
        normalize_error(format!(
            "normalizeWords: failed to gather words input: {err}"
        ))
    })?;
    let mut options = NormalizeOptions::default();
    let mut idx = 1;
    while idx < args.len() {
        let name = gather_if_needed_async(&args[idx]).await.map_err(|err| {
            normalize_error(format!(
                "normalizeWords: failed to gather option name: {err}"
            ))
        })?;
        let value = gather_if_needed_async(&args[idx + 1])
            .await
            .map_err(|err| {
                normalize_error(format!(
                    "normalizeWords: failed to gather option value: {err}"
                ))
            })?;
        let name = scalar_text(&name, "normalizeWords")
            .map_err(|err| normalize_error(err.to_string()))?
            .to_ascii_lowercase();
        match name.as_str() {
            "language" => {
                let value = scalar_text(&value, "normalizeWords")
                    .map_err(|err| normalize_error(err.to_string()))?;
                options.language = Language::parse(&value)?;
                options.language_explicit = true;
            }
            "style" => {
                let value = scalar_text(&value, "normalizeWords")
                    .map_err(|err| normalize_error(err.to_string()))?;
                options.style = Style::parse(&value)?
            }
            _ => {
                return Err(normalize_error(format!(
                    "normalizeWords: unsupported option '{name}'"
                )));
            }
        }
        idx += 2;
    }
    Ok((words, options))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NormalizeOptions {
    language: Language,
    language_explicit: bool,
    style: Style,
}

impl Default for NormalizeOptions {
    fn default() -> Self {
        Self {
            language: Language::English,
            language_explicit: false,
            style: Style::Stem,
        }
    }
}

impl NormalizeOptions {
    fn validate_standalone(self) -> BuiltinResult<()> {
        if self.style == Style::Lemma && self.language == Language::German {
            return Err(normalize_error(
                "normalizeWords: Style 'lemma' for standalone words supports English only; use tokenizedDocument for Japanese or Korean lemmatization",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    English,
    German,
}

impl Language {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "en" => Ok(Self::English),
            "de" => Ok(Self::German),
            "ja" | "ko" => Err(normalize_error(
                "normalizeWords: standalone word input supports Language 'en' or 'de'; use tokenizedDocument for Japanese or Korean",
            )),
            other => Err(normalize_error(format!(
                "normalizeWords: Language must be 'en' or 'de', got '{other}'"
            ))),
        }
    }

    fn parse_document_language(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "en" => Ok(Self::English),
            "de" => Ok(Self::German),
            "ja" | "ko" => Err(normalize_error(
                "normalizeWords: Japanese and Korean tokenizedDocument normalization requires MeCab-compatible token details and remains tracked",
            )),
            other => Err(normalize_error(format!(
                "normalizeWords: unsupported tokenizedDocument language '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Stem,
    Lemma,
}

impl Style {
    fn parse(value: &str) -> BuiltinResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "stem" => Ok(Self::Stem),
            "lemma" => Ok(Self::Lemma),
            other => Err(normalize_error(format!(
                "normalizeWords: Style must be 'stem' or 'lemma', got '{other}'"
            ))),
        }
    }
}

fn normalize_words_value(value: Value, options: NormalizeOptions) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            options.validate_standalone()?;
            Ok(Value::String(normalize_word_or_missing(&text, options)?))
        }
        Value::StringArray(array) => {
            options.validate_standalone()?;
            let data = array
                .data
                .iter()
                .map(|text| normalize_word_or_missing(text, options))
                .collect::<BuiltinResult<Vec<_>>>()?;
            StringArray::new(data, array.shape)
                .map(Value::StringArray)
                .map_err(|err| normalize_error(format!("normalizeWords: {err}")))
        }
        Value::CharArray(array) => {
            options.validate_standalone()?;
            let rows = (0..array.rows)
                .map(|row| {
                    normalize_word_or_missing(
                        &char_row_to_string_slice(&array.data, array.cols, row),
                        options,
                    )
                })
                .collect::<BuiltinResult<Vec<_>>>()?;
            char_rows(rows)
        }
        Value::Cell(cell) => {
            options.validate_standalone()?;
            let shape = cell.shape.clone();
            let data = cell
                .data
                .into_iter()
                .map(|item| normalize_cell_item(item, options))
                .collect::<BuiltinResult<Vec<_>>>()?;
            make_cell_with_shape(data, shape).map_err(|err| normalize_error(err.to_string()))
        }
        Value::Object(object) if object.is_class(TOKENIZED_DOCUMENT_CLASS) => {
            normalize_tokenized_document(&object, options)
        }
        Value::Object(object) => Err(normalize_error(format!(
            "normalizeWords: expected tokenizedDocument object, got {}",
            object.class_name
        ))),
        other => Err(normalize_error(format!(
            "normalizeWords: expected string scalar/array, character vector/array, or cell text input, got {other:?}"
        ))),
    }
}

fn normalize_tokenized_document(
    object: &runmat_builtins::ObjectInstance,
    mut options: NormalizeOptions,
) -> BuiltinResult<Value> {
    if options.language_explicit {
        return Err(normalize_error(
            "normalizeWords: tokenizedDocument input uses document Language metadata; the Language option is only supported for standalone words",
        ));
    }
    options.language = Language::parse_document_language(&tokenized_document_language(object))?;
    transform_tokenized_document(object, "normalizeWords", |token, token_type| {
        if matches!(
            token_type,
            DocumentTokenType::Letters | DocumentTokenType::Other
        ) {
            Ok(Some(normalize_word(token, options)))
        } else {
            Ok(Some(token.to_string()))
        }
    })
}

fn normalize_cell_item(value: Value, options: NormalizeOptions) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(normalize_word_or_missing(&text, options)?)),
        Value::StringArray(array) if array.data.len() == 1 => Ok(Value::StringArray(
            StringArray::new(
                vec![normalize_word_or_missing(&array.data[0], options)?],
                array.shape,
            )
            .map_err(|err| normalize_error(format!("normalizeWords: {err}")))?,
        )),
        Value::CharArray(array) if array.rows <= 1 => {
            let text = if array.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&array.data, array.cols, 0)
            };
            Ok(Value::CharArray(CharArray::new_row(&normalize_word_or_missing(
                &text, options,
            )?)))
        }
        other => Err(normalize_error(format!(
            "normalizeWords: cell elements must be string scalars or character vectors, got {other:?}"
        ))),
    }
}

fn normalize_word_or_missing(text: &str, options: NormalizeOptions) -> BuiltinResult<String> {
    if is_missing_string(text) {
        return Ok(text.to_string());
    }
    let trimmed = text.trim();
    if trimmed.split_whitespace().nth(1).is_some() {
        return Err(normalize_error(format!(
            "normalizeWords: each input element must contain a single word, got '{trimmed}'"
        )));
    }
    Ok(normalize_word(trimmed, options))
}

fn normalize_word(word: &str, options: NormalizeOptions) -> String {
    if !word.chars().any(|ch| ch.is_alphabetic()) {
        return word.to_string();
    }
    match (options.language, options.style) {
        (Language::English, Style::Stem) => porter_stem(&word.to_ascii_lowercase()),
        (Language::English, Style::Lemma) => english_lemma(&word.to_ascii_lowercase()),
        (Language::German, Style::Stem) => german_stem(&word.to_lowercase()),
        (Language::German, Style::Lemma) => word.to_string(),
    }
}

fn char_rows(rows: Vec<String>) -> BuiltinResult<Value> {
    let row_count = rows.len();
    let cols = rows
        .iter()
        .map(|row| row.chars().count())
        .max()
        .unwrap_or(0);
    let mut data = Vec::with_capacity(row_count * cols);
    for row in rows {
        let mut chars = row.chars().collect::<Vec<_>>();
        chars.resize(cols, ' ');
        data.extend(chars);
    }
    CharArray::new(data, row_count, cols)
        .map(Value::CharArray)
        .map_err(|err| normalize_error(format!("normalizeWords: {err}")))
}

pub(in crate::builtins::strings::text_analytics) fn english_lemma(word: &str) -> String {
    match word {
        "am" | "are" | "is" | "was" | "were" | "be" | "been" | "being" => "be".to_string(),
        "has" | "had" | "having" => "have".to_string(),
        "does" | "did" | "done" | "doing" => "do".to_string(),
        "ran" => "run".to_string(),
        "children" => "child".to_string(),
        "men" => "man".to_string(),
        "women" => "woman".to_string(),
        "mice" => "mouse".to_string(),
        "geese" => "goose".to_string(),
        "teeth" => "tooth".to_string(),
        "feet" => "foot".to_string(),
        "better" | "best" => "good".to_string(),
        "worse" | "worst" => "bad".to_string(),
        _ => english_lemma_rules(word),
    }
}

fn english_lemma_rules(word: &str) -> String {
    if word.len() <= 3 {
        return word.to_string();
    }
    if let Some(base) = word.strip_suffix("ies") {
        if base.len() > 1 {
            return format!("{base}y");
        }
    }
    if let Some(base) = word.strip_suffix("ves") {
        if base.len() > 1 {
            return format!("{base}f");
        }
    }
    if let Some(base) = word.strip_suffix("ing") {
        if base.len() >= 3 {
            return undouble_final_consonant(base);
        }
    }
    if let Some(base) = word.strip_suffix("ed") {
        if base.len() >= 3 {
            return undouble_final_consonant(base);
        }
    }
    if let Some(base) = word.strip_suffix("es") {
        if base.len() >= 3 {
            return base.to_string();
        }
    }
    if let Some(base) = word.strip_suffix('s') {
        if base.len() >= 3 && !base.ends_with('s') {
            return base.to_string();
        }
    }
    word.to_string()
}

fn undouble_final_consonant(text: &str) -> String {
    let mut chars = text.chars().collect::<Vec<_>>();
    if chars.len() >= 2 {
        let last = chars[chars.len() - 1];
        let prev = chars[chars.len() - 2];
        if last == prev && is_consonant_char(last) && !matches!(last, 's' | 'z' | 'l') {
            chars.pop();
        }
    }
    chars.into_iter().collect()
}

fn german_stem(word: &str) -> String {
    let mut stem = word
        .replace('ä', "a")
        .replace('ö', "o")
        .replace('ü', "u")
        .replace('ß', "ss");
    if stem.len() <= 3 {
        return stem;
    }
    for suffix in [
        "heiten", "keit", "lich", "isch", "ern", "em", "er", "en", "es", "e", "s",
    ] {
        if stem.ends_with(suffix) && stem.len() > suffix.len() + 2 {
            let keep = stem.len() - suffix.len();
            stem.truncate(keep);
            break;
        }
    }
    stem
}

fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }
    let mut stem = word.to_string();
    porter_step_1a(&mut stem);
    porter_step_1b(&mut stem);
    porter_step_1c(&mut stem);
    porter_step_2(&mut stem);
    porter_step_3(&mut stem);
    porter_step_4(&mut stem);
    porter_step_5(&mut stem);
    stem
}

fn porter_step_1a(stem: &mut String) {
    if replace_suffix(stem, "sses", "ss") {
        return;
    }
    if replace_suffix(stem, "ies", "i") || stem.ends_with("ss") {
        return;
    }
    if stem.ends_with('s') {
        stem.pop();
    }
}

fn porter_step_1b(stem: &mut String) {
    if stem.ends_with("eed") {
        let base = &stem[..stem.len() - 3];
        if measure(base) > 0 {
            stem.truncate(stem.len() - 1);
        }
        return;
    }
    let mut changed = false;
    if stem.ends_with("ed") {
        let base = &stem[..stem.len() - 2];
        if contains_vowel(base) {
            stem.truncate(stem.len() - 2);
            changed = true;
        }
    } else if stem.ends_with("ing") {
        let base = &stem[..stem.len() - 3];
        if contains_vowel(base) {
            stem.truncate(stem.len() - 3);
            changed = true;
        }
    }
    if changed {
        if stem.ends_with("at") || stem.ends_with("bl") || stem.ends_with("iz") {
            stem.push('e');
        } else if ends_double_consonant(stem)
            && !matches!(stem.chars().last(), Some('l' | 's' | 'z'))
        {
            stem.pop();
        } else if measure(stem) == 1 && cvc(stem) {
            stem.push('e');
        }
    }
}

fn porter_step_1c(stem: &mut String) {
    if stem.ends_with('y') {
        let base = &stem[..stem.len() - 1];
        if contains_vowel(base) {
            stem.pop();
            stem.push('i');
        }
    }
}

fn porter_step_2(stem: &mut String) {
    for (suffix, replacement) in [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("logi", "log"),
    ] {
        if replace_suffix_if_measure(stem, suffix, replacement, 0) {
            return;
        }
    }
}

fn porter_step_3(stem: &mut String) {
    for (suffix, replacement) in [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ] {
        if replace_suffix_if_measure(stem, suffix, replacement, 0) {
            return;
        }
    }
}

fn porter_step_4(stem: &mut String) {
    for suffix in [
        "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
        "ive", "ize", "al", "er", "ic",
    ] {
        if stem.ends_with(suffix) {
            let base = &stem[..stem.len() - suffix.len()];
            if measure(base) > 1 {
                stem.truncate(base.len());
            }
            return;
        }
    }
    if stem.ends_with("ion") {
        let base = &stem[..stem.len() - 3];
        if measure(base) > 1 && matches!(base.chars().last(), Some('s' | 't')) {
            stem.truncate(base.len());
        }
    }
}

fn porter_step_5(stem: &mut String) {
    if stem.ends_with('e') {
        let base = &stem[..stem.len() - 1];
        let m = measure(base);
        if m > 1 || (m == 1 && !cvc(base)) {
            stem.truncate(base.len());
        }
    }
    if measure(stem) > 1 && stem.ends_with("ll") {
        stem.pop();
    }
}

fn replace_suffix(stem: &mut String, suffix: &str, replacement: &str) -> bool {
    if stem.ends_with(suffix) {
        let base_len = stem.len() - suffix.len();
        stem.truncate(base_len);
        stem.push_str(replacement);
        true
    } else {
        false
    }
}

fn replace_suffix_if_measure(
    stem: &mut String,
    suffix: &str,
    replacement: &str,
    min_measure: usize,
) -> bool {
    if stem.ends_with(suffix) {
        let base = &stem[..stem.len() - suffix.len()];
        if measure(base) > min_measure {
            let base_len = base.len();
            stem.truncate(base_len);
            stem.push_str(replacement);
        }
        true
    } else {
        false
    }
}

fn measure(word: &str) -> usize {
    let chars = word.chars().collect::<Vec<_>>();
    let mut count = 0;
    let mut prev_vowel = false;
    for idx in 0..chars.len() {
        let vowel = is_vowel(&chars, idx);
        if !vowel && prev_vowel {
            count += 1;
        }
        prev_vowel = vowel;
    }
    count
}

fn contains_vowel(word: &str) -> bool {
    let chars = word.chars().collect::<Vec<_>>();
    (0..chars.len()).any(|idx| is_vowel(&chars, idx))
}

fn is_vowel(chars: &[char], idx: usize) -> bool {
    match chars[idx] {
        'a' | 'e' | 'i' | 'o' | 'u' => true,
        'y' => idx > 0 && !is_vowel(chars, idx - 1),
        _ => false,
    }
}

fn is_consonant_char(ch: char) -> bool {
    ch.is_ascii_alphabetic() && !matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn ends_double_consonant(word: &str) -> bool {
    let chars = word.chars().collect::<Vec<_>>();
    chars.len() >= 2
        && chars[chars.len() - 1] == chars[chars.len() - 2]
        && is_consonant_char(chars[chars.len() - 1])
}

fn cvc(word: &str) -> bool {
    let chars = word.chars().collect::<Vec<_>>();
    if chars.len() < 3 {
        return false;
    }
    let len = chars.len();
    !is_vowel(&chars, len - 1)
        && is_vowel(&chars, len - 2)
        && !is_vowel(&chars, len - 3)
        && !matches!(chars[len - 1], 'w' | 'x' | 'y')
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{CellArray, ObjectInstance, Tensor};

    use crate::builtins::strings::text_analytics::documents::documents_from_object;

    fn run(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(normalize_words_builtin(args))
    }

    fn tokenized_document(documents: Vec<Vec<&str>>, language: &str) -> Value {
        let mut object = ObjectInstance::new(TOKENIZED_DOCUMENT_CLASS.to_string());
        let cells = documents
            .iter()
            .map(|doc| {
                StringArray::new(
                    doc.iter().map(|token| (*token).to_string()).collect(),
                    vec![1, doc.len()],
                )
                .map(Value::StringArray)
                .unwrap()
            })
            .collect::<Vec<_>>();
        object.properties.insert(
            "Documents".to_string(),
            Value::Cell(CellArray::new(cells, documents.len(), 1).unwrap()),
        );
        object.properties.insert(
            "Shape".to_string(),
            Value::Tensor(Tensor::new(vec![documents.len() as f64, 1.0], vec![1, 2]).unwrap()),
        );
        object.properties.insert(
            "TokenizeMethod".to_string(),
            Value::String("unicode".to_string()),
        );
        object
            .properties
            .insert("Language".to_string(), Value::String(language.to_string()));
        Value::Object(object)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn stems_english_string_array_and_preserves_shape() {
        let input = StringArray::new(
            vec![
                "strongly".to_string(),
                "worded".to_string(),
                "collections".to_string(),
                "words".to_string(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let out = run(vec![Value::StringArray(input)]).expect("normalize");
        let Value::StringArray(array) = out else {
            panic!("expected string array");
        };
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec!["strongli", "word", "collect", "word"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lemmatizes_english_char_vector() {
        let out = run(vec![
            Value::CharArray(CharArray::new_row("running")),
            Value::String("Style".to_string()),
            Value::String("lemma".to_string()),
        ])
        .expect("normalize");
        assert_eq!(out, Value::CharArray(CharArray::new_row("run")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn stems_german_word_array() {
        let input =
            StringArray::new(vec!["Morgen".to_string(), "guter".to_string()], vec![2, 1]).unwrap();
        let out = run(vec![
            Value::StringArray(input),
            Value::CharArray(CharArray::new_row("Language")),
            Value::CharArray(CharArray::new_row("de")),
        ])
        .expect("normalize");
        let Value::StringArray(array) = out else {
            panic!("expected string array");
        };
        assert_eq!(array.data, vec!["morg", "gut"]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn repads_multi_row_char_array_after_normalization() {
        let input = CharArray::new("runningworded ".chars().collect(), 2, "running".len()).unwrap();
        let out = run(vec![Value::CharArray(input)]).expect("normalize");
        let Value::CharArray(array) = out else {
            panic!("expected char array");
        };
        assert_eq!(array.rows, 2);
        assert_eq!(array.cols, 4);
        assert_eq!(
            array.data.into_iter().collect::<String>(),
            "run word".to_string()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn preserves_cell_element_types() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("running")),
                Value::String("children".to_string()),
            ],
            1,
            2,
        )
        .unwrap();
        let out = run(vec![
            Value::Cell(cell),
            Value::String("Style".to_string()),
            Value::String("lemma".to_string()),
        ])
        .expect("normalize");
        let Value::Cell(cell) = out else {
            panic!("expected cell");
        };
        assert_eq!(cell.shape, vec![1, 2]);
        assert_eq!(cell.data[0], Value::CharArray(CharArray::new_row("run")));
        assert_eq!(cell.data[1], Value::String("child".to_string()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn preserves_missing_strings() {
        let input = StringArray::new(vec!["<missing>".to_string()], vec![1, 1]).unwrap();
        let out = run(vec![Value::StringArray(input)]).expect("normalize");
        assert!(matches!(out, Value::StringArray(array) if array.data == vec!["<missing>"]));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn normalizes_tokenized_documents_and_preserves_complex_tokens() {
        let out = run(vec![tokenized_document(
            vec![vec!["a", "strongly", "worded", ".", "https://example.com"]],
            "en",
        )])
        .expect("normalize documents");
        let Value::Object(object) = out else {
            panic!("expected tokenizedDocument object");
        };
        assert_eq!(
            documents_from_object(&object, "test").unwrap(),
            vec![vec!["a", "strongli", "word", ".", "https://example.com",]]
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_language_override_for_tokenized_documents() {
        let err = run(vec![
            tokenized_document(vec![vec!["word"]], "en"),
            Value::String("Language".to_string()),
            Value::String("de".to_string()),
        ])
        .expect_err("expected Language option rejection");
        assert!(err.to_string().contains("document Language metadata"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_unsupported_language_for_standalone_words() {
        let err = run(vec![
            Value::String("word".to_string()),
            Value::String("Language".to_string()),
            Value::String("ja".to_string()),
        ])
        .expect_err("expected unsupported language");
        assert!(err.to_string().contains("tokenizedDocument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_german_lemma_for_standalone_words() {
        let err = run(vec![
            Value::String("wort".to_string()),
            Value::String("Language".to_string()),
            Value::String("de".to_string()),
            Value::String("Style".to_string()),
            Value::String("lemma".to_string()),
        ])
        .expect_err("expected unsupported German lemma");
        assert!(err.to_string().contains("English only"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn wraps_non_text_option_errors_with_normalize_identifier() {
        let err = run(vec![
            Value::String("word".to_string()),
            Value::Num(1.0),
            Value::String("stem".to_string()),
        ])
        .expect_err("expected option name error");
        assert!(err.to_string().contains("normalizeWords"));
        assert_eq!(err.identifier(), Some("RunMat:normalizeWords:InvalidInput"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn rejects_multi_word_elements() {
        let err = run(vec![Value::String("two words".to_string())])
            .expect_err("expected single word error");
        assert!(err.to_string().contains("single word"));
    }
}
