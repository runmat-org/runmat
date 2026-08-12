//! MATLAB-compatible `stopWords` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StringArray, Value};

use crate::builtins::strings::core::compat::scalar_text;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const OUT_WORDS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "words",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Stop word list.",
}];

const LANGUAGE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("Language"),
        description: "Language option name.",
    },
    BuiltinParamDescriptor {
        name: "language",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("en"),
        description: "Stop word language: en, ja, de, or ko.",
    },
];

const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STOPWORDS.INVALID_INPUT",
    identifier: Some("RunMat:stopWords:InvalidInput"),
    when: "Inputs are not a supported stopWords form.",
    message: "stopWords: invalid input",
};

const ERRORS: [BuiltinErrorDescriptor; 1] = [ERROR_INVALID_INPUT];

pub const STOP_WORDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "words = stopWords",
            inputs: &[],
            outputs: &OUT_WORDS,
        },
        BuiltinSignatureDescriptor {
            label: "words = stopWords('Language', language)",
            inputs: &LANGUAGE_INPUTS,
            outputs: &OUT_WORDS,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn string_array_type(_args: &[Type], _ctx: &ResolveContext) -> Type {
    Type::String
}

fn stopwords_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("stopWords");
    if let Some(identifier) = ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "stopWords",
    category = "strings/text_analytics",
    summary = "Return common stop words for supported Text Analytics languages.",
    keywords = "stopWords,stop words,text analytics,language,string",
    accel = "sink",
    type_resolver(string_array_type),
    descriptor(crate::builtins::strings::text_analytics::stopwords::STOP_WORDS_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::text_analytics::stopwords"
)]
async fn stop_words_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let language = parse_language_args(args).await?;
    let words = stop_words_for_language(language);
    let strings = words.iter().map(|word| (*word).to_string()).collect();
    StringArray::new(strings, vec![words.len(), 1])
        .map(Value::StringArray)
        .map_err(|err| stopwords_error(format!("stopWords: {err}")))
}

async fn parse_language_args(args: Vec<Value>) -> BuiltinResult<StopWordsLanguage> {
    match args.len() {
        0 => Ok(StopWordsLanguage::English),
        2 => {
            let name = gather_if_needed_async(&args[0]).await.map_err(|err| {
                stopwords_error(format!("stopWords: failed to gather option name: {err}"))
            })?;
            let value = gather_if_needed_async(&args[1]).await.map_err(|err| {
                stopwords_error(format!("stopWords: failed to gather language value: {err}"))
            })?;
            let option = scalar_text(&name, "stopWords")?.to_ascii_lowercase();
            if option != "language" {
                return Err(stopwords_error(format!(
                    "stopWords: unsupported option '{option}'"
                )));
            }
            parse_language(&scalar_text(&value, "stopWords")?)
        }
        _ => Err(stopwords_error(
            "stopWords: expected stopWords or stopWords('Language', language)",
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::builtins::strings::text_analytics) enum StopWordsLanguage {
    English,
    Japanese,
    German,
    Korean,
}

fn parse_language(language: &str) -> BuiltinResult<StopWordsLanguage> {
    match language.to_ascii_lowercase().as_str() {
        "en" => Ok(StopWordsLanguage::English),
        "ja" => Ok(StopWordsLanguage::Japanese),
        "de" => Ok(StopWordsLanguage::German),
        "ko" => Ok(StopWordsLanguage::Korean),
        other => Err(stopwords_error(format!(
            "stopWords: language must be 'en', 'ja', 'de', or 'ko', got '{other}'"
        ))),
    }
}

pub(in crate::builtins::strings::text_analytics) fn stop_words_for_language(
    language: StopWordsLanguage,
) -> &'static [&'static str] {
    match language {
        StopWordsLanguage::English => ENGLISH_STOP_WORDS,
        StopWordsLanguage::Japanese => JAPANESE_STOP_WORDS,
        StopWordsLanguage::German => GERMAN_STOP_WORDS,
        StopWordsLanguage::Korean => KOREAN_STOP_WORDS,
    }
}

const ENGLISH_STOP_WORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "across",
    "after",
    "all",
    "along",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "arent",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cant",
    "cannot",
    "could",
    "couldn't",
    "couldnt",
    "did",
    "didn't",
    "didnt",
    "do",
    "does",
    "doesn't",
    "doesnt",
    "doing",
    "done",
    "don't",
    "dont",
    "during",
    "each",
    "either",
    "for",
    "from",
    "given",
    "had",
    "has",
    "have",
    "having",
    "he",
    "he'd",
    "hed",
    "he'll",
    "hell",
    "her",
    "here",
    "hers",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "i",
    "i'd",
    "id",
    "i'll",
    "ill",
    "i'm",
    "im",
    "if",
    "in",
    "instead",
    "into",
    "is",
    "isn't",
    "isnt",
    "it",
    "it'll",
    "itll",
    "it's",
    "its",
    "i've",
    "ive",
    "let's",
    "lets",
    "may",
    "me",
    "more",
    "most",
    "much",
    "must",
    "my",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "said",
    "says",
    "see",
    "she",
    "she'd",
    "shed",
    "she'll",
    "shell",
    "should",
    "since",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "therefore",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "towards",
    "under",
    "until",
    "us",
    "use",
    "used",
    "uses",
    "using",
    "very",
    "want",
    "was",
    "wasn't",
    "wasnt",
    "we",
    "we'd",
    "wed",
    "we'll",
    "well",
    "we're",
    "were",
    "we've",
    "weve",
    "what",
    "what's",
    "whats",
    "when",
    "where",
    "whether",
    "which",
    "while",
    "who",
    "who'll",
    "wholl",
    "who's",
    "whos",
    "who've",
    "whove",
    "will",
    "with",
    "within",
    "without",
    "won't",
    "wont",
    "would",
    "wouldn't",
    "wouldnt",
    "you",
    "you'd",
    "youd",
    "you'll",
    "youll",
    "you're",
    "youre",
    "you've",
    "youve",
    "your",
];

const JAPANESE_STOP_WORDS: &[&str] = &[
    "あそこ",
    "あたり",
    "あちら",
    "あっち",
    "あと",
    "あなた",
    "あれ",
    "いくつ",
    "いつ",
    "いま",
    "いや",
    "いろいろ",
    "うち",
    "おおまか",
    "おまえ",
    "おれ",
    "ここ",
    "こちら",
    "こっち",
    "こと",
    "これ",
    "それ",
    "そこ",
    "そちら",
    "そっち",
    "ため",
    "どこ",
    "どこか",
    "ところ",
    "もの",
    "よう",
    "私",
    "我々",
    "彼",
    "彼女",
    "誰",
    "何",
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
    "上",
    "下",
    "前",
    "後",
    "左",
    "右",
    "中",
    "外",
    "する",
    "した",
    "して",
    "される",
    "です",
    "ます",
    "ませ",
    "ない",
    "なる",
    "あり",
    "ある",
    "いる",
    "そして",
    "また",
    "から",
    "まで",
    "より",
    "ので",
    "のに",
    "ほど",
    "ばかり",
    "だけ",
    "など",
    "この",
    "その",
    "あの",
    "どの",
    "ここ",
    "そこ",
    "あそこ",
    "ところ",
    "場合",
    "今回",
    "以前",
    "以後",
];

const GERMAN_STOP_WORDS: &[&str] = &[
    "ab",
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "das",
    "dass",
    "daß",
    "dein",
    "deine",
    "deinem",
    "deiner",
    "deines",
    "dem",
    "den",
    "denn",
    "der",
    "derer",
    "des",
    "dessen",
    "dich",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "dir",
    "doch",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "er",
    "es",
    "euch",
    "euer",
    "eure",
    "eurem",
    "euren",
    "eures",
    "für",
    "ganz",
    "gar",
    "habe",
    "haben",
    "hat",
    "hatte",
    "hattest",
    "hattet",
    "hätte",
    "hättest",
    "hättet",
    "her",
    "hin",
    "ich",
    "ihm",
    "ihn",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "ihres",
    "im",
    "in",
    "ins",
    "ist",
    "ja",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jene",
    "jenem",
    "jenen",
    "jener",
    "jenes",
    "kann",
    "kannst",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "keines",
    "können",
    "könnte",
    "könnten",
    "könntest",
    "ließ",
    "man",
    "manche",
    "manchem",
    "manchen",
    "mancher",
    "manches",
    "mehr",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "meiner",
    "meines",
    "mich",
    "mir",
    "mit",
    "muss",
    "musst",
    "musste",
    "muß",
    "müssen",
    "müssten",
    "nach",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "seid",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "seines",
    "sich",
    "sie",
    "sind",
    "so",
    "um",
    "und",
    "uns",
    "unter",
    "vom",
    "von",
    "vor",
    "war",
    "waren",
    "warst",
    "warum",
    "was",
    "weil",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "welches",
    "wenn",
    "wer",
    "werde",
    "werden",
    "weshalb",
    "wie",
    "wieder",
    "wir",
    "wirst",
    "wo",
    "während",
    "wieso",
    "zu",
    "zum",
    "zur",
    "über",
];

const KOREAN_STOP_WORDS: &[&str] = &[
    "이",
    "그",
    "저",
    "것",
    "수",
    "등",
    "들",
    "및",
    "에서",
    "으로",
    "에게",
    "께서",
    "그리고",
    "그러나",
    "하지만",
    "또한",
    "또",
    "더",
    "더욱",
    "가장",
    "매우",
    "너무",
    "아주",
    "좀",
    "잘",
    "못",
    "안",
    "않다",
    "있다",
    "없다",
    "하다",
    "되다",
    "이다",
    "아니다",
    "같다",
    "위해",
    "통해",
    "대한",
    "대해",
    "까지",
    "부터",
    "보다",
    "처럼",
    "만큼",
    "하고",
    "하며",
    "해서",
    "하는",
    "한",
    "된",
    "되는",
    "있는",
    "없는",
    "입니다",
    "합니다",
    "있습니다",
    "그리고",
    "그러면",
    "따라서",
    "때문",
    "때문에",
    "우리",
    "너희",
    "그들",
    "자신",
    "각",
    "각각",
    "모든",
    "어떤",
    "무슨",
    "언제",
    "어디",
    "누구",
    "무엇",
    "왜",
    "어떻게",
];

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_value::CharArray;

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(stop_words_builtin(args))
    }

    fn words(value: Value) -> Vec<String> {
        match value {
            Value::StringArray(array) => {
                assert_eq!(array.cols, 1);
                array.data
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn stop_words_default_returns_english_column() {
        let out = words(call(Vec::new()).expect("stopWords"));
        assert!(out.len() > 100);
        assert_eq!(out.first().map(String::as_str), Some("a"));
        assert!(out.contains(&"and".to_string()));
        assert!(out.contains(&"the".to_string()));
        assert!(out.contains(&"you're".to_string()));
    }

    #[test]
    fn stop_words_language_option_supports_documented_languages() {
        let german = words(
            call(vec![
                Value::String("Language".into()),
                Value::CharArray(CharArray::new_row("de")),
            ])
            .expect("German stop words"),
        );
        assert!(german.contains(&"und".to_string()));
        assert!(german.contains(&"über".to_string()));

        let japanese = words(
            call(vec![
                Value::CharArray(CharArray::new_row("Language")),
                Value::String("ja".into()),
            ])
            .expect("Japanese stop words"),
        );
        assert!(japanese.contains(&"これ".to_string()));
        assert!(japanese.contains(&"する".to_string()));

        let korean = words(
            call(vec![
                Value::String("language".into()),
                Value::String("ko".into()),
            ])
            .expect("Korean stop words"),
        );
        assert!(korean.contains(&"그리고".to_string()));
        assert!(korean.contains(&"합니다".to_string()));
    }

    #[test]
    fn stop_words_rejects_unsupported_forms() {
        let err = call(vec![Value::String("de".into())]).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:stopWords:InvalidInput"));
        assert!(err.message().contains("expected stopWords"));

        let err = call(vec![
            Value::String("Locale".into()),
            Value::String("en".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:stopWords:InvalidInput"));
        assert!(err.message().contains("unsupported option"));

        let err = call(vec![
            Value::String("Language".into()),
            Value::String("fr".into()),
        ])
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:stopWords:InvalidInput"));
        assert!(err.message().contains("language must be"));
    }
}
