//! MATLAB-compatible `stopWords` builtin.

use runmat_builtins::STOP_WORDS_ERROR_INVALID_INPUT;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::strings::common::{
    contains_numeric_or_resident_text_input, scalar_text_input, text_input_admission,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[path = "stopwords/output.rs"]
mod output;

fn stopwords_error(message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("stopWords");
    if let Some(identifier) = STOP_WORDS_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "stopWords",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::text_analytics::stopwords",
    context = true
)]
async fn stop_words_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let language = parse_language_args(runtime.as_ref(), args)?;
    let words = stop_words_for_language(language);
    output::string_array(runtime.as_ref(), words).map(Value::StringArray)
}

fn parse_language_args(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<StopWordsLanguage> {
    match args.len() {
        0 => Ok(StopWordsLanguage::English),
        2 => {
            if args.iter().any(contains_numeric_or_resident_text_input) {
                return Err(stopwords_error(
                    "stopWords: expected host text option and language",
                ));
            }
            // The preflight above excludes resident values even inside aggregates.
            // Borrow host controls directly so an existing text payload is never
            // cloned before its operation-specific admission.
            let option = scalar_text_input(&args[0], runtime, |runtime, cause| {
                text_input_admission("stopWords", runtime, cause)
            })?
            .ok_or_else(|| stopwords_error("stopWords: expected text option"))?;
            if !option.as_str().eq_ignore_ascii_case("language") {
                return Err(stopwords_error(format!(
                    "stopWords: unsupported option '{}'",
                    option.as_str()
                )));
            }
            let language = scalar_text_input(&args[1], runtime, |runtime, cause| {
                text_input_admission("stopWords", runtime, cause)
            })?
            .ok_or_else(|| stopwords_error("stopWords: expected text language"))?;
            parse_language(language.as_str())
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
    if language.eq_ignore_ascii_case("en") {
        Ok(StopWordsLanguage::English)
    } else if language.eq_ignore_ascii_case("ja") {
        Ok(StopWordsLanguage::Japanese)
    } else if language.eq_ignore_ascii_case("de") {
        Ok(StopWordsLanguage::German)
    } else if language.eq_ignore_ascii_case("ko") {
        Ok(StopWordsLanguage::Korean)
    } else {
        Err(stopwords_error(format!(
            "stopWords: language must be 'en', 'ja', 'de', or 'ko', got '{language}'"
        )))
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
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use futures::executor::block_on;
    use runmat_value::CharArray;
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn managed_runtime(ledger: &ManagedHostEffectLedger) -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
    }

    fn call(args: Vec<Value>) -> BuiltinResult<Value> {
        block_on(stop_words_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
    }

    fn words(value: Value) -> runmat_value::StringArray {
        match value {
            Value::StringArray(array) => {
                assert_eq!(array.cols, 1);
                array
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[test]
    fn stop_words_default_returns_english_column() {
        let out = words(call(Vec::new()).expect("stopWords"));
        assert!(out.data().len() > 100);
        assert_eq!(out.data().first().and_then(|text| text.as_str()), Some("a"));
        assert!(out.data().iter().any(|text| text.as_str() == Some("and")));
        assert!(out.data().iter().any(|text| text.as_str() == Some("the")));
        assert!(out
            .data()
            .iter()
            .any(|text| text.as_str() == Some("you're")));
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
        assert!(german
            .data()
            .iter()
            .any(|text| text.as_str() == Some("und")));
        assert!(german
            .data()
            .iter()
            .any(|text| text.as_str() == Some("über")));

        let japanese = words(
            call(vec![
                Value::CharArray(CharArray::new_row("Language")),
                Value::String("ja".into()),
            ])
            .expect("Japanese stop words"),
        );
        assert!(japanese
            .data()
            .iter()
            .any(|text| text.as_str() == Some("これ")));
        assert!(japanese
            .data()
            .iter()
            .any(|text| text.as_str() == Some("する")));

        let korean = words(
            call(vec![
                Value::String("language".into()),
                Value::String("ko".into()),
            ])
            .expect("Korean stop words"),
        );
        assert!(korean
            .data()
            .iter()
            .any(|text| text.as_str() == Some("그리고")));
        assert!(korean
            .data()
            .iter()
            .any(|text| text.as_str() == Some("합니다")));
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

    #[test]
    fn managed_vocabulary_claims_original_text_until_last_alias_and_refuses_under_budget() {
        let denied = ManagedHostEffectLedger::new(1);
        let error = output::string_array(Some(&managed_runtime(&denied)), &["a", "über"])
            .expect_err("all output layers require pre-effect admission");
        assert!(error.message().contains("stopWords"));
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);

        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let context = managed_runtime(&ledger);
        let output = output::string_array(Some(&context), &["a", "über"])
            .expect("admitted static vocabulary");
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(output.data()[1].as_str(), Some("über"));
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > 0);
        let alias = output.clone();
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
