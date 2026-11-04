//! `.fea` semantic token classes from the one generated grammar
//! (`runmat_design_syntax::FEA_TEXTMATE_GRAMMAR`, built from the compiler's closed tables).
//!
//! The docs, the site and this server color the same words the same way: the grammar's
//! `vocabulary` lists each TextMate scope's words in match priority, and this module applies
//! them to lexer tokens. Classes map onto the shared RunMat legend (`semantic_tokens::legend`),
//! which Desktop also hard-codes, so no new token types are introduced.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use runmat_lexer::fea::{FeaToken, FeaTokenKind, FeaTriviaKind};

/// Legend indices (`semantic_tokens::legend`): keyword, function, variable, parameter, namespace,
/// string, number, operator, comment. Modifier bit 1 is `defaultLibrary`.
const KEYWORD: u32 = 0;
const FUNCTION: u32 = 1;
const VARIABLE: u32 = 2;
const PARAMETER: u32 = 3;
const NAMESPACE: u32 = 4;
const STRING: u32 = 5;
const NUMBER: u32 = 6;
const OPERATOR: u32 = 7;
const COMMENT: u32 = 8;
const DEFAULT_LIBRARY: u32 = 1 << 1;

/// A token's semantic type and modifier bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenClass {
    pub token_type: u32,
    pub modifiers: u32,
}

const fn class(token_type: u32, modifiers: u32) -> TokenClass {
    TokenClass {
        token_type,
        modifiers,
    }
}

/// The grammar scope a compiler word is colored with, mapped to a legend class.
fn scope_class(scope: &str) -> Option<TokenClass> {
    Some(match scope {
        "keyword.control.fea" => class(KEYWORD, 0),
        "storage.type.fea" => class(NAMESPACE, DEFAULT_LIBRARY),
        "support.function.fea" => class(FUNCTION, DEFAULT_LIBRARY),
        "constant.language.fea" => class(PARAMETER, DEFAULT_LIBRARY),
        "support.type.property-name.fea" => class(VARIABLE, DEFAULT_LIBRARY),
        "support.type.fea" | "support.type.unit.fea" => class(NAMESPACE, 0),
        _ => return None,
    })
}

struct Vocabulary {
    /// Single words, first scope in grammar priority wins.
    words: HashMap<String, TokenClass>,
    /// Dotted names (`structural.fixed`) by first segment, longest first.
    dotted: HashMap<String, Vec<Vec<String>>>,
    units: HashSet<String>,
    unit_class: TokenClass,
}

fn vocabulary() -> &'static Vocabulary {
    static VOCABULARY: OnceLock<Vocabulary> = OnceLock::new();
    VOCABULARY.get_or_init(|| {
        let grammar: serde_json::Value =
            serde_json::from_str(runmat_design_syntax::FEA_TEXTMATE_GRAMMAR)
                .expect("runmat-design-syntax grammar/fea.tmLanguage.json is valid JSON");
        let vocabulary = &grammar["vocabulary"];
        let mut words = HashMap::new();
        let mut dotted: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for entry in vocabulary["scopes"].as_array().into_iter().flatten() {
            let Some(token_class) = entry["scope"].as_str().and_then(scope_class) else {
                continue;
            };
            for word in entry["words"].as_array().into_iter().flatten() {
                let Some(word) = word.as_str() else { continue };
                if word.contains('.') {
                    let segments: Vec<String> = word.split('.').map(str::to_string).collect();
                    dotted
                        .entry(segments[0].clone())
                        .or_default()
                        .push(segments);
                } else {
                    words.entry(word.to_string()).or_insert(token_class);
                }
            }
        }
        for names in dotted.values_mut() {
            names.sort_by_key(|segments| std::cmp::Reverse(segments.len()));
        }
        let units = vocabulary["units"]["words"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|word| word.as_str().map(str::to_string))
            .collect();
        let unit_class = vocabulary["units"]["scope"]
            .as_str()
            .and_then(scope_class)
            .unwrap_or(class(NAMESPACE, 0));
        Vocabulary {
            words,
            dotted,
            units,
            unit_class,
        }
    })
}

fn punctuation_class(kind: FeaTokenKind) -> Option<TokenClass> {
    Some(match kind {
        FeaTokenKind::Keyword(_) => class(KEYWORD, 0),
        FeaTokenKind::String => class(STRING, 0),
        FeaTokenKind::Number => class(NUMBER, 0),
        FeaTokenKind::Trivia(FeaTriviaKind::Comment) => class(COMMENT, 0),
        FeaTokenKind::Equal
        | FeaTokenKind::EqualEqual
        | FeaTokenKind::BangEqual
        | FeaTokenKind::Less
        | FeaTokenKind::LessEqual
        | FeaTokenKind::Greater
        | FeaTokenKind::GreaterEqual
        | FeaTokenKind::Plus
        | FeaTokenKind::Minus
        | FeaTokenKind::Star
        | FeaTokenKind::Slash
        | FeaTokenKind::Caret => class(OPERATOR, 0),
        _ => return None,
    })
}

/// Classifies every token of a `.fea` source, returning `(token, class)` pairs in source order.
/// Identifiers take the grammar's class: compiler words by their scope, dotted support names
/// segment by segment, units after a number or `]` (and through `*`, `/`, `^` chains, and after
/// `units =`), and other names as plain variables.
pub(super) fn classify(source: &str, tokens: &[FeaToken]) -> Vec<(FeaToken, TokenClass)> {
    let vocabulary = vocabulary();
    let significant: Vec<FeaToken> = tokens
        .iter()
        .copied()
        .filter(|token| {
            !token.kind.is_trivia() || token.kind == FeaTokenKind::Trivia(FeaTriviaKind::Comment)
        })
        .collect();
    let code: Vec<usize> = (0..significant.len())
        .filter(|&index| !significant[index].kind.is_trivia())
        .collect();
    let text = |token: &FeaToken| source.get(token.span.start..token.span.end).unwrap_or("");
    let mut classes: Vec<Option<TokenClass>> = significant
        .iter()
        .map(|token| punctuation_class(token.kind))
        .collect();

    let mut unit_chain = false;
    let mut position = 0;
    while position < code.len() {
        let index = code[position];
        let token = significant[index];
        let previous = position
            .checked_sub(1)
            .map(|before| significant[code[before]]);
        if token.kind != FeaTokenKind::Identifier {
            let operator = matches!(
                token.kind,
                FeaTokenKind::Star | FeaTokenKind::Slash | FeaTokenKind::Caret
            );
            let exponent = token.kind == FeaTokenKind::Number
                && previous.is_some_and(|before| before.kind == FeaTokenKind::Caret);
            unit_chain = unit_chain && (operator || exponent);
            position += 1;
            continue;
        }
        let word = text(&token);

        // A dotted support name, matched whole when its segments are adjacent.
        if let Some(candidates) = vocabulary.dotted.get(word) {
            let matched = candidates.iter().find(|segments| {
                segments.iter().enumerate().skip(1).all(|(segment, name)| {
                    let dot = code
                        .get(position + 2 * segment - 1)
                        .map(|&i| significant[i]);
                    let next = code.get(position + 2 * segment).map(|&i| significant[i]);
                    let before = significant[code[position + 2 * segment - 2]];
                    matches!((dot, next), (Some(dot), Some(next))
                        if dot.kind == FeaTokenKind::Dot
                            && next.kind == FeaTokenKind::Identifier
                            && dot.span.start == before.span.end
                            && next.span.start == dot.span.end
                            && text(&next) == name)
                })
            });
            if let Some(segments) = matched {
                let function = class(FUNCTION, DEFAULT_LIBRARY);
                for segment in 0..segments.len() {
                    classes[code[position + 2 * segment]] = Some(function);
                }
                position += 2 * segments.len() - 1;
                unit_chain = false;
                continue;
            }
        }

        let after_quantity = previous.is_some_and(|before| {
            let quantity = matches!(
                before.kind,
                FeaTokenKind::Number | FeaTokenKind::RightBracket
            );
            let compound =
                unit_chain && matches!(before.kind, FeaTokenKind::Star | FeaTokenKind::Slash);
            let units_value = before.kind == FeaTokenKind::Equal
                && position >= 2
                && text(&significant[code[position - 2]]) == "units";
            quantity || compound || units_value
        });
        let is_unit = after_quantity && vocabulary.units.contains(word);
        classes[index] = Some(if is_unit {
            vocabulary.unit_class
        } else {
            vocabulary
                .words
                .get(word)
                .copied()
                .unwrap_or(class(VARIABLE, 0))
        });
        unit_chain = is_unit;
        position += 1;
    }

    significant
        .into_iter()
        .zip(classes)
        .filter_map(|(token, token_class)| token_class.map(|token_class| (token, token_class)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_parser::fea::parse_fea;

    fn class_of(source: &str, word: &str, nth: usize) -> Option<TokenClass> {
        let syntax = parse_fea(source.to_string());
        classify(source, &syntax.tokens.tokens)
            .into_iter()
            .filter(|(token, _)| &source[token.span.start..token.span.end] == word)
            .nth(nth)
            .map(|(_, class)| class)
    }

    #[test]
    fn tokens_take_the_generated_grammar_classes() {
        let source = "material steel {\n    mass_density = 7850 kg/m^3\n    elastic { youngs_modulus = 200 GPa }\n}\nmodel m {\n    constraint fixed { structural.fixed { target = geometry.beam.region.fixed_end } }\n}\nstudy s {\n    analysis static { structural static {} }\n    output d = max(magnitude(analysis.static.field.node.displacement), over = geometry.beam.region.tip)\n}\n";
        let unit = class(NAMESPACE, 0);
        assert_eq!(class_of(source, "kg", 0), Some(unit));
        assert_eq!(class_of(source, "m", 0), Some(unit));
        assert_eq!(class_of(source, "GPa", 0), Some(unit));
        assert_eq!(
            class_of(source, "mass_density", 0),
            Some(class(VARIABLE, DEFAULT_LIBRARY))
        );
        let support = class(FUNCTION, DEFAULT_LIBRARY);
        assert_eq!(class_of(source, "structural", 0), Some(support));
        assert_eq!(class_of(source, "fixed", 1), Some(support));
        assert_eq!(class_of(source, "max", 0), Some(support));
        assert_eq!(
            class_of(source, "static", 1),
            Some(class(NAMESPACE, DEFAULT_LIBRARY))
        );
        assert_eq!(
            class_of(source, "node", 0),
            Some(class(PARAMETER, DEFAULT_LIBRARY))
        );
        assert_eq!(
            class_of(source, "displacement", 0),
            Some(class(PARAMETER, DEFAULT_LIBRARY))
        );
        assert_eq!(class_of(source, "steel", 0), Some(class(VARIABLE, 0)));
        assert_eq!(class_of(source, "7850", 0), Some(class(NUMBER, 0)));
    }

    #[test]
    fn every_grammar_word_has_a_legend_class() {
        let vocabulary = vocabulary();
        assert!(vocabulary.words.len() > 200, "{}", vocabulary.words.len());
        assert!(vocabulary.units.contains("GPa") && vocabulary.units.contains("mm"));
        assert!(vocabulary.dotted.contains_key("structural"));
        let grammar: serde_json::Value =
            serde_json::from_str(runmat_design_syntax::FEA_TEXTMATE_GRAMMAR).unwrap();
        for entry in grammar["vocabulary"]["scopes"].as_array().unwrap() {
            let scope = entry["scope"].as_str().unwrap();
            assert!(scope_class(scope).is_some(), "no legend class for {scope}");
        }
    }
}
