use crate::core::position::offset_to_position;
use lsp_types::{
    SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokens, SemanticTokensLegend,
};
use runmat_lexer::{SpannedToken, Token};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentifierRole {
    Function,
    Parameter,
    Variable,
    Namespace,
}

#[derive(Clone, Copy, Debug)]
pub struct SemanticHint {
    pub start: usize,
    pub end: usize,
    pub role: IdentifierRole,
    pub declaration: bool,
    pub default_library: bool,
}

/// Return the semantic token legend (types/modifiers) shared by native and wasm.
pub fn legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: vec![
            SemanticTokenType::KEYWORD,
            SemanticTokenType::FUNCTION,
            SemanticTokenType::VARIABLE,
            SemanticTokenType::PARAMETER,
            SemanticTokenType::NAMESPACE,
            SemanticTokenType::STRING,
            SemanticTokenType::NUMBER,
            SemanticTokenType::OPERATOR,
            SemanticTokenType::COMMENT,
        ],
        token_modifiers: vec![
            SemanticTokenModifier::DECLARATION,
            SemanticTokenModifier::DEFAULT_LIBRARY,
        ],
    }
}

/// Build full-document semantic tokens from the lexer output.
pub fn full(text: &str, tokens: &[SpannedToken], hints: &[SemanticHint]) -> Option<SemanticTokens> {
    let legend = legend();
    let hint_map = hints
        .iter()
        .map(|hint| ((hint.start, hint.end), *hint))
        .collect::<HashMap<_, _>>();

    let mut data: Vec<SemanticToken> = Vec::new();
    let mut prev_line: u32 = 0;
    let mut prev_col: u32 = 0;
    let mut first = true;

    for tok in tokens {
        let (token_type, token_modifiers_bitset) = if let Some(hint) =
            hint_map.get(&(tok.start, tok.end))
        {
            let token_type = match hint.role {
                IdentifierRole::Function => SemanticTokenType::FUNCTION,
                IdentifierRole::Parameter => SemanticTokenType::PARAMETER,
                IdentifierRole::Variable => SemanticTokenType::VARIABLE,
                IdentifierRole::Namespace => SemanticTokenType::NAMESPACE,
            };
            let mut bitset = 0u32;
            if hint.declaration {
                bitset |= 1 << 0;
            }
            if hint.default_library {
                bitset |= 1 << 1;
            }
            (token_type, bitset)
        } else {
            let token_type = match tok.token {
                Token::Function
                | Token::If
                | Token::Else
                | Token::ElseIf
                | Token::For
                | Token::ParFor
                | Token::Spmd
                | Token::While
                | Token::Break
                | Token::Continue
                | Token::Return
                | Token::End
                | Token::ClassDef
                | Token::Properties
                | Token::Methods
                | Token::Events
                | Token::Enumeration
                | Token::Arguments
                | Token::Import
                | Token::Switch
                | Token::Case
                | Token::Otherwise
                | Token::Try
                | Token::Catch
                | Token::Global
                | Token::Persistent
                | Token::True
                | Token::False => SemanticTokenType::KEYWORD,
                Token::Ident => SemanticTokenType::VARIABLE,
                Token::Float | Token::Integer | Token::RadixInteger => SemanticTokenType::NUMBER,
                Token::Str => SemanticTokenType::STRING,
                Token::Section | Token::BlockComment | Token::LineComment | Token::Ellipsis => {
                    SemanticTokenType::COMMENT
                }
                Token::AndAnd
                | Token::OrOr
                | Token::Equal
                | Token::NotEqual
                | Token::LessEqual
                | Token::GreaterEqual
                | Token::Less
                | Token::Greater
                | Token::Plus
                | Token::Minus
                | Token::Star
                | Token::Slash
                | Token::Backslash
                | Token::Caret
                | Token::DotStar
                | Token::DotSlash
                | Token::DotBackslash
                | Token::DotCaret
                | Token::And
                | Token::Or
                | Token::Colon => SemanticTokenType::OPERATOR,
                _ => continue,
            };
            (token_type, 0)
        };

        let pos = offset_to_position(text, tok.start);
        let token_type_idx = legend
            .token_types
            .iter()
            .position(|t| t == &token_type)
            .unwrap_or(0) as u32;

        // LSP lengths are UTF-16 columns, not UTF-8 bytes. Split multiline
        // comments/continuations so no token extends past a physical line.
        for (line, content) in text.get(tok.start..tok.end)?.split('\n').enumerate() {
            let length = content.trim_end_matches('\r').encode_utf16().count() as u32;
            if length == 0 {
                continue;
            }
            let current_line = pos.line + line as u32;
            let column = if line == 0 { pos.character } else { 0 };
            let delta_line = if first {
                current_line
            } else {
                current_line - prev_line
            };
            let delta_start = if first || delta_line > 0 {
                column
            } else {
                column - prev_col
            };
            data.push(SemanticToken {
                delta_line,
                delta_start,
                length,
                token_type: token_type_idx,
                token_modifiers_bitset,
            });
            prev_line = current_line;
            prev_col = column;
            first = false;
        }
    }

    Some(SemanticTokens {
        result_id: None,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_and_multiline_comments_stay_within_utf16_line_bounds() {
        let text = "x = \"😀\"; ... trailing text\r\n%{\r\nβ comment\r\n%}\r\ny = 1;";
        let tokens = runmat_lexer::tokenize_detailed(text);
        let encoded = full(text, &tokens, &[]).unwrap();
        let lines = text.split('\n').collect::<Vec<_>>();
        let mut line = 0;
        let mut column = 0;
        let mut continuation_seen = false;
        for token in encoded.data {
            line += token.delta_line as usize;
            column = if token.delta_line > 0 {
                token.delta_start
            } else {
                column + token.delta_start
            };
            assert!(
                column + token.length
                    <= lines[line].trim_end_matches('\r').encode_utf16().count() as u32
            );
            if line == 0 && column == 4 {
                assert_eq!(token.length, 4, "quoted emoji spans four UTF-16 units");
            }
            if line == 0 && column == 10 {
                assert_eq!(
                    legend().token_types[token.token_type as usize],
                    SemanticTokenType::COMMENT
                );
                continuation_seen = true;
            }
        }
        assert!(continuation_seen);
    }
}
