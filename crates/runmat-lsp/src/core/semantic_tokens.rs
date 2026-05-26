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
        let (token_type, token_modifiers_bitset) =
            if let Some(hint) = hint_map.get(&(tok.start, tok.end)) {
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
                    Token::Float | Token::Integer => SemanticTokenType::NUMBER,
                    Token::Str => SemanticTokenType::STRING,
                    Token::Section | Token::BlockComment | Token::LineComment => {
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
        let len = (tok.end.saturating_sub(tok.start)) as u32;
        if len == 0 {
            continue;
        }
        let token_type_idx = legend
            .token_types
            .iter()
            .position(|t| t == &token_type)
            .unwrap_or(0) as u32;

        let delta_line = if first {
            pos.line
        } else {
            pos.line.saturating_sub(prev_line)
        };
        let delta_start = if first || delta_line > 0 {
            pos.character
        } else {
            pos.character.saturating_sub(prev_col)
        };

        data.push(SemanticToken {
            delta_line,
            delta_start,
            length: len,
            token_type: token_type_idx,
            token_modifiers_bitset,
        });

        prev_line = pos.line;
        prev_col = pos.character;
        first = false;
    }

    Some(SemanticTokens {
        result_id: None,
        data,
    })
}
