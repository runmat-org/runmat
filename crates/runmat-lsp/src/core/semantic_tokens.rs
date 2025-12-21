use crate::core::position::offset_to_position;
use lsp_types::{SemanticToken, SemanticTokenType, SemanticTokens, SemanticTokensLegend};
use runmat_lexer::{SpannedToken, Token};

/// Return the semantic token legend (types/modifiers) shared by native and wasm.
pub fn legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: vec![
            SemanticTokenType::KEYWORD,
            SemanticTokenType::FUNCTION,
            SemanticTokenType::VARIABLE,
            SemanticTokenType::PARAMETER,
            SemanticTokenType::STRING,
            SemanticTokenType::NUMBER,
            SemanticTokenType::OPERATOR,
            SemanticTokenType::COMMENT,
        ],
        token_modifiers: vec![],
    }
}

/// Build full-document semantic tokens from the lexer output.
pub fn full(text: &str, tokens: &[SpannedToken]) -> Option<SemanticTokens> {
    let legend = legend();
    let mut data: Vec<SemanticToken> = Vec::new();
    let mut prev_line: u32 = 0;
    let mut prev_col: u32 = 0;
    let mut first = true;

    for tok in tokens {
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
            Token::Section | Token::BlockComment | Token::LineComment => SemanticTokenType::COMMENT,
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
            token_modifiers_bitset: 0,
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
