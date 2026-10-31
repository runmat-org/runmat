//! Tolerant editor syntax context, with HIR owning resolved output counts.
use super::super::analysis::DocumentAnalysis;
use runmat_lexer::{SpannedToken, Token};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArgumentKind {
    Unknown,
    Integer,
    Number,
    Text,
    Logical,
    NumericArray,
    Callable,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    Parenthesis,
    Bracket,
    Brace,
}

fn opening(token: &Token) -> Option<Delimiter> {
    match token {
        Token::LParen => Some(Delimiter::Parenthesis),
        Token::LBracket => Some(Delimiter::Bracket),
        Token::LBrace => Some(Delimiter::Brace),
        _ => None,
    }
}

fn closing(token: &Token) -> Option<Delimiter> {
    match token {
        Token::RParen => Some(Delimiter::Parenthesis),
        Token::RBracket => Some(Delimiter::Bracket),
        Token::RBrace => Some(Delimiter::Brace),
        _ => None,
    }
}

fn argument_kind(tokens: &[SpannedToken]) -> ArgumentKind {
    if let [token] = tokens {
        return match token.token {
            Token::Integer | Token::RadixInteger => ArgumentKind::Integer,
            Token::Float => ArgumentKind::Number,
            Token::Str => ArgumentKind::Text,
            Token::True | Token::False => ArgumentKind::Logical,
            _ => ArgumentKind::Unknown,
        };
    }
    if tokens.first().is_some_and(|token| token.token == Token::At) {
        return ArgumentKind::Callable;
    }
    if tokens
        .first()
        .is_some_and(|token| token.token == Token::LBracket)
        && tokens.iter().all(|token| {
            matches!(
                token.token,
                Token::LBracket
                    | Token::RBracket
                    | Token::Comma
                    | Token::Semicolon
                    | Token::Integer
                    | Token::RadixInteger
                    | Token::Float
                    | Token::Plus
                    | Token::Minus
            )
        })
    {
        return ArgumentKind::NumericArray;
    }
    ArgumentKind::Unknown
}

pub(in crate::core) struct CallSite {
    pub(in crate::core) name: String,
    pub(in crate::core) active_argument: usize,
    pub(in crate::core) callee_range: std::ops::Range<usize>,
    pub(super) arguments: Vec<ArgumentKind>,
    pub(super) outputs: Option<usize>,
    pub(super) complete: bool,
}

fn callee_token(token: &Token) -> bool {
    matches!(token, Token::Ident | Token::True | Token::False)
}

fn callee_before(tokens: &[SpannedToken], open: usize) -> Option<(String, usize)> {
    let last = open.checked_sub(1)?;
    if !callee_token(&tokens.get(last)?.token) {
        return None;
    }
    let mut first = last;
    while first >= 2
        && tokens[first - 1].token == Token::Dot
        && tokens[first - 2].token == Token::Ident
    {
        first -= 2;
    }
    Some((
        tokens[first..=last]
            .iter()
            .map(|token| token.lexeme.as_str())
            .collect(),
        tokens[first].start,
    ))
}

fn call_open_at(tokens: &[SpannedToken], cursor: usize) -> Option<usize> {
    // Hovering the callee precedes its opening delimiter. Recognize that local
    // qualified name before looking for an enclosing argument list.
    let near = tokens
        .partition_point(|token| token.start <= cursor)
        .saturating_sub(1);
    if let Some(token) = tokens.get(near).filter(|token| token.end >= cursor) {
        if token.token == Token::LParen && callee_before(tokens, near).is_some() {
            return Some(near);
        }
        let mut last = near;
        if token.token == Token::Dot {
            last += 1;
        }
        if tokens
            .get(last)
            .is_some_and(|token| callee_token(&token.token))
        {
            while tokens
                .get(last + 1)
                .is_some_and(|token| token.token == Token::Dot)
                && tokens
                    .get(last + 2)
                    .is_some_and(|token| token.token == Token::Ident)
            {
                last += 2;
            }
            if tokens
                .get(last + 1)
                .is_some_and(|token| token.token == Token::LParen)
            {
                return Some(last + 1);
            }
        }
    }
    let mut stack = Vec::new();
    for (index, token) in tokens
        .iter()
        .enumerate()
        .take_while(|(_, token)| token.start < cursor)
    {
        if let Some(delimiter) = opening(&token.token) {
            stack.push((delimiter, index));
        } else if let Some(delimiter) = closing(&token.token) {
            if stack.pop().map(|entry| entry.0) != Some(delimiter) {
                stack.clear();
            }
        }
    }
    stack.into_iter().rev().find_map(|(delimiter, index)| {
        (delimiter == Delimiter::Parenthesis && callee_before(tokens, index).is_some())
            .then_some(index)
    })
}

fn syntactic_outputs(tokens: &[SpannedToken], callee_start: usize) -> Option<usize> {
    let first = tokens.partition_point(|token| token.start < callee_start);
    if first < 2 || tokens[first - 1].token != Token::Assign {
        return Some(1); // A nested/value expression consumes one result.
    }
    if tokens[first - 2].token == Token::Ident {
        return Some(1);
    }
    if tokens[first - 2].token != Token::RBracket {
        return None;
    }
    let mut index = first - 2;
    let mut count = 0;
    loop {
        index = index.checked_sub(1)?;
        if !matches!(tokens[index].token, Token::Ident | Token::Tilde) {
            return None;
        }
        count += 1;
        index = index.checked_sub(1)?;
        match tokens[index].token {
            Token::LBracket => return Some(count),
            Token::Comma => {}
            _ => return None, // Expanded/indexed outputs require the HIR authority.
        }
    }
}

impl CallSite {
    pub fn at(analysis: &DocumentAnalysis, cursor: usize) -> Option<Self> {
        let tokens = &analysis.tokens;
        let index = call_open_at(tokens, cursor)?;
        let (name, start) = callee_before(tokens, index)?;
        let mut stack = vec![Delimiter::Parenthesis];
        let mut begin = index + 1;
        let mut arguments = Vec::new();
        let mut active_argument = 0;
        let mut complete = false;
        let mut end = usize::MAX;
        for (next, argument) in tokens.iter().enumerate().skip(index + 1) {
            if let Some(delimiter) = opening(&argument.token) {
                stack.push(delimiter);
            } else if let Some(delimiter) = closing(&argument.token) {
                if stack.pop() != Some(delimiter) {
                    break;
                }
                if stack.is_empty() {
                    complete = next > begin || arguments.is_empty();
                    if next != begin || !arguments.is_empty() {
                        arguments.push(argument_kind(&tokens[begin..next]));
                    }
                    end = argument.end;
                    break;
                }
            } else if argument.token == Token::Comma && stack.len() == 1 {
                arguments.push(argument_kind(&tokens[begin..next]));
                begin = next + 1;
                if argument.start < cursor {
                    active_argument += 1;
                }
            } else if argument.token == Token::Semicolon && stack.len() == 1 {
                end = argument.start;
                break;
            }
        }
        if end == usize::MAX && (begin < tokens.len() || !arguments.is_empty()) {
            arguments.push(argument_kind(&tokens[begin..]));
        }
        if cursor >= end {
            return None;
        }
        let resolved = analysis.lowering.as_ref().and_then(|lowering| {
            lowering
                .hir_index
                .calls
                .iter()
                .filter(|call| call.span.start <= start && start < call.span.end)
                .filter(|call| call.name.display_name().as_deref() == Some(name.as_str()))
                .min_by_key(|call| call.span.end - call.span.start)
        });
        let outputs = resolved.map_or_else(
            || syntactic_outputs(tokens, start),
            |call| call.requested_outputs.known_count(),
        );
        Some(Self {
            name,
            active_argument,
            callee_range: start..tokens[index].start,
            arguments,
            outputs,
            complete,
        })
    }
}
