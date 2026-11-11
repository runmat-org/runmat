use super::FeaParser;
use crate::lexical::{FeaToken, FeaTokenKind, FeaTriviaKind};
use crate::syntax::FeaTokenRange;
use runmat_types::Span;

impl FeaParser {
    pub(super) fn statement_end(&self, start: usize) -> usize {
        let mut cursor = start;
        while !self.at(cursor, FeaTokenKind::EndOfFile)
            && !self.at(cursor, FeaTokenKind::Semicolon)
            && !self.is_newline(cursor)
        {
            cursor += 1;
        }
        cursor
    }

    pub(super) fn consume_terminator(&self, mut cursor: usize) -> usize {
        if self.at(cursor, FeaTokenKind::Semicolon) || self.is_newline(cursor) {
            cursor += 1;
        }
        cursor
    }

    pub(super) fn trimmed_range(&self, mut start: usize, mut end: usize) -> Option<FeaTokenRange> {
        while start < end && self.kind(start).is_some_and(FeaTokenKind::is_trivia) {
            start += 1;
        }
        while end > start && self.kind(end - 1).is_some_and(FeaTokenKind::is_trivia) {
            end -= 1;
        }
        (start < end).then(|| self.token_range(start, end))
    }

    pub(super) fn token_range(&self, start: usize, end: usize) -> FeaTokenRange {
        if start >= end {
            let position = self
                .stream
                .tokens
                .get(start)
                .map_or(self.stream.source.len(), |token| token.span.start);
            return FeaTokenRange {
                start,
                end,
                span: Span {
                    start: position,
                    end: position,
                },
            };
        }
        FeaTokenRange {
            start,
            end,
            span: Span {
                start: self.token(start).span.start,
                end: self.token(end - 1).span.end,
            },
        }
    }

    pub(super) fn range_span(&self, start: usize, end: usize) -> Span {
        self.token_range(start, end).span
    }

    pub(super) fn significant(&self, start: usize, end: usize) -> Vec<usize> {
        (start..end)
            .filter(|index| !self.kind(*index).is_some_and(FeaTokenKind::is_trivia))
            .collect()
    }

    pub(super) fn skip_trivia(&self, mut cursor: usize) -> usize {
        while self.kind(cursor).is_some_and(FeaTokenKind::is_trivia) {
            cursor += 1;
        }
        cursor
    }

    pub(super) fn next_significant(&self, cursor: usize) -> Option<usize> {
        let cursor = self.skip_trivia(cursor);
        (!self.at(cursor, FeaTokenKind::EndOfFile)).then_some(cursor)
    }

    pub(super) fn is_newline(&self, index: usize) -> bool {
        self.at(index, FeaTokenKind::Trivia(FeaTriviaKind::Newline))
    }

    pub(super) fn at(&self, index: usize, expected: FeaTokenKind) -> bool {
        self.kind(index) == Some(expected)
    }

    pub(super) fn kind(&self, index: usize) -> Option<FeaTokenKind> {
        self.stream.tokens.get(index).map(|token| token.kind)
    }

    pub(super) fn token(&self, index: usize) -> FeaToken {
        self.stream.tokens[index]
    }

    pub(super) fn lexeme(&self, index: usize) -> &str {
        self.stream.lexeme(self.token(index))
    }
}
