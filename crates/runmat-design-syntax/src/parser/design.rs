//! The design dialects' statement classifier: the generic forms over the shared engine.
//!
//! Blocks, statement boundaries, names and expressions use the same scanning and Pratt parser
//! as `.fea`; only the classification differs. Domain words are never consulted here: a
//! statement's shape decides its node (`<word> <name> = <value>` is a value statement whatever
//! the word), and each front end checks the words against its registry.

use super::expression::{self, ExpressionMode};
use super::grammar::{is_name_token, BlockMemberEnd};
use super::FeaParser;
use crate::dialect::{Dialect, GenericForm, CLAUSE_WORDS, STD_HEADER_WORD, VALUE_WORDS};
use crate::lexical::{FeaKeyword, FeaTokenKind};
use crate::tree::NodeKind;
use crate::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_types::Span;

impl FeaParser {
    pub(super) fn parse_design_file(&mut self, dialect: Dialect) {
        self.expression_mode = ExpressionMode::Design;
        let mut cursor = 0;
        let mut seen_item = false;
        while !self.at(cursor, FeaTokenKind::EndOfFile) {
            cursor = self.skip_trivia(cursor);
            if self.at(cursor, FeaTokenKind::EndOfFile) {
                break;
            }
            let start = cursor;
            match self.scan_block_member(start) {
                BlockMemberEnd::Nested(open) => {
                    seen_item = true;
                    let word = self.lexeme(start).to_string();
                    if word == GenericForm::Use.word() {
                        let (_, next) = self.parse_design_block(open);
                        self.parse_use_declaration(start, open, Some(next));
                        cursor = next;
                    } else {
                        if !dialect.is_declaration_word(&word) {
                            self.diagnostics.push(FeaSyntaxDiagnostic {
                                kind: FeaSyntaxDiagnosticKind::ExpectedDeclaration,
                                span: self.token(start).span,
                                message: format!(
                                    "`{word}` does not start a {} declaration; expected {}",
                                    dialect.spec().label.to_lowercase(),
                                    quoted_list(dialect.spec().declarations)
                                ),
                            });
                        }
                        let (_, next) = self.parse_design_block(open);
                        self.parse_block_header(start, open, true);
                        self.tree.node(NodeKind::Declaration, start, next);
                        cursor = next;
                    }
                }
                BlockMemberEnd::Statement(end) | BlockMemberEnd::Close(end) if end > start => {
                    let range = self.trimmed_range(start, end);
                    let significant = self.significant(start, end);
                    let word = self.lexeme(start).to_string();
                    let is_header = (word == GenericForm::VersionHeader.word()
                        || word == STD_HEADER_WORD)
                        && significant.len() >= 2;
                    if is_header {
                        if seen_item {
                            self.diagnostics.push(FeaSyntaxDiagnostic {
                                kind: FeaSyntaxDiagnosticKind::ExpectedDeclaration,
                                span: self.token(start).span,
                                message: "the version header comes before every other item".into(),
                            });
                        }
                        self.parse_version_header(&significant);
                    } else if word == GenericForm::Use.word() {
                        seen_item = true;
                        self.parse_use_declaration(start, end, None);
                    } else if self.at(start, FeaTokenKind::Keyword(FeaKeyword::Import)) {
                        seen_item = true;
                        if self.parse_import(start, end).is_some() {
                            if let Some(range) = range {
                                self.tree.node(NodeKind::Import, range.start, range.end);
                            }
                        } else if let Some(range) = range {
                            self.tree.node(NodeKind::Error, range.start, range.end);
                        }
                    } else {
                        seen_item = true;
                        self.diagnostics.push(FeaSyntaxDiagnostic {
                            kind: FeaSyntaxDiagnosticKind::ExpectedDeclaration,
                            span: self.token(start).span,
                            message: format!(
                                "expected a version header, `use`, `import` or a declaration ({})",
                                quoted_list(dialect.spec().declarations)
                            ),
                        });
                        if let Some(range) = range {
                            self.tree.node(NodeKind::Error, range.start, range.end);
                        }
                    }
                    cursor = self.consume_terminator(end);
                }
                BlockMemberEnd::Statement(_) | BlockMemberEnd::Close(_) => {
                    // A stray `}` at file level.
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::ExpectedDeclaration,
                        span: self.token(start).span,
                        message: "unmatched `}`".into(),
                    });
                    self.tree.node(NodeKind::Error, start, start + 1);
                    cursor = start + 1;
                }
            }
        }
    }

    fn parse_version_header(&mut self, significant: &[usize]) {
        let first = significant[0];
        let kind = if self.lexeme(first) == STD_HEADER_WORD {
            NodeKind::StdHeader
        } else {
            NodeKind::LanguageHeader
        };
        let version = significant[1];
        if matches!(
            self.kind(version),
            Some(FeaTokenKind::Number | FeaTokenKind::String | FeaTokenKind::Identifier)
        ) {
            self.tree.node(NodeKind::Version, version, version + 1);
        } else {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: self.token(version).span,
                message: "expected a version after the header word".into(),
            });
        }
        if let Some(extra) = significant.get(2) {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: self.token(*extra).span,
                message: "unexpected token after the version".into(),
            });
        }
        self.tree.node_over(kind, significant);
    }

    /// `{ … }` of a design dialect. Returns the cursor after the block.
    fn parse_design_block(&mut self, open: usize) -> (Span, usize) {
        let mut cursor = open + 1;
        loop {
            cursor = self.skip_trivia(cursor);
            if self.at(cursor, FeaTokenKind::RightBrace) {
                self.tree.node(NodeKind::Block, open, cursor + 1);
                return (self.range_span(open, cursor + 1), cursor + 1);
            }
            if self.at(cursor, FeaTokenKind::EndOfFile) {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::UnclosedBlock,
                    span: self.token(open).span,
                    message: "unclosed block".to_string(),
                });
                self.tree.node(NodeKind::Block, open, cursor);
                return (self.range_span(open, cursor), cursor);
            }
            let start = cursor;
            match self.scan_block_member(start) {
                BlockMemberEnd::Nested(open_block) => {
                    let (_, next) = self.parse_design_block(open_block);
                    self.parse_nested_statement(start, open_block, next);
                    cursor = next;
                }
                BlockMemberEnd::Statement(end) => {
                    self.parse_design_statement(start, end);
                    cursor = self.consume_terminator(end);
                }
                BlockMemberEnd::Close(end) => {
                    if end > start {
                        self.parse_design_statement(start, end);
                    }
                    cursor = end;
                }
            }
        }
    }

    fn parse_nested_statement(&mut self, start: usize, open: usize, next: usize) {
        let word = self.lexeme(start).to_string();
        let kind = match word.as_str() {
            w if w == GenericForm::When.word() => {
                let condition = self.significant(start + 1, open);
                if condition.is_empty() {
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                        span: self.token(start).span,
                        message: "`when` requires a condition before `{`".into(),
                    });
                } else {
                    self.design_expression(&condition, self.token(start).span);
                }
                NodeKind::When
            }
            w if w == GenericForm::Solved.word() => {
                if let Some(extra) = self.significant(start + 1, open).first() {
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                        span: self.token(*extra).span,
                        message: "`solved` takes no header; write `solved { … }`".into(),
                    });
                }
                NodeKind::Solved
            }
            w if w == GenericForm::Use.word() => {
                self.parse_use_declaration(start, open, Some(next));
                return;
            }
            w if w == GenericForm::Check.word() => {
                let significant = self.significant(start, open);
                self.required_name(&significant, 1, "`check` requires a name");
                NodeKind::Check
            }
            _ => {
                self.parse_block_header(start, open, false);
                NodeKind::BlockStatement
            }
        };
        self.tree.node(kind, start, next);
    }

    /// `<word> [<name>] [: <type>] [<clause word> <expr>]…` before a block.
    fn parse_block_header(&mut self, start: usize, open: usize, declaration: bool) {
        let significant = self.significant(start, open);
        let mut position = 1;
        if declaration {
            self.required_name(&significant, 1, "a declaration requires a name");
            position = 2;
        } else if significant.get(1).is_some_and(|index| {
            self.at(*index, FeaTokenKind::Identifier) || self.at(*index, FeaTokenKind::String)
        }) && !CLAUSE_WORDS.contains(&self.lexeme(significant[1]))
        {
            self.tree
                .node(NodeKind::Name, significant[1], significant[1] + 1);
            position = 2;
        }
        if let Some(range) =
            self.trimmed_range(significant.get(position).copied().unwrap_or(open), open)
        {
            self.tree.node(NodeKind::Header, range.start, range.end);
        }
        self.parse_header_tail(&significant[position.min(significant.len())..]);
    }

    fn parse_header_tail(&mut self, tail: &[usize]) {
        let mut cursor = 0;
        if tail
            .first()
            .is_some_and(|index| self.at(*index, FeaTokenKind::Colon))
        {
            let end = self.next_clause(tail, 1);
            if end > 1 {
                self.tree.node_over(NodeKind::Type, &tail[1..end]);
            } else {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                    span: self.token(tail[0]).span,
                    message: "expected a type after `:`".into(),
                });
            }
            cursor = end;
        }
        while cursor < tail.len() {
            let word = tail[cursor];
            if !CLAUSE_WORDS.contains(&self.lexeme(word)) && !is_name_token_at(self, word) {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                    span: self.token(word).span,
                    message: "expected a clause such as `on <reference>` in the header".into(),
                });
                return;
            }
            let end = self.next_clause(tail, cursor + 1);
            let value = &tail[cursor + 1..end];
            if value.is_empty() {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                    span: self.token(word).span,
                    message: format!("expected a value after `{}`", self.lexeme(word)),
                });
            } else {
                self.design_expression(value, self.token(word).span);
            }
            self.tree.node_over(NodeKind::Clause, &tail[cursor..end]);
            cursor = end;
        }
    }

    /// The index in `tail` of the next clause word at nesting depth zero, or its length.
    fn next_clause(&self, tail: &[usize], from: usize) -> usize {
        let mut depth = 0usize;
        for (position, index) in tail.iter().enumerate().skip(from) {
            match self.kind(*index) {
                Some(
                    FeaTokenKind::LeftParen | FeaTokenKind::LeftBracket | FeaTokenKind::LeftBrace,
                ) => depth += 1,
                Some(
                    FeaTokenKind::RightParen
                    | FeaTokenKind::RightBracket
                    | FeaTokenKind::RightBrace,
                ) => depth = depth.saturating_sub(1),
                _ if depth == 0
                    && position > from
                    && CLAUSE_WORDS.contains(&self.lexeme(*index))
                    && self.lexeme(*index) != "as" =>
                {
                    return position;
                }
                _ => {}
            }
        }
        tail.len()
    }

    fn parse_design_statement(&mut self, start: usize, end: usize) {
        let Some(range) = self.trimmed_range(start, end) else {
            return;
        };
        let significant = self.significant(range.start, range.end);
        let first = significant[0];
        let word = self.lexeme(first).to_string();
        let span = range.span;
        let relation = self.top_level_relation(&significant);
        let kind = match word.as_str() {
            w if w == GenericForm::Use.word() => {
                self.parse_use_declaration(range.start, range.end, None);
                return;
            }
            w if w == GenericForm::Constrain.word() => {
                let rest = &significant[1..];
                if rest.is_empty() {
                    self.missing_expression(span, "`constrain` requires an expression");
                } else {
                    let remaining = self.design_expression_prefix(rest);
                    if remaining < rest.len() {
                        if self.at(rest[remaining], FeaTokenKind::Equal) {
                            self.design_expression(&rest[remaining + 1..], span);
                        } else {
                            self.invalid_after_expression(rest[remaining]);
                        }
                    }
                }
                NodeKind::Constrain
            }
            w if (w == GenericForm::Publish.word() || w == GenericForm::Check.word())
                && significant.get(1).is_some_and(|index| {
                    is_name_token_at(self, *index) || self.at(*index, FeaTokenKind::String)
                }) =>
            {
                self.tree
                    .node(NodeKind::Name, significant[1], significant[1] + 1);
                match significant.get(2) {
                    Some(equal) if self.at(*equal, FeaTokenKind::Equal) => {
                        self.design_expression(&significant[3..], span);
                    }
                    Some(other) => self.invalid_after_expression(*other),
                    None => {}
                }
                if w == GenericForm::Publish.word() {
                    NodeKind::Publish
                } else {
                    NodeKind::Check
                }
            }
            w if w == GenericForm::Link.word() => {
                self.required_name(&significant, 1, "`link` requires a kind");
                let to = significant
                    .iter()
                    .position(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::To)));
                match to {
                    Some(to) if to > 2 => {
                        self.design_expression(&significant[2..to], span);
                        self.design_expression(&significant[to + 1..], span);
                    }
                    _ => {
                        self.diagnostics.push(FeaSyntaxDiagnostic {
                            kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                            span,
                            message: "a link is written `link <kind> <from> to <target>`".into(),
                        });
                    }
                }
                NodeKind::Link
            }
            _ if self.is_value_statement(&significant) => {
                self.tree
                    .node(NodeKind::Name, significant[1], significant[1] + 1);
                let equal = significant
                    .iter()
                    .position(|index| self.at(*index, FeaTokenKind::Equal));
                if significant
                    .get(2)
                    .is_some_and(|index| self.at(*index, FeaTokenKind::Colon))
                {
                    let type_end = equal.unwrap_or(significant.len());
                    if type_end > 3 {
                        self.tree
                            .node_over(NodeKind::Type, &significant[3..type_end]);
                    } else {
                        self.diagnostics.push(FeaSyntaxDiagnostic {
                            kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                            span: self.token(significant[2]).span,
                            message: "expected a type after `:`".into(),
                        });
                    }
                } else if let Some(extra) = significant.get(2).filter(|_| equal != Some(2)) {
                    self.invalid_after_expression(*extra);
                }
                if let Some(equal) = equal {
                    self.design_expression(&significant[equal + 1..], span);
                }
                NodeKind::ValueStatement
            }
            _ if relation.is_some() => {
                let position = relation.unwrap();
                if self
                    .parse_reference_indices(&significant[..position])
                    .is_none()
                {
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                        span: self.token(significant[0]).span,
                        message: "an assignment target is a name or a dotted path".into(),
                    });
                }
                self.design_expression(&significant[position + 1..], span);
                NodeKind::Assignment
            }
            _ => {
                if significant.len() > 1 && is_name_token_at(self, first) {
                    self.design_expression(&significant[1..], span);
                }
                NodeKind::Command
            }
        };
        self.tree.node(kind, range.start, range.end);
    }

    /// `<word> <identifier> (: … | = … | end)`, or any `let`/`input`/`property` statement.
    fn is_value_statement(&self, significant: &[usize]) -> bool {
        let [first, second, rest @ ..] = significant else {
            return false;
        };
        // A name that is not a word is quoted (`sheet "1" = { … }`). A word the FEA dialect
        // reserves (`signal`, `table`, `load`) is still a name here (`net_class signal = { … }`).
        let named = is_name_token_at(self, *second)
            || (self.at(*second, FeaTokenKind::String)
                && rest.first().is_some_and(|index| {
                    self.at(*index, FeaTokenKind::Colon) || self.at(*index, FeaTokenKind::Equal)
                }));
        if !is_name_token_at(self, *first) || !named {
            return false;
        }
        if VALUE_WORDS.contains(&self.lexeme(*first)) {
            return true;
        }
        rest.first().is_some_and(|index| {
            self.at(*index, FeaTokenKind::Colon) || self.at(*index, FeaTokenKind::Equal)
        })
    }

    fn top_level_relation(&self, significant: &[usize]) -> Option<usize> {
        let mut depth = 0usize;
        for (position, index) in significant.iter().enumerate() {
            match self.kind(*index)? {
                FeaTokenKind::LeftParen | FeaTokenKind::LeftBracket | FeaTokenKind::LeftBrace => {
                    depth += 1
                }
                FeaTokenKind::RightParen
                | FeaTokenKind::RightBracket
                | FeaTokenKind::RightBrace => depth = depth.saturating_sub(1),
                FeaTokenKind::Equal | FeaTokenKind::LessEqual | FeaTokenKind::GreaterEqual
                    if depth == 0 =>
                {
                    return Some(position);
                }
                _ => {}
            }
        }
        None
    }

    /// `use` in its forms; `body_end` is set when overrides follow in a block.
    fn parse_use_declaration(&mut self, start: usize, end: usize, body_end: Option<usize>) {
        let significant = self.significant(start, end);
        let span = self.range_span(start, end);
        let mut cursor = 1;
        match significant.get(1).map(|index| self.kind(*index)) {
            Some(Some(FeaTokenKind::String)) => {
                self.tree
                    .node(NodeKind::StringLiteral, significant[1], significant[1] + 1);
                cursor = 2;
            }
            Some(_)
                if significant.get(3).is_some_and(|index| {
                    self.at(*index, FeaTokenKind::Keyword(FeaKeyword::From))
                }) =>
            {
                self.required_name(&significant, 1, "`use` requires a kind");
                self.required_name(&significant, 2, "`use` requires a name");
                cursor = 4;
                match significant.get(4).map(|index| self.kind(*index)) {
                    Some(Some(FeaTokenKind::String)) => {
                        self.tree
                            .node(NodeKind::StringLiteral, significant[4], significant[4] + 1);
                        cursor = 5;
                    }
                    Some(_) => {
                        let path_end = significant[4..]
                            .iter()
                            .position(|index| {
                                self.at(*index, FeaTokenKind::At) || self.lexeme(*index) == "as"
                            })
                            .map_or(significant.len(), |offset| offset + 4);
                        if self
                            .parse_reference_indices(&significant[4..path_end])
                            .is_none()
                        {
                            self.diagnostics.push(FeaSyntaxDiagnostic {
                                kind: FeaSyntaxDiagnosticKind::ExpectedString,
                                span: self.token(significant[4]).span,
                                message: "expected a quoted folder or a package path after `from`"
                                    .into(),
                            });
                        }
                        cursor = path_end;
                    }
                    None => {
                        self.diagnostics.push(FeaSyntaxDiagnostic {
                            kind: FeaSyntaxDiagnosticKind::ExpectedString,
                            span,
                            message: "expected a source after `from`".into(),
                        });
                    }
                }
            }
            Some(_) => {
                let path_end = significant
                    .iter()
                    .position(|index| self.lexeme(*index) == "as")
                    .unwrap_or(significant.len());
                if self
                    .parse_reference_indices(&significant[1..path_end])
                    .is_none()
                {
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                        span,
                        message: "`use` takes `<kind> <name> from <source>`, a quoted folder, or a folder path"
                            .into(),
                    });
                }
                cursor = path_end;
            }
            None => {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                    span,
                    message: "`use` requires a source".into(),
                });
            }
        }
        if significant
            .get(cursor)
            .is_some_and(|index| self.at(*index, FeaTokenKind::At))
        {
            match significant.get(cursor + 1) {
                Some(pin) if self.at(*pin, FeaTokenKind::String) => {
                    self.tree.node(NodeKind::Version, *pin, *pin + 1);
                    cursor += 2;
                }
                _ => {
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::ExpectedString,
                        span: self.token(significant[cursor]).span,
                        message: "expected a quoted revision after `@`".into(),
                    });
                    cursor += 1;
                }
            }
        }
        if significant
            .get(cursor)
            .is_some_and(|index| self.lexeme(*index) == "as")
        {
            self.required_name(&significant, cursor + 1, "expected a name after `as`");
            cursor += 2;
        }
        if let Some(extra) = significant.get(cursor) {
            self.invalid_after_expression(*extra);
        }
        match body_end {
            Some(next) => self.tree.node(NodeKind::UseDeclaration, start, next),
            None => self.tree.node_over(NodeKind::UseDeclaration, &significant),
        }
    }

    fn required_name(&mut self, significant: &[usize], position: usize, message: &str) {
        match significant.get(position) {
            // A name that is not a word is quoted (`circuit "sensor-node" { … }`).
            Some(index)
                if is_name_token_at(self, *index) || self.at(*index, FeaTokenKind::String) =>
            {
                self.tree.node(NodeKind::Name, *index, *index + 1);
            }
            Some(index) => {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedIdentifier,
                    span: self.token(*index).span,
                    message: message.into(),
                });
            }
            None => {
                let span = significant
                    .last()
                    .map_or(Span::default(), |index| self.token(*index).span);
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedIdentifier,
                    span,
                    message: message.into(),
                });
            }
        }
    }

    /// A whole expression over `indices`; trailing tokens are a diagnostic.
    fn design_expression(&mut self, indices: &[usize], fallback: Span) {
        if indices.is_empty() {
            self.missing_expression(fallback, "expected a value");
            return;
        }
        let remaining = self.design_expression_prefix(indices);
        if remaining < indices.len() {
            self.invalid_after_expression(indices[remaining]);
        }
    }

    /// An expression from the start of `indices`; returns how many tokens it used.
    fn design_expression_prefix(&mut self, indices: &[usize]) -> usize {
        let (_, remaining) = expression::parse_expression(
            &self.stream,
            indices,
            &mut self.diagnostics,
            &mut self.tree,
            ExpressionMode::Design,
        );
        remaining
    }

    fn invalid_after_expression(&mut self, index: usize) {
        self.diagnostics.push(FeaSyntaxDiagnostic {
            kind: FeaSyntaxDiagnosticKind::InvalidExpression,
            span: self.token(index).span,
            message: "unexpected token here".into(),
        });
    }
}

fn is_name_token_at(parser: &FeaParser, index: usize) -> bool {
    parser.kind(index).is_some_and(is_name_token)
}

fn quoted_list(words: &[&str]) -> String {
    words
        .iter()
        .map(|word| format!("`{word}`"))
        .collect::<Vec<_>>()
        .join(", ")
}
