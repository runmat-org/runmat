use runmat_lexer::Token;

use crate::{CompatMode, Expr, SyntaxError};

use super::Parser;

#[derive(Clone, Copy)]
struct CommandVerb {
    name: &'static str,
    arg_kind: CommandArgKind,
}

#[derive(Clone, Copy)]
enum CommandArgKind {
    Keyword {
        allowed: &'static [&'static str],
        optional: bool,
    },
    Any,
    StringifyWords,
    PathWords {
        optional: bool,
    },
}

const REQUIRED_PATH_WORDS: CommandArgKind = CommandArgKind::PathWords { optional: false };
const OPTIONAL_PATH_WORDS: CommandArgKind = CommandArgKind::PathWords { optional: true };

const COMMAND_VERBS: &[CommandVerb] = &[
    CommandVerb {
        name: "hold",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off", "all", "reset"],
            optional: false,
        },
    },
    CommandVerb {
        name: "grid",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off", "minor"],
            optional: true,
        },
    },
    CommandVerb {
        name: "box",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off"],
            optional: false,
        },
    },
    CommandVerb {
        name: "axis",
        arg_kind: CommandArgKind::Keyword {
            allowed: &[
                "auto", "manual", "tight", "equal", "image", "ij", "xy", "on", "off",
            ],
            optional: false,
        },
    },
    CommandVerb {
        name: "shading",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["flat", "interp", "faceted"],
            optional: false,
        },
    },
    CommandVerb {
        name: "colormap",
        arg_kind: CommandArgKind::Keyword {
            allowed: &[
                "parula", "jet", "hsv", "hot", "cool", "spring", "summer", "autumn", "winter",
                "gray", "bone", "copper", "pink",
            ],
            optional: false,
        },
    },
    CommandVerb {
        name: "colorbar",
        arg_kind: CommandArgKind::Keyword {
            allowed: &["on", "off"],
            optional: true,
        },
    },
    CommandVerb {
        name: "figure",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "subplot",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "clf",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "cla",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "clc",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "pause",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "drawnow",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "syms",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "close",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "clear",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "clearvars",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "warning",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "print",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "addpath",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "cd",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "copyfile",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "delete",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "dir",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "exist",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "fullfile",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "genpath",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "getenv",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "load",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "ls",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "mkdir",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "movefile",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "path",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "pwd",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "rmdir",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "rmpath",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "run",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "save",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "savepath",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "setenv",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "tempdir",
        arg_kind: CommandArgKind::Any,
    },
    CommandVerb {
        name: "tempname",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "uigetfile",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "uiputfile",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "which",
        arg_kind: REQUIRED_PATH_WORDS,
    },
    CommandVerb {
        name: "who",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "whos",
        arg_kind: OPTIONAL_PATH_WORDS,
    },
    CommandVerb {
        name: "format",
        arg_kind: CommandArgKind::Keyword {
            allowed: &[
                "short", "long", "shortE", "longE", "shortG", "longG", "rat", "rational", "hex",
                "compact", "loose",
            ],
            optional: true,
        },
    },
];

impl Parser {
    pub(super) fn parse_command_stmt(&mut self) -> Result<crate::Stmt, SyntaxError> {
        if self.options.compat_mode == CompatMode::Strict {
            return Err(self.error(
                "Command syntax is disabled in strict compatibility mode; call functions with parentheses.",
            ));
        }
        let name_token = self.next().unwrap();
        let mut args = self.parse_command_args(&name_token.lexeme);
        if let Some(command) = self.lookup_command(&name_token.lexeme) {
            self.normalize_command_args(command, &mut args[..])?;
        }
        let end = self.last_token_end();
        let span = self.span_from(name_token.position, end);
        Ok(crate::Stmt::ExprStmt(
            Expr::CommandCall(name_token.lexeme, args, span),
            false,
            span,
        ))
    }

    pub(super) fn can_start_command_form(&self) -> bool {
        let Some(current) = self.tokens.get(self.pos) else {
            return false;
        };
        let verb = current.lexeme.as_str();
        let command = self.lookup_command(verb);
        let zero_arg_allowed = matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::Any,
                ..
            })
        ) || matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::Keyword { optional: true, .. },
                ..
            })
        ) || matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::StringifyWords,
                ..
            })
        ) || matches!(
            command,
            Some(CommandVerb {
                arg_kind: CommandArgKind::PathWords { optional: true },
                ..
            })
        );

        let mut i = 1;
        let mut saw_arg = false;
        self.skip_command_continuations(&mut i);
        if !self.command_arg_has_required_separator(i) {
            return false;
        }

        if self.has_malformed_syms_parameter_suffix(verb, i) {
            return false;
        }
        if self.try_skip_command_arg(verb, &mut i) {
            saw_arg = true;
        } else if !matches!(self.peek_token_at(i), Some(Token::Ellipsis)) && !zero_arg_allowed {
            return false;
        }

        loop {
            if self.has_malformed_syms_parameter_suffix(verb, i) {
                return false;
            }
            if self.try_skip_command_arg(verb, &mut i) {
                saw_arg = true;
            } else if matches!(self.peek_token_at(i), Some(Token::Ellipsis)) {
                self.skip_command_continuations(&mut i);
            } else {
                break;
            }
        }
        if !saw_arg {
            return zero_arg_allowed
                && matches!(
                    self.peek_token_at(i),
                    None | Some(Token::Semicolon) | Some(Token::Comma) | Some(Token::Newline)
                );
        }

        matches!(
            self.peek_token_at(i),
            None | Some(Token::Semicolon) | Some(Token::Comma) | Some(Token::Newline)
        )
    }

    fn parse_command_args(&mut self, verb: &str) -> Vec<Expr> {
        let mut args = Vec::new();
        loop {
            if matches!(self.peek_token(), Some(Token::Newline)) {
                break;
            }
            if self.consume(&Token::Ellipsis) {
                // `...` is a line-continuation; skip all following newlines and keep parsing.
                while self.consume(&Token::Newline) {}
                continue;
            }
            if self.can_parse_path_word_arg(verb) {
                if let Some(arg) = self.parse_path_word_arg() {
                    args.push(arg);
                    continue;
                }
            }
            match self.peek_token() {
                Some(Token::Ident) => {
                    let token = self.next().unwrap();
                    let start = token.position;
                    let mut end = token.end;
                    let mut word = token.lexeme;
                    if self.can_parse_syms_function_arg(verb)
                        && matches!(self.peek_token(), Some(Token::LParen))
                    {
                        if self.skip_syms_parameter_suffix(0).is_none() {
                            break;
                        }
                        if let Some((suffix, suffix_end)) = self.parse_syms_parameter_suffix() {
                            word.push_str(&suffix);
                            end = suffix_end;
                        }
                    }
                    if self.can_parse_dotted_word_arg(verb) {
                        while matches!(self.peek_token(), Some(Token::Dot))
                            && matches!(self.peek_token_at(1), Some(Token::Ident | Token::Integer))
                        {
                            self.next();
                            let Some(part) = self.next() else {
                                break;
                            };
                            word.push('.');
                            word.push_str(&part.lexeme);
                            end = part.end;
                        }
                    }
                    let span = self.span_from(start, end);
                    args.push(Expr::Ident(word, span));
                }
                // In command-form, accept 'end' as a literal identifier token for compatibility.
                Some(Token::End) => {
                    let token = &self.tokens[self.pos];
                    self.pos += 1;
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Ident("end".to_string(), span));
                }
                Some(Token::Integer) | Some(Token::Float) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Number(token.lexeme, span));
                }
                Some(Token::Str) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::String(token.lexeme, span));
                }
                Some(Token::Minus)
                    if self.can_parse_dash_option_arg(verb)
                        && matches!(self.peek_token_at(1), Some(Token::Ident)) =>
                {
                    let minus = self.next().unwrap();
                    let Some(option) = self.next() else {
                        break;
                    };
                    let span = self.span_from(minus.position, option.end);
                    args.push(Expr::Ident(format!("-{}", option.lexeme), span));
                }
                Some(Token::Slash)
                | Some(Token::Star)
                | Some(Token::Backslash)
                | Some(Token::Plus)
                | Some(Token::LParen)
                | Some(Token::Dot)
                | Some(Token::LBracket)
                | Some(Token::LBrace)
                | Some(Token::Transpose) => break,
                _ => break,
            }
        }
        args
    }

    fn try_skip_command_arg(&self, verb: &str, offset: &mut usize) -> bool {
        if self.try_skip_path_word_arg(verb, offset) {
            return true;
        }
        if self.try_skip_dash_option(verb, offset) {
            return true;
        }
        match self.peek_token_at(*offset) {
            Some(Token::Ident) => {
                if self.can_parse_syms_function_arg(verb)
                    && matches!(self.peek_token_at(*offset + 1), Some(Token::LParen))
                {
                    let Some(after_suffix) = self.skip_syms_parameter_suffix(*offset + 1) else {
                        return false;
                    };
                    *offset = after_suffix;
                } else {
                    *offset += 1;
                    self.skip_dotted_word_suffix(verb, offset);
                }
                true
            }
            Some(Token::Integer | Token::Float | Token::Str | Token::End) => {
                *offset += 1;
                true
            }
            _ => false,
        }
    }

    fn try_skip_dash_option(&self, verb: &str, offset: &mut usize) -> bool {
        if !self.can_parse_dash_option_arg(verb) {
            return false;
        }
        if !matches!(self.peek_token_at(*offset), Some(Token::Minus))
            || !matches!(self.peek_token_at(*offset + 1), Some(Token::Ident))
            || !self.tokens_adjacent(self.pos + *offset, self.pos + *offset + 1)
        {
            return false;
        }
        *offset += 2;
        true
    }

    fn can_parse_dash_option_arg(&self, verb: &str) -> bool {
        verb.eq_ignore_ascii_case("clearvars")
            || verb.eq_ignore_ascii_case("print")
            || self.can_parse_path_word_arg(verb)
    }

    fn can_parse_dotted_word_arg(&self, verb: &str) -> bool {
        verb.eq_ignore_ascii_case("print")
    }

    fn can_parse_path_word_arg(&self, verb: &str) -> bool {
        matches!(
            self.lookup_command(verb),
            Some(CommandVerb {
                arg_kind: CommandArgKind::PathWords { .. },
                ..
            })
        )
    }

    fn parse_path_word_arg(&mut self) -> Option<Expr> {
        let start_index = self.pos;
        if !self.can_start_path_word_arg_at(start_index) {
            return None;
        }
        let start = self.tokens.get(start_index)?.position;
        let mut text = String::new();
        let mut end = start;

        loop {
            let index = self.pos;
            let Some(token) = self.tokens.get(index) else {
                break;
            };
            if index > start_index && !self.tokens_adjacent(index - 1, index) {
                break;
            }
            if !is_path_word_token(&token.token) {
                break;
            }
            text.push_str(&token.lexeme);
            end = token.end;
            self.pos += 1;
        }

        if self.pos == start_index {
            None
        } else {
            Some(Expr::String(
                quote_command_string(&text),
                self.span_from(start, end),
            ))
        }
    }

    fn try_skip_path_word_arg(&self, verb: &str, offset: &mut usize) -> bool {
        if !self.can_parse_path_word_arg(verb) {
            return false;
        }

        let start = self.pos + *offset;
        if !self.can_start_path_word_arg_at(start) {
            return false;
        }
        let mut index = start;
        while let Some(token) = self.tokens.get(index) {
            if index > start && !self.tokens_adjacent(index - 1, index) {
                break;
            }
            if !is_path_word_token(&token.token) {
                break;
            }
            index += 1;
        }

        if index == start {
            false
        } else {
            *offset += index - start;
            true
        }
    }

    fn command_arg_has_required_separator(&self, offset: usize) -> bool {
        match self.peek_token_at(offset) {
            None | Some(Token::Semicolon) | Some(Token::Comma) | Some(Token::Newline) => true,
            _ => !self.tokens_adjacent(self.pos, self.pos + offset),
        }
    }

    fn can_start_path_word_arg_at(&self, index: usize) -> bool {
        let Some(token) = self.tokens.get(index) else {
            return false;
        };
        match token.token {
            Token::Plus | Token::Minus | Token::At => self
                .tokens
                .get(index + 1)
                .is_some_and(|_| self.tokens_adjacent(index, index + 1)),
            Token::DotStar | Token::DotSlash | Token::DotBackslash => true,
            _ => is_path_word_token(&token.token),
        }
    }

    fn can_parse_syms_function_arg(&self, verb: &str) -> bool {
        verb.eq_ignore_ascii_case("syms")
    }

    fn has_malformed_syms_parameter_suffix(&self, verb: &str, offset: usize) -> bool {
        self.can_parse_syms_function_arg(verb)
            && matches!(self.peek_token_at(offset), Some(Token::Ident))
            && matches!(self.peek_token_at(offset + 1), Some(Token::LParen))
            && self.skip_syms_parameter_suffix(offset + 1).is_none()
    }

    fn skip_syms_parameter_suffix(&self, offset: usize) -> Option<usize> {
        if !matches!(self.peek_token_at(offset), Some(Token::LParen)) {
            return None;
        }
        let mut i = offset + 1;
        loop {
            match self.peek_token_at(i) {
                Some(Token::Ident) => {
                    i += 1;
                    match self.peek_token_at(i) {
                        Some(Token::Comma) => i += 1,
                        Some(Token::RParen) => return Some(i + 1),
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }

    fn parse_syms_parameter_suffix(&mut self) -> Option<(String, usize)> {
        self.skip_syms_parameter_suffix(0)?;
        let open = self.next()?;
        if !matches!(open.token, Token::LParen) {
            return None;
        }
        let mut suffix = String::from("(");
        loop {
            let parameter = self.next()?;
            if !matches!(parameter.token, Token::Ident) {
                return None;
            }
            suffix.push_str(&parameter.lexeme);

            if self.consume(&Token::Comma) {
                suffix.push(',');
                continue;
            }

            let close = self.next()?;
            if !matches!(close.token, Token::RParen) {
                return None;
            }
            suffix.push(')');
            return Some((suffix, close.end));
        }
    }

    fn skip_dotted_word_suffix(&self, verb: &str, offset: &mut usize) {
        if !self.can_parse_dotted_word_arg(verb) {
            return;
        }
        while matches!(self.peek_token_at(*offset), Some(Token::Dot))
            && matches!(
                self.peek_token_at(*offset + 1),
                Some(Token::Ident | Token::Integer)
            )
        {
            *offset += 2;
        }
    }

    fn skip_command_continuations(&self, offset: &mut usize) {
        while matches!(self.peek_token_at(*offset), Some(Token::Ellipsis)) {
            *offset += 1;
            while matches!(self.peek_token_at(*offset), Some(Token::Newline)) {
                *offset += 1;
            }
        }
    }

    fn lookup_command(&self, name: &str) -> Option<&'static CommandVerb> {
        COMMAND_VERBS
            .iter()
            .find(|cmd| cmd.name.eq_ignore_ascii_case(name))
    }

    fn normalize_command_args(
        &self,
        command: &CommandVerb,
        args: &mut [Expr],
    ) -> Result<(), SyntaxError> {
        match command.arg_kind {
            CommandArgKind::Keyword { allowed, optional } => {
                if args.is_empty() {
                    if optional {
                        return Ok(());
                    }
                    return Err(self.error(&format!(
                        "'{}' command syntax requires an argument",
                        command.name
                    )));
                }
                if args.len() > 1 {
                    return Err(self.error(&format!(
                        "'{}' command syntax accepts only one argument",
                        command.name
                    )));
                }
                let keyword = extract_keyword(&args[0]).ok_or_else(|| {
                    self.error(&format!(
                        "'{}' command syntax expects a keyword argument",
                        command.name
                    ))
                })?;
                if allowed
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(&keyword))
                {
                    let span = args[0].span();
                    args[0] = Expr::String(format!("\"{}\"", keyword), span);
                } else {
                    return Err(self.error(&format!(
                        "'{}' command syntax does not support '{}'",
                        command.name, keyword
                    )));
                }
            }
            CommandArgKind::Any => {}
            CommandArgKind::StringifyWords | CommandArgKind::PathWords { .. } => {
                for arg in args {
                    let span = arg.span();
                    match arg {
                        Expr::Ident(word, _) => {
                            *arg = Expr::String(quote_command_string(word), span);
                        }
                        Expr::EndKeyword(_) => {
                            *arg = Expr::String("\"end\"".to_string(), span);
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

fn is_path_word_token(token: &Token) -> bool {
    matches!(
        token,
        Token::Ident
            | Token::Integer
            | Token::Float
            | Token::End
            | Token::Dot
            | Token::DotStar
            | Token::DotSlash
            | Token::DotBackslash
            | Token::Slash
            | Token::Backslash
            | Token::Colon
            | Token::Tilde
            | Token::At
            | Token::Plus
            | Token::Minus
            | Token::Star
            | Token::Question
    )
}

fn quote_command_string(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
}

fn extract_keyword(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(s, _) => Some(s.clone()),
        Expr::String(s, _) => Some(s.trim_matches(&['"', '\''][..]).to_string()),
        _ => None,
    }
}
