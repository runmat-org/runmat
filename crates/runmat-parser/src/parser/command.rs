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
}

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
            allowed: &["on", "off"],
            optional: false,
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
            allowed: &["auto", "manual", "tight", "equal", "ij", "xy"],
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
        name: "close",
        arg_kind: CommandArgKind::StringifyWords,
    },
    CommandVerb {
        name: "clear",
        arg_kind: CommandArgKind::StringifyWords,
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
        let mut args = self.parse_command_args();
        if let Some(command) = self.lookup_command(&name_token.lexeme) {
            self.normalize_command_args(command, &mut args[..])?;
        }
        let end = self.last_token_end();
        let span = self.span_from(name_token.position, end);
        Ok(crate::Stmt::ExprStmt(
            Expr::FuncCall(name_token.lexeme, args, span),
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
        );

        let mut i = 1;
        let mut saw_arg = false;
        self.skip_command_continuations(&mut i);

        if !matches!(
            self.peek_token_at(i),
            Some(Token::Ident | Token::Integer | Token::Float | Token::Str | Token::End)
        ) {
            if !zero_arg_allowed {
                return false;
            }
        } else {
            saw_arg = true;
        }

        loop {
            match self.peek_token_at(i) {
                Some(Token::Ident | Token::Integer | Token::Float | Token::Str | Token::End) => {
                    saw_arg = true;
                    i += 1;
                }
                Some(Token::Ellipsis) => self.skip_command_continuations(&mut i),
                _ => break,
            }
        }
        if !saw_arg && !zero_arg_allowed {
            return false;
        }

        match self.peek_token_at(i) {
            Some(Token::LParen)
            | Some(Token::Dot)
            | Some(Token::LBracket)
            | Some(Token::LBrace)
            | Some(Token::Transpose) => false,
            Some(Token::Assign) => false,
            None | Some(Token::Semicolon) | Some(Token::Comma) | Some(Token::Newline) => true,
            _ => true,
        }
    }

    fn parse_command_args(&mut self) -> Vec<Expr> {
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
            match self.peek_token() {
                Some(Token::Ident) => {
                    let token = self.next().unwrap();
                    let span = self.span_from(token.position, token.end);
                    args.push(Expr::Ident(token.lexeme, span));
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
                Some(Token::Slash)
                | Some(Token::Star)
                | Some(Token::Backslash)
                | Some(Token::Plus)
                | Some(Token::Minus)
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
            CommandArgKind::StringifyWords => {
                for arg in args {
                    let span = arg.span();
                    match arg {
                        Expr::Ident(word, _) => {
                            *arg = Expr::String(format!("\"{}\"", word), span);
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

fn extract_keyword(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(s, _) => Some(s.clone()),
        Expr::String(s, _) => Some(s.trim_matches(&['"', '\''][..]).to_string()),
        _ => None,
    }
}
