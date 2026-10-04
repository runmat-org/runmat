use super::*;

pub(super) async fn parse_textscan(
    text: &str,
    elements: &[FormatElement],
    repeat: Option<usize>,
    options: &TextscanOptions,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<ParsedTextscan> {
    let output_kinds: Vec<OutputKind> = elements
        .iter()
        .filter_map(|element| match element {
            FormatElement::Conversion(item) if !item.skip => Some(item.kind.output_kind()),
            _ => None,
        })
        .collect();
    let mut columns: Vec<ColumnData> = output_kinds.into_iter().map(ColumnData::new).collect();
    let mut scanner = TextScanner::new(text, options);
    scanner.skip_header_lines();
    let mut records = 0usize;
    while !scanner.is_eof() && repeat.map(|limit| records < limit).unwrap_or(true) {
        cooperation.completed(0).await?;
        scanner.skip_separators();
        if scanner.is_eof() {
            break;
        }
        let row_len = columns.first().map(ColumnData::len).unwrap_or(0);
        let mut output_idx = 0usize;
        let row_start = scanner.pos;
        let mut failed = false;
        for idx in 0..elements.len() {
            cooperation.completed(1).await?;
            match &elements[idx] {
                FormatElement::Whitespace => scanner.skip_format_whitespace(),
                FormatElement::Literal(literal) => {
                    if let Err(err) = scanner.consume_literal(literal) {
                        if options.return_on_error {
                            failed = true;
                            break;
                        }
                        return Err(err);
                    }
                }
                FormatElement::Conversion(item) => {
                    let next_literal = next_literal(&elements[idx + 1..]);
                    let parsed = match scanner.parse_conversion(item, next_literal) {
                        Ok(parsed) => parsed,
                        Err(_) if options.return_on_error => {
                            failed = true;
                            break;
                        }
                        Err(err) => return Err(err),
                    };
                    let Some(parsed) = parsed else {
                        continue;
                    };
                    match parsed {
                        ParsedValue::Number(value) => {
                            if !item.skip {
                                columns[output_idx].push_numeric(value);
                                output_idx += 1;
                            }
                        }
                        ParsedValue::Text(value) => {
                            if !item.skip {
                                columns[output_idx].push_text(value);
                                output_idx += 1;
                            }
                        }
                    }
                }
            }
        }
        if failed || output_idx < columns.len() {
            for column in &mut columns {
                column.truncate(row_len);
            }
            break;
        }
        if scanner.pos == row_start {
            break;
        }
        records += 1;
        cooperation
            .completed(scanner.pos.saturating_sub(row_start))
            .await?;
        scanner.skip_separators();
    }
    Ok(ParsedTextscan {
        columns,
        consumed_text_pos: scanner.pos,
    })
}

#[derive(Debug, Clone)]
pub(super) struct ParsedTextscan {
    pub(super) columns: Vec<ColumnData>,
    pub(super) consumed_text_pos: usize,
}

fn next_literal(elements: &[FormatElement]) -> Option<&str> {
    if let Some(element) = elements.first() {
        return match element {
            FormatElement::Literal(literal) => Some(literal),
            FormatElement::Whitespace | FormatElement::Conversion(_) => None,
        };
    }
    None
}

struct TextScanner<'a> {
    text: &'a str,
    pos: usize,
    options: &'a TextscanOptions,
    whitespace: HashSet<char>,
    delimiters: Vec<String>,
}

impl<'a> TextScanner<'a> {
    fn new(text: &'a str, options: &'a TextscanOptions) -> Self {
        let mut delimiters = options.delimiters.clone();
        delimiters.push("\r\n".to_string());
        delimiters.push("\n".to_string());
        delimiters.push("\r".to_string());
        delimiters.sort_by_key(|delimiter| std::cmp::Reverse(delimiter.len()));
        Self {
            text,
            pos: 0,
            options,
            whitespace: options.whitespace.chars().collect(),
            delimiters,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.text.len()
    }

    fn current_char(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn skip_header_lines(&mut self) {
        for _ in 0..self.options.header_lines {
            if self.current_char().is_none() {
                break;
            }
            while let Some(ch) = self.current_char() {
                self.pos += ch.len_utf8();
                if ch == '\n' {
                    break;
                }
            }
        }
    }

    fn skip_format_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
        }
    }

    fn skip_separators(&mut self) {
        loop {
            if self.skip_comment() {
                continue;
            }
            if let Some(delimiter) = self.match_delimiter() {
                self.pos += delimiter.len();
                continue;
            }
            let Some(ch) = self.current_char() else {
                break;
            };
            if ch.is_whitespace() || self.whitespace.contains(&ch) {
                self.pos += ch.len_utf8();
                continue;
            }
            break;
        }
    }

    fn consume_literal(&mut self, literal: &str) -> BuiltinResult<()> {
        if self.text[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            return Ok(());
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            format!("textscan: expected literal '{literal}'"),
        ))
    }

    fn parse_conversion(
        &mut self,
        item: &FormatItem,
        next_literal: Option<&str>,
    ) -> BuiltinResult<Option<ParsedValue>> {
        if !matches!(&item.kind, FormatKind::Char | FormatKind::CharSet { .. }) {
            self.skip_separators();
        }
        let parsed = match &item.kind {
            FormatKind::Float(dtype) => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                *dtype,
                parse_float_scalar,
            )?),
            FormatKind::SignedInt(dtype) => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                *dtype,
                parse_signed_scalar,
            )?),
            FormatKind::UnsignedInt(dtype) => ParsedValue::Number(self.parse_numeric_field(
                item.width,
                next_literal,
                *dtype,
                parse_unsigned_scalar,
            )?),
            FormatKind::RadixInt { dtype, radix } => {
                let radix = *radix;
                ParsedValue::Number(self.parse_numeric_field(
                    item.width,
                    next_literal,
                    *dtype,
                    |field, dtype| parse_radix_scalar(field, dtype, radix),
                )?)
            }
            FormatKind::String => {
                ParsedValue::Text(self.read_field(item.width, next_literal, true)?)
            }
            FormatKind::QuotedString => {
                ParsedValue::Text(self.read_quoted_or_field(item.width, next_literal)?)
            }
            FormatKind::Char => ParsedValue::Text(self.read_chars(item.width.unwrap_or(1))?),
            FormatKind::CharSet { chars, negated } => {
                ParsedValue::Text(self.read_charset(chars, *negated, item.width)?)
            }
        };
        if item.skip {
            Ok(None)
        } else {
            Ok(Some(parsed))
        }
    }

    fn parse_numeric_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        dtype: NumericDType,
        parse: impl FnOnce(&str, NumericDType) -> BuiltinResult<NumericScalar>,
    ) -> BuiltinResult<NumericScalar> {
        let field = self.read_field(width, next_literal, false)?;
        if self
            .options
            .treat_as_empty
            .iter()
            .any(|empty| empty == &field)
        {
            return Ok(empty_numeric_scalar(dtype));
        }
        parse(&field, dtype)
    }

    fn read_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        allow_treat_empty: bool,
    ) -> BuiltinResult<String> {
        let start = self.pos;
        let mut chars = 0usize;
        while !self.is_eof() {
            if width.map(|limit| chars >= limit).unwrap_or(false) {
                break;
            }
            if next_literal
                .filter(|literal| self.text[self.pos..].starts_with(*literal))
                .is_some()
            {
                break;
            }
            if self.is_at_separator() || self.is_at_comment() {
                break;
            }
            let Some(ch) = self.current_char() else {
                break;
            };
            self.pos += ch.len_utf8();
            chars += 1;
        }
        let field = self.text[start..self.pos].trim().to_string();
        if allow_treat_empty
            && self
                .options
                .treat_as_empty
                .iter()
                .any(|empty| empty == &field)
        {
            return Ok(String::new());
        }
        if field.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: empty field",
            ));
        }
        Ok(field)
    }

    fn read_quoted_or_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
    ) -> BuiltinResult<String> {
        if self.current_char() != Some('"') {
            return self.read_field(width, next_literal, true);
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(ch) = self.current_char() {
            self.pos += ch.len_utf8();
            if ch == '"' {
                if self.current_char() == Some('"') {
                    self.pos += 1;
                    out.push('"');
                    continue;
                }
                return Ok(out);
            }
            if width
                .map(|limit| out.chars().count() >= limit)
                .unwrap_or(false)
            {
                return Ok(out);
            }
            out.push(ch);
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            "textscan: unterminated quoted field",
        ))
    }

    fn read_chars(&mut self, count: usize) -> BuiltinResult<String> {
        let mut out = String::new();
        for _ in 0..count {
            let Some(ch) = self.current_char() else {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_PARSE,
                    "textscan: not enough characters for %c conversion",
                ));
            };
            self.pos += ch.len_utf8();
            out.push(ch);
        }
        Ok(out)
    }

    fn read_charset(
        &mut self,
        chars: &HashSet<char>,
        negated: bool,
        width: Option<usize>,
    ) -> BuiltinResult<String> {
        let mut out = String::new();
        while let Some(ch) = self.current_char() {
            if width
                .map(|limit| out.chars().count() >= limit)
                .unwrap_or(false)
            {
                break;
            }
            if !(chars.contains(&ch) ^ negated) {
                break;
            }
            self.pos += ch.len_utf8();
            out.push(ch);
        }
        if out.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: character set conversion matched no characters",
            ));
        }
        Ok(out)
    }

    fn is_at_separator(&self) -> bool {
        self.match_delimiter().is_some()
            || self
                .current_char()
                .map(|ch| ch.is_whitespace() || self.whitespace.contains(&ch))
                .unwrap_or(false)
    }

    fn match_delimiter(&self) -> Option<&str> {
        self.delimiters
            .iter()
            .find(|delimiter| self.text[self.pos..].starts_with(delimiter.as_str()))
            .map(String::as_str)
    }

    fn skip_comment(&mut self) -> bool {
        match &self.options.comment_style {
            CommentStyle::None => false,
            CommentStyle::Line(markers) => {
                if markers
                    .iter()
                    .any(|marker| !marker.is_empty() && self.text[self.pos..].starts_with(marker))
                {
                    while let Some(ch) = self.current_char() {
                        self.pos += ch.len_utf8();
                        if ch == '\n' {
                            break;
                        }
                    }
                    true
                } else {
                    false
                }
            }
            CommentStyle::Block { start, end } => {
                if start.is_empty() || !self.text[self.pos..].starts_with(start) {
                    return false;
                }
                let after_start = self.pos + start.len();
                if let Some(end_idx) = self.text[after_start..].find(end) {
                    self.pos = after_start + end_idx + end.len();
                } else {
                    self.pos = self.text.len();
                }
                true
            }
        }
    }

    fn is_at_comment(&self) -> bool {
        match &self.options.comment_style {
            CommentStyle::None => false,
            CommentStyle::Line(markers) => markers
                .iter()
                .any(|marker| !marker.is_empty() && self.text[self.pos..].starts_with(marker)),
            CommentStyle::Block { start, .. } => {
                !start.is_empty() && self.text[self.pos..].starts_with(start)
            }
        }
    }
}
