use super::*;

impl<'a> TextScanner<'a> {
    pub(super) async fn parse_conversion(
        &mut self,
        item: &FormatItem,
        next_literal: Option<&str>,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<Option<ParsedValue>> {
        if !matches!(&item.kind, FormatKind::Char | FormatKind::CharSet { .. }) {
            self.skip_separators(cooperation).await?;
        }
        let parsed = match &item.kind {
            FormatKind::Float(dtype) => ParsedValue::Number(
                self.parse_numeric_field(
                    item.width,
                    next_literal,
                    *dtype,
                    parse_float_scalar,
                    cooperation,
                )
                .await?,
            ),
            FormatKind::SignedInt(dtype) => ParsedValue::Number(
                self.parse_numeric_field(
                    item.width,
                    next_literal,
                    *dtype,
                    parse_signed_scalar,
                    cooperation,
                )
                .await?,
            ),
            FormatKind::UnsignedInt(dtype) => ParsedValue::Number(
                self.parse_numeric_field(
                    item.width,
                    next_literal,
                    *dtype,
                    parse_unsigned_scalar,
                    cooperation,
                )
                .await?,
            ),
            FormatKind::RadixInt { dtype, radix } => {
                let radix = *radix;
                ParsedValue::Number(
                    self.parse_numeric_field(
                        item.width,
                        next_literal,
                        *dtype,
                        |field, dtype| parse_radix_scalar(field, dtype, radix),
                        cooperation,
                    )
                    .await?,
                )
            }
            FormatKind::String => ParsedValue::Text(
                self.read_field(item.width, next_literal, true, cooperation)
                    .await?,
            ),
            FormatKind::QuotedString => ParsedValue::Text(
                self.read_quoted_or_field(item.width, next_literal, cooperation)
                    .await?,
            ),
            FormatKind::Char => ParsedValue::Text(
                self.read_chars(item.width.unwrap_or(1), cooperation)
                    .await?,
            ),
            FormatKind::CharSet { chars, negated } => ParsedValue::Text(
                self.read_charset(chars, *negated, item.width, cooperation)
                    .await?,
            ),
        };
        if item.skip {
            Ok(None)
        } else {
            Ok(Some(parsed))
        }
    }

    async fn parse_numeric_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        dtype: NumericDType,
        parse: impl FnOnce(&str, NumericDType) -> BuiltinResult<NumericScalar>,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<NumericScalar> {
        let field = self
            .read_field(width, next_literal, false, cooperation)
            .await?;
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

    async fn read_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        allow_treat_empty: bool,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<String> {
        let start = self.pos;
        let mut chars = 0usize;
        let mut work = 0usize;
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
            checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
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

    async fn read_quoted_or_field(
        &mut self,
        width: Option<usize>,
        next_literal: Option<&str>,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<String> {
        if self.current_char() != Some('"') {
            return self
                .read_field(width, next_literal, true, cooperation)
                .await;
        }
        self.pos += 1;
        let mut out = String::new();
        let mut chars = 0usize;
        let mut work = 0usize;
        while let Some(ch) = self.current_char() {
            self.pos += ch.len_utf8();
            checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
            if ch == '"' {
                if self.current_char() == Some('"') {
                    self.pos += 1;
                    out.push('"');
                    chars += 1;
                    continue;
                }
                return Ok(out);
            }
            if width.map(|limit| chars >= limit).unwrap_or(false) {
                return Ok(out);
            }
            out.push(ch);
            chars += 1;
        }
        Err(textscan_error_with(
            &TEXTSCAN_ERROR_PARSE,
            "textscan: unterminated quoted field",
        ))
    }

    async fn read_chars(
        &mut self,
        count: usize,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<String> {
        let mut out = String::new();
        let mut work = 0usize;
        for _ in 0..count {
            let Some(ch) = self.current_char() else {
                return Err(textscan_error_with(
                    &TEXTSCAN_ERROR_PARSE,
                    "textscan: not enough characters for %c conversion",
                ));
            };
            self.pos += ch.len_utf8();
            out.push(ch);
            checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
        }
        Ok(out)
    }

    async fn read_charset(
        &mut self,
        chars: &HashSet<char>,
        negated: bool,
        width: Option<usize>,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<String> {
        let mut out = String::new();
        let mut chars_read = 0usize;
        let mut work = 0usize;
        while let Some(ch) = self.current_char() {
            if width.map(|limit| chars_read >= limit).unwrap_or(false) {
                break;
            }
            if !(chars.contains(&ch) ^ negated) {
                break;
            }
            self.pos += ch.len_utf8();
            out.push(ch);
            chars_read += 1;
            checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
        }
        if out.is_empty() {
            return Err(textscan_error_with(
                &TEXTSCAN_ERROR_PARSE,
                "textscan: character set conversion matched no characters",
            ));
        }
        Ok(out)
    }
}
