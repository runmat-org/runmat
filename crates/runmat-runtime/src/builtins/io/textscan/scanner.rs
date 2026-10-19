use super::*;

mod fields;

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
    scanner.skip_header_lines(cooperation).await?;
    let mut records = 0usize;
    while !scanner.is_eof() && repeat.map(|limit| records < limit).unwrap_or(true) {
        cooperation.completed(0).await?;
        scanner.skip_separators(cooperation).await?;
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
                FormatElement::Whitespace => scanner.skip_format_whitespace(cooperation).await?,
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
                    let parsed = match scanner
                        .parse_conversion(item, next_literal, cooperation)
                        .await
                    {
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
        scanner.skip_separators(cooperation).await?;
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

    async fn skip_header_lines(
        &mut self,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut work = 0usize;
        for _ in 0..self.options.header_lines {
            if self.current_char().is_none() {
                break;
            }
            while let Some(ch) = self.current_char() {
                self.pos += ch.len_utf8();
                checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
                if ch == '\n' {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn skip_format_whitespace(
        &mut self,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut work = 0usize;
        while let Some(ch) = self.current_char() {
            if !ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
            checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
        }
        Ok(())
    }

    async fn skip_separators(
        &mut self,
        cooperation: &mut TextscanCooperation,
    ) -> BuiltinResult<()> {
        let mut work = 0usize;
        loop {
            if self.skip_comment(cooperation).await? {
                continue;
            }
            if let Some(length) = self.match_delimiter().map(str::len) {
                self.pos += length;
                checkpoint_scanned(&mut work, length, cooperation).await?;
                continue;
            }
            let Some(ch) = self.current_char() else {
                break;
            };
            if ch.is_whitespace() || self.whitespace.contains(&ch) {
                self.pos += ch.len_utf8();
                checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
                continue;
            }
            break;
        }
        Ok(())
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

    async fn skip_comment(&mut self, cooperation: &mut TextscanCooperation) -> BuiltinResult<bool> {
        match &self.options.comment_style {
            CommentStyle::None => Ok(false),
            CommentStyle::Line(markers) => {
                if markers
                    .iter()
                    .any(|marker| !marker.is_empty() && self.text[self.pos..].starts_with(marker))
                {
                    let mut work = 0usize;
                    while let Some(ch) = self.current_char() {
                        self.pos += ch.len_utf8();
                        checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
                        if ch == '\n' {
                            break;
                        }
                    }
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CommentStyle::Block { start, end } => {
                if start.is_empty() || !self.text[self.pos..].starts_with(start) {
                    return Ok(false);
                }
                self.pos += start.len();
                let mut work = start.len();
                while self.pos < self.text.len() {
                    if self.text[self.pos..].starts_with(end) {
                        self.pos += end.len();
                        return Ok(true);
                    }
                    let ch = self.text[self.pos..]
                        .chars()
                        .next()
                        .expect("valid char boundary");
                    self.pos += ch.len_utf8();
                    checkpoint_scanned(&mut work, ch.len_utf8(), cooperation).await?;
                }
                Ok(true)
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

async fn checkpoint_scanned(
    work: &mut usize,
    units: usize,
    cooperation: &mut TextscanCooperation,
) -> BuiltinResult<()> {
    *work = work.saturating_add(units);
    if *work >= 4096 {
        cooperation.completed(*work).await?;
        *work = 0;
    }
    Ok(())
}
