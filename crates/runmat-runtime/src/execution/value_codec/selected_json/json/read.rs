//! Canonical field-order reader for the private versioned record. Large
//! arrays/text are consumed incrementally, never as a serde_json Value tree.
use super::*;
use serde::de::DeserializeOwned;

pub(crate) struct Reader {
    bytes: Vec<u8>,
    at: usize,
}
impl Reader {
    pub(crate) fn field_is(&self, name: &'static str) -> bool {
        self.bytes.get(self.at) == Some(&b'"')
            && self
                .bytes
                .get(self.at.saturating_add(1)..self.at.saturating_add(1 + name.len()))
                == Some(name.as_bytes())
    }
    pub(crate) fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, at: 0 }
    }
    pub(crate) fn finish(self) -> BuiltinResult<()> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(error("selected capture has trailing record bytes"))
        }
    }
    pub(crate) fn token(&mut self, token: u8) -> BuiltinResult<()> {
        if self.bytes.get(self.at) != Some(&token) {
            return Err(error("selected capture JSON structure is invalid"));
        }
        self.at += 1;
        Ok(())
    }
    pub(crate) fn is(&self, token: u8) -> bool {
        self.bytes.get(self.at) == Some(&token)
    }
    pub(crate) fn null(&mut self) -> BuiltinResult<bool> {
        if self.bytes.get(self.at..self.at.saturating_add(4)) == Some(b"null") {
            self.at += 4;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub(crate) fn key(&mut self, name: &'static str, first: bool) -> BuiltinResult<()> {
        if !first {
            self.token(b',')?;
        }
        self.token(b'"')?;
        if self.bytes.get(self.at..self.at.saturating_add(name.len())) != Some(name.as_bytes()) {
            return Err(error("selected capture field order/schema is invalid"));
        }
        self.at += name.len();
        self.token(b'"')?;
        self.token(b':')
    }
    pub(crate) async fn metadata<T: DeserializeOwned>(
        &mut self,
        maximum: usize,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<T> {
        let first = self.at;
        self.skip_value(maximum, c).await?;
        serde_json::from_slice(&self.bytes[first..self.at])
            .map_err(|source| with_source("invalid selected capture metadata", source))
    }
    pub(crate) async fn array<T: DeserializeOwned>(
        &mut self,
        maximum_item: usize,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<T>> {
        self.array_limited(maximum_item, usize::MAX, c).await
    }
    pub(crate) async fn array_limited<T: DeserializeOwned>(
        &mut self,
        maximum_item: usize,
        maximum_items: usize,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<Vec<T>> {
        self.token(b'[')?;
        let mut values = Vec::new();
        while !self.is(b']') {
            if values.len() >= maximum_items {
                return Err(error("selected JSON array exceeds its admitted extent"));
            }
            if !values.is_empty() {
                self.token(b',')?;
            }
            values.try_reserve(1).map_err(|source| {
                with_source("selected capture array allocation failed", source)
            })?;
            values.push(self.metadata(maximum_item, c).await?);
        }
        self.token(b']')?;
        Ok(values)
    }
    pub(crate) async fn bits(&mut self, c: &mut Cooperation<'_>) -> BuiltinResult<Vec<f64>> {
        self.token(b'[')?;
        let mut values = Vec::new();
        while !self.is(b']') {
            if !values.is_empty() {
                self.token(b',')?;
            }
            values.try_reserve(1).map_err(|source| {
                with_source("selected capture numeric allocation failed", source)
            })?;
            values.push(f64::from_bits(self.metadata::<u64>(20, c).await?));
        }
        self.token(b']')?;
        Ok(values)
    }
    pub(crate) async fn string(&mut self, c: &mut Cooperation<'_>) -> BuiltinResult<String> {
        self.string_limited(usize::MAX, c).await
    }
    pub(crate) async fn string_limited(
        &mut self,
        maximum: usize,
        c: &mut Cooperation<'_>,
    ) -> BuiltinResult<String> {
        self.token(b'"')?;
        let mut result = String::new();
        let mut first = self.at;
        loop {
            let byte = *self
                .bytes
                .get(self.at)
                .ok_or_else(|| error("unterminated selected capture text"))?;
            if byte == b'"' {
                self.decode_text_chunk(first, self.at, &mut result, maximum)?;
                c.completed((self.at - first) as u64).await?;
                self.at += 1;
                return Ok(result);
            }
            if byte == b'\\' {
                self.at += 1;
                let escape = *self
                    .bytes
                    .get(self.at)
                    .ok_or_else(|| error("unterminated selected capture escape"))?;
                self.at += 1;
                if escape == b'u' {
                    let end = self
                        .at
                        .checked_add(4)
                        .ok_or_else(|| error("selected capture text extent overflow"))?;
                    let digits = self
                        .bytes
                        .get(self.at..end)
                        .ok_or_else(|| error("unterminated selected capture Unicode escape"))?;
                    let hex = std::str::from_utf8(digits).map_err(|source| {
                        with_source("invalid selected capture Unicode escape", source)
                    })?;
                    let code = u16::from_str_radix(hex, 16).map_err(|source| {
                        with_source("invalid selected capture Unicode escape", source)
                    })?;
                    self.at = end;
                    if (0xd800..=0xdbff).contains(&code) {
                        // Keep a UTF-16 surrogate pair inside one canonical
                        // serde string fragment, never normalize it locally.
                        if self.bytes.get(self.at..self.at.saturating_add(2)) != Some(b"\\u") {
                            return Err(error("selected capture text has an unpaired surrogate"));
                        }
                        self.at = self
                            .at
                            .checked_add(6)
                            .ok_or_else(|| error("selected capture text extent overflow"))?;
                        if self.at > self.bytes.len() {
                            return Err(error("unterminated selected capture surrogate"));
                        }
                    }
                }
            } else {
                // One scalar is bounded independently of the input extent.
                // Validate before advancing or copying a fragment: an invalid
                // continuation run must not postpone the cooperative boundary.
                let width = match byte {
                    0x20..=0x7f => 1,
                    0xc2..=0xdf => 2,
                    0xe0..=0xef => 3,
                    0xf0..=0xf4 => 4,
                    _ => return Err(error("invalid selected capture UTF-8 scalar")),
                };
                let end = self
                    .at
                    .checked_add(width)
                    .ok_or_else(|| error("selected capture text extent overflow"))?;
                let scalar = self
                    .bytes
                    .get(self.at..end)
                    .ok_or_else(|| error("unterminated selected capture UTF-8 scalar"))?;
                std::str::from_utf8(scalar).map_err(|source| {
                    with_source("invalid selected capture UTF-8 scalar", source)
                })?;
                self.at = end;
            }
            if self.at - first >= 1024 {
                self.decode_text_chunk(first, self.at, &mut result, maximum)?;
                c.completed((self.at - first) as u64).await?;
                first = self.at;
            }
        }
    }
    fn decode_text_chunk(
        &self,
        first: usize,
        end: usize,
        result: &mut String,
        maximum: usize,
    ) -> BuiltinResult<()> {
        let slice = self
            .bytes
            .get(first..end)
            .ok_or_else(|| error("selected capture text extent is invalid"))?;
        let mut quoted = Vec::new();
        quoted
            .try_reserve(slice.len().saturating_add(2))
            .map_err(|source| with_source("selected capture text allocation failed", source))?;
        quoted.push(b'"');
        quoted.extend_from_slice(slice);
        quoted.push(b'"');
        let part: String = serde_json::from_slice(&quoted)
            .map_err(|source| with_source("invalid selected capture text", source))?;
        if result
            .len()
            .checked_add(part.len())
            .is_none_or(|length| length > maximum)
        {
            return Err(error("selected JSON text exceeds its admitted extent"));
        }
        result
            .try_reserve(part.len())
            .map_err(|source| with_source("selected capture text allocation failed", source))?;
        result.push_str(&part);
        Ok(())
    }
    async fn skip_value(&mut self, maximum: usize, c: &mut Cooperation<'_>) -> BuiltinResult<()> {
        let first = self.at;
        let mut depth = 0usize;
        let mut quoted = false;
        let mut escaped = false;
        let mut work = 0u64;
        loop {
            let Some(&byte) = self.bytes.get(self.at) else {
                break;
            };
            if !quoted && depth == 0 && matches!(byte, b',' | b']' | b'}') {
                break;
            }
            if quoted {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    quoted = false;
                }
            } else {
                match byte {
                    b'"' => quoted = true,
                    b'[' | b'{' => {
                        depth = depth
                            .checked_add(1)
                            .ok_or_else(|| error("selected capture JSON depth overflow"))?
                    }
                    b']' | b'}' => {
                        depth = depth
                            .checked_sub(1)
                            .ok_or_else(|| error("selected capture JSON nesting is invalid"))?
                    }
                    _ => {}
                }
            }
            self.at += 1;
            work += 1;
            if self.at - first > maximum {
                return Err(error(
                    "selected capture metadata exceeds its typed wire extent",
                ));
            }
            if work == 1024 {
                c.completed(work).await?;
                work = 0;
            }
            if !quoted && depth == 0 && matches!(byte, b']' | b'}' | b'"') {
                break;
            }
        }
        c.completed(work).await?;
        if first == self.at || quoted || depth != 0 {
            return Err(error("selected capture JSON value is incomplete"));
        }
        Ok(())
    }
}
