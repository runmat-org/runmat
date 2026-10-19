use std::collections::BTreeMap;

use super::Ap242InspectionError;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PhysicalFile {
    pub schemas: Vec<String>,
    pub entities: BTreeMap<u64, Entity>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Entity {
    pub id: u64,
    pub components: Vec<Component>,
}

impl Entity {
    pub fn component(&self, name: &str) -> Option<&Component> {
        self.components.iter().find(|value| value.name == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.component(name).is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Component {
    pub name: String,
    pub arguments: Vec<P21Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum P21Value {
    String(String),
    Number(f64),
    Reference(u64),
    Enumeration(String),
    List(Vec<P21Value>),
    Typed(String, Vec<P21Value>),
    Null,
    Omitted,
}

impl P21Value {
    pub fn as_string(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }

    pub fn as_reference(&self) -> Option<u64> {
        if let Self::Reference(value) = self {
            Some(*value)
        } else {
            None
        }
    }

    pub fn as_list(&self) -> Option<&[P21Value]> {
        if let Self::List(values) = self {
            Some(values)
        } else {
            None
        }
    }
}

pub(super) fn parse(source: &str) -> Result<PhysicalFile, Ap242InspectionError> {
    if !source.to_ascii_uppercase().contains("ISO-10303-21") {
        return Err(syntax("missing ISO-10303-21 header"));
    }
    let statements = statements(source)?;
    let schemas = statements
        .iter()
        .find(|value| value.trim_start().starts_with("FILE_SCHEMA"))
        .map(|value| quoted_strings(value))
        .unwrap_or_default();
    let mut entities = BTreeMap::new();
    for statement in statements {
        let statement = statement.trim();
        let Some(rest) = statement.strip_prefix('#') else {
            continue;
        };
        let Some((id, rhs)) = rest.split_once('=') else {
            return Err(syntax("entity instance is missing `=`"));
        };
        let id = id
            .trim()
            .parse::<u64>()
            .map_err(|_| syntax("entity id is not an unsigned integer"))?;
        if id == 0 || entities.contains_key(&id) {
            return Err(syntax("entity ids must be unique and nonzero"));
        }
        let components = Parser::new(rhs.trim()).components()?;
        entities.insert(id, Entity { id, components });
    }
    Ok(PhysicalFile { schemas, entities })
}

fn statements(source: &str) -> Result<Vec<String>, Ap242InspectionError> {
    let mut values = Vec::new();
    let mut current = String::new();
    let mut chars = source.chars().peekable();
    let (mut quoted, mut comment) = (false, false);
    while let Some(ch) = chars.next() {
        if comment {
            if ch == '*' && chars.peek() == Some(&'/') {
                chars.next();
                comment = false;
            }
            continue;
        }
        if !quoted && ch == '/' && chars.peek() == Some(&'*') {
            chars.next();
            comment = true;
            continue;
        }
        if ch == '\'' {
            if quoted && chars.peek() == Some(&'\'') {
                current.push(ch);
                current.push(chars.next().expect("peeked quote"));
                continue;
            }
            quoted = !quoted;
        }
        if ch == ';' && !quoted {
            values.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    if quoted || comment {
        return Err(syntax("unterminated string or comment"));
    }
    if !current.trim().is_empty() {
        return Err(syntax("statement is missing `;`"));
    }
    Ok(values)
}

fn quoted_strings(source: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut chars = source.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\'' {
            continue;
        }
        let mut value = String::new();
        while let Some(ch) = chars.next() {
            if ch == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    value.push('\'');
                } else {
                    break;
                }
            } else {
                value.push(ch);
            }
        }
        values.push(value);
    }
    values
}

struct Parser<'a> {
    source: &'a [u8],
    cursor: usize,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            cursor: 0,
        }
    }

    fn components(mut self) -> Result<Vec<Component>, Ap242InspectionError> {
        self.space();
        let values = if self.take(b'(') {
            let mut values = Vec::new();
            loop {
                self.space();
                if self.take(b')') {
                    break;
                }
                values.push(self.component()?);
            }
            values
        } else {
            vec![self.component()?]
        };
        self.space();
        if self.cursor != self.source.len() {
            return Err(syntax("trailing entity bytes"));
        }
        Ok(values)
    }

    fn component(&mut self) -> Result<Component, Ap242InspectionError> {
        let name = self.identifier()?;
        self.expect(b'(')?;
        Ok(Component {
            name,
            arguments: self.values(b')')?,
        })
    }

    fn values(&mut self, close: u8) -> Result<Vec<P21Value>, Ap242InspectionError> {
        let mut values = Vec::new();
        self.space();
        if self.take(close) {
            return Ok(values);
        }
        loop {
            values.push(self.value()?);
            self.space();
            if self.take(close) {
                return Ok(values);
            }
            self.expect(b',')?;
        }
    }

    fn value(&mut self) -> Result<P21Value, Ap242InspectionError> {
        self.space();
        match self.peek() {
            Some(b'\'') => self.string().map(P21Value::String),
            Some(b'#') => {
                self.cursor += 1;
                Ok(P21Value::Reference(self.unsigned()?))
            }
            Some(b'.') => self.enumeration().map(P21Value::Enumeration),
            Some(b'(') => {
                self.cursor += 1;
                self.values(b')').map(P21Value::List)
            }
            Some(b'$') => {
                self.cursor += 1;
                Ok(P21Value::Null)
            }
            Some(b'*') => {
                self.cursor += 1;
                Ok(P21Value::Omitted)
            }
            Some(value) if value.is_ascii_digit() || matches!(value, b'+' | b'-') => {
                self.number().map(P21Value::Number)
            }
            Some(value) if value.is_ascii_alphabetic() || value == b'_' => {
                let name = self.identifier()?;
                self.space();
                if self.take(b'(') {
                    Ok(P21Value::Typed(name, self.values(b')')?))
                } else {
                    Ok(P21Value::Enumeration(name))
                }
            }
            _ => Err(syntax("unsupported or missing parameter value")),
        }
    }

    fn identifier(&mut self) -> Result<String, Ap242InspectionError> {
        self.space();
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|value| value.is_ascii_alphanumeric() || matches!(value, b'_' | b'-'))
        {
            self.cursor += 1;
        }
        if start == self.cursor {
            return Err(syntax("expected entity name"));
        }
        Ok(String::from_utf8_lossy(&self.source[start..self.cursor]).to_ascii_uppercase())
    }

    fn string(&mut self) -> Result<String, Ap242InspectionError> {
        self.expect(b'\'')?;
        let mut value = Vec::new();
        loop {
            let Some(ch) = self.peek() else {
                return Err(syntax("unterminated string"));
            };
            self.cursor += 1;
            if ch == b'\'' {
                if self.peek() == Some(b'\'') {
                    self.cursor += 1;
                    value.push(b'\'');
                } else {
                    return String::from_utf8(value)
                        .map_err(|_| syntax("supported strings must be UTF-8"));
                }
            } else {
                value.push(ch);
            }
        }
    }

    fn enumeration(&mut self) -> Result<String, Ap242InspectionError> {
        self.expect(b'.')?;
        let start = self.cursor;
        while self.peek().is_some_and(|value| value != b'.') {
            self.cursor += 1;
        }
        let value = String::from_utf8_lossy(&self.source[start..self.cursor]).to_ascii_uppercase();
        self.expect(b'.')?;
        Ok(value)
    }

    fn number(&mut self) -> Result<f64, Ap242InspectionError> {
        let start = self.cursor;
        while self.peek().is_some_and(|value| {
            value.is_ascii_digit() || matches!(value, b'+' | b'-' | b'.' | b'E' | b'e')
        }) {
            self.cursor += 1;
        }
        String::from_utf8_lossy(&self.source[start..self.cursor])
            .parse()
            .map_err(|_| syntax("invalid number"))
    }

    fn unsigned(&mut self) -> Result<u64, Ap242InspectionError> {
        let start = self.cursor;
        while self.peek().is_some_and(|value| value.is_ascii_digit()) {
            self.cursor += 1;
        }
        String::from_utf8_lossy(&self.source[start..self.cursor])
            .parse()
            .map_err(|_| syntax("invalid reference"))
    }

    fn expect(&mut self, expected: u8) -> Result<(), Ap242InspectionError> {
        self.space();
        if self.take(expected) {
            Ok(())
        } else {
            Err(syntax("unexpected STEP token"))
        }
    }
    fn take(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
    fn space(&mut self) {
        while self.peek().is_some_and(|value| value.is_ascii_whitespace()) {
            self.cursor += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.source.get(self.cursor).copied()
    }
}

fn syntax(message: &str) -> Ap242InspectionError {
    Ap242InspectionError::Syntax(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_multiline_simple_and_complex_entities() {
        let file = parse("ISO-10303-21;HEADER;FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));ENDSEC;DATA;#10=PRODUCT('A','',(#20));#11=(LENGTH_MEASURE_WITH_UNIT() MEASURE_REPRESENTATION_ITEM() MEASURE_WITH_UNIT(LENGTH_MEASURE(8.0),#20) REPRESENTATION_ITEM('Wall'));ENDSEC;END-ISO-10303-21;").unwrap();
        assert_eq!(file.schemas.len(), 1);
        assert!(file.entities[&10].has("PRODUCT"));
        assert!(file.entities[&11].has("MEASURE_WITH_UNIT"));
    }
}
