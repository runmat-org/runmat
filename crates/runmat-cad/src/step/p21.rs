//! ISO 10303-21 exchange structures: the DATA section's instances read into a map by id
//! (simple and complex instances, every parameter type), and new instances appended after the
//! highest id, so a kernel-written file gains entities without being rewritten.

use std::collections::BTreeMap;
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    /// `$`
    Unset,
    /// `*`
    Derived,
    Ref(u64),
    Int(i64),
    Real(f64),
    Str(String),
    /// `.NAME.` (enumerations and booleans: `.T.`, `.F.`, `.U.`).
    Enum(String),
    List(Vec<Param>),
    /// A typed value of a select: `LENGTH_MEASURE(0.05)`.
    Typed(String, Box<Param>),
}

impl Param {
    pub fn as_ref(&self) -> Option<u64> {
        match self {
            Param::Ref(r) => Some(*r),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Param::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_enum(&self) -> Option<&str> {
        match self {
            Param::Enum(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_list(&self) -> &[Param] {
        match self {
            Param::List(l) => l,
            _ => &[],
        }
    }
    /// A number, through a typed value (`LENGTH_MEASURE(0.05)`) too.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Param::Real(v) => Some(*v),
            Param::Int(v) => Some(*v as f64),
            Param::Typed(_, p) => p.as_f64(),
            _ => None,
        }
    }
}

/// One instance: a simple entity, or a complex one as its partial entities in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub parts: Vec<(String, Vec<Param>)>,
}

impl Instance {
    pub fn is_complex(&self) -> bool {
        self.parts.len() > 1
    }
    /// The simple entity's name, or `None` for a complex instance.
    pub fn name(&self) -> Option<&str> {
        (!self.is_complex()).then(|| self.parts[0].0.as_str())
    }
    /// The parameters of the partial entity `name` (the whole parameter list of a simple one).
    pub fn part(&self, name: &str) -> Option<&[Param]> {
        self.parts.iter().find(|(n, _)| n == name).map(|(_, p)| p.as_slice())
    }
    pub fn has(&self, name: &str) -> bool {
        self.parts.iter().any(|(n, _)| n == name)
    }
}

/// A file's DATA instances by id.
#[derive(Clone, Debug, Default)]
pub struct Data {
    pub instances: BTreeMap<u64, Instance>,
}

impl Data {
    pub fn get(&self, id: u64) -> Option<&Instance> {
        self.instances.get(&id)
    }
    /// Ids of simple instances named `name`, and of complex ones that include it.
    pub fn of<'a>(&'a self, name: &'a str) -> impl Iterator<Item = (u64, &'a Instance)> + 'a {
        self.instances.iter().filter(move |(_, i)| i.has(name)).map(|(k, v)| (*k, v))
    }
    pub fn max_id(&self) -> u64 {
        self.instances.keys().next_back().copied().unwrap_or(0)
    }
}

struct Lexer<'a> {
    s: &'a [u8],
    i: usize,
}

impl Lexer<'_> {
    fn skip(&mut self) {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"/*") {
                match self.s[self.i + 2..].windows(2).position(|w| w == b"*/") {
                    Some(k) => self.i += k + 4,
                    None => self.i = self.s.len(),
                }
            } else {
                return;
            }
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.s.get(self.i).copied()
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected `{}` at byte {}", c as char, self.i))
        }
    }
    fn keyword(&mut self) -> String {
        self.skip();
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_' || self.s[self.i] == b'!') {
            self.i += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.i]).into_owned()
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect(b'\'')?;
        let mut raw = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("an unterminated string")?;
            self.i += 1;
            if c == b'\'' {
                if self.s.get(self.i) == Some(&b'\'') {
                    raw.push(b'\'');
                    self.i += 1;
                    continue;
                }
                break;
            }
            raw.push(c);
        }
        Ok(decode_string(&String::from_utf8_lossy(&raw)))
    }
    fn param(&mut self) -> Result<Param, String> {
        match self.peek().ok_or("unexpected end of file")? {
            b'$' => {
                self.i += 1;
                Ok(Param::Unset)
            }
            b'*' => {
                self.i += 1;
                Ok(Param::Derived)
            }
            b'#' => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i]).unwrap_or("").parse().map(Param::Ref).map_err(|_| "a bad instance reference".into())
            }
            b'\'' => self.string().map(Param::Str),
            b'.' => {
                self.i += 1;
                let k = self.keyword();
                self.expect(b'.')?;
                Ok(Param::Enum(k))
            }
            b'(' => {
                self.i += 1;
                let mut items = Vec::new();
                if self.peek() == Some(b')') {
                    self.i += 1;
                    return Ok(Param::List(items));
                }
                loop {
                    items.push(self.param()?);
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b')') => {
                            self.i += 1;
                            return Ok(Param::List(items));
                        }
                        _ => return Err(format!("a bad list at byte {}", self.i)),
                    }
                }
            }
            b'"' => {
                // Binary: kept as a string of hex digits.
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'"' {
                    self.i += 1;
                }
                let v = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
                self.i += 1;
                Ok(Param::Str(v))
            }
            c if c == b'-' || c == b'+' || c.is_ascii_digit() => {
                let start = self.i;
                self.i += 1;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'.' | b'E' | b'e' | b'-' | b'+') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
                if t.contains(['.', 'E', 'e']) {
                    let t = if t.ends_with('.') { format!("{t}0") } else { t.replace(".E", ".0E") };
                    t.parse().map(Param::Real).map_err(|_| format!("a bad real `{t}`"))
                } else {
                    t.parse().map(Param::Int).map_err(|_| format!("a bad integer `{t}`"))
                }
            }
            c if c.is_ascii_alphabetic() => {
                let k = self.keyword();
                self.expect(b'(')?;
                let p = self.param()?;
                self.expect(b')')?;
                Ok(Param::Typed(k, Box::new(p)))
            }
            c => Err(format!("unexpected `{}` at byte {}", c as char, self.i)),
        }
    }
    fn params(&mut self) -> Result<Vec<Param>, String> {
        match self.param()? {
            Param::List(l) => Ok(l),
            _ => Err("an entity's parameters are not a list".into()),
        }
    }
}

/// `\X2\…\X0\` (UCS-2 hex) and `\X\hh` decoded; other text as written.
fn decode_string(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(k) = rest.find('\\') {
        out.push_str(&rest[..k]);
        rest = &rest[k..];
        if let Some(r) = rest.strip_prefix("\\X2\\") {
            let end = r.find("\\X0\\").unwrap_or(r.len());
            let units: Vec<u16> = r[..end].as_bytes().chunks(4).filter_map(|c| u16::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok()).collect();
            out.push_str(&String::from_utf16_lossy(&units));
            rest = r.get(end + 4..).unwrap_or("");
        } else if let Some(r) = rest.strip_prefix("\\X\\") {
            if let Some(b) = r.get(..2).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(b as char);
                rest = &r[2..];
            } else {
                out.push('\\');
                rest = &rest[1..];
            }
        } else if let Some(r) = rest.strip_prefix("\\\\") {
            out.push('\\');
            rest = r;
        } else {
            out.push('\\');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// A string as Part 21 writes it: quotes doubled, backslashes escaped, non-ASCII as `\X2\`.
pub fn encode_string(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        match c {
            '\'' => out.push_str("''"),
            '\\' => out.push_str("\\\\"),
            c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
            c => {
                out.push_str("\\X2\\");
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "{u:04X}");
                }
                out.push_str("\\X0\\");
            }
        }
    }
    out.push('\'');
    out
}

/// A real as Part 21 writes it (always with a decimal point).
pub fn real(v: f64) -> String {
    let s = format!("{v:?}");
    let (m, e) = match s.split_once('e') {
        Some((m, e)) => (m.to_owned(), Some(e.to_owned())),
        None => (s, None),
    };
    let m = if m.contains('.') { m } else { format!("{m}.") };
    match e {
        Some(e) => format!("{m}E{e}"),
        None => m,
    }
}

/// The DATA section's instances.
pub fn parse(text: &str) -> Result<Data, String> {
    let start = text.find("DATA;").ok_or("no DATA section")? + 5;
    let mut lx = Lexer { s: text.as_bytes(), i: start };
    let mut data = Data::default();
    loop {
        match lx.peek() {
            Some(b'#') => {
                lx.i += 1;
                let id_start = lx.i;
                while lx.i < lx.s.len() && lx.s[lx.i].is_ascii_digit() {
                    lx.i += 1;
                }
                let id: u64 = std::str::from_utf8(&lx.s[id_start..lx.i]).unwrap_or("").parse().map_err(|_| "a bad instance id")?;
                lx.expect(b'=')?;
                let parts = if lx.peek() == Some(b'(') {
                    lx.i += 1;
                    let mut parts = Vec::new();
                    while lx.peek() != Some(b')') {
                        let name = lx.keyword();
                        if name.is_empty() {
                            return Err(format!("#{id}: a bad complex instance"));
                        }
                        parts.push((name, lx.params()?));
                    }
                    lx.i += 1;
                    parts
                } else {
                    let name = lx.keyword();
                    vec![(name, lx.params()?)]
                };
                lx.expect(b';')?;
                data.instances.insert(id, Instance { parts });
            }
            Some(_) => {
                if lx.keyword() == "ENDSEC" {
                    return Ok(data);
                }
                return Err(format!("unexpected text in DATA at byte {}", lx.i));
            }
            None => return Err("the DATA section has no ENDSEC".into()),
        }
    }
}

/// New instances appended to a file: [`Appender::add`] returns each one's id, and
/// [`Appender::finish`] writes them before the DATA section's `ENDSEC;`.
pub struct Appender {
    next: u64,
    lines: String,
}

impl Appender {
    pub fn new(data: &Data) -> Self {
        Appender { next: data.max_id() + 1, lines: String::new() }
    }
    /// Add `#id = <body>;` and return the id.
    pub fn add(&mut self, body: impl AsRef<str>) -> u64 {
        let id = self.next;
        self.next += 1;
        let _ = writeln!(self.lines, "#{id}={};", body.as_ref());
        id
    }
    pub fn finish(self, text: &str) -> Result<String, String> {
        let data = text.find("DATA;").ok_or("no DATA section")?;
        let end = text[data..].find("ENDSEC;").ok_or("the DATA section has no ENDSEC")? + data;
        Ok(format!("{}{}{}", &text[..end], self.lines, &text[end..]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_and_complex_instances_with_every_parameter_kind() {
        let t = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1=ADVANCED_FACE('it''s \\X2\\00E9\\X0\\',(#2),#3,.T.);\n/* c */#2=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\n#3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(0.05),#2);\n#4=X(1.E-06,-2,$,());\nENDSEC;\nEND-ISO-10303-21;\n";
        let d = parse(t).unwrap();
        let f = d.get(1).unwrap();
        assert_eq!(f.name(), Some("ADVANCED_FACE"));
        assert_eq!(f.parts[0].1[0].as_str(), Some("it's é"));
        assert_eq!(f.parts[0].1[3].as_enum(), Some("T"));
        let u = d.get(2).unwrap();
        assert!(u.is_complex() && u.has("SI_UNIT"));
        assert_eq!(u.part("SI_UNIT").unwrap()[0].as_enum(), Some("MILLI"));
        assert_eq!(d.get(3).unwrap().parts[0].1[0].as_f64(), Some(0.05));
        assert_eq!(d.get(4).unwrap().parts[0].1[0].as_f64(), Some(1e-6));
        let mut a = Appender::new(&d);
        assert_eq!(a.add("DATUM('A','',#9,.F.,'A')"), 5);
        let t2 = a.finish(t).unwrap();
        assert_eq!(parse(&t2).unwrap().get(5).unwrap().name(), Some("DATUM"));
        assert_eq!(encode_string("a'é"), "'a''\\X2\\00E9\\X0\\'");
        assert_eq!(real(10.0), "10.0");
        assert_eq!(real(5e-5), "5.E-5");
    }
}
