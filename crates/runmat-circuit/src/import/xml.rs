//! A small XML reader for the interchange formats that are XML documents: elements with
//! attributes and text, the five predefined entities and numeric character references.
//! Comments, processing instructions, CDATA and the document type declaration are read past.

use std::collections::BTreeMap;

/// One element: its name, attributes, child elements in order and its text.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Element {
    pub name: String,
    pub attrs: BTreeMap<String, String>,
    pub children: Vec<Element>,
    pub text: String,
}

impl Element {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.get(name).map(String::as_str)
    }
    /// An attribute as a number (`None` when absent or not a number).
    pub fn num(&self, name: &str) -> Option<f64> {
        self.attr(name).and_then(|v| v.trim().parse().ok())
    }
    pub fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.name == name)
    }
    pub fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }
    /// The children of the first `outer` child named `inner` (`<parts><part/>…</parts>`).
    pub fn list<'a>(&'a self, outer: &str, inner: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
        self.child(outer).into_iter().flat_map(move |o| o.all(inner))
    }
    /// The first element at a path of child names (`drawing/board`).
    pub fn path(&self, path: &str) -> Option<&Element> {
        path.split('/').try_fold(self, |at, name| at.child(name))
    }
}

/// Parses a document to its root element.
pub fn parse(text: &str) -> Result<Element, String> {
    let mut p = Parser { s: text.as_bytes(), at: 0, text };
    p.skip_prolog()?;
    let root = p.element()?;
    Ok(root)
}

struct Parser<'a> {
    s: &'a [u8],
    at: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn err<T>(&self, what: &str) -> Result<T, String> {
        let line = self.text[..self.at.min(self.text.len())].matches('\n').count() + 1;
        Err(format!("line {line}: {what}"))
    }
    fn starts(&self, pat: &str) -> bool {
        self.s[self.at..].starts_with(pat.as_bytes())
    }
    fn skip_ws(&mut self) {
        while self.at < self.s.len() && self.s[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }
    fn skip_past(&mut self, end: &str) -> Result<(), String> {
        match self.text[self.at..].find(end) {
            Some(i) => {
                self.at += i + end.len();
                Ok(())
            }
            None => self.err(&format!("no closing `{end}`")),
        }
    }
    /// Skips declarations, comments and the DOCTYPE (with an internal subset) before the root.
    fn skip_prolog(&mut self) -> Result<(), String> {
        if self.starts("\u{feff}") {
            self.at += 3;
        }
        loop {
            self.skip_ws();
            if self.starts("<?") {
                self.skip_past("?>")?;
            } else if self.starts("<!--") {
                self.skip_past("-->")?;
            } else if self.starts("<!DOCTYPE") {
                let mut depth = 0;
                while self.at < self.s.len() {
                    match self.s[self.at] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        b'>' if depth == 0 => break,
                        _ => {}
                    }
                    self.at += 1;
                }
                self.at += 1;
            } else {
                return Ok(());
            }
        }
    }
    fn name(&mut self) -> Result<String, String> {
        let start = self.at;
        while self.at < self.s.len() {
            let c = self.s[self.at];
            if c.is_ascii_whitespace() || matches!(c, b'>' | b'/' | b'=') {
                break;
            }
            self.at += 1;
        }
        if start == self.at {
            return self.err("expected a name");
        }
        Ok(self.text[start..self.at].to_owned())
    }
    fn element(&mut self) -> Result<Element, String> {
        if !self.starts("<") {
            return self.err("expected an element");
        }
        self.at += 1;
        let mut el = Element { name: self.name()?, ..Default::default() };
        loop {
            self.skip_ws();
            if self.starts("/>") {
                self.at += 2;
                return Ok(el);
            }
            if self.starts(">") {
                self.at += 1;
                break;
            }
            let key = self.name()?;
            self.skip_ws();
            if !self.starts("=") {
                return self.err(&format!("attribute `{key}` has no value"));
            }
            self.at += 1;
            self.skip_ws();
            let quote = match self.s.get(self.at) {
                Some(b'"') => "\"",
                Some(b'\'') => "'",
                _ => return self.err(&format!("attribute `{key}` is not quoted")),
            };
            self.at += 1;
            let end = match self.text[self.at..].find(quote) {
                Some(i) => self.at + i,
                None => return self.err("unclosed attribute value"),
            };
            let value = unescape(&self.text[self.at..end]);
            self.at = end + 1;
            el.attrs.insert(key, value);
        }
        loop {
            if self.at >= self.s.len() {
                return self.err(&format!("`<{}>` is not closed", el.name));
            }
            if self.starts("</") {
                self.at += 2;
                let name = self.name()?;
                if name != el.name {
                    return self.err(&format!("`<{}>` closed by `</{name}>`", el.name));
                }
                self.skip_ws();
                self.at += 1;
                return Ok(el);
            }
            if self.starts("<!--") {
                self.skip_past("-->")?;
            } else if self.starts("<![CDATA[") {
                let start = self.at + 9;
                self.skip_past("]]>")?;
                el.text.push_str(&self.text[start..self.at - 3]);
            } else if self.starts("<?") {
                self.skip_past("?>")?;
            } else if self.starts("<") {
                el.children.push(self.element()?);
            } else {
                let start = self.at;
                while self.at < self.s.len() && self.s[self.at] != b'<' {
                    self.at += 1;
                }
                el.text.push_str(&unescape(&self.text[start..self.at]));
            }
        }
    }
}

/// Replaces the predefined entities and numeric character references.
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';') else { break };
        let entity = &rest[1..end];
        let ch = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            e if e.starts_with("#x") || e.starts_with("#X") => u32::from_str_radix(&e[2..], 16).ok().and_then(char::from_u32),
            e if e.starts_with('#') => e[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_elements_attributes_and_entities() {
        let doc = parse("<?xml version=\"1.0\"?>\n<!DOCTYPE eagle SYSTEM \"eagle.dtd\">\n<a x='1' y=\"&lt;2&gt;\"><!-- c --><b n=\"&#65;&amp;\"/>t&amp;u<b/></a>").unwrap();
        assert_eq!(doc.name, "a");
        assert_eq!(doc.attr("y"), Some("<2>"));
        assert_eq!(doc.all("b").count(), 2);
        assert_eq!(doc.child("b").unwrap().attr("n"), Some("A&"));
        assert_eq!(doc.text, "t&u");
        assert!(parse("<a><b></a>").is_err());
    }
}
