//! A small XML 1.0 reader for the exchange formats: elements with attributes and text, entity
//! and character references, comments, CDATA and processing instructions, and namespaces
//! resolved to their URIs (`{uri}local`). No DTDs.

use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Element {
    /// `{namespace-uri}local`, or `local` when the name has no namespace.
    pub name: String,
    pub attributes: BTreeMap<String, String>,
    pub children: Vec<Element>,
    pub text: String,
}

impl Element {
    pub fn local(&self) -> &str {
        self.name.rsplit('}').next().unwrap_or(&self.name)
    }
    pub fn namespace(&self) -> Option<&str> {
        self.name
            .strip_prefix('{')
            .and_then(|s| s.split('}').next())
    }
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes.get(name).map(String::as_str)
    }
    /// Children with this local name.
    pub fn all<'a>(&'a self, local: &'a str) -> impl Iterator<Item = &'a Element> + 'a {
        self.children.iter().filter(move |c| c.local() == local)
    }
    pub fn first(&self, local: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.local() == local)
    }
}

fn unescape(s: &str) -> Result<String, String> {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let j = rest[i..]
            .find(';')
            .ok_or("an unterminated entity reference")?
            + i;
        let ent = &rest[i + 1..j];
        match ent {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = if let Some(h) = ent.strip_prefix("#x") {
                    u32::from_str_radix(h, 16).ok()
                } else if let Some(d) = ent.strip_prefix('#') {
                    d.parse().ok()
                } else {
                    None
                };
                out.push(
                    code.and_then(char::from_u32)
                        .ok_or_else(|| format!("unknown entity &{ent};"))?,
                );
            }
        }
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

struct Raw {
    name: String,
    attributes: Vec<(String, String)>,
    children: Vec<Element>,
    text: String,
    scope: BTreeMap<String, String>,
}

fn resolve(
    name: &str,
    scope: &BTreeMap<String, String>,
    attribute: bool,
) -> Result<String, String> {
    if name == "xmlns" || name.starts_with("xmlns:") {
        return Ok(name.to_owned());
    }
    match name.split_once(':') {
        Some(("xml", l)) => Ok(format!("{{http://www.w3.org/XML/1998/namespace}}{l}")),
        Some((p, l)) => scope
            .get(p)
            .map(|u| format!("{{{u}}}{l}"))
            .ok_or_else(|| format!("the prefix `{p}` has no namespace")),
        // Unprefixed attributes have no namespace; unprefixed elements take the default.
        None if attribute => Ok(name.to_owned()),
        None => Ok(scope
            .get("")
            .filter(|u| !u.is_empty())
            .map(|u| format!("{{{u}}}{name}"))
            .unwrap_or_else(|| name.to_owned())),
    }
}

fn finish(raw: Raw) -> Result<Element, String> {
    let mut attributes = BTreeMap::new();
    for (k, v) in raw.attributes {
        attributes.insert(resolve(&k, &raw.scope, true)?, v);
    }
    Ok(Element {
        name: resolve(&raw.name, &raw.scope, false)?,
        attributes,
        children: raw.children,
        text: raw.text,
    })
}

/// The document's root element.
pub fn parse(text: &str) -> Result<Element, String> {
    let s = text.strip_prefix('\u{feff}').unwrap_or(text);
    let b = s.as_bytes();
    let mut i = 0;
    let mut stack: Vec<Raw> = Vec::new();
    let mut root = None;
    while i < b.len() {
        if b[i] != b'<' {
            let j = s[i..].find('<').map(|k| k + i).unwrap_or(b.len());
            if let Some(top) = stack.last_mut() {
                top.text.push_str(&unescape(&s[i..j])?);
            } else if !s[i..j].trim().is_empty() {
                return Err("text outside the root element".into());
            }
            i = j;
            continue;
        }
        let rest = &s[i..];
        if rest.starts_with("<?") {
            i += rest
                .find("?>")
                .ok_or("an unterminated processing instruction")?
                + 2;
        } else if rest.starts_with("<!--") {
            i += rest.find("-->").ok_or("an unterminated comment")? + 3;
        } else if rest.starts_with("<![CDATA[") {
            let j = rest.find("]]>").ok_or("an unterminated CDATA section")?;
            if let Some(top) = stack.last_mut() {
                top.text.push_str(&rest[9..j]);
            }
            i += j + 3;
        } else if rest.starts_with("<!") {
            return Err("document type declarations are not read".into());
        } else if rest.starts_with("</") {
            let j = rest.find('>').ok_or("an unterminated end tag")?;
            let name = rest[2..j].trim();
            let raw = stack.pop().ok_or("an end tag without a start tag")?;
            if raw.name != name {
                return Err(format!("</{name}> closes <{}>", raw.name));
            }
            let e = finish(raw)?;
            match stack.last_mut() {
                Some(p) => p.children.push(e),
                None => root = Some(e),
            }
            i += j + 1;
        } else {
            // A start tag: name, attributes (quoted values may hold `>`), `>` or `/>`.
            let mut k = 1;
            while k < rest.len() && !matches!(b[i + k], b' ' | b'\t' | b'\r' | b'\n' | b'>' | b'/')
            {
                k += 1;
            }
            let name = rest[1..k].to_owned();
            let mut attributes = Vec::new();
            let empty;
            loop {
                while k < rest.len() && b[i + k].is_ascii_whitespace() {
                    k += 1;
                }
                if rest[k..].starts_with("/>") {
                    empty = true;
                    k += 2;
                    break;
                }
                if rest[k..].starts_with('>') {
                    empty = false;
                    k += 1;
                    break;
                }
                let eq = rest[k..]
                    .find('=')
                    .ok_or_else(|| format!("<{name}>: an attribute without a value"))?
                    + k;
                let key = rest[k..eq].trim().to_owned();
                let mut q = eq + 1;
                while q < rest.len() && b[i + q].is_ascii_whitespace() {
                    q += 1;
                }
                let quote = *b.get(i + q).ok_or("an unterminated start tag")?;
                if quote != b'"' && quote != b'\'' {
                    return Err(format!("<{name}>: the value of {key} is not quoted"));
                }
                let end = rest[q + 1..]
                    .find(quote as char)
                    .ok_or("an unterminated attribute value")?
                    + q
                    + 1;
                attributes.push((key, unescape(&rest[q + 1..end])?));
                k = end + 1;
            }
            let mut scope = stack.last().map(|r| r.scope.clone()).unwrap_or_default();
            for (k, v) in &attributes {
                if k == "xmlns" {
                    scope.insert(String::new(), v.clone());
                } else if let Some(p) = k.strip_prefix("xmlns:") {
                    scope.insert(p.to_owned(), v.clone());
                }
            }
            let raw = Raw {
                name,
                attributes,
                children: Vec::new(),
                text: String::new(),
                scope,
            };
            if empty {
                let e = finish(raw)?;
                match stack.last_mut() {
                    Some(p) => p.children.push(e),
                    None => root = Some(e),
                }
            } else {
                stack.push(raw);
            }
            i += k;
        }
    }
    if let Some(open) = stack.last() {
        return Err(format!("<{}> is not closed", open.name));
    }
    root.ok_or_else(|| "no root element".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements_attributes_namespaces_and_entities() {
        let e = parse(
            "<?xml version=\"1.0\"?>\n<!-- c --><m xmlns=\"urn:a\" xmlns:p=\"urn:p\" u='x &amp; y'><i p:id=\"1\"/><t>a&lt;b<![CDATA[<c>]]>&#65;</t></m>",
        )
        .unwrap();
        assert_eq!(e.name, "{urn:a}m");
        assert_eq!(e.attr("u"), Some("x & y"));
        assert_eq!(e.first("i").unwrap().attr("{urn:p}id"), Some("1"));
        assert_eq!(e.first("t").unwrap().text, "a<b<c>A");
        assert!(parse("<a><b></a>").is_err());
    }
}
