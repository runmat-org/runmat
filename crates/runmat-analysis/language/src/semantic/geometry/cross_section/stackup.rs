//! The stackup of a board layout (`.circuit.layout`), read for a cross-section.
//!
//! A layout lists each stackup top to bottom:
//!
//! ```text
//! stackup rigid {
//!     finish = ENIG
//!     layer "F.Mask" = { kind = mask, thickness = 0.02, material = solder_mask }
//!     layer "F.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
//!     layer "dielectric 1" = { kind = prepreg, thickness = 0.2, material = { name = "prepreg", dk = 4.1, ... } }
//! }
//! ```
//!
//! A bare length is in millimetres; `um`, `mm`, `mil` and `in` name other units. A material is a
//! built-in name or a record with its `dk`. Only the fields a cross-section uses are read: each
//! layer's name, kind, thickness and Dk.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Copper,
    Dielectric,
    Mask,
    /// Silkscreen and paste: not part of the electrical section.
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Layer {
    pub name: String,
    pub kind: Kind,
    pub thickness_m: f64,
    pub dk: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Stackup {
    pub layers: Vec<Layer>,
}

/// The stackup named `name`, or the layout's only stackup.
pub(super) fn read(text: &str, name: Option<&str>) -> Result<Stackup, String> {
    let tokens = tokens(text)?;
    let mut found = Vec::new();
    let mut index = 0;
    while index + 2 < tokens.len() {
        if tokens[index] == Token::Word("stackup".into()) && tokens[index + 2] == Token::Open {
            let stackup_name = match &tokens[index + 1] {
                Token::Word(word) | Token::Text(word) => word.clone(),
                _ => return Err("a stackup is written `stackup <name> { ... }`".into()),
            };
            let (layers, next) = stackup_body(&tokens, index + 3)?;
            found.push((stackup_name, layers));
            index = next;
        } else {
            index += 1;
        }
    }
    let names = || {
        found
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    match name {
        Some(name) => found
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, layers)| Stackup {
                layers: layers.clone(),
            })
            .ok_or_else(|| format!("has no stackup `{name}`; its stackups are {}", names())),
        None => match found.as_slice() {
            [(_, layers)] => Ok(Stackup {
                layers: layers.clone(),
            }),
            [] => Err("has no stackup".into()),
            _ => Err(format!(
                "has several stackups ({}); name one with `stackup_name`",
                names()
            )),
        },
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Text(String),
    Open,
    Close,
    Equal,
    Comma,
}

fn tokens(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() || c == ';' => {
                chars.next();
            }
            '/' if text_starts_comment(&mut chars) => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '{' | '}' | '=' | ',' => {
                chars.next();
                tokens.push(match c {
                    '{' => Token::Open,
                    '}' => Token::Close,
                    '=' => Token::Equal,
                    _ => Token::Comma,
                });
            }
            '"' => {
                chars.next();
                let mut value = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => value.extend(chars.next()),
                        Some(c) => value.push(c),
                        None => return Err("has an unclosed string".into()),
                    }
                }
                tokens.push(Token::Text(value));
            }
            _ => {
                let mut word = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_whitespace() || matches!(c, '{' | '}' | '=' | ',' | '"' | ';') {
                        break;
                    }
                    word.push(c);
                    chars.next();
                }
                tokens.push(Token::Word(word));
            }
        }
    }
    Ok(tokens)
}

fn text_starts_comment(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> bool {
    let mut ahead = chars.clone();
    ahead.next();
    ahead.peek() == Some(&'/')
}

/// The layers of a stackup body that opens before `start`; returns the index after its `}`.
fn stackup_body(tokens: &[Token], mut index: usize) -> Result<(Vec<Layer>, usize), String> {
    let mut layers = Vec::new();
    loop {
        match tokens.get(index) {
            None => return Err("has an unclosed stackup".into()),
            Some(Token::Close) => return Ok((layers, index + 1)),
            Some(Token::Word(word)) if word == "layer" => {
                let name = match tokens.get(index + 1) {
                    Some(Token::Text(name) | Token::Word(name)) => name.clone(),
                    _ => return Err("a stackup layer is written `layer \"<name>\" = { ... }`".into()),
                };
                if tokens.get(index + 2) != Some(&Token::Equal)
                    || tokens.get(index + 3) != Some(&Token::Open)
                {
                    return Err(format!("layer \"{name}\" is written `layer \"{name}\" = {{ ... }}`"));
                }
                let (record, next) = record(tokens, index + 4)?;
                layers.push(layer(name, &record)?);
                index = next;
            }
            Some(Token::Word(_)) => {
                // `finish = ENIG` and other stackup settings.
                index = skip_value(tokens, index + 1);
            }
            Some(_) => index += 1,
        }
    }
}

#[derive(Debug, Clone)]
enum Value {
    Plain(Vec<String>),
    Record(Vec<(String, Value)>),
}

/// `key = value, ...` up to the closing brace of a record that opened before `index`.
fn record(tokens: &[Token], mut index: usize) -> Result<(Vec<(String, Value)>, usize), String> {
    let mut fields = Vec::new();
    loop {
        match tokens.get(index) {
            None => return Err("has an unclosed record".into()),
            Some(Token::Close) => return Ok((fields, index + 1)),
            Some(Token::Comma) => index += 1,
            Some(Token::Word(key)) if tokens.get(index + 1) == Some(&Token::Equal) => {
                index += 2;
                if tokens.get(index) == Some(&Token::Open) {
                    let (inner, next) = record(tokens, index + 1)?;
                    fields.push((key.clone(), Value::Record(inner)));
                    index = next;
                } else {
                    let mut words = Vec::new();
                    while let Some(Token::Word(word) | Token::Text(word)) = tokens.get(index) {
                        words.push(word.clone());
                        index += 1;
                    }
                    fields.push((key.clone(), Value::Plain(words)));
                }
            }
            Some(_) => return Err("has a record field that is not `key = value`".into()),
        }
    }
}

fn skip_value(tokens: &[Token], mut index: usize) -> usize {
    if tokens.get(index) == Some(&Token::Equal) {
        index += 1;
    }
    let mut depth = 0usize;
    while let Some(token) = tokens.get(index) {
        match token {
            Token::Open => depth += 1,
            Token::Close if depth == 0 => return index,
            Token::Close => {
                depth -= 1;
                if depth == 0 {
                    return index + 1;
                }
            }
            Token::Word(_) | Token::Text(_) if depth == 0 => {
                // One plain value, then the next statement.
                return index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    index
}

fn layer(name: String, fields: &[(String, Value)]) -> Result<Layer, String> {
    let get = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let plain = |key: &str| match get(key) {
        Some(Value::Plain(words)) => Ok(words.clone()),
        _ => Err(format!("layer \"{name}\" needs `{key}`")),
    };
    let kind = match plain("kind")?.as_slice() {
        [kind] => match kind.as_str() {
            "copper" => Kind::Copper,
            "core" | "prepreg" => Kind::Dielectric,
            "mask" => Kind::Mask,
            "silk" | "paste" => Kind::Other,
            other => return Err(format!("layer \"{name}\" has kind `{other}`")),
        },
        _ => return Err(format!("layer \"{name}\" has an invalid `kind`")),
    };
    let thickness_m = length(&plain("thickness")?)
        .ok_or_else(|| format!("layer \"{name}\" has an invalid `thickness`"))?;
    let dk = match get("material") {
        Some(Value::Plain(words)) => match words.as_slice() {
            [material] => builtin_dk(material)
                .ok_or_else(|| format!("layer \"{name}\" names unknown material `{material}`"))?,
            _ => return Err(format!("layer \"{name}\" has an invalid `material`")),
        },
        Some(Value::Record(material)) => material
            .iter()
            .find(|(key, _)| key == "dk")
            .and_then(|(_, value)| match value {
                Value::Plain(words) if words.len() == 1 => words[0].parse::<f64>().ok(),
                _ => None,
            })
            .ok_or_else(|| format!("layer \"{name}\" material has no numeric `dk`"))?,
        None => return Err(format!("layer \"{name}\" needs `material`")),
    };
    Ok(Layer {
        name,
        kind,
        thickness_m,
        dk,
    })
}

/// Dk of the layout's built-in materials.
fn builtin_dk(name: &str) -> Option<f64> {
    Some(match name {
        "copper" => 1.0,
        "fr4" => 4.3,
        "polyimide" => 3.4,
        "solder_mask" => 3.8,
        _ => return None,
    })
}

/// A bare number in millimetres, or a number and a unit.
fn length(words: &[String]) -> Option<f64> {
    let (value, scale) = match words {
        [value] => (value, 1e-3),
        [value, unit] => (
            value,
            match unit.as_str() {
                "mm" => 1e-3,
                "um" => 1e-6,
                "mil" => 25.4e-6,
                "in" => 25.4e-3,
                _ => return None,
            },
        ),
        _ => return None,
    };
    let value = value.parse::<f64>().ok()?;
    (value.is_finite() && value >= 0.0).then_some(value * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYOUT: &str = r#"
layout esc {
    stackup rigid {
        finish = ENIG
        layer "F.Mask" = { kind = mask, thickness = 0.02, material = solder_mask }
        layer "F.Cu" = { kind = copper, thickness = 35 um, weight = 1, material = copper }
        layer "dielectric 1" = { kind = prepreg, thickness = 0.2, material = { name = "FR-4 prepreg", dk = 4.1, df = 0.02 } }
        layer "In1.Cu" = { kind = copper, thickness = 0.035, weight = 1, material = copper }
        layer "dielectric 2" = { kind = core, thickness = 1.2, material = fr4 }
    }
    // a second, flex stackup
    stackup flex {
        layer "F.Cu" = { kind = copper, thickness = 0.018, material = copper }
        layer "core" = { kind = core, thickness = 2 mil, material = polyimide }
    }
}
"#;

    #[test]
    fn reads_a_named_stackup_with_units_and_materials() {
        let stackup = read(LAYOUT, Some("rigid")).unwrap();
        let names = stackup.layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>();
        assert_eq!(
            names,
            ["F.Mask", "F.Cu", "dielectric 1", "In1.Cu", "dielectric 2"]
        );
        assert_eq!(stackup.layers[0].kind, Kind::Mask);
        assert!((stackup.layers[0].dk - 3.8).abs() < 1e-12);
        assert!((stackup.layers[1].thickness_m - 35e-6).abs() < 1e-15);
        assert!((stackup.layers[2].dk - 4.1).abs() < 1e-12);
        assert!((stackup.layers[2].thickness_m - 0.2e-3).abs() < 1e-15);
        assert!((stackup.layers[4].dk - 4.3).abs() < 1e-12);
        let flex = read(LAYOUT, Some("flex")).unwrap();
        assert!((flex.layers[1].thickness_m - 50.8e-6).abs() < 1e-15);
    }

    #[test]
    fn several_stackups_need_a_name() {
        let error = read(LAYOUT, None).unwrap_err();
        assert!(error.contains("rigid, flex"), "{error}");
        assert!(read(LAYOUT, Some("none")).unwrap_err().contains("no stackup `none`"));
    }
}
