//! Where the cursor sits in a design file's text: the word under it, the call it is inside
//! (`hole(`, `counterbore(`, `line(`), the argument being written and the operation the
//! argument registry ([`runmat_cad::syntax::arguments`]) has for that call.
//!
//! The reading is lexical, so it works while the text is mid-edit and does not parse.

use runmat_cad::syntax::arguments::{self, Allowed, Argument, Group, Operation};

/// The identifier around `offset`: its start and end.
pub(super) fn word_at(source: &str, offset: usize) -> Option<(usize, usize)> {
    let bytes = source.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = offset.min(bytes.len());
    while start > 0 && is_word(bytes[start - 1]) {
        start -= 1;
    }
    let mut end = offset.min(bytes.len());
    while end < bytes.len() && is_word(bytes[end]) {
        end += 1;
    }
    (start < end).then_some((start, end))
}

/// An open bracket before the cursor.
#[derive(Clone, Debug)]
pub(super) struct Open {
    pub at: usize,
    pub bracket: u8,
}

/// The brackets open at `offset`, outermost first. Strings and `//` comments are skipped.
pub(super) fn open_brackets(source: &str, offset: usize) -> Vec<Open> {
    let bytes = source.as_bytes();
    let end = offset.min(bytes.len());
    let mut stack: Vec<Open> = Vec::new();
    let mut i = 0;
    while i < end {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < end && bytes[i] != b'"' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < end && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b @ (b'(' | b'[' | b'{') => stack.push(Open { at: i, bracket: b }),
            b')' | b']' | b'}' => {
                stack.pop();
            }
            _ => {}
        }
        i += 1;
    }
    stack
}

/// The dotted callee written just before `paren` (`plane.offset`, `hole`), with its start.
fn callee_before(source: &str, paren: usize) -> Option<(usize, String)> {
    let head = source[..paren].trim_end();
    let start = head
        .char_indices()
        .rev()
        .find(|(_, c)| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '.'))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let callee = &head[start..];
    (!callee.is_empty() && !callee.starts_with('.')).then(|| (start, callee.to_string()))
}

/// Whether `offset` is inside a `sketch … { }` block.
fn in_sketch(source: &str, stack: &[Open]) -> bool {
    stack.iter().any(|o| {
        o.bracket == b'{' && {
            let line_start = source[..o.at].rfind('\n').map_or(0, |i| i + 1);
            source[line_start..o.at].trim_start().starts_with("sketch ")
        }
    })
}

/// What a call's arguments are documented by.
#[derive(Clone, Copy, Debug)]
pub(super) enum Callee {
    Operation(&'static Operation),
    /// A choice written as a call under an operation's argument (`form = counterbore(…)`).
    Choice(&'static Operation, &'static Argument, &'static Allowed),
}

impl Callee {
    pub fn arguments(&self) -> &'static [Argument] {
        match self {
            Callee::Operation(op) => op.arguments,
            Callee::Choice(_, _, c) => c.fields,
        }
    }

    pub fn word(&self) -> &'static str {
        match self {
            Callee::Operation(op) => op.word,
            Callee::Choice(_, _, c) => c.word,
        }
    }

    pub fn summary(&self) -> &'static str {
        match self {
            Callee::Operation(op) => op.summary,
            Callee::Choice(_, _, c) => c.summary,
        }
    }

    pub fn argument(&self, name: &str) -> Option<&'static Argument> {
        self.arguments().iter().find(|a| a.name == name)
    }

    /// `hole(face, at, …)`.
    pub fn signature(&self) -> (String, Vec<&'static Argument>) {
        let args: Vec<&'static Argument> =
            self.arguments().iter().filter(|a| !a.is_nested()).collect();
        let names: Vec<&str> = args.iter().map(|a| a.name).collect();
        (format!("{}({})", self.word(), names.join(", ")), args)
    }
}

/// A call the cursor is inside.
#[derive(Clone, Debug)]
pub(super) struct Call {
    pub callee: Callee,
    /// Where the call's `(` is.
    pub open: usize,
    /// The argument being written: its name when written `name =`, else by position.
    pub argument: Option<String>,
    pub position: usize,
    /// Arguments already written in the call.
    pub written: Vec<String>,
    /// Whether the cursor is after `name =` (writing a value).
    pub in_value: bool,
}

/// The operation a word names where it is written: a feature (or a sketch entity or constraint
/// inside a sketch block), a reference function, or a choice under the enclosing call's
/// argument.
pub(super) fn callee(source: &str, at: usize, word: &str) -> Option<Callee> {
    let stack = open_brackets(source, at);
    let sketch = in_sketch(source, &stack);
    let groups: &[Group] = if sketch {
        &[
            Group::SketchEntity,
            Group::SketchConstraint,
            Group::Reference,
        ]
    } else {
        &[Group::Feature, Group::Reference]
    };
    // A choice under the enclosing call's argument comes first: `form = counterbore(…)`,
    // `layout = sketch(…)` inside a pattern.
    if let Some(outer) = call_at(source, at) {
        if let (Callee::Operation(op), Some(name)) = (outer.callee, outer.argument.as_deref()) {
            if let Some(argument) = op.argument(name) {
                if let Some(choice) = argument.choice(word) {
                    return Some(Callee::Choice(op, argument, choice));
                }
            }
        }
    }
    groups
        .iter()
        .find_map(|g| arguments::operation_in(*g, word))
        .map(Callee::Operation)
}

/// The innermost registered call `offset` is inside.
pub(super) fn call_at(source: &str, offset: usize) -> Option<Call> {
    let stack = open_brackets(source, offset);
    let open = stack.iter().rev().find(|o| o.bracket == b'(')?;
    // A record or list inside the call is part of one argument's value.
    let (start, name) = callee_before(source, open.at)?;
    let callee = callee(source, start, &name)?;
    let inner = &source[open.at + 1..offset];
    let mut position = 0;
    let mut written = Vec::new();
    let mut segment_start = 0;
    let mut depth = 0i32;
    let mut quoted = false;
    for (i, c) in inner.char_indices() {
        match c {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                if let Some(n) = segment_name(&inner[segment_start..i]) {
                    written.push(n);
                }
                position += 1;
                segment_start = i + 1;
            }
            _ => {}
        }
    }
    let segment = &inner[segment_start..];
    let (argument, in_value) = match segment.split_once('=') {
        Some((name, _)) if is_name(name.trim()) => (Some(name.trim().to_string()), true),
        // The cursor inside a name followed by `=`: that argument, before its value.
        _ => {
            let rest = &source[offset.min(source.len())..];
            let tail: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            let name = format!("{}{tail}", segment.trim());
            let after = rest[tail.len()..].trim_start();
            let named = is_name(&name) && after.starts_with('=') && !after.starts_with("==");
            (named.then_some(name), false)
        }
    };
    Some(Call {
        callee,
        open: open.at,
        argument,
        position,
        written,
        in_value,
    })
}

/// The argument name a written segment sets: `name = …` or a bare flag.
fn segment_name(segment: &str) -> Option<String> {
    let text = segment.trim();
    let name = match text.split_once('=') {
        Some((name, _)) => name.trim(),
        None => text,
    };
    is_name(name).then(|| name.to_string())
}

fn is_name(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_know_their_operation_and_argument() {
        let text = "feature h = hole(face = feature.body.face.end, diameter = 5 mm, form = counterbore(depth = ";
        let call = call_at(text, text.len()).unwrap();
        assert_eq!(call.callee.word(), "counterbore");
        assert_eq!(call.argument.as_deref(), Some("depth"));
        assert!(call.in_value);
        let outer = call_at(text, text.find("diameter").unwrap() + 3).unwrap();
        assert_eq!(outer.callee.word(), "hole");
        assert_eq!(outer.written, vec!["face".to_string()]);
        assert_eq!(outer.position, 1);
    }

    #[test]
    fn sketch_blocks_read_sketch_words() {
        let text = "sketch s on origin.xy {\n    c1 = circle(center = [0 mm, 0 mm], ";
        let call = call_at(text, text.len()).unwrap();
        assert_eq!(call.callee.word(), "circle");
        let outside = "feature p = pattern(features = [h], layout = sketch(";
        let call = call_at(outside, outside.len()).unwrap();
        assert!(matches!(call.callee, Callee::Choice(_, a, _) if a.name == "layout"));
    }
}
