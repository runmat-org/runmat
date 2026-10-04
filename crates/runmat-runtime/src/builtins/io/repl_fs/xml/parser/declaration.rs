//! XML 1.0 declaration policy, separate from the event reader's syntax framing.

use super::*;
use quick_xml::events::BytesDecl;
use quick_xml::events::Event;
use quick_xml::Reader;

pub(super) struct XmlDeclaration {
    pub(super) encoding: Option<String>,
    pub(super) standalone: Option<bool>,
}

pub(super) async fn parse(
    declaration: &BytesDecl<'_>,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlDeclaration> {
    let raw: &[u8] = declaration.as_ref();
    if raw.contains(&b'&') {
        return Err(malformed(
            "references are not allowed in XML declaration values",
        ));
    }
    let attributes = super::attributes::parse_ordered(
        raw.strip_prefix(b"xml")
            .ok_or_else(|| malformed("invalid XML declaration target"))?,
        cooperation,
    )
    .await?;
    let mut fields = attributes.into_iter();
    match fields.next() {
        Some((name, value)) if name == "version" && value == "1.0" => {}
        _ => return Err(malformed("XML 1.0 version must be first in declaration")),
    }
    let mut encoding = None;
    let mut standalone = None;
    for (name, value) in fields {
        match name.as_str() {
            "encoding" if encoding.is_none() && standalone.is_none() => {
                let mut chars = value.chars();
                if !chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
                    || !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
                {
                    return Err(malformed("invalid XML encoding label"));
                }
                encoding = Some(value);
            }
            "standalone" if standalone.is_none() && matches!(value.as_str(), "yes" | "no") => {
                standalone = Some(value == "yes");
            }
            _ => return Err(malformed("invalid XML declaration field or order")),
        }
    }
    Ok(XmlDeclaration {
        encoding,
        standalone,
    })
}

fn malformed(reason: &str) -> RuntimeError {
    compat_error("xmlread", format!("xmlread: malformed XML ({reason})"))
}

/// The public file form writes UTF-8. Rebuild only a leading, typed XML
/// declaration; never publish UTF-8 bytes still labelled as UTF-16 or a
/// legacy source encoding. The returned text form follows the same policy.
pub(in crate::builtins::io::repl_fs::xml) async fn normalize_utf8_output(
    source: String,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<String> {
    let Some((prefix, suffix_start)) = normalized_leading_declaration(&source, cooperation).await?
    else {
        return Ok(source);
    };
    let had_bom = source.starts_with('\u{FEFF}');
    let body = source.strip_prefix('\u{FEFF}').unwrap_or(&source);
    let mut out = prefix;
    let suffix = &body[suffix_start..];
    let mut position = 0usize;
    while position < suffix.len() {
        let mut end = position.saturating_add(4096).min(suffix.len());
        while end < suffix.len() && !suffix.is_char_boundary(end) {
            end += 1;
        }
        out.try_reserve(end - position).map_err(|error| {
            build_runtime_error("xmlwrite: cannot reserve UTF-8 XML output")
                .with_builtin("xmlwrite")
                .with_source(error)
                .build()
        })?;
        out.push_str(&suffix[position..end]);
        cooperation.completed(end - position).await?;
        position = end;
    }
    debug_assert_eq!(had_bom, out.starts_with('\u{FEFF}'));
    Ok(out)
}

/// A Document's tree is the serialization authority. Its retained Text may
/// contribute only a parsed declaration, never a stale copy of its body.
pub(in crate::builtins::io::repl_fs::xml) async fn document_declaration_prefix(
    source: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Option<String>> {
    Ok(normalized_leading_declaration(source, cooperation)
        .await?
        .map(|(prefix, _)| prefix))
}

async fn normalized_leading_declaration(
    source: &str,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Option<(String, usize)>> {
    let had_bom = source.starts_with('\u{FEFF}');
    let body = source.strip_prefix('\u{FEFF}').unwrap_or(source);
    let mut reader = Reader::from_reader(super::reader::CooperativeXmlBuf::new(
        body.as_bytes(),
        cooperation.budget_owner(),
    ));
    let mut buffer = Vec::new();
    let event = reader
        .read_event_into_async(&mut buffer)
        .await
        .map_err(|error| {
            build_runtime_error("xmlwrite: invalid XML declaration")
                .with_builtin("xmlwrite")
                .with_source(super::events::quick_xml_error(error, cooperation))
                .build()
        })?;
    let Event::Decl(declaration) = event else {
        return Ok(None);
    };
    let fields = parse(&declaration, cooperation).await.map_err(|error| {
        build_runtime_error("xmlwrite: invalid XML declaration")
            .with_builtin("xmlwrite")
            .with_source(error)
            .build()
    })?;
    let suffix_start = usize::try_from(reader.buffer_position())
        .ok()
        .filter(|position| *position <= body.len())
        .ok_or_else(|| {
            build_runtime_error("xmlwrite: XML declaration extent is invalid")
                .with_builtin("xmlwrite")
                .build()
        })?;
    let mut out = String::new();
    if had_bom {
        out.push('\u{FEFF}');
    }
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"");
    if let Some(standalone) = fields.standalone {
        out.push_str(if standalone {
            " standalone=\"yes\""
        } else {
            " standalone=\"no\""
        });
    }
    out.push_str("?>");
    Ok(Some((out, suffix_start)))
}
