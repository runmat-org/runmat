//! XML 1.0 declaration policy, separate from the event reader's syntax framing.

use super::*;
use quick_xml::events::BytesDecl;

pub(super) struct XmlDeclaration {
    pub(super) encoding: Option<String>,
}

pub(super) async fn parse(
    declaration: &BytesDecl<'_>,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<XmlDeclaration> {
    let raw = declaration.as_ref();
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
    let mut standalone_seen = false;
    for (name, value) in fields {
        match name.as_str() {
            "encoding" if encoding.is_none() && !standalone_seen => {
                let mut chars = value.chars();
                if !chars.next().is_some_and(|ch| ch.is_ascii_alphabetic())
                    || !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
                {
                    return Err(malformed("invalid XML encoding label"));
                }
                encoding = Some(value);
            }
            "standalone" if !standalone_seen && matches!(value.as_str(), "yes" | "no") => {
                standalone_seen = true;
            }
            _ => return Err(malformed("invalid XML declaration field or order")),
        }
    }
    Ok(XmlDeclaration { encoding })
}

fn malformed(reason: &str) -> RuntimeError {
    compat_error("xmlread", format!("xmlread: malformed XML ({reason})"))
}
