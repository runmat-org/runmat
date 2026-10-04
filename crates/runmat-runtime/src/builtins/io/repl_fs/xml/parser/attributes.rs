//! Bounded attribute projection from a quick-xml-delimited start event.
//! The mature reader owns tag framing; this cursor prevents one enormous
//! quoted value from becoming an uninterruptible attribute-iterator call.

use super::*;
use crate::builtins::io::repl_fs::xml::names::{name_char, name_start};
use std::collections::BTreeMap;

pub(super) async fn name(raw: &[u8], cooperation: &mut XmlCooperation) -> BuiltinResult<String> {
    let mut out = String::new();
    let mut position = 0usize;
    while position < raw.len() {
        let mut end = position.saturating_add(4096).min(raw.len());
        let mut continuation = 0;
        while end < raw.len() && raw[end] & 0b1100_0000 == 0b1000_0000 && continuation < 3 {
            end += 1;
            continuation += 1;
        }
        let part = std::str::from_utf8(&raw[position..end]).map_err(|error| {
            build_runtime_error("xmlread: malformed XML (invalid name encoding)")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        out.try_reserve(part.len()).map_err(|error| {
            build_runtime_error("xmlread: cannot reserve XML name")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        for character in part.chars() {
            let valid = if out.is_empty() {
                name_start(character)
            } else {
                name_char(character)
            };
            if !valid {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (invalid name)",
                ));
            }
            out.push(character);
        }
        cooperation.completed(end - position).await?;
        position = end;
    }
    if out.is_empty() {
        return Err(compat_error(
            "xmlread",
            "xmlread: malformed XML (empty name)",
        ));
    }
    Ok(out)
}

pub(super) async fn parse(
    raw: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<BTreeMap<String, String>> {
    Ok(parse_ordered(raw, cooperation).await?.into_iter().collect())
}

pub(super) async fn parse_ordered(
    raw: &[u8],
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<Vec<(String, String)>> {
    let mut attributes = Vec::<(String, String)>::new();
    let mut position = 0usize;
    while position < raw.len() {
        skip_space(raw, &mut position, cooperation).await?;
        if position == raw.len() {
            break;
        }
        let name_start = position;
        scan_until(
            raw,
            &mut position,
            |byte| byte == b'=' || xml_space(byte),
            cooperation,
        )
        .await?;
        let key = name(&raw[name_start..position], cooperation).await?;
        if attributes.iter().any(|(existing, _)| existing == &key) {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (duplicate attribute)",
            ));
        }
        skip_space(raw, &mut position, cooperation).await?;
        if raw.get(position) != Some(&b'=') {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (attribute requires '=')",
            ));
        }
        position += 1;
        skip_space(raw, &mut position, cooperation).await?;
        let quote = match raw.get(position) {
            Some(b'\'' | b'"') => raw[position],
            _ => {
                return Err(compat_error(
                    "xmlread",
                    "xmlread: malformed XML (attribute value must be quoted)",
                ))
            }
        };
        position += 1;
        let value_start = position;
        scan_until(raw, &mut position, |byte| byte == quote, cooperation).await?;
        if position == raw.len() {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (unterminated attribute)",
            ));
        }
        let value =
            super::decode::decode_attribute(&raw[value_start..position], cooperation).await?;
        position += 1;
        if raw.get(position).is_some_and(|byte| !xml_space(*byte)) {
            return Err(compat_error(
                "xmlread",
                "xmlread: malformed XML (attributes require whitespace separation)",
            ));
        }
        attributes.try_reserve(1).map_err(|error| {
            build_runtime_error("xmlread: cannot reserve XML attributes")
                .with_builtin("xmlread")
                .with_source(error)
                .build()
        })?;
        attributes.push((key, value));
    }
    Ok(attributes)
}

async fn skip_space(
    raw: &[u8],
    position: &mut usize,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<()> {
    scan_until(raw, position, |byte| !xml_space(byte), cooperation).await
}

async fn scan_until(
    raw: &[u8],
    position: &mut usize,
    stop: impl Fn(u8) -> bool,
    cooperation: &mut XmlCooperation,
) -> BuiltinResult<()> {
    while *position < raw.len() {
        let end = (*position).saturating_add(4096).min(raw.len());
        let length = raw[*position..end]
            .iter()
            .position(|byte| stop(*byte))
            .unwrap_or(end - *position);
        *position += length;
        cooperation.completed(length).await?;
        if *position < end {
            break;
        }
    }
    Ok(())
}

fn xml_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_attributes(raw: &str) -> BuiltinResult<BTreeMap<String, String>> {
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        futures::executor::block_on(parse(raw.as_bytes(), &mut cooperation))
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn quoted_gt_unicode_names_and_entities_preserve_values() {
        let attrs = parse_attributes(" id='a>b' 温度=\"2 &amp; 3\" ").unwrap();
        assert_eq!(attrs.get("id").map(String::as_str), Some("a>b"));
        assert_eq!(attrs.get("温度").map(String::as_str), Some("2 & 3"));
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn incomplete_duplicate_or_unseparated_attributes_are_rejected() {
        for raw in ["a", "a='x' a='y'", "a='x'b='y'", "a=x", "a='x"] {
            assert!(parse_attributes(raw).is_err(), "invalid: {raw}");
        }
        let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlRead);
        let invalid_name = [b'a', 0x80, 0x80, 0x80, 0x80];
        assert!(futures::executor::block_on(name(&invalid_name, &mut cooperation)).is_err());
    }

    #[test]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn long_single_attribute_observes_execution_cancellation_during_scan() {
        use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
        use runmat_execution::cancellation::CancellationSource;
        use runmat_execution::CancellationReason;
        use std::num::NonZeroU64;

        let source = CancellationSource::new();
        let canceller = source.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || {
                canceller.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
        let raw = format!(" id='{}'", "x".repeat(16 * 1024));
        let error = futures::executor::block_on(parse(raw.as_bytes(), &mut cooperation))
            .expect_err("large attribute scan should yield");
        assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
    }
}
