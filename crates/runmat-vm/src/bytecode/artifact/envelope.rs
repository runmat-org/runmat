use super::error::invalid;
use super::{InterpreterPayloadError, InterpreterPayloadForm, InterpreterRevisionField};

const HEADER_TERMINATOR: &[u8] = b"\n\n";
const MAX_REVISION_HEADER_BYTES: usize = 512;
const MAX_INTERPRETER_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;

pub(super) fn encode_payload<T: serde::Serialize>(
    form: InterpreterPayloadForm,
    preamble: &[u8],
    revisions: &[(&str, u16)],
    payload: &T,
) -> Result<Vec<u8>, InterpreterPayloadError> {
    let value = serde_json::to_value(payload).map_err(|error| invalid(form, error.to_string()))?;
    let body = serde_json::to_vec(&value).map_err(|error| invalid(form, error.to_string()))?;
    let header_capacity = preamble.len() + revisions.len() * 32 + HEADER_TERMINATOR.len();
    if body.len().saturating_add(header_capacity).saturating_add(1) > MAX_INTERPRETER_PAYLOAD_BYTES
    {
        return Err(invalid(
            form,
            format!("payload exceeds the {MAX_INTERPRETER_PAYLOAD_BYTES}-byte limit"),
        ));
    }
    let mut encoded = Vec::with_capacity(header_capacity + body.len() + 1);
    encoded.extend_from_slice(preamble);
    for (name, revision) in revisions {
        encoded.extend_from_slice(name.as_bytes());
        encoded.push(b'=');
        encoded.extend_from_slice(revision.to_string().as_bytes());
        encoded.push(b'\n');
    }
    encoded.push(b'\n');
    encoded.extend_from_slice(&body);
    encoded.push(b'\n');
    Ok(encoded)
}

pub(super) fn decode_envelope<'a>(
    bytes: &'a [u8],
    form: InterpreterPayloadForm,
    preamble: &[u8],
    expected_revisions: &[(&str, InterpreterRevisionField, u16)],
) -> Result<&'a [u8], InterpreterPayloadError> {
    decode_envelope_bounded(
        bytes,
        form,
        preamble,
        expected_revisions,
        MAX_INTERPRETER_PAYLOAD_BYTES,
    )
}

pub(super) fn decode_envelope_bounded<'a>(
    bytes: &'a [u8],
    form: InterpreterPayloadForm,
    preamble: &[u8],
    expected_revisions: &[(&str, InterpreterRevisionField, u16)],
    maximum_bytes: usize,
) -> Result<&'a [u8], InterpreterPayloadError> {
    if !bytes.starts_with(preamble) {
        if bytes.starts_with(b"runmat-interpreter-") {
            return Err(invalid(
                form,
                "payload preamble names a different artifact form",
            ));
        }
        return Err(InterpreterPayloadError::LegacyPayload { expected: form });
    }
    let after_preamble = &bytes[preamble.len()..];
    let header_end = after_preamble[..after_preamble.len().min(MAX_REVISION_HEADER_BYTES)]
        .windows(HEADER_TERMINATOR.len())
        .position(|window| window == HEADER_TERMINATOR)
        .ok_or_else(|| invalid(form, "missing header terminator"))?;
    let header = &after_preamble[..header_end];
    let mut lines = header.split(|byte| *byte == b'\n');
    for (name, field, expected) in expected_revisions {
        let line = lines
            .next()
            .ok_or_else(|| invalid(form, format!("missing {name} revision")))?;
        let value = line
            .strip_prefix(name.as_bytes())
            .and_then(|suffix| suffix.strip_prefix(b"="))
            .ok_or_else(|| invalid(form, format!("expected {name}=<revision>")))?;
        let value = std::str::from_utf8(value)
            .map_err(|_| invalid(form, format!("{name} revision is not UTF-8")))?;
        let actual = value
            .parse::<u16>()
            .map_err(|_| invalid(form, format!("{name} revision is not a u16")))?;
        if actual != *expected {
            return Err(InterpreterPayloadError::UnsupportedRevision {
                field: *field,
                actual,
                expected: *expected,
            });
        }
    }
    if lines.next().is_some() {
        return Err(invalid(form, "unexpected revision header field"));
    }
    if bytes.len() > maximum_bytes {
        return Err(invalid(
            form,
            format!("payload exceeds the {maximum_bytes}-byte limit"),
        ));
    }
    let encoded_body = &after_preamble[header_end + HEADER_TERMINATOR.len()..];
    let body = encoded_body.strip_suffix(b"\n").ok_or_else(|| {
        invalid(
            form,
            "canonical payload body must end with exactly one newline",
        )
    })?;
    if body.is_empty() {
        return Err(invalid(form, "missing canonical payload body"));
    }
    Ok(body)
}

pub(super) fn decode_canonical_body<T>(
    body: &[u8],
    form: InterpreterPayloadForm,
) -> Result<T, InterpreterPayloadError>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|error| invalid(form, error.to_string()))?;
    let canonical = serde_json::to_vec(&value).map_err(|error| invalid(form, error.to_string()))?;
    if canonical != body {
        return Err(invalid(
            form,
            "payload body is valid JSON but not canonical RunMat JSON",
        ));
    }
    let decoded: T =
        serde_json::from_value(value.clone()).map_err(|error| invalid(form, error.to_string()))?;
    let normalized =
        serde_json::to_value(&decoded).map_err(|error| invalid(form, error.to_string()))?;
    if normalized != value {
        return Err(invalid(
            form,
            "payload body omits or aliases fields from the current canonical representation",
        ));
    }
    Ok(decoded)
}
