use std::collections::BTreeSet;

use crate::{
    BuiltinExampleFixtureId, BuiltinFilesystemEntry, BuiltinFilesystemFixture,
    BuiltinFixtureContent, BuiltinLoopbackScenario,
};

use super::{push, BuiltinCatalogValidationError};

const MAX_FILESYSTEM_ENTRIES: usize = 64;
const MAX_LOOPBACK_EXCHANGES: usize = 32;
const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;
const MAX_RELATIVE_PATH_BYTES: usize = 1024;
const MAX_PATH_COMPONENT_BYTES: usize = 240;
const MAX_HTTP_HEADERS: usize = 64;
const MAX_HTTP_PATH_BYTES: usize = 2048;
const MAX_HTTP_HEADER_NAME_BYTES: usize = 256;
const MAX_HTTP_HEADER_VALUE_BYTES: usize = 8192;

pub(super) fn validate_filesystem(
    builtin: &'static str,
    fixture: BuiltinFilesystemFixture,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    validate_fixture_id(builtin, fixture.id, errors);
    validate_filesystem_entries(builtin, fixture.entries, true, errors);
}

pub(super) fn validate_filesystem_entries(
    builtin: &'static str,
    entries: &[BuiltinFilesystemEntry],
    require_nonempty: bool,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if (require_nonempty && entries.is_empty()) || entries.len() > MAX_FILESYSTEM_ENTRIES {
        push(
            errors,
            builtin,
            if require_nonempty {
                "filesystem fixture must contain between 1 and 64 entries"
            } else {
                "filesystem fixture must contain no more than 64 entries"
            },
        );
    }
    let mut paths = BTreeSet::new();
    let mut ordered_paths = Vec::new();
    let mut aggregate_payload = 0usize;
    for entry in entries {
        let (path, payload_len, is_file) = match entry {
            BuiltinFilesystemEntry::Directory { relative_path } => (*relative_path, 0, false),
            BuiltinFilesystemEntry::File {
                relative_path,
                content,
            } => {
                let length = match content {
                    BuiltinFixtureContent::Utf8(value) => value.len(),
                    BuiltinFixtureContent::Bytes(value) => value.len(),
                };
                (*relative_path, length, true)
            }
        };
        if !valid_relative_path(path) {
            push(
                errors,
                builtin,
                "filesystem fixture paths must be normalized relative paths",
            );
        }
        if matches!(path, "example.m" | "runmat.toml") {
            push(
                errors,
                builtin,
                "filesystem fixture path is reserved by the example runner",
            );
        }
        if !paths.insert(path) {
            push(errors, builtin, "filesystem fixture paths must be unique");
        }
        if payload_len > MAX_PAYLOAD_BYTES {
            push(
                errors,
                builtin,
                "filesystem fixture payload exceeds the one-megabyte bound",
            );
        }
        aggregate_payload = aggregate_payload.saturating_add(payload_len);
        ordered_paths.push((path, is_file));
    }
    if !ordered_paths.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        push(
            errors,
            builtin,
            "filesystem fixture entries must use canonical path order",
        );
    }
    if ordered_paths.iter().any(|(file, is_file)| {
        *is_file
            && ordered_paths.iter().any(|(path, _)| {
                path.starts_with(*file) && path.as_bytes().get(file.len()) == Some(&b'/')
            })
    }) {
        push(
            errors,
            builtin,
            "filesystem fixture cannot place entries below a file",
        );
    }
    if aggregate_payload > MAX_PAYLOAD_BYTES {
        push(
            errors,
            builtin,
            "filesystem fixture aggregate payload exceeds the one-megabyte bound",
        );
    }
}

pub(super) fn validate_loopback(
    builtin: &'static str,
    scenario: BuiltinLoopbackScenario,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    match scenario {
        BuiltinLoopbackScenario::Http(scenario) => {
            if scenario.exchanges.is_empty() || scenario.exchanges.len() > MAX_LOOPBACK_EXCHANGES {
                push(
                    errors,
                    builtin,
                    "HTTP fixture must contain between 1 and 32 exchanges",
                );
            }
            let mut aggregate_payload = 0usize;
            for exchange in scenario.exchanges {
                if !exchange.request.path.starts_with('/')
                    || exchange.request.path.contains(char::is_whitespace)
                    || exchange.request.path.len() > MAX_HTTP_PATH_BYTES
                {
                    push(errors, builtin, "HTTP fixture request path is invalid");
                }
                if exchange.response.status < 100 || exchange.response.status > 599 {
                    push(errors, builtin, "HTTP fixture response status is invalid");
                }
                if exchange.request.body.map_or(0, |body| body.len()) > MAX_PAYLOAD_BYTES
                    || exchange.response.body.len() > MAX_PAYLOAD_BYTES
                {
                    push(errors, builtin, "HTTP fixture payload exceeds its bound");
                }
                aggregate_payload = aggregate_payload
                    .saturating_add(exchange.request.body.map_or(0, |body| body.len()))
                    .saturating_add(exchange.response.body.len());
                let mut headers = BTreeSet::new();
                if exchange.response.headers.len() > MAX_HTTP_HEADERS {
                    push(
                        errors,
                        builtin,
                        "HTTP fixture has too many response headers",
                    );
                }
                for header in exchange.response.headers {
                    let folded = header.name.to_ascii_lowercase();
                    if !valid_http_header_name(header.name)
                        || header.value.chars().any(|character| {
                            character == '\u{7f}' || (character.is_control() && character != '\t')
                        })
                        || header.name.len() > MAX_HTTP_HEADER_NAME_BYTES
                        || header.value.len() > MAX_HTTP_HEADER_VALUE_BYTES
                        || !headers.insert(folded)
                    {
                        push(errors, builtin, "HTTP fixture response header is invalid");
                    }
                }
            }
            if aggregate_payload > MAX_PAYLOAD_BYTES {
                push(
                    errors,
                    builtin,
                    "HTTP fixture aggregate payload exceeds its bound",
                );
            }
        }
        BuiltinLoopbackScenario::Tcp(scenario) => {
            if scenario.exchanges.is_empty() || scenario.exchanges.len() > MAX_LOOPBACK_EXCHANGES {
                push(
                    errors,
                    builtin,
                    "TCP fixture must contain between 1 and 32 exchanges",
                );
            }
            let mut aggregate_payload = 0usize;
            for exchange in scenario.exchanges {
                if exchange.client_bytes.len() > MAX_PAYLOAD_BYTES
                    || exchange.server_bytes.len() > MAX_PAYLOAD_BYTES
                {
                    push(errors, builtin, "TCP fixture payload exceeds its bound");
                }
                aggregate_payload = aggregate_payload
                    .saturating_add(exchange.client_bytes.len())
                    .saturating_add(exchange.server_bytes.len());
            }
            if aggregate_payload > MAX_PAYLOAD_BYTES {
                push(
                    errors,
                    builtin,
                    "TCP fixture aggregate payload exceeds its bound",
                );
            }
        }
    }
}

pub(super) fn validate_fixture_id(
    builtin: &'static str,
    id: BuiltinExampleFixtureId,
    errors: &mut Vec<BuiltinCatalogValidationError>,
) {
    if id.local_name.is_empty()
        || id.local_name.len() > 64
        || !id
            .local_name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        push(
            errors,
            builtin,
            "fixture local name must use lowercase ASCII letters, digits, and hyphens",
        );
    }
}

pub(super) fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= MAX_RELATIVE_PATH_BYTES
        && !path.starts_with('/')
        && !path.contains('\\')
        && !path.contains('\0')
        && path.split('/').all(valid_path_component)
}

fn valid_path_component(component: &str) -> bool {
    if component.is_empty()
        || component.len() > MAX_PATH_COMPONENT_BYTES
        || component == "."
        || component == ".."
        || component.ends_with('.')
        || component.ends_with(' ')
        || component
            .chars()
            .any(|character| character.is_control() || ":*?\"<>|".contains(character))
    {
        return false;
    }
    let stem = component.split('.').next().unwrap_or(component);
    !matches!(
        stem.to_ascii_uppercase().as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

fn valid_http_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}
