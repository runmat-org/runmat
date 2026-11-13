//! Shared, typed admission and filtering for workspace-name queries.
//! Selector values are text/container metadata; numeric resident payloads are
//! rejected before any provider transfer.

use glob::Pattern;
use regex::Regex;
use runmat_builtins::{BuiltinDescriptor, BuiltinErrorDescriptor, WHOS_DESCRIPTOR, WHO_DESCRIPTOR};
use runmat_value::{CharArray, Value};
use std::path::PathBuf;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkspaceQueryOwner {
    Who,
    Whos,
}

impl WorkspaceQueryOwner {
    fn name(self) -> &'static str {
        match self {
            Self::Who => "who",
            Self::Whos => "whos",
        }
    }

    fn descriptor(self) -> &'static BuiltinDescriptor {
        match self {
            Self::Who => &WHO_DESCRIPTOR,
            Self::Whos => &WHOS_DESCRIPTOR,
        }
    }

    fn error(self, kind: QueryError) -> &'static BuiltinErrorDescriptor {
        &self.descriptor().errors[kind as usize]
    }

    fn reject(self, kind: QueryError) -> RuntimeError {
        crate::runtime_descriptor_error(self.name(), self.error(kind))
    }

    fn detail(self, kind: QueryError, detail: impl AsRef<str>) -> RuntimeError {
        let descriptor = self.error(kind);
        let mut builder =
            build_runtime_error(format!("{} {}", descriptor.message, detail.as_ref()))
                .with_builtin(self.name());
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

// The two canonical descriptors have the same ordered error roles. This enum
// indexes those declared roles; it does not manufacture identifier strings.
#[repr(usize)]
#[derive(Clone, Copy)]
enum QueryError {
    FileRequiresFilename,
    FileDuplicate,
    RegexpRequiresPattern,
    RegexpEmptyPattern,
    RegexpInvalid,
    UnsupportedOption,
    FilenameType,
    CellMemberType,
    SelectorType,
    WildcardInvalid,
}

#[derive(Debug)]
pub(super) enum WorkspaceQuerySource {
    Workspace,
    File(PathBuf),
}

#[derive(Debug)]
enum NameSelector {
    Exact(String),
    Wildcard(Pattern),
}

#[derive(Debug)]
pub(super) struct WorkspaceQuery {
    pub source: WorkspaceQuerySource,
    selectors: Vec<NameSelector>,
    regex_patterns: Vec<Regex>,
    pub only_global: bool,
}

impl WorkspaceQuery {
    pub fn matches(&self, name: &str) -> bool {
        if self.selectors.is_empty() && self.regex_patterns.is_empty() {
            return true;
        }
        self.selectors.iter().any(|selector| match selector {
            NameSelector::Exact(expected) => name == expected,
            NameSelector::Wildcard(pattern) => pattern.matches(name),
        }) || self.regex_patterns.iter().any(|regex| regex.is_match(name))
    }
}

#[derive(Debug)]
enum OptionToken {
    File,
    Regexp,
    Unsupported(String),
}

fn option_token(value: &Value, owner: WorkspaceQueryOwner) -> BuiltinResult<Option<OptionToken>> {
    // A matrix supplies one selector per row, not a scalar option token.
    // Scalar-only positions (filenames and cell members) still reject it.
    if matches!(value, Value::CharArray(array) if array.rows != 1) {
        return Ok(None);
    }
    let Some(text) = value_to_string_scalar(value, owner, QueryError::SelectorType)? else {
        return Ok(None);
    };
    if !text.starts_with('-') {
        return Ok(None);
    }
    Ok(Some(match text.to_ascii_lowercase().as_str() {
        "-file" => OptionToken::File,
        "-regexp" => OptionToken::Regexp,
        _ => OptionToken::Unsupported(text.to_ascii_lowercase()),
    }))
}

pub(super) fn parse_request(
    values: crate::call::FunctionArgumentView<'_>,
    owner: WorkspaceQueryOwner,
) -> BuiltinResult<WorkspaceQuery> {
    let mut index = 0usize;
    let mut path: Option<&Value> = None;
    let mut names = Vec::new();
    let mut regex_patterns = Vec::new();
    let mut only_global = false;
    while index < values.len() {
        if let Some(token) = option_token(&values[index], owner)? {
            match token {
                OptionToken::File => {
                    index += 1;
                    if index >= values.len() {
                        return Err(owner.reject(QueryError::FileRequiresFilename));
                    }
                    if path.is_some() {
                        return Err(owner.reject(QueryError::FileDuplicate));
                    }
                    path = Some(&values[index]);
                    index += 1;
                    continue;
                }
                OptionToken::Regexp => {
                    index += 1;
                    let before = regex_patterns.len();
                    while index < values.len() && option_token(&values[index], owner)?.is_none() {
                        let candidates = extract_name_list(&values[index], owner)?;
                        if candidates.is_empty() {
                            return Err(owner.reject(QueryError::RegexpEmptyPattern));
                        }
                        for candidate in candidates {
                            let regex = Regex::new(&candidate).map_err(|cause| {
                                let mut error = build_runtime_error(format!(
                                    "{}: invalid regular expression '{candidate}': {cause}",
                                    owner.name()
                                ))
                                .with_builtin(owner.name());
                                if let Some(identifier) =
                                    owner.error(QueryError::RegexpInvalid).identifier
                                {
                                    error = error.with_identifier(identifier);
                                }
                                error.with_source(cause).build()
                            })?;
                            regex_patterns.push(regex);
                        }
                        index += 1;
                    }
                    if regex_patterns.len() == before {
                        return Err(owner.reject(QueryError::RegexpRequiresPattern));
                    }
                    continue;
                }
                OptionToken::Unsupported(other) => {
                    return Err(owner.detail(QueryError::UnsupportedOption, format!("'{other}'")));
                }
            }
        }
        let extracted = extract_name_list(&values[index], owner)?;
        if extracted.len() == 1
            && extracted[0].eq_ignore_ascii_case("global")
            && names.is_empty()
            && regex_patterns.is_empty()
            && path.is_none()
        {
            only_global = true;
        } else {
            names.extend(extracted);
        }
        index += 1;
    }

    let source = if let Some(value) = path {
        let text = value_to_string_scalar(value, owner, QueryError::FilenameType)?
            .ok_or_else(|| owner.reject(QueryError::FilenameType))?;
        let mut path = PathBuf::from(text);
        if path.extension().is_none() {
            path.set_extension("mat");
        }
        WorkspaceQuerySource::File(path)
    } else {
        WorkspaceQuerySource::Workspace
    };
    let mut selectors = Vec::with_capacity(names.len());
    for name in names {
        if name.chars().any(|ch| matches!(ch, '*' | '?' | '[')) {
            let pattern = Pattern::new(&name).map_err(|cause| {
                let mut error = build_runtime_error(format!(
                    "{}: invalid pattern '{name}': {cause}",
                    owner.name()
                ))
                .with_builtin(owner.name());
                if let Some(identifier) = owner.error(QueryError::WildcardInvalid).identifier {
                    error = error.with_identifier(identifier);
                }
                error.with_source(cause).build()
            })?;
            selectors.push(NameSelector::Wildcard(pattern));
        } else {
            selectors.push(NameSelector::Exact(name));
        }
    }
    Ok(WorkspaceQuery {
        source,
        selectors,
        regex_patterns,
        only_global,
    })
}

fn extract_name_list(value: &Value, owner: WorkspaceQueryOwner) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(vec![text.as_str().to_owned()])
        }
        Value::CharArray(array) => char_array_rows_as_strings(array, owner),
        Value::StringArray(array) if array.data().iter().all(|text| !text.is_missing()) => {
            Ok(array
                .data()
                .iter()
                .map(|text| text.as_str().expect("checked nonmissing string").to_owned())
                .collect())
        }
        Value::Cell(array) => array
            .data()
            .iter()
            .map(|item| {
                value_to_string_scalar(item, owner, QueryError::CellMemberType)?
                    .ok_or_else(|| owner.reject(QueryError::CellMemberType))
            })
            .collect(),
        _ => Err(owner.reject(QueryError::SelectorType)),
    }
}

fn value_to_string_scalar(
    value: &Value,
    owner: WorkspaceQueryOwner,
    error: QueryError,
) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(array) if array.rows == 1 => {
            Ok(char_array_rows_as_strings(array, owner)?.into_iter().next())
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        Value::CharArray(_) => Err(owner.reject(error)),
        _ => Ok(None),
    }
}

pub(super) fn char_array_rows_as_strings(
    array: &CharArray,
    owner: WorkspaceQueryOwner,
) -> BuiltinResult<Vec<String>> {
    let mut rows = Vec::with_capacity(array.rows);
    for row in 0..array.rows {
        let start = row * array.cols;
        let value =
            super::text::decode_utf16_with_error(&array.data()[start..start + array.cols], || {
                owner.reject(QueryError::SelectorType)
            })?;
        rows.push(value.trim_end_matches([' ', '\0']).to_string());
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{CellArray, Tensor};

    #[test]
    fn typed_parser_preserves_union_and_owner() {
        let query = parse_request(
            crate::call::FunctionArgumentView::from_slice(&[
                Value::from("a*"),
                Value::from("-regexp"),
                Value::from("^b"),
            ]),
            WorkspaceQueryOwner::Who,
        )
        .expect("query");
        assert!(query.matches("alpha"));
        assert!(query.matches("beta"));
        assert!(!query.matches("gamma"));
        let invalid = parse_request(
            crate::call::FunctionArgumentView::from_slice(&[Value::from("-regexp")]),
            WorkspaceQueryOwner::Whos,
        )
        .expect_err("missing regex");
        assert_eq!(invalid.context.builtin.as_deref(), Some("whos"));
    }

    #[test]
    fn canonical_descriptor_roles_match_both_query_owners() {
        for owner in [WorkspaceQueryOwner::Who, WorkspaceQueryOwner::Whos] {
            assert!(owner.descriptor().errors.len() >= 10);
            assert!(owner
                .error(QueryError::FileRequiresFilename)
                .code
                .ends_with("FILE_REQUIRES_FILENAME"));
            assert!(owner
                .error(QueryError::SelectorType)
                .code
                .ends_with("SELECTOR_ARG_TYPE"));
            assert!(owner
                .error(QueryError::WildcardInvalid)
                .code
                .ends_with("PATTERN_INVALID"));
        }
    }

    #[test]
    fn numeric_and_nested_numeric_selectors_reject_without_provider_access() {
        let numeric = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("tensor"));
        assert!(parse_request(
            crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(&numeric)),
            WorkspaceQueryOwner::Who
        )
        .is_err());
        let cell = CellArray::new(vec![numeric], 1, 1).expect("cell");
        assert!(parse_request(
            crate::call::FunctionArgumentView::from_slice(&[Value::Cell(cell)]),
            WorkspaceQueryOwner::Whos
        )
        .is_err());
    }
}
