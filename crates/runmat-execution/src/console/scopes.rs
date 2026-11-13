//! Independent source inventories never imply ordering between attempts.
use super::ConsoleTranscript;
use crate::identity::AttemptId;
use crate::{ExecutionScopeId, PoolId, TaskId};
use serde::{Deserialize, Serialize};

pub const MAX_CONSOLE_SCOPES: usize = 1024;

/// Resolve the physical root after source identity and ownership validation.
pub fn original_root<'a>(
    root: Option<&'a ConsoleTranscript>,
    scopes: &'a [ScopedConsoleTranscript],
    delegated: Option<ConsoleSourceIdentity>,
) -> Option<&'a ConsoleTranscript> {
    root.or_else(|| {
        let identity = delegated?;
        scopes
            .iter()
            .find(|scope| scope.identity() == identity)
            .map(|scope| scope.transcript.as_ref())
    })
}

/// The enclosing invocation owns its root controls. Descendants retain their
/// independently admitted controls; their total inventory is bounded separately.
pub fn validate_source_controls(
    root: Option<&ConsoleTranscript>,
    scopes: &[ScopedConsoleTranscript],
    delegated: Option<ConsoleSourceIdentity>,
    expected: Option<super::ConsoleLimits>,
    enclosing_attempt: AttemptId,
) -> Result<(), crate::ContractError> {
    if root.is_some_and(|capture| Some(capture.limits()) != expected)
        || (expected.is_none() && (!scopes.is_empty() || delegated.is_some()))
        || scopes
            .iter()
            .any(|scope| scope.attempt_id == enclosing_attempt)
    {
        return Err(crate::ContractError::invalid(
            "console controls",
            "source ownership or requested root controls differ",
        ));
    }
    if let Some(identity) = delegated {
        let original = scopes
            .iter()
            .find(|scope| scope.identity() == identity)
            .ok_or_else(|| {
                crate::ContractError::invalid(
                    "console controls",
                    "delegated root source is missing",
                )
            })?;
        if Some(original.transcript.limits()) != expected {
            return Err(crate::ContractError::invalid(
                "console controls",
                "delegated root controls differ from the invocation",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsoleSourceIdentity {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub task_id: TaskId,
    pub attempt_id: AttemptId,
}

impl ScopedConsoleTranscript {
    pub fn identity(&self) -> ConsoleSourceIdentity {
        ConsoleSourceIdentity {
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            task_id: self.task_id,
            attempt_id: self.attempt_id,
        }
    }
}

pub fn deserialize_source_identity<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<ConsoleSourceIdentity>, D::Error> {
    Option::<ConsoleSourceIdentity>::deserialize(deserializer)
}

pub fn validate_delegated_root(
    root: Option<&ConsoleTranscript>,
    scopes: &[ScopedConsoleTranscript],
    delegated: Option<ConsoleSourceIdentity>,
    successful: bool,
) -> Result<(), crate::ContractError> {
    if let Some(identity) = delegated {
        if root.is_some() {
            return Err(crate::ContractError::invalid(
                "delegated console root",
                "also carries an outer root transcript",
            ));
        }
        let mut matches = scopes.iter().filter(|scope| scope.identity() == identity);
        let source = matches.next().ok_or_else(|| {
            crate::ContractError::invalid(
                "delegated console root",
                "original physical root source is missing",
            )
        })?;
        if matches.next().is_some()
            || (successful && source.transcript.completion() != super::ConsoleCompletion::Complete)
        {
            return Err(crate::ContractError::invalid(
                "delegated console root",
                "duplicate or incomplete original physical root source",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedConsoleTranscript {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub task_id: TaskId,
    pub attempt_id: AttemptId,
    pub transcript: Box<ConsoleTranscript>,
}

#[derive(Serialize)]
pub struct ScopedConsoleTranscriptRef<'a> {
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub task_id: TaskId,
    pub attempt_id: AttemptId,
    pub transcript: &'a ConsoleTranscript,
}

/// Required, finite inventory; omitted evidence is never an empty-complete claim.
pub fn deserialize_scopes<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<ScopedConsoleTranscript>, D::Error> {
    struct Scopes;
    impl<'de> serde::de::Visitor<'de> for Scopes {
        type Value = Vec<ScopedConsoleTranscript>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a bounded per-attempt console inventory")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            let mut result = Vec::new();
            let mut records = 0_u64;
            let mut text = 0_u64;
            while let Some(scope) = sequence.next_element()? {
                if result.len() >= MAX_CONSOLE_SCOPES {
                    return Err(serde::de::Error::custom(
                        "console scope inventory exceeds its bound",
                    ));
                }
                let scope: ScopedConsoleTranscript = scope;
                scope
                    .transcript
                    .validate()
                    .map_err(serde::de::Error::custom)?;
                records += scope.transcript.records().len() as u64;
                text += scope.transcript.text_bytes();
                if records > u64::from(super::MAX_CONSOLE_RECORDS)
                    || text > super::MAX_CONSOLE_TEXT_BYTES
                {
                    return Err(serde::de::Error::custom(
                        "console source inventory exceeds its supported aggregate bound",
                    ));
                }
                result.push(scope);
            }
            Ok(result)
        }
    }
    deserializer.deserialize_seq(Scopes)
}

pub fn validate_scopes(scopes: &[ScopedConsoleTranscript]) -> Result<(), crate::ContractError> {
    let mut records = 0_u64;
    let mut text = 0_u64;
    let mut identities = std::collections::BTreeSet::new();
    if scopes.len() > MAX_CONSOLE_SCOPES {
        return Err(crate::ContractError::invalid(
            "console scopes",
            "inventory exceeds its bound",
        ));
    }
    for scope in scopes {
        scope.transcript.validate()?;
        if !identities.insert((
            scope.scope_id,
            scope.pool_id,
            scope.task_id,
            scope.attempt_id,
        )) {
            return Err(crate::ContractError::invalid(
                "console scopes",
                "duplicate source identity",
            ));
        }
        records += scope.transcript.records().len() as u64;
        text += scope.transcript.text_bytes();
        if records > u64::from(super::MAX_CONSOLE_RECORDS) || text > super::MAX_CONSOLE_TEXT_BYTES {
            return Err(crate::ContractError::invalid(
                "console scopes",
                "inventory exceeds its supported aggregate bound",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod delegated_root_tests {
    use super::*;
    #[test]
    fn descendant_controls_are_independent_but_delegated_root_controls_are_exact() {
        let parent = super::super::ConsoleLimits {
            max_records: 32,
            max_text_bytes: 4096,
        };
        let child = super::super::ConsoleLimits {
            max_records: 8,
            max_text_bytes: 1000,
        };
        let root = super::super::ConsoleCapture::new(parent)
            .unwrap()
            .finish(true);
        let scope = ScopedConsoleTranscript {
            scope_id: ExecutionScopeId::derive(&[b"child"]),
            pool_id: PoolId::derive(&[b"child"]),
            task_id: TaskId::derive(&[b"child"]),
            attempt_id: AttemptId::derive(&[b"child"]),
            transcript: Box::new(
                super::super::ConsoleCapture::new(child)
                    .unwrap()
                    .finish(true),
            ),
        };
        let enclosing = AttemptId::derive(&[b"parent"]);
        let marker = scope.identity();
        let scopes = [scope];
        assert!(
            validate_source_controls(Some(&root), &scopes, None, Some(parent), enclosing).is_ok()
        );
        assert!(
            validate_source_controls(None, &scopes, Some(marker), Some(child), enclosing).is_ok()
        );
        assert!(
            validate_source_controls(None, &scopes, Some(marker), Some(parent), enclosing).is_err()
        );
        assert!(validate_source_controls(None, &scopes, None, None, enclosing).is_err());
    }
    #[test]
    fn delegated_root_requires_one_exact_original_complete_source_on_success() {
        let identity = ConsoleSourceIdentity {
            scope_id: ExecutionScopeId::derive(&[b"physical"]),
            pool_id: PoolId::derive(&[b"physical"]),
            task_id: TaskId::derive(&[b"physical"]),
            attempt_id: AttemptId::derive(&[b"physical"]),
        };
        let make_scope = |completed| ScopedConsoleTranscript {
            scope_id: identity.scope_id,
            pool_id: identity.pool_id,
            task_id: identity.task_id,
            attempt_id: identity.attempt_id,
            transcript: Box::new(
                super::super::ConsoleCapture::new(super::super::ConsoleLimits {
                    max_records: 4,
                    max_text_bytes: 64,
                })
                .unwrap()
                .finish(completed),
            ),
        };
        assert!(validate_delegated_root(None, &[make_scope(true)], Some(identity), true).is_ok());
        assert!(validate_delegated_root(None, &[], Some(identity), true).is_err());
        assert!(validate_delegated_root(
            None,
            &[make_scope(true), make_scope(true)],
            Some(identity),
            true
        )
        .is_err());
        assert!(validate_delegated_root(None, &[make_scope(false)], Some(identity), true).is_err());
        assert!(validate_delegated_root(None, &[make_scope(false)], Some(identity), false).is_ok());
        let root = make_scope(true);
        assert!(validate_delegated_root(
            Some(&root.transcript),
            &[make_scope(true)],
            Some(identity),
            true
        )
        .is_err());
    }
}
