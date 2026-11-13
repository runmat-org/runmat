use super::{
    lease_source_context_with_lexical_file, source_info_in_context, DigestStream,
    RuntimeSourceLease, SourceIdExhausted, SourceInfo,
};
use runmat_types::SourceId;
use std::sync::Arc;

/// File identity inherited by dynamically evaluated source.
///
/// Dynamic text has its own source name and text for diagnostics, while
/// file-oriented introspection continues to describe the file whose lexical
/// workspace is evaluating that text.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct SourceFileIdentity {
    name: Arc<str>,
    fullpath_name: Option<Arc<str>>,
}

#[derive(Debug)]
pub enum NestedSourceIdentityError {
    SourceUnavailable(SourceId),
    Exhausted(SourceIdExhausted),
}

impl std::fmt::Display for NestedSourceIdentityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceUnavailable(source_id) => write!(
                formatter,
                "lexical source identity {} is unavailable in the selected runtime",
                source_id.0
            ),
            Self::Exhausted(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NestedSourceIdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceUnavailable(_) => None,
            Self::Exhausted(error) => Some(error),
        }
    }
}

impl SourceInfo {
    pub fn lexical_file_name(&self) -> &str {
        self.lexical_file
            .as_ref()
            .map_or(self.name.as_ref(), |file| file.name.as_ref())
    }

    pub fn lexical_file_fullpath_name(&self) -> Option<&str> {
        match &self.lexical_file {
            Some(file) => file.fullpath_name.as_deref(),
            None => self.fullpath_name.as_deref(),
        }
    }

    pub(super) fn lexical_file_identity(&self) -> SourceFileIdentity {
        self.lexical_file
            .clone()
            .unwrap_or_else(|| SourceFileIdentity {
                name: Arc::clone(&self.name),
                fullpath_name: self.fullpath_name.clone(),
            })
    }
}

pub(super) fn update_identity_digest(info: &SourceInfo, digest: &mut DigestStream) {
    match &info.lexical_file {
        Some(file) => {
            digest.update(&[1]);
            for field in [Some(file.name.as_ref()), file.fullpath_name.as_deref()] {
                match field {
                    Some(value) => {
                        digest.update(&[1]);
                        digest.update(&(value.len() as u64).to_le_bytes());
                        digest.update(value.as_bytes());
                    }
                    None => digest.update(&[0]),
                }
            }
        }
        None => digest.update(&[0]),
    }
}

/// Issue identity for source text evaluated within the selected runtime's
/// current lexical file. The dynamic source remains distinct in diagnostics;
/// only file-oriented introspection inherits the caller's file identity.
pub fn lease_nested_source_context_in_context(
    runtime: &crate::context::RuntimeContext,
    name: impl Into<Arc<str>>,
    text: impl Into<Arc<str>>,
    lexical_source_id: Option<SourceId>,
) -> Result<RuntimeSourceLease, NestedSourceIdentityError> {
    let lexical_file = lexical_source_id
        .map(|source_id| {
            source_info_in_context(runtime, source_id)
                .map(|source| source.lexical_file_identity())
                .ok_or(NestedSourceIdentityError::SourceUnavailable(source_id))
        })
        .transpose()?;
    lease_source_context_with_lexical_file(runtime, name, None, text, lexical_file)
        .map_err(NestedSourceIdentityError::Exhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use crate::execution::RuntimeExecutionService;
    use crate::source_context::{lease_source_context_in_context, source_info_in_context};
    use std::rc::Rc;

    fn runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    #[test]
    fn nested_source_identity_binds_execution_and_lexical_provenance() {
        let runtime = runtime();
        let _runtime = runtime.enter();
        let first_outer = lease_source_context_in_context(
            &runtime,
            "first.m",
            Some(Arc::from("/project/first.m")),
            "value = eval('shared');",
        )
        .expect("first outer source");
        let first_nested = lease_nested_source_context_in_context(
            &runtime,
            "<eval>",
            "shared",
            Some(first_outer.source_id()),
        )
        .expect("first nested source");
        let second_outer = lease_source_context_in_context(
            &runtime,
            "second.m",
            Some(Arc::from("/project/second.m")),
            "value = eval('shared');",
        )
        .expect("second outer source");
        let second_nested = lease_nested_source_context_in_context(
            &runtime,
            "<eval>",
            "shared",
            Some(second_outer.source_id()),
        )
        .expect("second nested source");

        let first = first_nested.source_info();
        assert_eq!(first.name.as_ref(), "<eval>");
        assert_eq!(first.fullpath_name.as_deref(), None);
        assert_eq!(first.lexical_file_name(), "first.m");
        assert_eq!(first.lexical_file_fullpath_name(), Some("/project/first.m"));
        assert_ne!(
            first_nested.source_ref().semantic_content_revision(),
            second_nested.source_ref().semantic_content_revision(),
            "the same dynamic text in different lexical files is distinct provenance"
        );

        let child = runtime
            .fork_parallel_lab(RuntimeServicePorts::default())
            .expect("idle session fork");
        drop(first_nested);
        let inherited = source_info_in_context(&child, first.source_id.expect("nested source id"))
            .expect("parallel snapshot retains admitted nested provenance");
        assert_eq!(inherited.name.as_ref(), "<eval>");
        assert_eq!(inherited.lexical_file_name(), "first.m");
    }

    #[test]
    fn unavailable_lexical_source_rejects_before_issuing_nested_identity() {
        let runtime = runtime();
        let unavailable = SourceId(77);
        let error = lease_nested_source_context_in_context(
            &runtime,
            "<eval>",
            "untrusted",
            Some(unavailable),
        )
        .expect_err("unavailable lexical source must fail closed");
        assert!(matches!(
            error,
            NestedSourceIdentityError::SourceUnavailable(source_id)
                if source_id == unavailable
        ));

        let first = lease_source_context_in_context(&runtime, "first.m", None, "first")
            .expect("rejection does not consume source identity");
        assert_eq!(first.source_id(), SourceId(0));
    }
}
