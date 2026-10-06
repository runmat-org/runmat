use std::sync::Arc;

use super::{ProgramSourceBindingError, ProgramSourceRecord};

/// Bind one validated portable source map to fresh identities owned by this
/// invocation. The portable ordinals never enter the session issuer as IDs.
pub fn bind_portable_source_map_v2(
    runtime: crate::context::RuntimeContext,
    map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<crate::context::RuntimeContext, PortableSourceBindingError> {
    map.validate()
        .map_err(PortableSourceBindingError::Contract)?;
    runtime
        .with_program_sources(map.entries.iter().map(|entry| ProgramSourceRecord {
            ordinal: entry.source_id,
            name: Arc::from(entry.display_name.as_str()),
            fullpath_name: None,
            text: Arc::from(entry.text.as_str()),
        }))
        .map_err(PortableSourceBindingError::Issue)
}

#[derive(Debug)]
pub enum PortableSourceBindingError {
    Contract(runmat_execution::ContractError),
    Issue(ProgramSourceBindingError),
}

impl std::fmt::Display for PortableSourceBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "invalid source map: {error}"),
            Self::Issue(error) => write!(formatter, "cannot bind source map: {error}"),
        }
    }
}

impl std::error::Error for PortableSourceBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Issue(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{Digest, ExecutableSourceRecordV2};
    use runmat_types::ProgramSourceId;
    use std::rc::Rc;

    fn runtime() -> crate::context::RuntimeContext {
        crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
    }

    fn map() -> runmat_execution::ExecutableSourceMapV2 {
        runmat_execution::ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "main.m".into(),
            display_name: "main.m".into(),
            text: "answer = 7;".into(),
            content_revision: Digest::sha256(b"answer = 7;"),
        }])
        .expect("valid fixture")
    }

    #[test]
    fn portable_ordinal_gets_fresh_lease_without_importing_a_numeric_id() {
        let session = runtime();
        let existing = super::super::lease_source_context_in_context(
            &session,
            "<pending>",
            None,
            "pending = true;",
        )
        .expect("existing lease");
        let first = bind_portable_source_map_v2(session.clone(), &map()).expect("first binding");
        let second = bind_portable_source_map_v2(session, &map()).expect("second binding");
        let ordinal = ProgramSourceId(0);
        let first_id = first.program_source_id(ordinal).expect("first id");
        let second_id = second.program_source_id(ordinal).expect("second id");
        assert_ne!(first_id, existing.source_id());
        assert_ne!(second_id, existing.source_id());
        assert_ne!(first_id, second_id);
        assert_eq!(
            super::super::source_info_in_context(&first, first_id)
                .expect("first source")
                .text
                .as_ref(),
            "answer = 7;"
        );
    }

    #[test]
    fn forged_text_rejected_before_issuing_any_source() {
        let session = runtime();
        let mut forged = map();
        forged.entries[0].text = "answer = 8;".into();
        assert!(matches!(
            bind_portable_source_map_v2(session.clone(), &forged),
            Err(PortableSourceBindingError::Contract(_))
        ));
        let issued =
            super::super::lease_source_context_in_context(&session, "<next>", None, "next = 1;")
                .expect("first issued source");
        assert_eq!(issued.source_id().0, 0);
    }
}
