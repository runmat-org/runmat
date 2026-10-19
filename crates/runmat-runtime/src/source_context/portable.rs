use std::sync::Arc;

use super::{ProgramSourceBindingError, ProgramSourceRecord};

/// Bind one validated portable source map to fresh identities owned by this
/// invocation. The portable ordinals never enter the session issuer as IDs.
pub fn bind_portable_source_map_v2(
    runtime: crate::context::RuntimeContext,
    map: &runmat_execution::ExecutableSourceMapV2,
) -> Result<crate::context::RuntimeContext, PortableSourceBindingError> {
    bind_portable_source_map_v2_with_records(
        runtime,
        map,
        map.entries.iter().map(|entry| ProgramSourceRecord {
            ordinal: entry.source_id,
            name: Arc::from(entry.display_name.as_str()),
            fullpath_name: None,
            text: Arc::from(entry.text.as_str()),
        }),
    )
}

/// A host may retain an exact absolute diagnostic path locally, but portable
/// authority is the validated owner/relative/revision/text record. Check the
/// two views before issuing any runtime source identity.
pub fn bind_portable_source_map_v2_with_records(
    runtime: crate::context::RuntimeContext,
    map: &runmat_execution::ExecutableSourceMapV2,
    records: impl IntoIterator<Item = ProgramSourceRecord>,
) -> Result<crate::context::RuntimeContext, PortableSourceBindingError> {
    let admitted = super::ValidatedPortableSourceMap::new(Arc::new(map.clone()))
        .map_err(PortableSourceBindingError::Contract)?;
    let records = records.into_iter().collect::<Vec<_>>();
    if records.len() != map.entries.len()
        || records.iter().zip(&map.entries).any(|(record, entry)| {
            record.ordinal != entry.source_id
                || record.name.as_ref() != entry.display_name
                || record.text.as_ref() != entry.text
        })
    {
        return Err(PortableSourceBindingError::RecordMismatch);
    }
    let mut runtime = runtime
        .with_program_sources(records)
        .map_err(PortableSourceBindingError::Issue)?;
    runtime
        .retain_portable_program_source_map(&admitted)
        .map_err(PortableSourceBindingError::Provenance)?;
    Ok(runtime)
}

/// Attach a one-source portable product to the exact source lease already
/// issued for dynamic evaluation. Neither the source ID nor its text is
/// reissued, and the validated record must match that lease before mutation.
pub fn bind_portable_source_map_v2_with_existing_lease(
    runtime: crate::context::RuntimeContext,
    map: &runmat_execution::ExecutableSourceMapV2,
    lease: super::RuntimeSourceLease,
) -> Result<crate::context::RuntimeContext, PortableSourceBindingError> {
    let admitted = super::ValidatedPortableSourceMap::new(Arc::new(map.clone()))
        .map_err(PortableSourceBindingError::Contract)?;
    let [record] = map.entries.as_slice() else {
        return Err(PortableSourceBindingError::RecordMismatch);
    };
    let info = lease.source_info();
    if info.name.as_ref() != record.display_name || info.text.as_ref() != record.text {
        return Err(PortableSourceBindingError::RecordMismatch);
    }
    let mut runtime = runtime
        .with_existing_program_source(record.source_id, lease)
        .map_err(PortableSourceBindingError::Issue)?;
    runtime
        .retain_portable_program_source_map(&admitted)
        .map_err(PortableSourceBindingError::Provenance)?;
    Ok(runtime)
}

#[derive(Debug)]
pub enum PortableSourceBindingError {
    Contract(runmat_execution::ContractError),
    RecordMismatch,
    Issue(ProgramSourceBindingError),
    Provenance(super::PortableSourceIdentityError),
}

impl std::fmt::Display for PortableSourceBindingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "invalid source map: {error}"),
            Self::RecordMismatch => {
                formatter.write_str("local source records differ from the validated portable map")
            }
            Self::Issue(error) => write!(formatter, "cannot bind source map: {error}"),
            Self::Provenance(error) => {
                write!(formatter, "cannot retain source provenance: {error}")
            }
        }
    }
}

impl std::error::Error for PortableSourceBindingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::RecordMismatch => None,
            Self::Issue(error) => Some(error),
            Self::Provenance(error) => Some(error),
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
        assert_eq!(first.portable_program_source_map(), Some(&map()));
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

    #[test]
    fn mismatched_host_record_fails_before_issuing_and_fork_keeps_exact_map() {
        let session = runtime();
        let mismatched = [ProgramSourceRecord {
            ordinal: ProgramSourceId(0),
            name: Arc::from("main.m"),
            fullpath_name: Some(Arc::from("/host/main.m")),
            text: Arc::from("answer = 8;"),
        }];
        assert!(matches!(
            bind_portable_source_map_v2_with_records(session.clone(), &map(), mismatched),
            Err(PortableSourceBindingError::RecordMismatch)
        ));
        let first =
            super::super::lease_source_context_in_context(&session, "first.m", None, "first = 1;")
                .expect("no source issued for rejected record");
        assert_eq!(first.source_id().0, 0);

        let bound = bind_portable_source_map_v2_with_records(
            session,
            &map(),
            [ProgramSourceRecord {
                ordinal: ProgramSourceId(0),
                name: Arc::from("main.m"),
                fullpath_name: Some(Arc::from("/host/main.m")),
                text: Arc::from("answer = 7;"),
            }],
        )
        .expect("host path is local presentation only");
        let fork = bound
            .fork_parallel_lab(crate::context::RuntimeServicePorts::default())
            .expect("idle session fork");
        assert_eq!(fork.portable_program_source_map(), Some(&map()));
        assert_eq!(
            super::super::source_info_in_context(
                &fork,
                fork.program_source_id(ProgramSourceId(0)).unwrap()
            )
            .unwrap()
            .fullpath_name
            .as_deref(),
            Some("/host/main.m")
        );
    }

    #[test]
    fn existing_eval_lease_binds_portable_product_without_reissuing_identity() {
        let session = runtime();
        let lease =
            super::super::lease_source_context_in_context(&session, "main.m", None, "answer = 7;")
                .expect("issue eval source once");
        let selected = lease.source_ref();
        let bound =
            bind_portable_source_map_v2_with_existing_lease(session.clone(), &map(), lease.clone())
                .expect("bind exact existing lease");
        assert_eq!(
            bound.program_source_id(ProgramSourceId(0)),
            Some(lease.source_id())
        );
        assert_eq!(bound.program_source_ref(ProgramSourceId(0)), Some(selected));
        assert_eq!(bound.portable_program_source_map(), Some(&map()));
        assert!(super::super::portable_source_identity_in_context(&bound, selected).is_some());
        assert_eq!(
            crate::context::SourceEvaluationProvenance::runtime_source_lease(&lease)
                .expect("typed source-evaluation lease")
                .source_ref(),
            selected
        );
    }

    #[test]
    fn existing_eval_lease_rejects_foreign_and_mismatched_source_before_binding() {
        let session = runtime();
        let foreign = runtime();
        let lease =
            super::super::lease_source_context_in_context(&session, "main.m", None, "answer = 7;")
                .expect("issue eval source");
        assert!(matches!(
            bind_portable_source_map_v2_with_existing_lease(foreign, &map(), lease.clone()),
            Err(PortableSourceBindingError::Issue(
                ProgramSourceBindingError::ForeignOrRetired
            ))
        ));
        let mut mismatched = map();
        mismatched.entries[0].display_name = "other.m".into();
        assert!(matches!(
            bind_portable_source_map_v2_with_existing_lease(
                session.clone(),
                &mismatched,
                lease.clone()
            ),
            Err(PortableSourceBindingError::RecordMismatch)
        ));
        assert!(lease.source_ref().resolve_in(&session).is_some());
        assert!(
            super::super::portable_source_identity_in_context(&session, lease.source_ref())
                .is_none()
        );
    }
}
