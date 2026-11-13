use sha2::{Digest as _, Sha256};

use super::ExecutableSource;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableRevision {
    pub program_revision: runmat_execution::ProgramRevision,
    pub source_digest: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExecutableSourceSelection {
    ScriptSection { start_byte: u32, end_byte: u32 },
}

impl ExecutableRevision {
    pub fn derive(
        source: &ExecutableSource,
        program_revision: Option<runmat_execution::ProgramRevision>,
        environment: runmat_execution::ProgramEnvironment,
    ) -> Self {
        Self::derive_with_selection(source, program_revision, environment, None)
    }

    pub(crate) fn derive_with_selection(
        source: &ExecutableSource,
        program_revision: Option<runmat_execution::ProgramRevision>,
        environment: runmat_execution::ProgramEnvironment,
        selection: Option<ExecutableSourceSelection>,
    ) -> Self {
        let mut digest = Sha256::new();
        frame(&mut digest, source.owner_identity.as_bytes());
        frame(&mut digest, source.relative_path.as_bytes());
        frame(&mut digest, source.text.as_bytes());
        if let Some(ExecutableSourceSelection::ScriptSection {
            start_byte,
            end_byte,
        }) = selection
        {
            frame(&mut digest, b"runmat-test-script-section-v1");
            frame(&mut digest, &start_byte.to_be_bytes());
            frame(&mut digest, &end_byte.to_be_bytes());
        }
        let source_digest = runmat_execution::Digest::from_bytes(digest.finalize().into());
        let program_revision = program_revision.unwrap_or_else(|| {
            let mut graph = Sha256::new();
            frame(&mut graph, b"runmat-loose-source-graph-v1");
            frame(&mut graph, source.owner_identity.as_bytes());
            runmat_execution::ProgramRevision::new(
                runmat_execution::Digest::from_bytes(graph.finalize().into()),
                source_digest,
                environment,
            )
            .expect("Core source identity and program environment are valid")
        });
        Self {
            program_revision,
            source_digest: source_digest.to_string(),
        }
    }
}

fn frame(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_execution::{Digest, ProgramEnvironment};

    use super::*;

    fn environment() -> ProgramEnvironment {
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap()
    }

    #[test]
    fn frozen_physical_lookup_path_does_not_enter_portable_source_identity() {
        let first = ExecutableSource::frozen(
            "package@1",
            "src/+loads/pulse.m",
            "/host-a/project/src/+loads/pulse.m",
            "function y = pulse(x); y = x; end",
        );
        let second = ExecutableSource::frozen(
            "package@1",
            "src/+loads/pulse.m",
            "D:\\project\\src\\+loads\\pulse.m",
            "function y = pulse(x); y = x; end",
        );
        assert_eq!(
            ExecutableRevision::derive(&first, None, environment()),
            ExecutableRevision::derive(&second, None, environment())
        );
        assert_ne!(first.lookup_path(), second.lookup_path());
    }

    #[test]
    fn frozen_physical_lookup_path_does_not_enter_portable_executable() {
        let lookup_path = "/private/machine-specific/project/src/+loads/pulse.m";
        let source = ExecutableSource::frozen(
            "package@1",
            "src/+loads/pulse.m",
            lookup_path,
            "function y = pulse(x)\narguments\n    x double\nend\ny = x;\nend\n",
        );
        let mut session = crate::RunMatSession::with_options(false, false).unwrap();
        let unit = block_on(session.compile_executable_unit(source, None)).unwrap();
        let envelope = unit.portable_envelope_for(Some("pulse")).unwrap();
        assert_eq!(
            unit.entrypoint_identity(Some("pulse")).unwrap(),
            (
                envelope.manifest.identity.entrypoint_kind,
                envelope.manifest.identity.entrypoint_function,
            )
        );
        assert!(unit.entrypoint_identity(Some("absent")).is_err());
        let bytes = envelope.canonical_bytes().unwrap();
        assert!(!bytes
            .windows(lookup_path.len())
            .any(|window| window == lookup_path.as_bytes()));
    }

    #[test]
    fn selected_sections_of_one_frozen_file_have_distinct_executable_products() {
        let source = ExecutableSource::new(
            "path:workspace",
            "tests/sectionTest.m",
            "%% one\na=1\n%% two\na=2\n",
        );
        let environment = crate::program_environment(crate::CompatMode::Matlab);
        let frozen = runmat_execution::ProgramRevision::new(
            runmat_execution::Digest::sha256(b"graph"),
            runmat_execution::Digest::sha256(b"snapshot"),
            environment.clone(),
        )
        .unwrap();
        let first = ExecutableRevision::derive_with_selection(
            &source,
            Some(frozen.clone()),
            environment.clone(),
            Some(ExecutableSourceSelection::ScriptSection {
                start_byte: 7,
                end_byte: 11,
            }),
        );
        let second = ExecutableRevision::derive_with_selection(
            &source,
            Some(frozen.clone()),
            environment,
            Some(ExecutableSourceSelection::ScriptSection {
                start_byte: 18,
                end_byte: 22,
            }),
        );
        assert_ne!(first.source_digest, second.source_digest);
        assert_eq!(first.program_revision, frozen);
        assert_eq!(second.program_revision, frozen);
    }
}
