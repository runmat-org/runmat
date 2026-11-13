//! Cold composition must retain each child's original context, including absence.
use super::*;
use runmat_analysis_fea::execution::SweepCompositionPolicy;
use runmat_execution::{Digest, EnvironmentSnapshot};

struct Store(BTreeMap<Digest, Vec<u8>>);
impl CacheImport for Store {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

#[test]
fn cold_sweep_preserves_distinct_original_child_environments() {
    let authority = "sweep-environment";
    let source = Digest::sha256(b"portable exact CAD source");
    let digests = [Digest::sha256(b"first"), Digest::sha256(b"second")];
    let studies = digests
        .iter()
        .map(|digest| {
            crate::exact_structural_study_root::tests::prepared(authority, source, *digest)
        })
        .collect();
    let definition = SweepDefinition::new(
        Digest::sha256(b"sweep"),
        digests
            .iter()
            .map(|digest| StudyIdentity::from_authored_digest(*digest, None).unwrap())
            .collect(),
        SweepCompositionPolicy::FailFast,
    )
    .unwrap();
    let mut prepared = prepare_exact_structural_sweep_root(ExactStructuralSweepRootPreparation {
        program_revision: crate::exact_structural_study_root::tests::revision(),
        artifact_access: crate::exact_structural_study_root::tests::access(authority),
        definition,
        studies,
    })
    .unwrap();
    let access = crate::exact_structural_study_root::tests::access(authority);
    let ValuePayload::Object(reference) = &prepared.request.arguments[0] else {
        panic!("missing root")
    };
    let original = prepared
        .objects
        .iter()
        .find(|object| object.reference.logical_digest == reference.logical_digest)
        .unwrap();
    let mut root = ExactStructuralSweepRootRequest::decode(&original.bytes).unwrap();
    let contexts = [
        EnvironmentSnapshot::new(vec![("RUNMAT_CHILD".into(), "captured".into())]).unwrap(),
        EnvironmentSnapshot::default(),
    ];
    for (entry, context) in root.entries.iter_mut().zip(&contexts) {
        entry.descriptor.environment = context.clone();
    }
    let object = LogicalObject::new(
        ObjectNamespace::InputValue,
        "environment-root",
        ROOT_MEDIA_TYPE,
        root.canonical_bytes().unwrap(),
    )
    .unwrap();
    let reference = access
        .reference(&object, ValueRefKind::DriverObject, ROOT_VALUE_SCHEMA)
        .unwrap();
    prepared.request.arguments[0] = ValuePayload::Object(Box::new(reference));
    let mut store = Store(
        prepared
            .objects
            .into_iter()
            .map(|object| (object.reference.logical_digest, object.bytes))
            .collect(),
    );
    store.0.insert(object.descriptor.digest, object.bytes);
    let bytes = serde_json::to_vec(&prepared.request).unwrap();
    let cold: ProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
    let decoded = decode_exact_structural_sweep_root(&cold, &store).unwrap();
    for (study, context) in decoded.studies.iter().zip(contexts) {
        assert_eq!(study.environment, context);
    }
}
