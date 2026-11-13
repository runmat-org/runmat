//! Reseal a well-formed outer root with a different, independently valid heat result.
use super::*;
use runmat_analysis_execution::{prepare_thermal_study_root, CompletedThermalStudy};
use runmat_execution_artifact::ArtifactResult;

pub(crate) fn assert_foreign_thermal_substitution_rejected(
    source: &impl CacheImport,
    response: &ProgramExecutionResponse,
    expected: &crate::StructuralStudyResultExpectation,
    foreign_source: &impl CacheImport,
    foreign: &CompletedThermalStudy,
) {
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        panic!("expected externalized study result");
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        panic!("expected one study root");
    };
    let access = &expected.artifact_access;
    let mut publication =
        ExactStructuralStudyPublication::decode(&read(source, root).unwrap(), access).unwrap();
    let foreign = prepare_thermal_study_root(foreign).unwrap();
    assert_ne!(publication.thermal.as_ref().unwrap().root, foreign.root);
    let mut overlay = Overlay {
        source,
        objects: foreign
            .catalog
            .objects
            .iter()
            .map(|reference| {
                (
                    reference.logical_digest,
                    read(foreign_source, reference).unwrap(),
                )
            })
            .collect(),
    };
    // Prove the donor is independently valid under the same object authority. A malformed
    // object, missing dependency, or wrong authorization is not substitution evidence.
    runmat_analysis_execution::import_thermal_study_root(
        &overlay,
        &foreign.root,
        foreign.root.logical_digest,
        access,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    publication.thermal = Some(thermal_prefix::ThermalPrefixClosure {
        root: foreign.root,
        catalog: foreign.catalog,
    });
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/exact-structural-study-publications",
        PUBLICATION_MEDIA_TYPE,
        publication.canonical_bytes(access).unwrap(),
    )
    .unwrap();
    let root = access
        .reference(
            &object,
            ValueRefKind::ResultObject,
            PUBLICATION_VALUE_SCHEMA,
        )
        .unwrap();
    overlay.objects.insert(root.logical_digest, object.bytes);
    // Rebuild the exact declared inventory rather than leave stale donor/recipient roots.
    let result_objects = unique_references(
        publication
            .meshing_objects
            .iter()
            .chain(&publication.region_resolution_objects)
            .chain(&publication.input_objects)
            .chain(publication.analysis.objects())
            .chain(&publication.thermal.as_ref().unwrap().catalog.objects)
            .chain(std::iter::once(&root))
            .cloned(),
    )
    .unwrap();
    for reference in &result_objects {
        read(&overlay, reference).unwrap();
    }
    let substituted = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root))],
        result_objects,
    };
    let error = import_structural_study_result(
        &overlay,
        substituted,
        expected,
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("thermal prefix differs from containing compiled execution context"),
        "expected authenticated context rejection, got {error}"
    );
}

struct Overlay<'a, S> {
    source: &'a S,
    objects: BTreeMap<Digest, Vec<u8>>,
}

impl<S: CacheImport> CacheImport for Overlay<'_, S> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            assert_eq!(Digest::sha256(bytes), digest);
            Ok(Some(bytes.clone()))
        } else {
            self.source.read_verified(digest)
        }
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        self.read_verified_bounded_with_admission(digest, maximum_bytes, &mut |_| Ok(()))
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if let Some(bytes) = self.objects.get(&digest) {
            runmat_execution_artifact::cache::copy_verified_bounded_with_admission(
                digest,
                Some(bytes.as_slice()),
                maximum_bytes,
                admit,
            )
        } else {
            self.source
                .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
        }
    }
}
