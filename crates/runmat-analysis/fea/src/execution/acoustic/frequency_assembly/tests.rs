use super::*;
use crate::topology::test_meshes::slab_mesh;
use runmat_analysis_core::IndependentDomain;

fn none(count: usize) -> AcousticFrequencyAssemblyRequest {
    AcousticFrequencyAssemblyRequest::new(vec![None; count]).unwrap()
}

#[test]
fn fan_in_preserves_every_original_source_and_rejects_incomplete_inventory() {
    let mesh = slab_mesh(3);
    let mut harmonic = super::super::tests::workload();
    harmonic.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.1, 0.2, 0.3]);
    let workload = AcousticFrequencyWorkload::new(harmonic).unwrap();
    let mut entries = workload
        .plan()
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| workload.entry(entry).unwrap().execute(&mesh).unwrap())
        .collect::<Vec<_>>();
    let ordered =
        assemble_acoustic_frequency_result(workload.clone(), &mesh, entries.clone(), &none(3))
            .unwrap();
    entries.reverse();
    let reversed =
        assemble_acoustic_frequency_result(workload.clone(), &mesh, entries.clone(), &none(3))
            .unwrap();
    assert_eq!(ordered.result, reversed.result);
    assert_eq!(ordered.publication, reversed.publication);
    assert_eq!(ordered.fields, reversed.fields);
    for ((original, sample), fields) in ordered
        .result
        .results()
        .iter()
        .zip(ordered.publication.samples())
        .zip(&ordered.fields)
    {
        assert_eq!(
            sample.recovery_request.source,
            crate::execution::FieldRecoverySource::SolverResult(
                original.source_result_identity().unwrap()
            )
        );
        assert_eq!(sample.fields, fields.manifest().unwrap());
        assert_ne!(
            original.source_result_identity().unwrap(),
            ordered.result.source_result_identity().unwrap()
        );
    }
    let cold = crate::execution::FrequencySamplePublicationManifest::from_canonical_bytes(
        &ordered.publication.canonical_bytes().unwrap(),
    )
    .unwrap();
    cold.validate_for(&ordered.result.publication_authority().unwrap())
        .unwrap();
    assert!(assemble_acoustic_frequency_result(
        workload.clone(),
        &mesh,
        entries[..2].to_vec(),
        &none(3)
    )
    .is_err());
    let mut duplicate = entries.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(
        assemble_acoustic_frequency_result(workload.clone(), &mesh, duplicate, &none(3)).is_err()
    );
    let mut other = workload
        .entry(workload.plan().entries().unwrap()[0])
        .unwrap();
    other.model.materials[0]
        .acoustic
        .as_mut()
        .unwrap()
        .speed_of_sound_m_per_s = runmat_types::physical::SpeedOfSound::new(3.0);
    entries[0] = other.execute(&mesh).unwrap();
    assert!(assemble_acoustic_frequency_result(workload, &mesh, entries, &none(3)).is_err());
}
