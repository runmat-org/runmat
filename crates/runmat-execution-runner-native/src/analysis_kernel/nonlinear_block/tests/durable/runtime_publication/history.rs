//! Independent cold public-field references; no solver or return-map replay.
use super::*;
use runmat_analysis_core::AnalysisField;
use runmat_analysis_fea::{contracts::FEA_FIELD_STRUCTURAL_STRESS, FeaRunResult};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Copy)]
pub(crate) enum Reference {
    Uniaxial,
    Layered,
}

fn values(
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
    field: &AnalysisField,
) -> Vec<f64> {
    let selection = analysis::AnalysisFieldSelection::new(field.field_id.clone(), None);
    let mut values = Vec::new();
    let mut total = None;
    loop {
        let offset = values.len();
        let AnalysisFieldPageLookup::Page(page) = store
            .load_field_page(
                run_id,
                &selection,
                analysis::AnalysisFieldRequestOptions {
                    offset: Some(offset),
                    limit: Some(256),
                },
            )
            .unwrap()
        else {
            panic!("original history page {} missing", field.field_id);
        };
        assert_eq!(page.run_id, run_id);
        assert_eq!(page.field_id, field.field_id);
        assert_eq!(page.offset, offset);
        if let Some(total) = total {
            assert_eq!(page.total_count, total);
        } else {
            total = Some(page.total_count);
        }
        let analysis::AnalysisFieldPageValues::Real(next) = page.values else {
            panic!("real original history required");
        };
        assert_eq!(page.count, next.len());
        assert!(page.count > 0 && page.count <= 256);
        values.extend(next);
        assert!(values.len() <= page.total_count);
        assert_eq!(
            page.truncated,
            offset > 0 || values.len() < page.total_count
        );
        if values.len() == page.total_count {
            return values;
        }
    }
}

fn near(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "complete original field inventory"
    );
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "entry {index}: {a} != {e} within {tolerance}"
        );
    }
}

pub(super) fn verify(
    reference: Reference,
    mesh: &SolverMeshArtifact,
    history: &analysis::NonlinearResultsData,
    run: &FeaRunResult,
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
) {
    let count = history.load_factors.len();
    for snapshots in [
        &history.displacement_snapshots,
        &history.plastic_strain_snapshots,
        &history.equivalent_plastic_strain_snapshots,
        &history.von_mises_snapshots,
    ] {
        assert_eq!(snapshots.len(), count);
        let identities = snapshots
            .iter()
            .map(|field| &field.field_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(
            identities.len(),
            count,
            "accepted output ordinals cannot alias"
        );
    }
    assert_eq!(history.equivalent_plastic_strain_snapshots.len(), count);
    for (ordinal, field) in history
        .equivalent_plastic_strain_snapshots
        .iter()
        .enumerate()
    {
        let factor = history.load_factors[ordinal];
        let alpha = values(store, run_id, field);
        match reference {
            Reference::Uniaxial => {
                let expected = ((5.6 * factor - 5.0) / 30.0).max(0.0);
                near(
                    &alpha,
                    &vec![expected; mesh.topology.volume_elements.len()],
                    2e-9,
                );
            }
            Reference::Layered => verify_layered(mesh, history, ordinal, store, run_id, &alpha),
        }
    }
    if matches!(reference, Reference::Layered) {
        let terminal = run
            .field(FEA_FIELD_STRUCTURAL_STRESS)
            .expect("original terminal stress");
        let stress = layered_state(1.0);
        let expected = mesh
            .topology
            .volume_elements
            .iter()
            .flat_map(|e| stress[layer(e)].2)
            .collect::<Vec<_>>();
        near(&values(store, run_id, terminal), &expected, 2e-7);
    }
}

fn layered_state(factor: f64) -> [(f64, f64, [f64; 6]); 2] {
    [
        super::super::super::layered::reference::state(2.0, factor),
        super::super::super::layered::reference::state(3.5, factor),
    ]
}
fn layer(element: &runmat_meshing_core::SolverVolumeElement) -> usize {
    match element.region_id.source_topology_id.as_str() {
        "layer_0" => 0,
        "layer_1" => 1,
        other => panic!("foreign reference region {other}"),
    }
}
fn verify_layered(
    mesh: &SolverMeshArtifact,
    history: &analysis::NonlinearResultsData,
    ordinal: usize,
    store: &FilesystemAnalysisArtifactStore,
    run_id: &str,
    alpha: &[f64],
) {
    let references = layered_state(history.load_factors[ordinal]);
    let elements = &mesh.topology.volume_elements;
    let expected_alpha = elements
        .iter()
        .map(|e| references[layer(e)].1)
        .collect::<Vec<_>>();
    near(alpha, &expected_alpha, 2e-9);
    let expected_plastic = elements
        .iter()
        .flat_map(|e| {
            let a = references[layer(e)].1;
            [a, -a / 2.0, -a / 2.0, 0.0, 0.0, 0.0]
        })
        .collect::<Vec<_>>();
    near(
        &values(store, run_id, &history.plastic_strain_snapshots[ordinal]),
        &expected_plastic,
        2e-9,
    );
    let expected_u = mesh
        .topology
        .nodes
        .iter()
        .flat_map(|n| {
            let x = n.coordinates_m.canonical_value()[0];
            [
                references[0].0 * x.min(0.5) + references[1].0 * (x - 0.5).max(0.0),
                0.0,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    near(
        &values(store, run_id, &history.displacement_snapshots[ordinal]),
        &expected_u,
        2e-9,
    );
    assert_eq!(
        history.von_mises_snapshots.len(),
        history.load_factors.len()
    );
    let expected_vm = elements
        .iter()
        .map(|e| {
            let s = references[layer(e)].2;
            (s[0] - s[1]).abs()
        })
        .collect::<Vec<_>>();
    near(
        &values(store, run_id, &history.von_mises_snapshots[ordinal]),
        &expected_vm,
        2e-7,
    );
}
