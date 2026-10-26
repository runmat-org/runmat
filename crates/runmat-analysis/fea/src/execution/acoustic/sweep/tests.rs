use super::*;
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};

fn workload() -> AcousticFrequencyWorkload {
    let mut template = super::super::tests::workload();
    template.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.1, 0.2, 0.3]);
    AcousticFrequencyWorkload::new(template).unwrap()
}

#[test]
fn acoustic_frequency_fan_in_preserves_independent_wave_fields_and_cold_identity() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            quadratic_slab_mesh(2)
        } else {
            slab_mesh(4)
        };
        let authored = workload();
        let cold =
            AcousticFrequencyWorkload::from_canonical_bytes(&authored.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(cold.identity().unwrap(), authored.identity().unwrap());
        assert_eq!(cold.plan().frequencies_hz(), &[0.1, 0.2, 0.3]);
        let entries = cold.plan().entries().unwrap();
        let mut results = entries
            .into_iter()
            .map(|entry| cold.entry(entry).unwrap().execute(&mesh).unwrap())
            .collect::<Vec<_>>();
        let ordered =
            AcousticFrequencyResult::assemble(cold.clone(), &mesh, results.clone()).unwrap();
        results.reverse();
        let reversed = AcousticFrequencyResult::assemble(cold.clone(), &mesh, results).unwrap();
        assert_eq!(
            ordered.canonical_bytes().unwrap(),
            reversed.canonical_bytes().unwrap()
        );
        let reopened =
            AcousticFrequencyResult::from_canonical_bytes(&reversed.canonical_bytes().unwrap())
                .unwrap();
        reopened.validate_for(&authored, &mesh).unwrap();
        assert_eq!(
            ordered.source_result_identity().unwrap(),
            reopened.source_result_identity().unwrap()
        );
        for result in reopened.results() {
            // Exact matched plane wave p=exp(-ikx), rho=3, c=2. The
            // reference uses neither the computed operator nor a recovered field.
            let mut pressure_error = 0.0;
            for (node, pressure) in mesh.topology.nodes.iter().zip(result.pressure_pa()) {
                let phase = std::f64::consts::PI
                    * result.frequency().canonical_value()
                    * node.coordinates_m.canonical_value()[0];
                pressure_error += (pressure.real - phase.cos()).powi(2)
                    + (pressure.imaginary + phase.sin()).powi(2);
            }
            assert!((pressure_error / result.pressure_pa().len() as f64).sqrt() < 0.01);
            assert!(result
                .intensity_w_per_m2()
                .iter()
                .all(|i| (i[0] - 1.0 / 12.0).abs() < 0.01));
            assert!(result
                .energy_density_j_per_m3()
                .iter()
                .all(|e| (e - 1.0 / 24.0).abs() < 0.01));
        }

        let mut missing = ordered.results().to_vec();
        missing.pop();
        assert!(AcousticFrequencyResult::assemble(cold.clone(), &mesh, missing).is_err());
        let mut duplicate = ordered.results().to_vec();
        duplicate[1] = duplicate[0].clone();
        assert!(AcousticFrequencyResult::assemble(cold.clone(), &mesh, duplicate).is_err());
        let mut changed = cold.template.clone();
        changed.power_balance_relative_tolerance = 1e-7;
        let foreign = AcousticFrequencyWorkload::new(changed).unwrap();
        assert!(ordered.validate_for(&foreign, &mesh).is_err());
        let mut substituted = ordered.results().to_vec();
        substituted[0] = foreign
            .entry(foreign.plan().entries().unwrap()[0])
            .unwrap()
            .execute(&mesh)
            .unwrap();
        assert!(AcousticFrequencyResult::assemble(cold.clone(), &mesh, substituted).is_err());
        let mut other_mesh = mesh.clone();
        for node in &mut other_mesh.topology.nodes {
            {
                let quantity_change = 1.0;
                let quantity_index = 0;
                let quantity_slot = &mut node.coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
        }
        other_mesh.seal_canonical_digest().unwrap();
        assert!(ordered.validate_for(&cold, &other_mesh).is_err());
        for modification in 0..3 {
            let mut malformed = serde_json::to_value(&ordered).unwrap();
            let entries = malformed["results"].as_array_mut().unwrap();
            match modification {
                0 => {
                    entries.pop();
                }
                1 => {
                    entries.swap(0, 1);
                }
                _ => {
                    entries[1] = entries[0].clone();
                }
            }
            assert!(serde_json::from_value::<AcousticFrequencyResult>(malformed).is_err());
        }
    }
}

#[test]
fn acoustic_frequency_plan_cannot_add_unauthored_frequencies_or_foreign_entries() {
    let source = workload();
    let mut unordered = source.template.clone();
    unordered.model.sequence.steps[0].independent_domain =
        IndependentDomain::FrequencyHertz(vec![0.3, 0.1, 0.2]);
    assert!(AcousticFrequencyWorkload::new(unordered).is_err());
    let other = FrequencyResponsePlan::new(0.4, &[0.1, 0.2, 0.4], &[]).unwrap();
    assert!(source.entry(other.entries().unwrap()[0]).is_err());
    let mut wire = serde_json::to_value(&source).unwrap();
    wire["plan"] = serde_json::to_value(other).unwrap();
    assert!(serde_json::from_value::<AcousticFrequencyWorkload>(wire).is_err());
    let mut additional_step = source.template.clone();
    additional_step
        .model
        .sequence
        .steps
        .push(ProcedureStep::acoustic_harmonic("other"));
    assert!(AcousticFrequencyWorkload::new(additional_step).is_err());
}
