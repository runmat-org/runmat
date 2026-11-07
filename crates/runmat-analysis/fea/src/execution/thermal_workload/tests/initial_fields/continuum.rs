//! Constant-property continuum references, not a discrete operator or restart witness.
use super::*;
use runmat_analysis_core::*;

mod evidence;
mod fixture;
mod reference_coordinates;
mod spatial_window;

const REFERENCE: f64 = 300.0;
const GENERATION: f64 = 80.0;
const FINAL_TIME: f64 = 0.1;
const RESIDUAL: f64 = 1e-12;

// witness: FEA-THERMAL-ACCURACY-001/time @ thermal_transient/tet4|tet10/public_study;
//   check: ramp-heating; boundary: Both-order uniform ramp heating, independent continuum error and energy law; no
//   nonuniform mesh convergence credit.
#[test]
fn uniform_ramp_heating_has_continuum_backward_euler_time_convergence() {
    // rho=c=k=1, q(t)=2Bt on an insulated unit cube: T=300+B t².
    // Backward Euler's right-end source integration has EXACT endpoint error B t_final dt.
    const B: f64 = 20.0;
    for quadratic in [false, true] {
        let mesh = fixture::mesh(quadratic, 2);
        let mut errors = Vec::new();
        for intervals in [8, 16, 32] {
            let mut input = fixture::base();
            fixture::transient(&mut input, intervals);
            fixture::insulate_all(&mut input);
            fixture::generation(
                &mut input,
                SignalRepresentation::Tabulated(TabulatedSignal {
                    domain: TabulatedSignalDomain::TimeSeconds,
                    knots: [0.0, FINAL_TIME]
                        .into_iter()
                        .map(|t| {
                            SignalKnot::continuous(
                                t,
                                SignalOrdinate::Real(SignalValue::Scalar(
                                    HeatGenerationDensity::new(2.0 * B * t),
                                )),
                            )
                        })
                        .collect(),
                    interpolation: SignalInterpolation::Linear,
                    extrapolation: SignalExtrapolation::Reject,
                }),
            );
            input.model.initial_conditions =
                vec![InitialCondition::Temperature(InitialConditionValue {
                    initial_condition_id: "uniform-start".into(),
                    target: InitialConditionTarget::WholeModel,
                    frame: CoordinateFrame::Global,
                    value: InitialValue::Uniform(Temperature::new(REFERENCE)),
                    applies_before_step_id: input.step_id.0.clone(),
                })];
            let cold =
                ThermalWorkload::from_canonical_bytes(&input.canonical_bytes().unwrap()).unwrap();
            assert_eq!(cold.identity().unwrap(), input.identity().unwrap());
            let result = cold.execute(&mesh).unwrap();
            result.validate_complete().unwrap();
            let dt = FINAL_TIME / intervals as f64;
            assert_eq!(result.time_points_s.len(), intervals + 1);
            for (index, (&t, field)) in result
                .time_points_s
                .iter()
                .zip(&result.temperature_snapshots)
                .enumerate()
            {
                let discrete_expected = REFERENCE + B * (t * t + t * dt);
                assert!((t - index as f64 * dt).abs() < 1e-14);
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| (value - discrete_expected).abs() < 1e-8));
            }
            for (index, balance) in result.energy_balances.iter().enumerate() {
                let power = 2.0 * B * result.time_points_s[index + 1];
                assert!((balance.applied_power_w.canonical_value() - power).abs() < 1e-9);
                assert!((balance.storage_rate_w.canonical_value() - power).abs() < 1e-8);
                assert!(balance.prescribed_boundary_power_w.canonical_value().abs() < 1e-10);
                assert!(balance.boundary_loss_w.canonical_value().abs() < 1e-10);
                assert!(balance.relative_residual < 1e-10, "{balance:?}");
            }
            for field in result
                .temperature_gradient_snapshots
                .iter()
                .chain(&result.heat_flux_snapshots)
            {
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| value.abs() < 1e-8));
            }
            let endpoint = result
                .temperature_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap();
            let exact = REFERENCE + B * FINAL_TIME.powi(2);
            let error = (endpoint
                .iter()
                .map(|value| (value - exact).powi(2))
                .sum::<f64>()
                / endpoint.len() as f64)
                .sqrt();
            assert!((error - B * FINAL_TIME * dt).abs() < 1e-8);
            errors.push(error);
        }
        println!("continuum ramp: quadratic={quadratic}, errors={errors:?}");
        assert!(errors.windows(2).all(|pair| pair[1] < 0.56 * pair[0]));
        assert!(errors
            .windows(2)
            .all(|pair| (pair[0] / pair[1]).log2() > 0.8));
    }
}

// One cell is outside the asymptotic range for both orders (a Tet4 cube of one cell has no free
// thermal node), so the predeclared order gates are measured on two, four and eight cells.
#[test]
fn solved_heated_cube_cold_cooling_tet10_converges_to_continuum_diffusion() {
    cooling_convergence(true, [2, 4, 8]);
}

// witness: FEA-THERMAL-ACCURACY-001/space @ thermal_transient/tet4/public_study;
//   boundary: Tet4 heated-cube cold cooling continuum temperature, flux, stored heat and power with
//   temporal contamination bound; does not prove Tet10 spatial gate.
#[test]
fn resolved_tet4_heated_cube_cold_cooling_has_continuum_spatial_convergence() {
    cooling_convergence(false, [2, 4, 8]);
}

fn cooling_convergence(quadratic: bool, cell_counts: [usize; 3]) {
    // The initial field is actually solved, not fabricated from the analytic polynomial.
    // Richardson combinations ONLY isolate spatial error; they do not advertise a new integrator.
    // Three dt levels expose the remaining O(dt²) contamination of this measurement.
    let mut errors = Vec::new();
    for cells in cell_counts {
        let mesh = fixture::mesh(quadratic, cells);
        let mut producer_input = fixture::base();
        fixture::ends(&mut producer_input);
        producer_input.procedure = ThermalProcedureControl::Steady;
        producer_input.model.sequence = AnalysisSequence::from_steps(
            "heated-source",
            vec![ProcedureStep::thermal_steady("heat_a")],
        );
        fixture::generation(
            &mut producer_input,
            SignalRepresentation::Constant(SignalValue::Scalar(HeatGenerationDensity::new(
                GENERATION,
            ))),
        );
        let producer = producer_input.execute(&mesh).unwrap();
        producer.validate_complete().unwrap();
        let source_balance = producer.energy_balances[0];
        assert!((source_balance.applied_power_w.canonical_value() - GENERATION).abs() < 1e-8);
        assert!(
            (source_balance.prescribed_boundary_power_w.canonical_value() - GENERATION).abs()
                < 1e-8
        );
        assert!(source_balance.relative_residual < 1e-10);
        if quadratic {
            // The actual quadratic-space steady producer is independently exact; retain
            // this check without substituting these values into the consumer binding.
            for (node, value) in mesh
                .topology
                .nodes
                .iter()
                .zip(producer.temperature_snapshots[0].as_host_f64().unwrap())
            {
                let x = node.coordinates_m.canonical_value()[0];
                assert!((value - REFERENCE - GENERATION * x * (1.0 - x) / 2.0).abs() < 1e-8);
            }
        }
        let binding = solved_temperature_binding(&producer, &mesh);
        let producer_bytes = binding.artifact().canonical_bytes().unwrap();
        let mut results = Vec::new();
        // Quadratic spatial errors fall below the O(dt^2) floor of one Richardson elimination by
        // eight cells, so the quadratic measurement adds a fourth time level and a second
        // elimination (O(dt^3) floor), keeping the floor under a tenth of the spatial error.
        let interval_levels: &[usize] = if quadratic {
            &[64, 128, 256, 512]
        } else {
            &[64, 128, 256]
        };
        for &intervals in interval_levels {
            let mut consumer = fixture::base();
            fixture::ends(&mut consumer);
            fixture::transient(&mut consumer, intervals);
            consumer.model.initial_conditions =
                vec![InitialCondition::Temperature(InitialConditionValue {
                    initial_condition_id: "solved-heated-field".into(),
                    target: InitialConditionTarget::WholeModel,
                    frame: CoordinateFrame::Global,
                    value: InitialValue::FieldArtifact(binding.reference().clone()),
                    applies_before_step_id: consumer.step_id.0.clone(),
                })];
            consumer.initial_fields = vec![binding.clone()];
            let bytes = consumer.canonical_bytes().unwrap();
            let cold = ThermalWorkload::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(cold.canonical_bytes().unwrap(), bytes);
            let result = cold.execute(&mesh).unwrap();
            result.validate_complete().unwrap();
            assert_eq!(result.source_workload, cold.identity().unwrap());
            assert_eq!(result.time_points_s.len(), intervals + 1);
            assert_eq!(result.energy_balances.len(), intervals);
            assert!((result.time_points_s.last().unwrap() - FINAL_TIME).abs() < 1e-14);
            assert_eq!(
                result.temperature_snapshots[0].as_host_f64().unwrap(),
                producer
                    .temperature_snapshots
                    .last()
                    .unwrap()
                    .as_host_f64()
                    .unwrap()
            );
            for balance in &result.energy_balances {
                assert!(balance.applied_power_w.canonical_value().abs() < 1e-10);
                assert!(balance.boundary_loss_w.canonical_value().abs() < 1e-10);
                assert!(balance.relative_residual < 1e-10, "{balance:?}");
                assert!(balance.storage_rate_w.canonical_value() <= 1e-9);
                assert!(balance.prescribed_boundary_power_w.canonical_value() >= -1e-9);
            }
            results.push(result);
        }
        assert_eq!(
            binding.artifact().canonical_bytes().unwrap(),
            producer_bytes
        );
        let measured = evidence::measure(&mesh, &results);
        println!("continuum cooling: quadratic={quadratic}, cells={cells}, {measured:?}");
        // The Tet4 one-cell fixture has no free thermal nodes: a zero field is a genuine
        // coarse spatial error, not an opportunity to replace the producer analytically.
        assert!(measured.temperature_contamination < 0.1 * measured.temperature);
        assert!(measured.flux_contamination < 0.1 * measured.flux);
        assert!(measured.heat_contamination < 0.1 * measured.heat);
        assert!(measured.power_contamination < 0.1 * measured.power);
        errors.push(measured);
    }
    let finest = errors.last().unwrap();
    // Predeclared approximation gates: P1 temperature L2 ~h²; P2 ~h³;
    // recovered element-mean flux has its own, weaker projection-specific gate.
    let (temperature_order, flux_order, temperature_limit, flux_limit) = if quadratic {
        (2.5, 1.5, 0.005, 0.02)
    } else {
        (1.7, 0.8, 0.06, 0.15)
    };
    for pair in errors.windows(2) {
        assert!(
            (pair[0].temperature / pair[1].temperature).log2() > temperature_order,
            "temperature: {errors:?}"
        );
        assert!(
            (pair[0].flux / pair[1].flux).log2() > flux_order,
            "flux: {errors:?}"
        );
        assert!(pair[1].heat < 0.6 * pair[0].heat, "heat: {errors:?}");
        assert!(pair[1].power < 0.6 * pair[0].power, "power: {errors:?}");
    }
    assert!(finest.temperature < temperature_limit);
    assert!(finest.flux < flux_limit);
    assert!(finest.heat < if quadratic { 0.005 } else { 0.08 });
    assert!(finest.power < if quadratic { 0.02 } else { 0.15 });
}
