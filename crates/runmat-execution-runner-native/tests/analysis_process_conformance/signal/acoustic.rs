//! Real compiled complex spectra feeding canonical acoustic execution, not a continuum oracle.
use super::*;
use runmat_analysis_core::{Pressure, SignalKnot, TabulatedSignalDomain, Velocity};
use runmat_analysis_execution::admit_named_signal_generator;

#[path = "acoustic/physics.rs"]
mod physics;

#[derive(Clone, Copy, Debug)]
enum Drive {
    Pressure,
    NormalVelocity,
    IncidentPressure,
}

pub(super) async fn run() {
    for drive in [
        Drive::Pressure,
        Drive::NormalVelocity,
        Drive::IncidentPressure,
    ] {
        let fixture = prepare(drive).await;
        assert_eq!(fixture.prepared.program.chunks.len(), 2);
        let expected = canonical_tabulated_signal_content_digest(&fixture.explicit).unwrap();
        let mut generated = Vec::new();
        for (placement, realization) in [
            ("inline", fixture.inline().await),
            ("isolated", fixture.isolated().await),
            ("durable", fixture.durable().await),
            ("encrypted remote", fixture.encrypted_remote().await),
        ] {
            assert_realization(&fixture, &realization, expected, placement);
            let table = materialize_signal_for_solver(
                &fixture.prepared.authored,
                &fixture.prepared.admitted,
                &realization,
            )
            .unwrap();
            assert!(table.generator().is_none());
            match &table {
                PhysicalSignal::Pressure(signal) => {
                    assert!(matches!(signal.provenance, SignalProvenance::Generated(_)));
                }
                PhysicalSignal::Velocity(signal) => {
                    assert!(matches!(signal.provenance, SignalProvenance::Generated(_)));
                }
                _ => unreachable!(),
            }
            generated.push((placement, table));
        }
        physics::compare(drive, &fixture, &generated);
        println!("acoustic spectrum {drive:?}: four generator placements, both element orders, five frequencies, two result chunk sizes PASS");
    }
}

async fn prepare(drive: Drive) -> SignalFixture {
    // Binary-exact knots and coefficients make byte equality meaningful, including an
    // interpolated solver sample at 3/16 Hz. Complex values come from RunMat execution.
    let source = "function y = pulse(f)\narguments\n f (:,1) double\nend\ny = complex(1 + 2*f, 0.5 - f);\nend\n";
    let compilation::CompiledSignalFixture {
        session: _session,
        project_root,
        project,
        revision,
        unit,
        owner_identity,
    } = compilation::compile(source).await;
    let executable = unit.portable_envelope_for(Some("pulse")).unwrap();
    let function = unit
        .function_analysis(
            executable
                .manifest
                .identity
                .entrypoint_function
                .expect("selected acoustic function"),
        )
        .unwrap();
    let (authored, explicit) = signals(drive);
    let admitted = admit_named_signal_generator(
        &authored,
        revision.clone(),
        BundleCallable {
            owner_identity,
            qualified_name: "loads.pulse".into(),
            source_digest: Digest::sha256(source),
        },
        runmat_analysis_execution::named_callable::CompiledCallableBinding::from_selected_identity(
            &executable.manifest.identity,
            "loads.pulse",
        )
        .unwrap(),
        function,
        &BTreeSet::new(),
    )
    .unwrap();
    let chunks = plan_signal_generator_chunks(&authored, &admitted, None, 4096 + 2 * 16, |_| {
        panic!("no artifact argument in acoustic spectrum")
    })
    .unwrap();
    let prepared = prepare_signal_materialization(
        authored,
        admitted,
        executable,
        chunks.chunks,
        Vec::new(),
        chunks.input_admissions,
        None,
    )
    .unwrap();
    SignalFixture {
        _project_root: project_root,
        project,
        revision,
        prepared,
        explicit,
    }
}

fn signals(drive: Drive) -> (PhysicalSignal, PhysicalSignal) {
    fn typed<Q>(
        make: impl Fn(f64) -> Q,
        wrap: impl Fn(Signal<Q>) -> PhysicalSignal,
    ) -> (PhysicalSignal, PhysicalSignal) {
        let signal = Signal {
            signal_id: SignalId("acoustic-spectrum".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Generator(SignalGenerator {
                invocation: runmat_analysis_core::GeneratorInvocation {
                    callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                        qualified_name: "loads.pulse".into(),
                    },
                    arguments: Vec::new(),
                    effects: GeneratorEffectPolicy::Pure,
                },
                domain: GeneratorDomain::FrequencyHertz,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.125,
                    end: 0.5,
                    sample_count: 4,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Complex64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 0,
            },
            provenance: SignalProvenance::Authored,
        };
        let explicit_representation = SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: [0.125, 0.25, 0.375, 0.5]
                .into_iter()
                .map(|f| {
                    SignalKnot::continuous(
                        f,
                        SignalOrdinate::Complex {
                            real: SignalValue::Scalar(make(1.0 + 2.0 * f)),
                            imaginary: SignalValue::Scalar(make(0.5 - f)),
                        },
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        });
        // Construct each typed signal separately without imposing Clone on physical units.
        let authored = wrap(signal);
        let explicit = Signal {
            signal_id: SignalId("acoustic-spectrum".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: explicit_representation,
            precision: SignalPrecision::Complex64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 4,
            },
            provenance: SignalProvenance::Authored,
        };
        (authored, wrap(explicit))
    }
    match drive {
        Drive::Pressure | Drive::IncidentPressure => typed(Pressure::new, PhysicalSignal::Pressure),
        Drive::NormalVelocity => typed(Velocity::new, PhysicalSignal::Velocity),
    }
}
