use std::collections::BTreeSet;

use runmat_analysis_core::{
    GeneratorArgument, GeneratorEffectPolicy, GeneratorProvenance, PhysicalSignal,
    SignalSamplingPolicy,
};
use runmat_analysis_fea::execution::{GeneratorIdentity, SignalMaterializationIdentity};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{CapabilityRequirement, EffectKind, ValueKindFact};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

const ADMISSION_PREFIX: &[u8] = b"runmat-analysis-generator-admission/v1\0";
const ADMISSION_LIMITS: CanonicalLimits = CanonicalLimits::new(8 * 1024 * 1024, 65_536, 4096, 32);

#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedSignalGenerator {
    pub program_revision: ProgramRevision,
    pub callable: BundleCallable,
    pub function: runmat_types::ProgramFunctionId,
    pub generator_identity: GeneratorIdentity,
    pub materialization_identity: SignalMaterializationIdentity,
    pub sample_points: Vec<f64>,
    pub required_capabilities: BTreeSet<CapabilityRequirement>,
    pub provenance: GeneratorProvenance,
}

/// Admits one named project function against the exact static facts embedded in its executable.
///
/// The caller resolves `callable` and `function` from the same frozen project and immutable
/// executable unit. This boundary deliberately accepts no source path and no live runtime closure.
pub fn admit_named_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    program_revision.validate()?;
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    if callable.owner_identity.trim().is_empty()
        || callable.qualified_name != generator.qualified_function
    {
        return Err(AnalysisBridgeError::Identity(
            "resolved callable differs from the authored generator",
        ));
    }
    validate_static_contract(generator, function, available_capabilities)?;

    let callable_digest = contract_digest(
        "callable",
        &(
            &program_revision,
            &callable.owner_identity,
            &callable.qualified_name,
            callable.source_digest,
            function.function,
        ),
    )?;
    let capture_digest = contract_digest("named-captures", &function.callable.captures)?;
    let generator_contract_digest = contract_digest("generator", generator)?;
    let generator_identity =
        GeneratorIdentity::new(callable_digest, capture_digest, generator_contract_digest)?;
    let authored_signal_digest = contract_digest("authored-signal", signal)?;
    let sampling_policy_digest = contract_digest("sampling", &generator.sampling)?;
    let frozen_input_digests = generator
        .arguments
        .iter()
        .filter_map(|argument| match argument {
            GeneratorArgument::Artifact(artifact) => Some(
                artifact
                    .content_digest
                    .parse::<Digest>()
                    .map_err(AnalysisBridgeError::Platform),
            ),
            _ => None,
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let materialization_identity = SignalMaterializationIdentity::new(
        authored_signal_digest,
        Some(generator_identity),
        sampling_policy_digest,
        frozen_input_digests,
    )?;
    let frozen_input_digest = contract_digest("frozen-inputs", &generator.arguments)?;

    Ok(AdmittedSignalGenerator {
        program_revision: program_revision.clone(),
        callable,
        function: function.function,
        generator_identity,
        materialization_identity,
        sample_points: sample_points(&generator.sampling)?,
        required_capabilities: function.capabilities.0.clone(),
        provenance: GeneratorProvenance {
            project_revision: program_revision.canonical_identity(),
            callable_id: callable_digest.to_string(),
            frozen_input_digest: frozen_input_digest.to_string(),
        },
    })
}

fn validate_static_contract(
    generator: &runmat_analysis_core::SignalGenerator,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<()> {
    let callable = &function.callable;
    let expected_inputs =
        generator
            .arguments
            .len()
            .checked_add(1)
            .ok_or(AnalysisBridgeError::Invalid(
                "generator argument count overflows".into(),
            ))?;
    if function.convergence != FunctionConvergence::Exact
        || !callable.parameters_complete
        || callable.variadic_inputs
        || callable.parameters.len() != expected_inputs
        || !callable.outputs_complete
        || callable.variadic_outputs
        || callable.outputs.len() != 1
        || function.outputs.len() != 1
    {
        return Err(AnalysisBridgeError::Invalid(format!(
            "generator `{}` requires an exact, non-variadic ({expected_inputs} inputs, 1 output) static signature",
            generator.qualified_function
        )));
    }
    if !callable.captures_complete || !callable.captures.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "named project generators cannot carry lexical captures".into(),
        ));
    }
    if !matches!(function.outputs[0].kind, ValueKindFact::Numeric(_)) {
        return Err(AnalysisBridgeError::Invalid(
            "signal generator output must be statically numeric".into(),
        ));
    }

    if matches!(
        generator.effects,
        GeneratorEffectPolicy::SeededRandom { .. }
    ) {
        return Err(AnalysisBridgeError::Invalid(
            "seeded-random generators require a task-scoped deterministic RNG context".into(),
        ));
    }
    let permitted_effects = BTreeSet::from([EffectKind::MayThrow]);
    if !function.effects.0.is_subset(&permitted_effects) {
        let forbidden = function
            .effects
            .0
            .difference(&permitted_effects)
            .copied()
            .collect::<Vec<_>>();
        return Err(AnalysisBridgeError::Invalid(format!(
            "signal generator has effects forbidden by its policy: {forbidden:?}"
        )));
    }
    if matches!(
        generator.effects,
        GeneratorEffectPolicy::DeclaredImmutableArtifacts
    ) && !generator
        .arguments
        .iter()
        .any(|argument| matches!(argument, GeneratorArgument::Artifact(_)))
    {
        return Err(AnalysisBridgeError::Invalid(
            "declared-immutable-artifacts policy requires an artifact argument".into(),
        ));
    }
    if !function.capabilities.0.is_subset(available_capabilities) {
        let unavailable = function
            .capabilities
            .0
            .difference(available_capabilities)
            .copied()
            .collect::<Vec<_>>();
        return Err(AnalysisBridgeError::Invalid(format!(
            "signal generator requires unavailable capabilities: {unavailable:?}"
        )));
    }
    Ok(())
}

fn sample_points(policy: &SignalSamplingPolicy) -> AnalysisBridgeResult<Vec<f64>> {
    let points = match policy {
        SignalSamplingPolicy::Uniform {
            start,
            end,
            sample_count,
        } => {
            let intervals = sample_count.checked_sub(1).ok_or_else(|| {
                AnalysisBridgeError::Invalid("uniform signal sampling requires two points".into())
            })?;
            (0..*sample_count)
                .map(|index| start + (end - start) * index as f64 / intervals as f64)
                .collect()
        }
        SignalSamplingPolicy::CanonicalPoints(points) => points.clone(),
        SignalSamplingPolicy::Adaptive { .. } => {
            return Err(AnalysisBridgeError::Invalid(
                "adaptive callable materialization requires the bounded refinement executor".into(),
            ))
        }
    };
    if points.len() < 2
        || points.iter().any(|point| !point.is_finite())
        || points.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(AnalysisBridgeError::Invalid(
            "signal materialization points are not finite and strictly increasing".into(),
        ));
    }
    Ok(points)
}

fn contract_digest<T: serde::Serialize>(
    domain: &'static str,
    value: &T,
) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        ADMISSION_PREFIX,
        domain,
        value,
        ADMISSION_LIMITS,
    )?))
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        Force, GeneratorDomain, Signal, SignalApproximationEvidence, SignalExtrapolation,
        SignalGenerator, SignalId, SignalInterpolation, SignalPrecision, SignalProvenance,
        SignalRepresentation, SignalShape,
    };
    use runmat_execution::ProgramEnvironment;
    use runmat_types::{
        CallableFact, CapabilitySet, EffectSet, NumericClass, NumericDomain, NumericFact,
        ProgramFunctionId, ValueFact,
    };

    use super::*;

    fn digest(label: &str) -> Digest {
        Digest::sha256(label)
    }

    fn revision() -> ProgramRevision {
        ProgramRevision::new(
            digest("graph"),
            digest("source"),
            ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
        )
        .unwrap()
    }

    fn numeric() -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }))
    }

    fn signal() -> PhysicalSignal {
        PhysicalSignal::Force(Signal::<Force> {
            signal_id: SignalId("pulse".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Generator(SignalGenerator {
                qualified_function: "loads.pulse".into(),
                arguments: vec![GeneratorArgument::Real(4.0)],
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 1.0,
                    sample_count: 5,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
                effects: GeneratorEffectPolicy::Pure,
            }),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: None,
                max_relative_error: None,
                derivative_max_absolute_error: None,
                sample_count: 0,
            },
            provenance: SignalProvenance::Authored,
        })
    }

    fn callable() -> BundleCallable {
        BundleCallable {
            owner_identity: "project:loads".into(),
            qualified_name: "loads.pulse".into(),
            source_digest: digest("loads.m"),
        }
    }

    fn analysis() -> FunctionAnalysis {
        let output = numeric();
        FunctionAnalysis {
            function: ProgramFunctionId(7),
            callable: CallableFact {
                identity: None,
                parameters: vec![numeric(), numeric()],
                parameters_complete: true,
                outputs: vec![output.clone()],
                outputs_complete: true,
                variadic_inputs: false,
                variadic_outputs: false,
                captures: Vec::new(),
                captures_complete: true,
            },
            outputs: vec![output],
            effects: EffectSet(BTreeSet::from([EffectKind::MayThrow])),
            capabilities: CapabilitySet::default(),
            convergence: FunctionConvergence::Exact,
        }
    }

    #[test]
    fn admits_exact_named_generator_and_derives_stable_points_and_identity() {
        let first = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let second = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sample_points, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(first.function, ProgramFunctionId(7));
    }

    #[test]
    fn admission_identity_changes_with_frozen_source_or_authored_arguments() {
        let baseline = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let mut changed_callable = callable();
        changed_callable.source_digest = digest("changed-loads.m");
        let changed_source = admit_named_signal_generator(
            &signal(),
            revision(),
            changed_callable,
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let mut changed_signal = signal();
        let PhysicalSignal::Force(value) = &mut changed_signal else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut value.representation else {
            unreachable!()
        };
        generator.arguments[0] = GeneratorArgument::Real(8.0);
        let changed_argument = admit_named_signal_generator(
            &changed_signal,
            revision(),
            callable(),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_ne!(
            baseline.generator_identity,
            changed_source.generator_identity
        );
        assert_ne!(
            baseline.materialization_identity,
            changed_argument.materialization_identity
        );
    }

    #[test]
    fn rejects_dynamic_effectful_capturing_or_capability_unavailable_functions() {
        let mut cases = Vec::new();
        let mut widened = analysis();
        widened.convergence = FunctionConvergence::Widened;
        cases.push(widened);
        let mut capturing = analysis();
        capturing.callable.captures.push(numeric());
        cases.push(capturing);
        let mut networked = analysis();
        networked.effects.0.insert(EffectKind::Network);
        cases.push(networked);
        let mut accelerator = analysis();
        accelerator
            .capabilities
            .0
            .insert(CapabilityRequirement::Accelerator);
        cases.push(accelerator);

        for facts in cases {
            assert!(admit_named_signal_generator(
                &signal(),
                revision(),
                callable(),
                &facts,
                &BTreeSet::new(),
            )
            .is_err());
        }
    }

    #[test]
    fn unscoped_seeded_randomness_and_unimplemented_adaptive_sampling_fail_closed() {
        let mut seeded_signal = signal();
        {
            let PhysicalSignal::Force(value) = &mut seeded_signal else {
                unreachable!()
            };
            let SignalRepresentation::Generator(generator) = &mut value.representation else {
                unreachable!()
            };
            generator.effects = GeneratorEffectPolicy::SeededRandom {
                seed: 42,
                distribution: runmat_analysis_core::RandomDistribution::Normal,
            };
        }
        let mut random = analysis();
        random.effects.0.insert(EffectKind::Randomness);
        assert!(admit_named_signal_generator(
            &seeded_signal,
            revision(),
            callable(),
            &random,
            &BTreeSet::new(),
        )
        .is_err());

        let PhysicalSignal::Force(value) = &mut seeded_signal else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut value.representation else {
            unreachable!()
        };
        generator.sampling = SignalSamplingPolicy::Adaptive {
            start: 0.0,
            end: 1.0,
            max_absolute_error: 1e-6,
            max_samples: 100,
        };
        assert!(admit_named_signal_generator(
            &seeded_signal,
            revision(),
            callable(),
            &random,
            &BTreeSet::new(),
        )
        .is_err());
    }
}
