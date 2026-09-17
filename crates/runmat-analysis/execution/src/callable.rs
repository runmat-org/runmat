use std::collections::BTreeSet;

mod capture;

use capture::validate_capture_contract;

use runmat_analysis_core::{
    GeneratorArgument, GeneratorEffectPolicy, GeneratorInvocation, GeneratorProvenance,
    PhysicalSignal, RandomExcitationSpectrum, ResponseSpectrum, SignalSamplingPolicy,
};
use runmat_analysis_fea::execution::{
    GeneratorIdentity, NumericalMaterializationIdentity, SignalMaterializationIdentity,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::value::{CallableValue, ValueLimits, ValuePayload};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{
    CapabilityRequirement, DimensionFact, EffectKind, NumericClass, NumericDomain, ShapeFact,
    ValueFact, ValueKindFact,
};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

const ADMISSION_PREFIX: &[u8] = b"runmat-analysis-generator-admission/v1\0";
const ADMISSION_LIMITS: CanonicalLimits = CanonicalLimits::new(8 * 1024 * 1024, 65_536, 4096, 32);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedSignalGenerator {
    pub program_revision: ProgramRevision,
    pub callable: BundleCallable,
    pub executable_source_digest: Digest,
    pub function: runmat_types::ProgramFunctionId,
    pub captures: Vec<ValuePayload>,
    pub generator_identity: GeneratorIdentity,
    pub materialization_identity: SignalMaterializationIdentity,
    pub sample_points: Vec<f64>,
    pub required_capabilities: BTreeSet<CapabilityRequirement>,
    pub provenance: GeneratorProvenance,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedGeneratorInvocation {
    pub program_revision: ProgramRevision,
    pub callable: BundleCallable,
    pub executable_source_digest: Digest,
    pub function: runmat_types::ProgramFunctionId,
    pub captures: Vec<ValuePayload>,
    pub generator_identity: GeneratorIdentity,
    pub required_capabilities: BTreeSet<CapabilityRequirement>,
    pub provenance: GeneratorProvenance,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalGeneratorKind {
    ResponseSpectrum {
        spectrum_id: runmat_analysis_core::ResponseSpectrumId,
    },
    RandomSpectrum {
        excitation_id: runmat_analysis_core::RandomExcitationId,
        channel_count: usize,
    },
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedNumericalGenerator {
    pub invocation: AdmittedGeneratorInvocation,
    pub materialization_identity: NumericalMaterializationIdentity,
    pub kind: NumericalGeneratorKind,
    pub sample_points: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GeneratorOutputDomain {
    Real,
    RealOrComplex,
}

impl GeneratorOutputDomain {
    fn accepts(self, kind: &ValueKindFact) -> bool {
        match kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(numeric) => match self {
                Self::Real => numeric.domain == NumericDomain::Real,
                Self::RealOrComplex => true,
            },
            _ => false,
        }
    }
}

#[derive(Default)]
struct FrozenCallableBinding {
    captures: Vec<ValuePayload>,
    callable_value_digest: Option<Digest>,
}

/// Admits one named project function against the exact static facts embedded in its executable.
///
/// The caller resolves `callable` and `function` from the same frozen project and immutable
/// executable unit. This boundary deliberately accepts no source path and no live runtime closure.
pub fn admit_named_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    if !matches!(
        &generator.invocation.callable,
        runmat_analysis_core::GeneratorCallable::ProjectFunction { .. }
            | runmat_analysis_core::GeneratorCallable::ImportedFunction(_)
    ) {
        return Err(AnalysisBridgeError::Identity(
            "named project admission requires a project-function generator identity",
        ));
    }
    admit_signal_generator(
        signal,
        program_revision,
        callable,
        executable_source_digest,
        function,
        FrozenCallableBinding::default(),
        available_capabilities,
    )
}

/// Admit a semantically bound RunMat function handle or anonymous closure.
///
/// The runtime freezes the closure through its ordinary portable value codec before calling this
/// boundary. Dynamic workspace handles are deliberately excluded: the callable must name the exact
/// function in the frozen executable, and every capture must be a closed portable value.
pub fn admit_callable_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    frozen_callable: CallableValue,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    frozen_callable.validate_identity()?;
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    if !matches!(
        &generator.invocation.callable,
        runmat_analysis_core::GeneratorCallable::BoundFunction { function: bound, .. }
            if *bound == function.function
    ) {
        return Err(AnalysisBridgeError::Identity(
            "runtime callable function differs from the authored bound identity",
        ));
    }
    let expected_owner = format!("bound:{}", function.function.0);
    if frozen_callable.owner_identity != expected_owner
        || frozen_callable.qualified_name != callable.qualified_name
    {
        return Err(AnalysisBridgeError::Identity(
            "runtime callable is not bound to the frozen executable function",
        ));
    }
    for capture in &frozen_callable.captures {
        capture.validate(ValueLimits::default())?;
    }
    let callable_value_digest = frozen_callable.callable_digest;
    admit_signal_generator(
        signal,
        program_revision,
        callable,
        executable_source_digest,
        function,
        FrozenCallableBinding {
            captures: frozen_callable.captures,
            callable_value_digest: Some(callable_value_digest),
        },
        available_capabilities,
    )
}

pub fn admit_named_response_spectrum_generator(
    spectrum: &ResponseSpectrum,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedNumericalGenerator> {
    let invocation = spectrum.ordinates.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("response spectrum is not callable-backed".into())
    })?;
    admit_named_numerical_generator(
        invocation,
        program_revision,
        callable,
        executable_source_digest,
        function,
        GeneratorOutputDomain::Real,
        spectrum_axis(&spectrum.abscissa),
        NumericalGeneratorKind::ResponseSpectrum {
            spectrum_id: spectrum.spectrum_id.clone(),
        },
        contract_digest(
            "response-spectrum-materialization-contract",
            &(
                &spectrum.spectrum_id,
                &spectrum.abscissa,
                spectrum.damping,
                spectrum.response,
                spectrum.envelope,
                spectrum.interpolation,
                spectrum.extrapolation,
                &spectrum.direction,
                spectrum.modal_combination,
                spectrum.directional_combination,
                spectrum.truncation,
                spectrum.rigid_response,
                &spectrum.construction,
                &spectrum.evidence,
            ),
        )?,
        available_capabilities,
    )
}

pub fn admit_named_random_spectrum_generator(
    spectrum: &RandomExcitationSpectrum,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedNumericalGenerator> {
    let invocation = spectrum.spectral_density.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("random spectrum is not callable-backed".into())
    })?;
    admit_named_numerical_generator(
        invocation,
        program_revision,
        callable,
        executable_source_digest,
        function,
        GeneratorOutputDomain::RealOrComplex,
        &spectrum.frequencies_hz,
        NumericalGeneratorKind::RandomSpectrum {
            excitation_id: spectrum.excitation_id.clone(),
            channel_count: spectrum.channels.len(),
        },
        contract_digest(
            "random-spectrum-materialization-contract",
            &(
                &spectrum.excitation_id,
                &spectrum.frequencies_hz,
                &spectrum.channels,
                spectrum.convention,
                spectrum.interpolation,
                spectrum.extrapolation,
                &spectrum.evidence,
            ),
        )?,
        available_capabilities,
    )
}

fn admit_named_numerical_generator(
    invocation: &GeneratorInvocation,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    output_domain: GeneratorOutputDomain,
    sample_points: &[f64],
    kind: NumericalGeneratorKind,
    authored_contract_digest: Digest,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedNumericalGenerator> {
    if !matches!(
        invocation.callable,
        runmat_analysis_core::GeneratorCallable::ProjectFunction { .. }
            | runmat_analysis_core::GeneratorCallable::ImportedFunction(_)
    ) {
        return Err(AnalysisBridgeError::Identity(
            "named project admission requires a project-function generator identity",
        ));
    }
    validate_canonical_points(sample_points)?;
    let invocation = admit_generator_invocation(
        invocation,
        program_revision,
        callable,
        executable_source_digest,
        function,
        FrozenCallableBinding::default(),
        output_domain,
        available_capabilities,
    )?;
    let materialization_identity = NumericalMaterializationIdentity::new(
        authored_contract_digest,
        invocation.generator_identity,
    )?;
    Ok(AdmittedNumericalGenerator {
        invocation,
        materialization_identity,
        kind,
        sample_points: sample_points.to_vec(),
    })
}

fn spectrum_axis(axis: &runmat_analysis_core::SpectrumAbscissa) -> &[f64] {
    match axis {
        runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(values)
        | runmat_analysis_core::SpectrumAbscissa::PeriodSeconds(values) => values,
    }
}

fn validate_canonical_points(points: &[f64]) -> AnalysisBridgeResult<()> {
    if points.len() < 2
        || points.iter().any(|point| !point.is_finite())
        || points.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(AnalysisBridgeError::Invalid(
            "generator axis must contain at least two finite, strictly increasing points".into(),
        ));
    }
    Ok(())
}

fn admit_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    binding: FrozenCallableBinding,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    let admitted = admit_generator_invocation(
        &generator.invocation,
        program_revision,
        callable,
        executable_source_digest,
        function,
        binding,
        GeneratorOutputDomain::RealOrComplex,
        available_capabilities,
    )?;
    let authored_signal_digest = contract_digest(
        "authored-signal-contract",
        &(
            signal.signal_id(),
            signal.quantity(),
            signal.shape(),
            signal.frame(),
            signal.precision(),
        ),
    )?;
    let realization_policy_digest = contract_digest(
        "realization-policy",
        &(
            &generator.domain,
            &generator.sampling,
            generator.interpolation,
            generator.extrapolation,
        ),
    )?;
    let materialization_identity = SignalMaterializationIdentity::new(
        authored_signal_digest,
        Some(admitted.generator_identity),
        realization_policy_digest,
    )?;
    Ok(AdmittedSignalGenerator {
        program_revision: admitted.program_revision,
        callable: admitted.callable,
        executable_source_digest: admitted.executable_source_digest,
        function: admitted.function,
        captures: admitted.captures,
        generator_identity: admitted.generator_identity,
        materialization_identity,
        sample_points: sample_points(&generator.sampling)?,
        required_capabilities: admitted.required_capabilities,
        provenance: admitted.provenance,
    })
}

fn admit_generator_invocation(
    invocation: &GeneratorInvocation,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    function: &FunctionAnalysis,
    binding: FrozenCallableBinding,
    output_domain: GeneratorOutputDomain,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedGeneratorInvocation> {
    let FrozenCallableBinding {
        captures,
        callable_value_digest,
    } = binding;
    program_revision.validate()?;
    if callable.owner_identity.trim().is_empty()
        || callable.qualified_name != invocation.callable.display_name()
    {
        return Err(AnalysisBridgeError::Identity(
            "resolved callable differs from the authored generator",
        ));
    }
    validate_static_contract(
        invocation,
        function,
        captures.len(),
        output_domain,
        available_capabilities,
    )?;
    validate_capture_contract(&captures, &function.callable.captures)?;
    for argument in &invocation.arguments {
        if let GeneratorArgument::Artifact(artifact) = argument {
            artifact
                .content_digest
                .parse::<Digest>()
                .map_err(AnalysisBridgeError::Platform)?;
        }
    }

    let callable_digest = contract_digest(
        "callable",
        &(
            &program_revision,
            &callable.owner_identity,
            &callable.qualified_name,
            callable.source_digest,
            executable_source_digest,
            function.function,
            callable_value_digest,
        ),
    )?;
    let capture_digest = contract_digest("captures", &captures)?;
    let invocation_digest = contract_digest("generator-invocation", invocation)?;
    let generator_identity =
        GeneratorIdentity::new(callable_digest, capture_digest, invocation_digest)?;
    let frozen_input_digest =
        contract_digest("frozen-inputs", &(&captures, &invocation.arguments))?;

    Ok(AdmittedGeneratorInvocation {
        program_revision: program_revision.clone(),
        callable,
        executable_source_digest,
        function: function.function,
        captures,
        generator_identity,
        required_capabilities: function.capabilities.0.clone(),
        provenance: GeneratorProvenance {
            project_revision: program_revision.canonical_identity(),
            callable_id: callable_digest.to_string(),
            frozen_input_digest: frozen_input_digest.to_string(),
        },
    })
}

fn validate_static_contract(
    invocation: &GeneratorInvocation,
    function: &FunctionAnalysis,
    capture_count: usize,
    output_domain: GeneratorOutputDomain,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<()> {
    let callable = &function.callable;
    let expected_inputs =
        invocation
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
            invocation.callable.display_name()
        )));
    }
    if !callable.captures_complete || callable.captures.len() != capture_count {
        return Err(AnalysisBridgeError::Invalid(
            "frozen callable capture values do not match its exact static capture contract".into(),
        ));
    }
    if !output_domain.accepts(&function.outputs[0].kind) {
        return Err(AnalysisBridgeError::Invalid(
            "generator output has an incompatible static numeric domain".into(),
        ));
    }
    validate_axis_parameter(&callable.parameters[0])?;
    for (argument, parameter) in invocation.arguments.iter().zip(&callable.parameters[1..]) {
        validate_authored_parameter(argument, parameter)?;
    }

    if matches!(
        invocation.effects,
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
            "analysis generator has effects forbidden by its policy: {forbidden:?}"
        )));
    }
    if matches!(
        invocation.effects,
        GeneratorEffectPolicy::DeclaredImmutableArtifacts
    ) && !invocation
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
            "analysis generator requires unavailable capabilities: {unavailable:?}"
        )));
    }
    Ok(())
}

fn validate_axis_parameter(parameter: &ValueFact) -> AnalysisBridgeResult<()> {
    if !accepts_f64(parameter) || !accepts_column(&parameter.shape) {
        return Err(AnalysisBridgeError::Invalid(
            "analysis generator axis parameter must accept an N-by-1 real double array".into(),
        ));
    }
    Ok(())
}

fn validate_authored_parameter(
    argument: &GeneratorArgument,
    parameter: &ValueFact,
) -> AnalysisBridgeResult<()> {
    let valid = match argument {
        GeneratorArgument::Boolean(_) => {
            accepts_kind(parameter, |kind| matches!(kind, ValueKindFact::Logical))
                && accepts_scalar(&parameter.shape)
        }
        GeneratorArgument::SignedInteger(_) => {
            accepts_numeric_class(parameter, NumericClass::Int64)
                && accepts_scalar(&parameter.shape)
        }
        GeneratorArgument::UnsignedInteger(_) => {
            accepts_numeric_class(parameter, NumericClass::UInt64)
                && accepts_scalar(&parameter.shape)
        }
        GeneratorArgument::Real(_) => accepts_f64(parameter) && accepts_scalar(&parameter.shape),
        GeneratorArgument::Text(_) => {
            accepts_kind(parameter, |kind| matches!(kind, ValueKindFact::String))
                && accepts_scalar(&parameter.shape)
        }
        GeneratorArgument::RealVector(_) => {
            accepts_f64(parameter) && accepts_column(&parameter.shape)
        }
        GeneratorArgument::Artifact(_) => true,
    };
    if !valid {
        return Err(AnalysisBridgeError::Invalid(
            "analysis generator parameter does not accept its frozen argument representation"
                .into(),
        ));
    }
    Ok(())
}

fn accepts_f64(parameter: &ValueFact) -> bool {
    accepts_numeric_class(parameter, NumericClass::Double)
}

fn accepts_numeric_class(parameter: &ValueFact, expected: NumericClass) -> bool {
    accepts_kind(parameter, |kind| {
        matches!(
            kind,
            ValueKindFact::Numeric(numeric)
                if numeric.class == expected && numeric.domain == NumericDomain::Real
        )
    })
}

fn accepts_kind(parameter: &ValueFact, expected: impl FnOnce(&ValueKindFact) -> bool) -> bool {
    matches!(&parameter.kind, ValueKindFact::Unknown) || expected(&parameter.kind)
}

fn accepts_scalar(shape: &ShapeFact) -> bool {
    matches!(shape, ShapeFact::Unknown | ShapeFact::Scalar)
        || matches!(
            shape,
            ShapeFact::Shaped { dims }
                if !dims.is_empty()
                    && dims
                        .iter()
                        .all(|dimension| matches!(dimension, DimensionFact::Known(1)))
        )
}

fn accepts_column(shape: &ShapeFact) -> bool {
    match shape {
        ShapeFact::Unknown | ShapeFact::Ranked { rank: 2 } => true,
        ShapeFact::Shaped { dims } => {
            dims.len() == 2
                && !matches!(dims[0], DimensionFact::Known(_))
                && matches!(dims[1], DimensionFact::Known(1))
        }
        ShapeFact::Scalar | ShapeFact::Ranked { .. } => false,
    }
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
    use runmat_execution::value::InlineValue;
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

    fn numeric_column() -> ValueFact {
        let mut fact = numeric();
        fact.shape = ShapeFact::Shaped {
            dims: vec![DimensionFact::Unknown, DimensionFact::Known(1)],
        };
        fact
    }

    fn signal() -> PhysicalSignal {
        PhysicalSignal::Force(Signal::<Force> {
            signal_id: SignalId("pulse".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Generator(SignalGenerator {
                invocation: runmat_analysis_core::GeneratorInvocation {
                    callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                        qualified_name: "loads.pulse".into(),
                    },
                    arguments: vec![GeneratorArgument::Real(4.0)],
                    effects: GeneratorEffectPolicy::Pure,
                },
                domain: GeneratorDomain::TimeSeconds,
                sampling: SignalSamplingPolicy::Uniform {
                    start: 0.0,
                    end: 1.0,
                    sample_count: 5,
                },
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
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
                parameters: vec![numeric_column(), numeric()],
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
            digest("executable-source"),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let second = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            digest("executable-source"),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sample_points, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(first.function, ProgramFunctionId(7));
    }

    #[test]
    fn admission_identity_invalidates_only_the_affected_layers() {
        let baseline = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            digest("executable-source"),
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
            digest("executable-source"),
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
        generator.invocation.arguments[0] = GeneratorArgument::Real(8.0);
        let changed_argument = admit_named_signal_generator(
            &changed_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let mut changed_sampling_signal = signal();
        let PhysicalSignal::Force(value) = &mut changed_sampling_signal else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut value.representation else {
            unreachable!()
        };
        generator.sampling = SignalSamplingPolicy::Uniform {
            start: 0.0,
            end: 1.0,
            sample_count: 9,
        };
        let changed_sampling = admit_named_signal_generator(
            &changed_sampling_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let mut renamed_signal = signal();
        let PhysicalSignal::Force(value) = &mut renamed_signal else {
            unreachable!()
        };
        value.signal_id = SignalId("renamed-pulse".into());
        let renamed = admit_named_signal_generator(
            &renamed_signal,
            revision(),
            callable(),
            digest("executable-source"),
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
            changed_source.materialization_identity
        );
        assert_ne!(
            baseline.generator_identity,
            changed_argument.generator_identity
        );
        assert_ne!(
            baseline.materialization_identity,
            changed_argument.materialization_identity
        );
        assert_eq!(
            baseline.generator_identity, changed_sampling.generator_identity,
            "sampling is a realization policy, not callable provenance"
        );
        assert_ne!(
            baseline.materialization_identity,
            changed_sampling.materialization_identity
        );
        assert_eq!(
            baseline.generator_identity, renamed.generator_identity,
            "the authored signal label does not change the callable"
        );
        assert_ne!(
            baseline.materialization_identity,
            renamed.materialization_identity
        );
    }

    #[test]
    fn admits_only_semantically_bound_portable_closures_and_hashes_capture_values() {
        let mut facts = analysis();
        facts.callable.captures = vec![numeric()];
        let mut bound_signal = signal();
        let PhysicalSignal::Force(signal) = &mut bound_signal else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = &mut signal.representation else {
            unreachable!()
        };
        generator.invocation.callable = runmat_analysis_core::GeneratorCallable::BoundFunction {
            function: runmat_types::ProgramFunctionId(7),
            display_name: "loads.pulse".into(),
        };
        let frozen = |capture: f64| CallableValue {
            owner_identity: "bound:7".into(),
            qualified_name: "loads.pulse".into(),
            callable_digest: CallableValue::identity_digest("bound:7", "loads.pulse"),
            captures: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
                capture.to_bits(),
            )))],
        };
        let first = admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &facts,
            frozen(2.0),
            &BTreeSet::new(),
        )
        .unwrap();
        let changed_capture = admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &facts,
            frozen(3.0),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(first.captures, frozen(2.0).captures);
        assert_ne!(first.generator_identity, changed_capture.generator_identity);
        assert_ne!(
            first.materialization_identity,
            changed_capture.materialization_identity
        );

        let mut unresolved = frozen(2.0);
        unresolved.owner_identity = "workspace".into();
        unresolved.callable_digest =
            CallableValue::identity_digest(&unresolved.owner_identity, &unresolved.qualified_name);
        assert!(admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &facts,
            unresolved,
            &BTreeSet::new(),
        )
        .is_err());

        let mut wrong_capture_type = frozen(2.0);
        wrong_capture_type.captures = vec![ValuePayload::Inline(Box::new(InlineValue::String(
            "not numeric".into(),
        )))];
        assert!(admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            digest("executable-source"),
            &facts,
            wrong_capture_type,
            &BTreeSet::new(),
        )
        .is_err());
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
                digest("executable-source"),
                &facts,
                &BTreeSet::new(),
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_parameters_that_cannot_accept_the_materialization_wire_values() {
        let mut scalar_axis = analysis();
        scalar_axis.callable.parameters[0] = numeric();
        assert!(admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            digest("executable-source"),
            &scalar_axis,
            &BTreeSet::new(),
        )
        .is_err());

        let mut wrong_argument_class = analysis();
        wrong_argument_class.callable.parameters[1].kind = ValueKindFact::Logical;
        assert!(admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            digest("executable-source"),
            &wrong_argument_class,
            &BTreeSet::new(),
        )
        .is_err());
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
            generator.invocation.effects = GeneratorEffectPolicy::SeededRandom {
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
            digest("executable-source"),
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
            digest("executable-source"),
            &random,
            &BTreeSet::new(),
        )
        .is_err());
    }
}
