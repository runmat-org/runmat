use std::collections::BTreeSet;

mod capture;
pub(crate) mod graph;
pub(crate) mod executable;

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
    CapabilityRequirement, DimensionFact, NumericClass, NumericDomain, ShapeFact, ValueFact,
    ValueKindFact,
};

use crate::{AnalysisBridgeError, AnalysisBridgeResult};

use crate::named_callable::CompiledCallableBinding;

const ADMISSION_PREFIX: &[u8] = b"runmat-analysis-generator-admission/v3\0";
const ADMISSION_LIMITS: CanonicalLimits = CanonicalLimits::new(8 * 1024 * 1024, 65_536, 4096, 32);

/// Exactly one frozen callable input authority. A shared activation cannot be
/// represented simultaneously by an independent flat capture vector.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum FrozenGeneratorInputs {
    DirectCaptures(Vec<ValuePayload>),
    Graph(Box<runmat_execution_artifact::GraphExecutionRequest>),
}
impl FrozenGeneratorInputs {
    /// The direct executable ABI prefix. Graph targets own their captures and
    /// receive only actual function arguments, so this prefix is empty there.
    pub fn direct_captures(&self) -> &[ValuePayload] {
        match self {
            Self::DirectCaptures(values) => values,
            Self::Graph(_) => &[],
        }
    }
    pub fn graph(&self) -> Option<&runmat_execution_artifact::GraphExecutionRequest> {
        match self {
            Self::DirectCaptures(_) => None,
            Self::Graph(graph) => Some(graph),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedSignalGenerator {
    pub schema_version: u16,
    pub value_layout: crate::CanonicalGeneratorValueLayout,
    pub program_revision: ProgramRevision,
    pub callable: BundleCallable,
    pub executable_source_digest: Digest,
    pub compiled_binding: CompiledCallableBinding,
    pub function: runmat_types::ProgramFunctionId,
    pub frozen_inputs: FrozenGeneratorInputs,
    pub generator_identity: GeneratorIdentity,
    pub materialization_identity: SignalMaterializationIdentity,
    pub sample_points:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
    pub required_capabilities: BTreeSet<CapabilityRequirement>,
    pub provenance: GeneratorProvenance,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmittedGeneratorInvocation {
    pub program_revision: ProgramRevision,
    pub callable: BundleCallable,
    pub executable_source_digest: Digest,
    pub compiled_binding: CompiledCallableBinding,
    pub function: runmat_types::ProgramFunctionId,
    pub frozen_inputs: FrozenGeneratorInputs,
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
    pub schema_version: u16,
    pub invocation: AdmittedGeneratorInvocation,
    pub materialization_identity: NumericalMaterializationIdentity,
    pub kind: NumericalGeneratorKind,
    pub sample_points:
        runmat_types::physical::Quantity<runmat_types::physical::DynamicDimension, Vec<f64>>,
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

struct FrozenCallableBinding {
    inputs: FrozenGeneratorInputs,
    callable_value_digest: Option<Digest>,
}
impl Default for FrozenCallableBinding {
    fn default() -> Self {
        Self {
            inputs: FrozenGeneratorInputs::DirectCaptures(Vec::new()),
            callable_value_digest: None,
        }
    }
}

struct CallableAdmission<'a> {
    program_revision: ProgramRevision,
    callable: BundleCallable,
    executable_source_digest: Digest,
    compiled_binding: CompiledCallableBinding,
    function: &'a FunctionAnalysis,
    available_capabilities: &'a BTreeSet<CapabilityRequirement>,
}

struct NumericalGeneratorAdmission<'a> {
    output_domain: GeneratorOutputDomain,
    sample_points: &'a [f64],
    axis: runmat_types::physical::DynamicDimension,
    kind: NumericalGeneratorKind,
    authored_contract_digest: Digest,
}

/// Admits one named project function against the exact static facts embedded in its executable.
///
/// The caller resolves `callable` and `function` from the same frozen project and immutable
/// executable unit. This boundary deliberately accepts no source path and no live runtime closure.
pub fn admit_named_signal_generator(
    signal: &PhysicalSignal,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    let executable_source_digest = compiled_binding.source_digest();
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
        CallableAdmission {
            program_revision,
            callable,
            executable_source_digest,
            compiled_binding,
            function,
            available_capabilities,
        },
        FrozenCallableBinding::default(),
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
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    frozen_callable: CallableValue,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    let executable_source_digest = compiled_binding.source_digest();
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
    let expected_owner = runmat_execution::value::CallableOwner::Bound(
        runmat_execution::FunctionBindingOrigin::from_revision(
            &program_revision,
            function.function,
        )?,
    );
    if frozen_callable.owner != expected_owner
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
        CallableAdmission {
            program_revision,
            callable,
            executable_source_digest,
            compiled_binding,
            function,
            available_capabilities,
        },
        FrozenCallableBinding {
            inputs: FrozenGeneratorInputs::DirectCaptures(frozen_callable.captures),
            callable_value_digest: Some(callable_value_digest),
        },
    )
}

pub fn admit_named_response_spectrum_generator(
    spectrum: &ResponseSpectrum,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedNumericalGenerator> {
    let executable_source_digest = compiled_binding.source_digest();
    let invocation = spectrum.ordinates.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("response spectrum is not callable-backed".into())
    })?;
    admit_named_numerical_generator(
        invocation,
        CallableAdmission {
            program_revision,
            callable,
            executable_source_digest,
            compiled_binding,
            function,
            available_capabilities,
        },
        NumericalGeneratorAdmission {
            output_domain: GeneratorOutputDomain::Real,
            sample_points: spectrum_axis(&spectrum.abscissa),
            axis: spectrum_axis_descriptor(&spectrum.abscissa)?,
            kind: NumericalGeneratorKind::ResponseSpectrum {
                spectrum_id: spectrum.spectrum_id.clone(),
            },
            authored_contract_digest: contract_digest(
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
        },
    )
}

pub fn admit_named_random_spectrum_generator(
    spectrum: &RandomExcitationSpectrum,
    program_revision: ProgramRevision,
    callable: BundleCallable,
    compiled_binding: CompiledCallableBinding,
    function: &FunctionAnalysis,
    available_capabilities: &BTreeSet<CapabilityRequirement>,
) -> AnalysisBridgeResult<AdmittedNumericalGenerator> {
    let executable_source_digest = compiled_binding.source_digest();
    let invocation = spectrum.spectral_density.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("random spectrum is not callable-backed".into())
    })?;
    admit_named_numerical_generator(
        invocation,
        CallableAdmission {
            program_revision,
            callable,
            executable_source_digest,
            compiled_binding,
            function,
            available_capabilities,
        },
        NumericalGeneratorAdmission {
            output_domain: GeneratorOutputDomain::RealOrComplex,
            sample_points: spectrum.frequencies_hz.canonical_values(),
            axis: runmat_analysis_core::GeneratorDomain::FrequencyHertz
                .axis_descriptor(None)
                .map_err(AnalysisBridgeError::Invalid)?,
            kind: NumericalGeneratorKind::RandomSpectrum {
                excitation_id: spectrum.excitation_id.clone(),
                channel_count: spectrum.channels.len(),
            },
            authored_contract_digest: contract_digest(
                "random-spectrum-materialization-contract",
                &(
                    &spectrum.excitation_id,
                    spectrum.frequencies_hz.canonical_values(),
                    &spectrum.channels,
                    spectrum.convention,
                    spectrum.interpolation,
                    spectrum.extrapolation,
                    &spectrum.evidence,
                ),
            )?,
        },
    )
}

fn admit_named_numerical_generator(
    invocation: &GeneratorInvocation,
    callable: CallableAdmission<'_>,
    numerical: NumericalGeneratorAdmission<'_>,
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
    validate_canonical_points(numerical.sample_points)?;
    let invocation = admit_generator_invocation(
        invocation,
        callable,
        FrozenCallableBinding::default(),
        numerical.output_domain,
    )?;
    let materialization_identity = NumericalMaterializationIdentity::new(
        numerical.authored_contract_digest,
        invocation.generator_identity,
    )?;
    Ok(AdmittedNumericalGenerator {
        schema_version: 2,
        invocation,
        materialization_identity,
        kind: numerical.kind,
        sample_points: runmat_types::physical::Quantity::try_from_descriptor(
            numerical.axis,
            numerical.sample_points.to_vec(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
    })
}

pub(crate) fn spectrum_axis(axis: &runmat_analysis_core::SpectrumAbscissa) -> &[f64] {
    axis.canonical_points()
}

pub(crate) fn spectrum_axis_descriptor(
    axis: &runmat_analysis_core::SpectrumAbscissa,
) -> AnalysisBridgeResult<runmat_types::physical::DynamicDimension> {
    match axis {
        runmat_analysis_core::SpectrumAbscissa::FrequencyHertz(_) => {
            runmat_analysis_core::GeneratorDomain::FrequencyHertz
        }
        runmat_analysis_core::SpectrumAbscissa::PeriodSeconds(_) => {
            runmat_analysis_core::GeneratorDomain::TimeSeconds
        }
    }
    .axis_descriptor(None)
    .map_err(AnalysisBridgeError::Invalid)
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
    callable: CallableAdmission<'_>,
    binding: FrozenCallableBinding,
) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    runmat_analysis_core::validate_physical_signals(std::slice::from_ref(signal))
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let generator = signal.generator().ok_or_else(|| {
        AnalysisBridgeError::Invalid("signal is not backed by a callable generator".into())
    })?;
    let admitted = admit_generator_invocation(
        &generator.invocation,
        callable,
        binding,
        GeneratorOutputDomain::RealOrComplex,
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
            crate::CanonicalGeneratorValueLayout::SampleMajorRowMajor,
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
        schema_version: 2,
        value_layout: crate::CanonicalGeneratorValueLayout::SampleMajorRowMajor,
        program_revision: admitted.program_revision,
        callable: admitted.callable,
        executable_source_digest: admitted.executable_source_digest,
        compiled_binding: admitted.compiled_binding,
        function: admitted.function,
        frozen_inputs: admitted.frozen_inputs,
        generator_identity: admitted.generator_identity,
        materialization_identity,
        sample_points: runmat_types::physical::Quantity::try_from_descriptor(
            generator
                .domain
                .axis_descriptor(None)
                .map_err(AnalysisBridgeError::Invalid)?,
            sample_points(&generator.sampling)?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        required_capabilities: admitted.required_capabilities,
        provenance: admitted.provenance,
    })
}

fn admit_generator_invocation(
    invocation: &GeneratorInvocation,
    callable: CallableAdmission<'_>,
    binding: FrozenCallableBinding,
    output_domain: GeneratorOutputDomain,
) -> AnalysisBridgeResult<AdmittedGeneratorInvocation> {
    let CallableAdmission {
        program_revision,
        callable,
        executable_source_digest,
        compiled_binding,
        function,
        available_capabilities,
    } = callable;
    let FrozenCallableBinding {
        inputs,
        callable_value_digest,
    } = binding;
    let captures = inputs.direct_captures();
    program_revision.validate()?;
    compiled_binding.validate_admission(&callable, function.function, executable_source_digest)?;
    if callable.owner_identity.trim().is_empty()
        || callable.qualified_name != invocation.callable.display_name()
    {
        return Err(AnalysisBridgeError::Identity(
            "resolved callable differs from the authored generator",
        ));
    }
    let graph_captures = inputs
        .graph()
        .map(|graph| graph::captures(graph, &program_revision, function))
        .transpose()?;
    validate_static_contract(
        invocation,
        function,
        graph_captures.as_ref().map_or(captures.len(), Vec::len),
        output_domain,
        available_capabilities,
    )?;
    if let Some(graph_captures) = graph_captures.as_ref() {
        if !captures.is_empty() {
            return Err(AnalysisBridgeError::Identity(
                "graph callable has a redundant flat capture ABI",
            ));
        }
        capture::validate_capture_contract_iter(
            graph_captures.iter().copied(),
            &function.callable.captures,
        )?;
    } else {
        if function
            .captures
            .iter()
            .any(|capture| capture.source != runmat_mir::analysis::FunctionCaptureSource::ByValue)
        {
            return Err(AnalysisBridgeError::Invalid("callable requires canonical shared activation or unavailable capture source evidence".into()));
        }
        validate_capture_contract(captures, &function.callable.captures)?;
    }
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
            &compiled_binding,
            function.function,
            callable_value_digest,
        ),
    )?;
    let capture_digest = match graph_captures.as_ref() {
        Some(values) => contract_digest("captures", values)?,
        None => contract_digest("captures", &captures)?,
    };
    drop(graph_captures);
    let invocation_digest = contract_digest("generator-invocation", invocation)?;
    let generator_identity =
        GeneratorIdentity::new(callable_digest, capture_digest, invocation_digest)?;
    let frozen_input_digest = if let Some(graph) = inputs.graph() {
        contract_digest(
            "frozen-graph-inputs",
            &(graph, &captures, &invocation.arguments),
        )?
    } else {
        contract_digest("frozen-inputs", &(&captures, &invocation.arguments))?
    };

    Ok(AdmittedGeneratorInvocation {
        program_revision: program_revision.clone(),
        callable,
        executable_source_digest,
        compiled_binding,
        function: function.function,
        frozen_inputs: inputs,
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
    if function.captures.len() != callable.captures.len()
        || function
            .captures
            .iter()
            .zip(&callable.captures)
            .any(|(capture, value)| capture.value != *value)
    {
        return Err(AnalysisBridgeError::Invalid(
            "generator capture facts lack one consistent ordered compiler source contract".into(),
        ));
    }
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
    if !function.effects.0.iter().all(|effect| {
        effect.is_side_effect_free()
            || (*effect == runmat_types::EffectKind::MaySuspend && function.cooperative_suspension)
    }) {
        let forbidden = function
            .effects
            .0
            .iter()
            .filter(|effect| {
                !effect.is_side_effect_free()
                    && (**effect != runmat_types::EffectKind::MaySuspend
                        || !function.cooperative_suspension)
            })
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
        CallableFact, CapabilitySet, EffectKind, EffectSet, NumericClass, NumericDomain,
        NumericFact, ProgramFunctionId, ValueFact,
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

    #[test]
    fn logical_array_capture_checks_logical_class_and_actual_dimensions() {
        let capture = ValuePayload::Inline(Box::new(InlineValue::Dense(
            runmat_execution::value::DenseValue {
                element_type: runmat_execution::value::ElementType::Logical,
                shape: vec![2, 1],
                little_endian_data: vec![1, 0],
            },
        )));
        let mut fact = ValueFact::scalar(ValueKindFact::Logical);
        fact.shape = ShapeFact::Shaped {
            dims: vec![DimensionFact::Known(2), DimensionFact::Known(1)],
        };
        assert!(validate_capture_contract(
            std::slice::from_ref(&capture),
            std::slice::from_ref(&fact)
        )
        .is_ok());
        fact.shape = ShapeFact::Scalar;
        assert!(validate_capture_contract(
            std::slice::from_ref(&capture),
            std::slice::from_ref(&fact)
        )
        .is_err());
        assert!(validate_capture_contract(&[capture], &[numeric()]).is_err());
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
            captures: Vec::new(),
            callable: CallableFact {
                capabilities: CapabilitySet::default(),
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
            cooperative_suspension: false,
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let second = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.sample_points.canonical_values(),
            &vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );
        assert_eq!(first.function, ProgramFunctionId(7));
    }

    #[test]
    fn callable_admission_rejects_invalid_typed_physical_contracts() {
        let PhysicalSignal::Force(template) = signal() else {
            unreachable!()
        };
        let mut invalid_evidence = template.clone();
        invalid_evidence.evidence.max_absolute_error = Some(f64::NAN);
        let mut invalid_shape = template.clone();
        invalid_shape.shape = SignalShape::Tensor(vec![0, 3]);
        let mut invalid_frame = template;
        invalid_frame.shape = SignalShape::Vector3;
        for invalid in [invalid_evidence, invalid_shape, invalid_frame] {
            assert!(admit_named_signal_generator(
                &PhysicalSignal::Force(invalid),
                revision(),
                callable(),
                CompiledCallableBinding::fixture(
                    "pulse",
                    "loads.pulse",
                    runmat_types::ProgramFunctionId(7),
                    digest("executable-source")
                ),
                &analysis(),
                &BTreeSet::new(),
            )
            .is_err());
        }
    }

    #[test]
    fn admission_identity_invalidates_only_the_affected_layers() {
        let baseline = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        let changed_registry_binding = admit_named_signal_generator(
            &signal(),
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "registeredPulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &analysis(),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_ne!(
            baseline.generator_identity,
            changed_registry_binding.generator_identity
        );
        assert_eq!(baseline.callable, changed_registry_binding.callable);
        let mut changed_callable = callable();
        changed_callable.source_digest = digest("changed-loads.m");
        let changed_source = admit_named_signal_generator(
            &signal(),
            revision(),
            changed_callable,
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
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
        facts.captures = vec![runmat_mir::analysis::FunctionCaptureAnalysis {
            source: runmat_mir::analysis::FunctionCaptureSource::ByValue,
            value: numeric(),
        }];
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
        let owner = runmat_execution::value::CallableOwner::Bound(
            runmat_execution::FunctionBindingOrigin::from_revision(
                &revision(),
                ProgramFunctionId(7),
            )
            .unwrap(),
        );
        let form = runmat_execution::value::CallableForm::Closure;
        let source_role = runmat_types::FunctionSourceRole::Primary;
        let frozen = |capture: f64| CallableValue {
            owner,
            form,
            source_role,
            source_file: None,
            qualified_name: "loads.pulse".into(),
            callable_digest: CallableValue::identity_digest(
                owner,
                "loads.pulse",
                form,
                source_role,
                None,
            ),
            captures: vec![ValuePayload::Inline(Box::new(InlineValue::F64Bits(
                capture.to_bits(),
            )))],
        };
        let first = admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &facts,
            frozen(2.0),
            &BTreeSet::new(),
        )
        .unwrap();
        let changed_capture = admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source"),
            ),
            &facts,
            frozen(3.0),
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(
            first.frozen_inputs,
            FrozenGeneratorInputs::DirectCaptures(frozen(2.0).captures)
        );
        assert_ne!(first.generator_identity, changed_capture.generator_identity);
        assert_ne!(
            first.materialization_identity,
            changed_capture.materialization_identity
        );

        let mut unresolved = frozen(2.0);
        unresolved.owner = runmat_execution::value::CallableOwner::Workspace;
        unresolved.callable_digest = CallableValue::identity_digest(
            unresolved.owner,
            &unresolved.qualified_name,
            unresolved.form,
            unresolved.source_role,
            unresolved.source_file.as_deref(),
        );
        assert!(admit_callable_signal_generator(
            &bound_signal,
            revision(),
            callable(),
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
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
                CompiledCallableBinding::fixture(
                    "pulse",
                    "loads.pulse",
                    runmat_types::ProgramFunctionId(7),
                    digest("executable-source")
                ),
                &facts,
                &BTreeSet::new(),
            )
            .is_err());
        }
    }

    #[test]
    fn numerical_suspension_requires_proof_and_never_covers_ambient_effects() {
        let PhysicalSignal::Force(signal) = signal() else {
            unreachable!()
        };
        let SignalRepresentation::Generator(generator) = signal.representation else {
            unreachable!()
        };
        let mut facts = analysis();
        facts.effects.0.insert(EffectKind::MaySuspend);
        assert!(validate_static_contract(
            &generator.invocation,
            &facts,
            0,
            GeneratorOutputDomain::Real,
            &BTreeSet::new()
        )
        .is_err());
        facts.cooperative_suspension = true;
        assert!(validate_static_contract(
            &generator.invocation,
            &facts,
            0,
            GeneratorOutputDomain::Real,
            &BTreeSet::new()
        )
        .is_ok());
        for forbidden in [
            EffectKind::Clock,
            EffectKind::Network,
            EffectKind::HostCallback,
            EffectKind::Unknown,
        ] {
            let mut ambient = facts.clone();
            ambient.effects.0.insert(forbidden);
            assert!(validate_static_contract(
                &generator.invocation,
                &ambient,
                0,
                GeneratorOutputDomain::Real,
                &BTreeSet::new()
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
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
            CompiledCallableBinding::fixture(
                "pulse",
                "loads.pulse",
                runmat_types::ProgramFunctionId(7),
                digest("executable-source")
            ),
            &random,
            &BTreeSet::new(),
        )
        .is_err());
    }
}
