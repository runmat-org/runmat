use super::*;
use runmat_analysis_core::*;
use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableIdentity, ExecutableUnitManifest, ProgramEnvironment,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{
    CallableFact, CapabilitySet, EffectKind, EffectSet, NumericClass, NumericDomain, NumericFact,
    ProgramFunctionId, ValueFact, ValueKindFact,
};

pub(crate) fn signal() -> PhysicalSignal {
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("pulse".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Generator(SignalGenerator {
            invocation: GeneratorInvocation {
                callable: GeneratorCallable::ProjectFunction {
                    qualified_name: "loads.pulse".into(),
                },
                arguments: Vec::new(),
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

pub(crate) fn callable() -> BundleCallable {
    BundleCallable {
        owner_identity: "project:loads".into(),
        qualified_name: "loads.pulse".into(),
        source_digest: Digest::sha256("loads.m"),
    }
}

pub(crate) fn analysis() -> AnalysisStore {
    let output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut parameter = output.clone();
    parameter.shape = runmat_types::ShapeFact::Shaped {
        dims: vec![
            runmat_types::DimensionFact::Unknown,
            runmat_types::DimensionFact::Known(1),
        ],
    };
    AnalysisStore {
        functions: vec![FunctionAnalysis {
            function: ProgramFunctionId(7),
            captures: Vec::new(),
            callable: CallableFact {
                capabilities: CapabilitySet::default(),
                identity: None,
                parameters: vec![parameter],
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
        }],
        ..AnalysisStore::default()
    }
}

pub(crate) fn revision() -> ProgramRevision {
    let current = analysis().revision;
    ProgramRevision::new(
        Digest::sha256("graph"),
        Digest::sha256("source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::from_bytes(current.catalog_fingerprint),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

pub(crate) fn envelope(analysis: &AnalysisStore) -> ExecutableUnitEnvelope {
    let current = runmat_mir::analysis::AnalysisRevision::current();
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            let bytes = if kind == ExecutableComponentKind::Analysis {
                serde_json::to_vec(&serde_json::to_value(analysis).unwrap()).unwrap()
            } else {
                b"{}".to_vec()
            };
            ExecutableComponentPayload::new(kind, bytes).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            let version = if payload.kind == ExecutableComponentKind::Analysis {
                current.schema_version
            } else {
                1
            };
            ExecutableComponentDescriptor::from_payload(payload.kind, version, &payload.bytes)
                .unwrap()
        })
        .collect();
    ExecutableUnitEnvelope::new(
        ExecutableUnitManifest {
            schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: ExecutableIdentity {
                declaration_class: None,
                program: revision(),
                root_package: callable().owner_identity,
                entrypoint: "pulse".into(),
                entrypoint_function: Some(ProgramFunctionId(7)),
                entrypoint_kind: ExecutableEntrypointKind::Function,
                source_digest: Digest::sha256("executable-source"),
            },
            revisions: ExecutableComponentRevisions {
                catalog_schema: u16::try_from(current.catalog_schema).unwrap(),
                catalog_fingerprint: Digest::from_bytes(current.catalog_fingerprint),
                contract_schema: current.fact_schema_major,
                contract_fingerprint: Digest::sha256("contracts"),
                analysis_schema: current.schema_version,
                mir_schema: 1,
                bytecode_schema: 1,
                vm_layout_schema: 1,
                function_registry_schema: 1,
                source_map_schema: 1,
                region_schema: runmat_types::REGION_CONTRACT_SCHEMA_VERSION,
                interop_schema: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
                parallel_schema: runmat_types::PARALLEL_MANIFEST_SCHEMA_VERSION,
            },
            components,
            dependencies: Vec::new(),
            capabilities: CapabilitySet::default(),
            regions: Vec::new(),
            interop: runmat_types::InteropManifest::empty(),
            parallel: runmat_types::ParallelManifest::empty(),
            optional_sections: Vec::new(),
        },
        payloads,
    )
    .unwrap()
}

fn admit(envelope: &ExecutableUnitEnvelope) -> AnalysisBridgeResult<AdmittedSignalGenerator> {
    admit_named_signal_generator_from_executable(
        &signal(),
        revision(),
        callable(),
        envelope,
        &BTreeSet::new(),
    )
}

#[test]
fn recovers_exact_named_admission_from_authenticated_analysis_component() {
    let analysis = analysis();
    let executable = envelope(&analysis);
    let admitted = admit(&executable).unwrap();
    let expected = super::super::admit_named_signal_generator(
        &signal(),
        revision(),
        callable(),
        crate::named_callable::CompiledCallableBinding::from_selected_identity(
            &executable.manifest.identity,
            "loads.pulse",
        )
        .unwrap(),
        &analysis.functions[0],
        &BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(admitted, expected);
    let reopened =
        ExecutableUnitEnvelope::from_canonical_bytes(&executable.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(admit(&reopened).unwrap(), admitted);
}

#[test]
fn descriptor_compares_original_registry_binding_not_qualified_label_or_candidate() {
    let executable = envelope(&analysis());
    let admitted = admit(&executable).unwrap();
    assert_eq!(admitted.callable.qualified_name, "loads.pulse");
    assert_eq!(executable.manifest.identity.entrypoint, "pulse");
    crate::prepare_generator_descriptor(&admitted, executable.clone()).unwrap();
    // Keep the exact function/program/root/source metadata: only the registry
    // name changes. Rejection must come from the real name check.
    let mut other_name = executable.clone();
    other_name.manifest.identity.entrypoint = "other".into();
    assert!(matches!(
        crate::prepare_generator_descriptor(&admitted, other_name),
        Err(AnalysisBridgeError::Identity(
            "portable executable registry entrypoint differs from the admitted frozen callable"
        ))
    ));
    let mut other_function = executable;
    other_function.manifest.identity.entrypoint_function = Some(ProgramFunctionId(8));
    assert!(crate::prepare_generator_descriptor(&admitted, other_function).is_err());
}

#[test]
fn rejects_substituted_analysis_revision_inventory_and_static_behavior() {
    let mut changed = analysis();
    changed.revision.schema_version += 1;
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.revision.catalog_fingerprint[0] ^= 1;
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.functions.push(changed.functions[0].clone());
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    let mut prior = changed.functions[0].clone();
    prior.function = ProgramFunctionId(6);
    changed.functions.push(prior);
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.functions.clear();
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.functions[0]
        .effects
        .0
        .insert(EffectKind::WorkspaceWrite);
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.functions[0].callable.parameters_complete = false;
    assert!(admit(&envelope(&changed)).is_err());
    let mut changed = analysis();
    changed.functions[0]
        .capabilities
        .0
        .insert(CapabilityRequirement::Filesystem);
    assert!(admit(&envelope(&changed)).is_err());
}

#[test]
fn rejects_foreign_envelope_identity_and_unauthenticated_payload() {
    let base = envelope(&analysis());
    let mut changed = base.clone();
    changed.manifest.identity.root_package = "foreign".into();
    assert!(admit(&changed).is_err());
    let mut changed = base.clone();
    changed.manifest.identity.entrypoint_kind = ExecutableEntrypointKind::Script;
    assert!(admit(&changed).is_err());
    let mut changed = base.clone();
    changed.manifest.identity.entrypoint_function = Some(ProgramFunctionId(8));
    assert!(admit(&changed).is_err());
    let mut changed = base.clone();
    changed.manifest.identity.program = ProgramRevision::new(
        Digest::sha256("foreign-graph"),
        Digest::sha256("source"),
        revision().environment().clone(),
    )
    .unwrap();
    assert!(admit(&changed).is_err());
    let mut changed = base;
    changed
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == ExecutableComponentKind::Analysis)
        .unwrap()
        .bytes[0] ^= 1;
    assert!(admit(&changed).is_err());
}
