use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::{
    Digest, ExecutableComponentDescriptor, ExecutableComponentKind, ExecutableComponentPayload,
    ExecutableComponentRevisions, ExecutableEntrypointKind, ExecutableIdentity,
    ExecutableUnitEnvelope, ExecutableUnitManifest, ProgramEnvironment, ProgramRevision,
    EXECUTABLE_UNIT_SCHEMA_VERSION,
};
use runmat_execution_artifact::{ArtifactAccess, BundleCallable, ProgramExecutionResponse};
use runmat_geometry_core::{
    GeneratedGeometryArgument, GeneratedGeometryProduct, GeometryAuthoringValue, GeometryDigest,
    GeometryFeature, GeometryFeatureGraph, GeometryFeatureOperation, GeometryFeatureRef,
    GeometryScalarExpression,
};
use runmat_geometry_ops::ResolvedGeneratedGeometryRequest;
use runmat_mir::analysis::{FunctionAnalysis, FunctionConvergence};
use runmat_types::{
    CallableFact, CapabilitySet, EffectKind, EffectSet, InteropManifest, NumericClass,
    NumericDomain, NumericFact, ObjectFact, ParallelManifest, ProgramFunctionId, ShapeFact,
    StorageFact, ValueFact, ValueKindFact, INTEROP_MANIFEST_SCHEMA_VERSION,
    PARALLEL_MANIFEST_SCHEMA_VERSION, REGION_CONTRACT_SCHEMA_VERSION,
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

fn scalar(value: f64) -> GeometryScalarExpression {
    GeometryScalarExpression::Literal { value }
}

fn request() -> ResolvedGeneratedGeometryRequest {
    ResolvedGeneratedGeometryRequest {
        module_path: "project/makeBracket.m".into(),
        function: "makeBracket".into(),
        source_digest: GeometryDigest::from_bytes(*digest("makeBracket.m").bytes()),
        arguments: vec![
            GeneratedGeometryArgument {
                parameter: "width".into(),
                value: GeometryAuthoringValue::Scalar(scalar(0.08)),
            },
            GeneratedGeometryArgument {
                parameter: "depth".into(),
                value: GeometryAuthoringValue::Scalar(scalar(0.02)),
            },
        ],
    }
}

fn callable() -> BundleCallable {
    BundleCallable {
        owner_identity: "project".into(),
        qualified_name: "makeBracket".into(),
        source_digest: digest("makeBracket.m"),
    }
}

fn numeric_scalar() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn geometry_output() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::ClassIdentity::new("geometry.Generated").unwrap()),
            properties: BTreeMap::new(),
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn analysis() -> FunctionAnalysis {
    let output = geometry_output();
    FunctionAnalysis {
        function: ProgramFunctionId(7),
        captures: Vec::new(),
        callable: CallableFact {
            capabilities: CapabilitySet::default(),
            identity: None,
            parameters: vec![numeric_scalar(), numeric_scalar()],
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

fn admitted() -> AdmittedGeometryGenerator {
    admit_named_geometry_generator(
        request(),
        revision(),
        callable(),
        crate::named_callable::CompiledCallableBinding::fixture(
            "makeBracket",
            "makeBracket",
            ProgramFunctionId(7),
            digest("executable"),
        ),
        &analysis(),
        &BTreeSet::new(),
    )
    .unwrap()
}

fn executable(admitted: &AdmittedGeometryGenerator) -> ExecutableUnitEnvelope {
    let payloads = ExecutableComponentKind::REQUIRED
        .into_iter()
        .map(|kind| {
            ExecutableComponentPayload::new(kind, format!("{kind:?}").into_bytes()).unwrap()
        })
        .collect::<Vec<_>>();
    let components = payloads
        .iter()
        .map(|payload| {
            ExecutableComponentDescriptor::from_payload(payload.kind, 1, &payload.bytes).unwrap()
        })
        .collect();
    ExecutableUnitEnvelope::new(
        ExecutableUnitManifest {
            schema_version: EXECUTABLE_UNIT_SCHEMA_VERSION,
            identity: ExecutableIdentity {
                declaration_class: None,
                program: admitted.invocation.program_revision.clone(),
                root_package: admitted.invocation.callable.owner_identity.clone(),
                entrypoint: admitted.invocation.callable.qualified_name.clone(),
                entrypoint_function: Some(admitted.invocation.function),
                entrypoint_kind: ExecutableEntrypointKind::Function,
                source_digest: admitted.invocation.executable_source_digest,
            },
            revisions: ExecutableComponentRevisions {
                catalog_schema: 1,
                catalog_fingerprint: *admitted.invocation.program_revision.catalog_fingerprint(),
                contract_schema: 1,
                contract_fingerprint: digest("contracts"),
                analysis_schema: 1,
                mir_schema: 1,
                bytecode_schema: 1,
                vm_layout_schema: 1,
                function_registry_schema: 1,
                source_map_schema: 1,
                region_schema: REGION_CONTRACT_SCHEMA_VERSION,
                interop_schema: INTEROP_MANIFEST_SCHEMA_VERSION,
                parallel_schema: PARALLEL_MANIFEST_SCHEMA_VERSION,
            },
            components,
            dependencies: Vec::new(),
            capabilities: CapabilitySet::default(),
            regions: Vec::new(),
            interop: InteropManifest::empty(),
            parallel: ParallelManifest::empty(),
            optional_sections: Vec::new(),
        },
        payloads,
    )
    .unwrap()
}

fn product() -> GeneratedGeometryProduct {
    GeneratedGeometryProduct {
        graph: GeometryFeatureGraph {
            features: vec![GeometryFeature {
                feature_id: "stock".into(),
                operation: GeometryFeatureOperation::Box {
                    width_m: scalar(0.08),
                    depth_m: scalar(0.02),
                    height_m: scalar(0.01),
                },
            }],
            result: GeometryFeatureRef("stock".into()),
        },
        region_outputs: BTreeMap::new(),
    }
}

#[test]
fn admission_binds_ordered_arguments_and_rederives_every_identity() {
    let admitted = admitted();
    admitted.validate().unwrap();
    assert_eq!(admitted.request.arguments[0].parameter, "width");
    assert_eq!(admitted.request.arguments[1].parameter, "depth");

    let mut reordered = admitted.clone();
    reordered.request.arguments.swap(0, 1);
    assert!(reordered.validate().is_err());

    let mut substituted = admitted;
    substituted.arguments.swap(0, 1);
    assert!(substituted.validate().is_err());
}

#[test]
fn admission_rejects_mutable_or_effectful_generator_contracts() {
    let mut mutable = analysis();
    let ValueKindFact::Object(object) = &mut mutable.outputs[0].kind else {
        unreachable!();
    };
    object.handle_semantics = Some(true);
    assert!(admit_named_geometry_generator(
        request(),
        revision(),
        callable(),
        crate::named_callable::CompiledCallableBinding::fixture(
            "makeBracket",
            "makeBracket",
            ProgramFunctionId(7),
            digest("executable")
        ),
        &mutable,
        &BTreeSet::new(),
    )
    .is_err());

    let mut effectful = analysis();
    effectful.effects.0.insert(EffectKind::FilesystemRead);
    assert!(admit_named_geometry_generator(
        request(),
        revision(),
        callable(),
        crate::named_callable::CompiledCallableBinding::fixture(
            "makeBracket",
            "makeBracket",
            ProgramFunctionId(7),
            digest("executable")
        ),
        &effectful,
        &BTreeSet::new(),
    )
    .is_err());
}

#[test]
fn portable_program_is_bound_to_the_exact_admitted_executable() {
    let admitted = admitted();
    let prepared = prepare_geometry_generator(admitted.clone(), executable(&admitted)).unwrap();
    assert_eq!(prepared.program.chunks.len(), 1);
    assert_eq!(
        prepared.program.chunks[0].plan.arguments,
        admitted.arguments
    );

    let mut substituted = executable(&admitted);
    substituted.manifest.identity.source_digest = digest("different executable");
    assert!(prepare_geometry_generator(admitted, substituted).is_err());
}

#[test]
fn typed_product_and_receipt_are_canonical_and_substitution_resistant() {
    let product = product();
    let value = encode_generated_geometry_value(&product).unwrap();
    assert_eq!(decode_generated_geometry_value(&value).unwrap(), product);

    let admitted = admitted();
    let access = ArtifactAccess {
        authorization_scope: "geometry:test".into(),
        encryption_context: digest("encryption"),
    };
    let prepared = prepare_generated_geometry_product(
        &admitted,
        ProgramExecutionResponse::Success {
            outputs: vec![value],
        },
        &access,
    )
    .unwrap();
    prepared.receipt.validate().unwrap();
    assert_eq!(prepared.product, product);
    assert_eq!(
        prepared.root().logical_digest,
        prepared.object.descriptor.digest
    );

    let other_access = ArtifactAccess {
        authorization_scope: "geometry:other-scope".into(),
        encryption_context: digest("other encryption"),
    };
    let other = prepare_generated_geometry_product(
        &admitted,
        ProgramExecutionResponse::Success {
            outputs: vec![encode_generated_geometry_value(&product).unwrap()],
        },
        &other_access,
    )
    .unwrap();
    assert_eq!(prepared.source, other.source);
    assert_ne!(
        prepared.root().authorization_scope,
        other.root().authorization_scope
    );

    let mut tampered = prepared.receipt;
    tampered.admission.request.function = "differentFunction".into();
    assert!(tampered.validate().is_err());
}
