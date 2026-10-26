use super::*;

#[test]
fn constructorless_class_has_admitted_declaration_product_without_callable_entrypoint() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let unit = block_on(session.compile_executable_unit(
        ExecutableSource::new(
            "declaration-only@1",
            "NoBodyBase.m",
            "classdef NoBodyBase < handle\n properties\n seed = 13\n end\nend\n",
        ),
        None,
    ))
    .expect("compile class source");
    let class = runmat_types::ClassIdentity::new("NoBodyBase").expect("class identity");
    let primary = unit
        .source_map()
        .entries()
        .first()
        .expect("primary source")
        .source_id;
    let envelope = unit
        .portable_envelope_for_declaration_source(&class, primary)
        .expect("declaration-only product");
    assert_eq!(
        envelope.manifest.identity.entrypoint_kind,
        runmat_execution::ExecutableEntrypointKind::DeclarationOnly
    );
    assert_eq!(envelope.manifest.identity.entrypoint_function, None);
    let admitted = runmat_vm::admit_executable_unit(&envelope.canonical_bytes().unwrap())
        .expect("admit class metadata without callable identity");
    assert_eq!(admitted.declaration_class(), Some(&class));
    assert!(admitted.entrypoint_function().is_none());
    assert!(admitted.entrypoint_origin().is_err());

    let mut forged = envelope.clone();
    let foreign = runmat_types::ClassIdentity::new("ForeignBase").unwrap();
    forged.manifest.identity.entrypoint = foreign.display_name().into();
    forged.manifest.identity.declaration_class = Some(foreign);
    assert!(runmat_vm::admit_executable_unit(&forged.canonical_bytes().unwrap()).is_err());

    // Hash refresh cannot supply missing or contradictory source provenance.
    for analysis in [false, true] {
        let mut altered = envelope.clone();
        let kind = if analysis {
            runmat_execution::ExecutableComponentKind::Analysis
        } else {
            runmat_execution::ExecutableComponentKind::Mir
        };
        let mut payload: serde_json::Value =
            serde_json::from_slice(&altered.component(kind).unwrap().bytes).unwrap();
        if analysis {
            payload["classes"][0]["declaration"]["source"] = u32::MAX.into();
        } else {
            payload["classes"][0]["source"] = u32::MAX.into();
        }
        replace_component(&mut altered, kind, &payload);
        assert!(runmat_vm::admit_executable_unit(&altered.canonical_bytes().unwrap()).is_err());
    }
    let mut missing = envelope.clone();
    let kind = runmat_execution::ExecutableComponentKind::Mir;
    let mut payload: serde_json::Value =
        serde_json::from_slice(&missing.component(kind).unwrap().bytes).unwrap();
    payload["classes"][0]
        .as_object_mut()
        .unwrap()
        .remove("source");
    replace_component(&mut missing, kind, &payload);
    assert!(runmat_vm::admit_executable_unit(&missing.canonical_bytes().unwrap()).is_err());
    let mut source_mismatch = envelope;
    let component = source_mismatch
        .component(runmat_execution::ExecutableComponentKind::Bytecode)
        .expect("bytecode component");
    let mut bytecode: runmat_vm::Bytecode = serde_json::from_slice(&component.bytes).unwrap();
    bytecode.source_id = None;
    replace_component(
        &mut source_mismatch,
        runmat_execution::ExecutableComponentKind::Bytecode,
        &bytecode,
    );
    assert!(runmat_vm::admit_executable_unit(&source_mismatch.canonical_bytes().unwrap()).is_err());
}

#[test]
fn portable_function_request_invokes_the_admitted_local_registry_body() {
    use runmat_execution::{OutputContract, ProgramCallable, ProgramInvocationContext};
    use runmat_execution_artifact::{
        ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
        ProgramExecutionResponse, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
    };

    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let unit = block_on(session.compile_executable_unit(
        ExecutableSource::new(
            "portable-function-dispatch@1",
            "portableFunctionDispatch.m",
            "function result = portableFunctionDispatch()\nresult = 42;\nend\n",
        ),
        None,
    ))
    .expect("compile function unit");
    let envelope = unit.portable_envelope().expect("portable function product");
    assert_eq!(
        envelope.manifest.identity.entrypoint_kind,
        runmat_execution::ExecutableEntrypointKind::Function
    );
    assert_eq!(
        envelope.manifest.identity.entrypoint, "portableFunctionDispatch",
        "an exact registered source name remains the portable entrypoint"
    );
    let registry_payload = envelope
        .component(runmat_execution::ExecutableComponentKind::FunctionRegistry)
        .expect("function registry component");
    let registry: runmat_vm::FunctionRegistry =
        serde_json::from_slice(&registry_payload.bytes).expect("decode function registry");
    let local_function = registry
        .resolve_name("portableFunctionDispatch")
        .expect("selected local body");
    let portable_function = envelope.manifest.identity.entrypoint_function.unwrap();
    assert_eq!(
        registry.get(local_function).unwrap().portable_function,
        portable_function
    );
    assert_ne!(
        local_function.0, portable_function.0 as usize,
        "the round trip must exercise distinct local and portable IDs"
    );

    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("portable-function-dispatch-test"),
        interop: envelope.manifest.interop.clone(),
        foreign_policy: None,
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV4,
        envelope.canonical_bytes().expect("canonical unit"),
    )
    .expect("program artifact");
    let request = ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(portable_function, None),
        context: ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    let mut wrong_callable = request.clone();
    wrong_callable.callable = ProgramCallable::semantic(
        runmat_types::ProgramFunctionId(portable_function.0 + 1),
        None,
    );
    assert!(matches!(
        block_on(runmat_vm::execute_program_request(wrong_callable)),
        ProgramExecutionResponse::Failure { .. }
    ));

    let response = block_on(runmat_vm::execute_program_request(request));
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("exact function unit must execute: {response:?}");
    };
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        runmat_runtime::execution::value_codec::decode_inline_value(&outputs[0])
            .expect("inline function result"),
        runmat_value::Value::Num(42.0)
    );
}

#[test]
fn class_constructor_manifest_uses_its_exact_admitted_registry_name() {
    for relative_path in ["SeedDerived.m", "+pkg/SeedDerived.m"] {
        let mut session = RunMatSession::with_options(false, false).expect("session init");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::new(
                format!("constructor-entrypoint-{relative_path}"),
                relative_path,
                "classdef SeedDerived < handle\n methods\n function obj = SeedDerived()\n end\n end\nend",
            ),
            None,
        ))
        .expect("compile class source");
        // A package request enters the dynamic loader under pkg.SeedDerived,
        // whose source-local class declaration is still SeedDerived.
        let envelope = unit
            .portable_envelope_for(Some("SeedDerived"))
            .expect("class constructor product");
        assert_eq!(
            envelope.manifest.identity.entrypoint,
            "SeedDerived.SeedDerived"
        );
        let registry_payload = envelope
            .component(runmat_execution::ExecutableComponentKind::FunctionRegistry)
            .expect("function registry component");
        let registry: runmat_vm::FunctionRegistry =
            serde_json::from_slice(&registry_payload.bytes).expect("decode function registry");
        let selected = registry
            .resolve_name(&envelope.manifest.identity.entrypoint)
            .expect("constructor has an exact registered name");
        let selected_body = registry.get(selected).expect("registered constructor body");
        assert_eq!(
            selected_body.portable_function,
            envelope.manifest.identity.entrypoint_function.unwrap()
        );
        assert_eq!(
            selected_body.binding_origin,
            Some(
                runmat_execution::FunctionBindingOrigin::from_revision(
                    &envelope.manifest.identity.program,
                    selected_body.portable_function,
                )
                .expect("constructor origin")
            )
        );
        let bytes = envelope.canonical_bytes().expect("canonical class product");
        runmat_vm::admit_executable_unit(&bytes).expect("admit class constructor product");
        crate::ExecutableUnit::from_portable_envelope(&envelope)
            .expect("registered constructor alias survives an exact-origin round trip");

        let mut wrong_name = envelope.clone();
        wrong_name.manifest.identity.entrypoint = "SeedDerived".into();
        let error = runmat_vm::admit_executable_unit(
            &wrong_name
                .canonical_bytes()
                .expect("canonical name mismatch"),
        )
        .expect_err("a public class alias is not a registered constructor name");
        assert!(
            error.to_string().contains(
                "executable entrypoint identity disagrees with its admitted function body"
            ),
            "{error}"
        );

        let mut wrong_function = envelope;
        wrong_function.manifest.identity.entrypoint_function = Some(
            runmat_types::ProgramFunctionId(selected_body.portable_function.0 + 1),
        );
        let error = runmat_vm::admit_executable_unit(
            &wrong_function
                .canonical_bytes()
                .expect("canonical function mismatch"),
        )
        .expect_err("the admitted registry must match the portable function identity");
        assert!(
            error.to_string().contains(
                "executable entrypoint identity disagrees with its admitted function body"
            ),
            "{error}"
        );
    }
}

fn replace_component(
    envelope: &mut runmat_execution::ExecutableUnitEnvelope,
    kind: runmat_execution::ExecutableComponentKind,
    value: &impl serde::Serialize,
) {
    let bytes = serde_json::to_vec(value).expect("encode mismatched component");
    envelope
        .payloads
        .iter_mut()
        .find(|payload| payload.kind == kind)
        .expect("component payload")
        .bytes = bytes.clone();
    let descriptor = envelope
        .manifest
        .components
        .iter_mut()
        .find(|descriptor| descriptor.kind == kind)
        .expect("component descriptor");
    *descriptor = runmat_execution::ExecutableComponentDescriptor::from_payload(
        kind,
        descriptor.schema_version,
        &bytes,
    )
    .expect("updated component descriptor");
}

#[test]
fn portable_script_entrypoint_rejects_registry_and_layout_identity_mismatches() {
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let unit = block_on(session.compile_executable_unit(
        ExecutableSource::new(
            "core-script-entrypoint-mismatch-test@1",
            "script_entrypoint_mismatch.m",
            "answer = localPlusOne(41);\nfunction output = localPlusOne(input)\noutput = input + 1;\nend\n",
        ),
        None,
    ))
    .expect("compile script with a local function");
    let envelope = unit.portable_envelope().expect("portable script product");
    let registry_payload = envelope
        .component(runmat_execution::ExecutableComponentKind::FunctionRegistry)
        .expect("function registry payload");
    let registry: runmat_vm::FunctionRegistry =
        serde_json::from_slice(&registry_payload.bytes).expect("decode function registry");
    let local_function = registry
        .resolve_name("localPlusOne")
        .and_then(|function| registry.get(function))
        .expect("local function body");

    let mut registry_identity = envelope.clone();
    registry_identity.manifest.identity.entrypoint_function =
        Some(local_function.portable_function);
    let error = runmat_vm::admit_executable_unit(
        &registry_identity
            .canonical_bytes()
            .expect("canonical registry-identity mismatch"),
    )
    .expect_err("a script entrypoint cannot name a registry function body");
    assert!(
        error
            .to_string()
            .contains("script entrypoint does not identify synthetic MIR"),
        "{error}"
    );

    let local_mir_function = unit
        .functions()
        .resolve_name("localPlusOne")
        .expect("local MIR function identity");
    assert_ne!(
        unit.mir().functions[&local_mir_function].kind,
        runmat_hir::FunctionKind::SyntheticEntrypoint
    );
    let mut nonsynthetic_identity = envelope.clone();
    nonsynthetic_identity.manifest.identity.entrypoint_function = Some(
        unit.functions()
            .get(local_mir_function)
            .unwrap()
            .portable_function,
    );
    let error = runmat_vm::admit_executable_unit(
        &nonsynthetic_identity
            .canonical_bytes()
            .expect("canonical nonsynthetic MIR mismatch"),
    )
    .expect_err("a script entrypoint must identify synthetic MIR");
    assert!(
        error
            .to_string()
            .contains("script entrypoint does not identify synthetic MIR"),
        "{error}"
    );

    let mut kind_mismatch = envelope.clone();
    kind_mismatch.manifest.identity.entrypoint_kind =
        runmat_execution::ExecutableEntrypointKind::Function;
    let error = runmat_vm::admit_executable_unit(
        &kind_mismatch
            .canonical_bytes()
            .expect("canonical entrypoint-kind mismatch"),
    )
    .expect_err("a synthetic script entrypoint is not a registry function");
    assert!(
        error
            .to_string()
            .contains("executable entrypoint identity disagrees with its admitted function body: original origin is absent"),
        "{error}"
    );

    let mut layout_mismatch = envelope.clone();
    let layout_payload = layout_mismatch
        .component(runmat_execution::ExecutableComponentKind::VmLayout)
        .expect("VM layout payload");
    let mut layout: runmat_vm::VmAssemblyLayout =
        serde_json::from_slice(&layout_payload.bytes).expect("decode VM layout");
    let script_function = runmat_hir::FunctionId(
        usize::try_from(
            layout_mismatch
                .manifest
                .identity
                .entrypoint_function
                .unwrap()
                .0,
        )
        .expect("script function identity fits this host"),
    );
    let local_function = registry
        .resolve_name("localPlusOne")
        .expect("local function identity");
    let script_layout = layout
        .entrypoints
        .values_mut()
        .find(|entrypoint| entrypoint.target == script_function)
        .expect("script VM entrypoint");
    script_layout.target = local_function;
    replace_component(
        &mut layout_mismatch,
        runmat_execution::ExecutableComponentKind::VmLayout,
        &layout,
    );

    let error = runmat_vm::admit_executable_unit(
        &layout_mismatch
            .canonical_bytes()
            .expect("canonical VM-layout mismatch"),
    )
    .expect_err("script MIR and VM entrypoints must agree");
    assert!(
        error
            .to_string()
            .contains("script entrypoint identity disagrees with its admitted VM layout"),
        "{error}"
    );

    let bytecode_payload = envelope
        .component(runmat_execution::ExecutableComponentKind::Bytecode)
        .expect("bytecode payload");
    let bytecode: runmat_vm::Bytecode =
        serde_json::from_slice(&bytecode_payload.bytes).expect("decode bytecode");
    for source in [
        None,
        Some(runmat_vm::BytecodeSourceRef::Program(
            runmat_types::ProgramSourceId(1),
        )),
    ] {
        let mut source_mismatch = envelope.clone();
        let mut mismatched_bytecode = bytecode.clone();
        mismatched_bytecode.source_id = source;
        replace_component(
            &mut source_mismatch,
            runmat_execution::ExecutableComponentKind::Bytecode,
            &mismatched_bytecode,
        );
        let error = runmat_vm::admit_executable_unit(
            &source_mismatch
                .canonical_bytes()
                .expect("canonical source-identity mismatch"),
        )
        .expect_err("script bytecode must retain its exact primary source");
        assert!(
            error
                .to_string()
                .contains("script bytecode does not identify its primary MIR source"),
            "{error}"
        );
    }
}
