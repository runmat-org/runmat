//! Real pure assessment functions crossing a bounded native process boundary.
use super::*;
use runmat_analysis_core::{
    AssessmentEvidenceRequirement, AssessmentFunctionEvaluation, AssessmentFunctionOutcome,
    AssessmentFunctionReference, AssessmentQuantity, AssessmentRule, AuthoredAssessment,
};
use runmat_analysis_execution::assessment_program::{
    build_assessment_task_submission, FrozenAssessmentCallable,
};
use runmat_analysis_execution::named_callable::NamedCallableExecutionContext;
use runmat_execution_artifact::BundleCallable;

pub(super) async fn run() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("runmat.toml"),
        "[package]\nname = \"assessment-process-conformance\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let path = directory.path().join("src/verify.m");
    let source = "function result = verify(value, limit)\narguments\n value (1,1) double\n limit (1,1) double\nend\nresult = struct('evaluated', true, 'accepted', value <= limit);\nend\n";
    std::fs::write(&path, source).unwrap();
    let project = runmat_package::build_frozen_project(
        &directory.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let symbols = runmat_package::source_symbols_from_frozen(&project, &path);
    let definition = symbols
        .definitions
        .iter()
        .find(|definition| definition.name == "verify")
        .unwrap();
    let source_id = definition.source_id.as_ref().unwrap();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest().bytes()),
        Digest::from_bytes(*project.source_revision().bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let callable = BundleCallable {
        owner_identity: source_id.package_instance.to_string(),
        qualified_name: definition.qualified_name.clone(),
        source_digest: Digest::sha256(source),
    };
    let mut compiler = runmat_core::RunMatSession::with_options(false, false).unwrap();
    compiler
        .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
        .await
        .unwrap();
    let unit = compiler
        .compile_executable_unit(
            runmat_core::ExecutableSource::frozen(
                callable.owner_identity.clone(),
                source_id.relative_path.as_str(),
                path.to_string_lossy(),
                source,
            ),
            Some(revision.clone()),
        )
        .await
        .unwrap();
    let executable = unit.portable_envelope_for(Some("verify")).unwrap();
    let criterion = AuthoredAssessment {
        assessment_id: "deflection".into(),
        output_id: "peak".into(),
        rule: AssessmentRule::Function {
            function: AssessmentFunctionReference {
                module_path: "project/src/verify.m".into(),
                symbol: callable.qualified_name.clone(),
                source_digest: *callable.source_digest.bytes(),
            },
            parameters: vec![AssessmentQuantity::try_from_descriptor(
                (runmat_types::PhysicalDimension::LENGTH).into(),
                0.001,
            )
            .expect("checked finite assessment quantity")],
        },
        evidence: AssessmentEvidenceRequirement::AuthoredSamples,
    };
    let frozen =
        FrozenAssessmentCallable::new(criterion.clone(), revision, callable, executable).unwrap();
    // A detached completion restores its submitted executable, rather than reopening live source.
    let restored: FrozenAssessmentCallable =
        serde_json::from_slice(&serde_json::to_vec(&frozen).unwrap()).unwrap();
    restored.validate().unwrap();
    std::fs::write(&path, "function result = verify(value, limit)\nresult = struct('evaluated', true, 'accepted', false);\nend\n").unwrap();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.store_root = directory.path().join("processes");
    config.max_workers = 1;
    let session = NativeProgramSession::new(config).unwrap();
    let resources = ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 0,
        max_wall_millis: 30_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
    };
    for (measured, accepted) in [(0.0005, true), (0.002, false)] {
        let study = Digest::sha256("checked-study");
        let evidence = Digest::sha256("immutable-observation");
        let prepared = restored
            .prepare(&criterion, measured, study, evidence)
            .unwrap();
        let submission = build_assessment_task_submission(
            &prepared,
            BTreeSet::new(),
            NamedCallableExecutionContext {
                host: super::phase_execution_support::host_requirement(prepared.request()),
                scope_id: session.scope_id(),
                pool_id: session.pool_id(),
                resources: resources.clone(),
                retry: RetryPolicy::IdempotentInfrastructure,
                deadline_unix_millis: None,
                priority: 0,
            },
        )
        .unwrap();
        let result = wait_for_native_task(
            &session
                .submit(prepared.request().clone(), submission)
                .unwrap(),
        )
        .await;
        let receipt = prepared.admit_response(&success_response(&result)).unwrap();
        assert_eq!(
            receipt
                .validate_binding(&criterion, measured, study, evidence)
                .unwrap(),
            AssessmentFunctionEvaluation::Completed(AssessmentFunctionOutcome {
                evaluated: true,
                accepted
            })
        );
        assert!(receipt
            .validate_binding(&criterion, measured * 2.0, study, evidence)
            .is_err());
        assert!(receipt
            .validate_binding(
                &criterion,
                measured,
                study,
                Digest::sha256("other-evidence")
            )
            .is_err());
    }
    println!("assessment-process-conformance: passed");
}
