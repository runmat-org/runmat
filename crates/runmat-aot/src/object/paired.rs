//! Paired selected native objects from one frozen Core compilation input.

use std::collections::{BTreeMap, BTreeSet};

use runmat_execution_artifact::target_test::{
    NativeCompiledEntry, NativePairedArtifactManifest, NativePairedBuildBinding,
    NativePairedProductHeader, NativePairedProductRole, NativeTestEntry,
    NATIVE_PAIRED_TEST_SCHEMA_VERSION,
};
use runmat_execution_artifact::{ProgramArtifact, ProgramBuildRecipe};
use runmat_native_codegen::aot::{RelocatableNativeObject, SelectedNativeProduct};
use runmat_test::identity::{TestId, TestIdentityInput};
use runmat_test::plan::{validate_plan, TestPlan};

use super::{object_data_options, NativeObjectOptions};
use crate::{AotError, AotResult};

pub struct PairedNativeObjects {
    binding: NativePairedBuildBinding,
    application_header: NativePairedProductHeader,
    test_header: NativePairedProductHeader,
    application: RelocatableNativeObject,
    tests: RelocatableNativeObject,
}

impl PairedNativeObjects {
    pub fn binding(&self) -> &NativePairedBuildBinding {
        &self.binding
    }
    pub fn application_header(&self) -> &NativePairedProductHeader {
        &self.application_header
    }
    pub fn test_header(&self) -> &NativePairedProductHeader {
        &self.test_header
    }
    pub fn application(&self) -> &RelocatableNativeObject {
        &self.application
    }
    pub fn tests(&self) -> &RelocatableNativeObject {
        &self.tests
    }

    pub(crate) fn validate(&self) -> AotResult<()> {
        validate_pair(self)
    }
}

pub struct PairedNativeArtifacts {
    pub manifest: NativePairedArtifactManifest,
    pub application: ProgramArtifact,
    pub tests: ProgramArtifact,
}

/// Materialize two exact object envelopes and their external content-addressed
/// pair. The common binding is embedded in the linked objects, while artifact
/// IDs live only here to avoid a self-referential object hash.
pub fn materialize_selected_paired_native_artifacts(
    pair: &PairedNativeObjects,
    application_recipe: ProgramBuildRecipe,
    test_recipe: ProgramBuildRecipe,
) -> AotResult<PairedNativeArtifacts> {
    pair.validate()?;
    if application_recipe.program_revision != pair.binding.program_revision
        || test_recipe.program_revision != pair.binding.program_revision
    {
        return Err(invalid(
            "paired artifact recipes differ from frozen program revision",
        ));
    }
    let (application_recipe, application) =
        crate::materialize_native_object_artifact(application_recipe, &pair.application)?;
    let (test_recipe, tests) = crate::materialize_native_object_artifact(test_recipe, &pair.tests)?;
    let manifest = NativePairedArtifactManifest {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        binding: pair.binding.clone(),
        application_recipe,
        test_recipe,
        application_artifact: application.id,
        test_artifact: tests.id,
    };
    manifest
        .validate_with_artifacts(&application, &tests)
        .map_err(|error| invalid(format!("paired artifacts: {error}")))?;
    Ok(PairedNativeArtifacts {
        manifest,
        application,
        tests,
    })
}

fn validate_pair(pair: &PairedNativeObjects) -> AotResult<()> {
    pair.binding
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    pair.application_header
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    pair.test_header
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if pair.application_header.binding != pair.binding
        || pair.test_header.binding != pair.binding
        || pair.application_header.role != NativePairedProductRole::Application
        || pair.test_header.role != NativePairedProductRole::Test
    {
        return Err(invalid(
            "paired object headers differ from their shared build",
        ));
    }
    for (object, header) in [
        (&pair.application, &pair.application_header),
        (&pair.tests, &pair.test_header),
    ] {
        object
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let expected = runmat_execution::Digest::sha256(
            &header
                .to_canonical_bytes()
                .map_err(|error| invalid(error.to_string()))?,
        );
        if object.manifest.product.selected() != Some(&pair.binding.selected_product)
            || object
                .manifest
                .data
                .iter()
                .filter(|data| data.symbol == runmat_native_codegen::aot::AOT_PAIRED_PRODUCT_SYMBOL)
                .count()
                != 1
            || object
                .manifest
                .data
                .iter()
                .find(|data| data.symbol == runmat_native_codegen::aot::AOT_PAIRED_PRODUCT_SYMBOL)
                .is_none_or(|data| data.digest != expected)
        {
            return Err(invalid("paired object has no exact embedded role header"));
        }
    }
    Ok(())
}

/// Test procedures are resolved against one frozen Core compiler inventory;
/// caller-supplied function ordinals never become a selection authority.
pub fn emit_selected_paired_native_objects(
    unit: &runmat_core::ExecutableUnit,
    selected: &SelectedNativeProduct<'_>,
    plan: &TestPlan,
    options: NativeObjectOptions,
) -> AotResult<PairedNativeObjects> {
    validate_plan(plan).map_err(|error| invalid(error.to_string()))?;
    if plan.program_revision != unit.revision().program_revision {
        return Err(invalid("test plan and frozen executable revisions differ"));
    }
    if options.optimization
        != selected
            .product()
            .selected()
            .expect("verified selected constructor")
            .optimization
    {
        return Err(invalid("paired optimization differs from selected target"));
    }
    if options.retained_functions.is_some() {
        return Err(invalid(
            "paired role retention is derived from the frozen test inventory",
        ));
    }
    if plan.tests().count() == 0 {
        return Err(invalid("paired native test plan has no procedures"));
    }

    // This is the sole frontend preparation. Both products fork its exact MIR,
    // analysis, source records and captured program, then retain their own
    // role's reachable functions without independently recompiling source.
    let frozen = unit
        .prepare_native_compilation_for_with_interop(None, options.interop.clone())
        .map_err(|error| invalid(format!("frozen native input: {error}")))?;
    let common = frozen
        .lower(selected.target().clone())
        .map_err(|error| invalid(format!("common native lowering: {error}")))?;
    selected
        .validate_assembly(&common)
        .map_err(|error| invalid(format!("selected union: {error}")))?;
    let bodies = common
        .functions
        .iter()
        .map(|function| {
            (
                function.id,
                (function.portable_id, function.mir_body_digest),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let (application_function, application_body) = bodies
        .get(&frozen.entrypoint())
        .copied()
        .ok_or_else(|| invalid("application entrypoint is absent from common Native IR"))?;
    let application = NativeCompiledEntry {
        function: application_function,
        compiled_function: frozen.entrypoint(),
        body: application_body,
    };
    let mut tests = Vec::with_capacity(plan.tests().count());
    for descriptor in plan.tests() {
        let function = frozen
            .resolve_test_procedure(&descriptor.procedure)
            .map_err(|error| invalid(format!("frozen test procedure: {error}")))?;
        let source = &descriptor.procedure.source;
        let parameter_identity = descriptor
            .parameters
            .iter()
            .map(|parameter| parameter.id.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let scheme = match descriptor.procedure.kind {
            runmat_test::descriptor::ProcedureKind::Function => "function",
            runmat_test::descriptor::ProcedureKind::Method => "class-method",
            _ => return Err(invalid("unsupported native test procedure kind")),
        };
        let derived = TestId::derive(&TestIdentityInput {
            owner_identity: &source.owner_identity,
            relative_source_identity: &source.relative_path,
            semantic_scheme: scheme,
            semantic_item_path: &descriptor.procedure.semantic_path,
            parameter_identity: &parameter_identity,
            fixture_identity: descriptor.fixture_group_id.as_str(),
        });
        if derived != descriptor.id {
            return Err(invalid("test identity differs from its frozen procedure"));
        }
        let (portable_function, body) = *bodies
            .get(&function)
            .ok_or_else(|| invalid("typed test compiler entry is absent from common Native IR"))?;
        tests.push(NativeTestEntry {
            suite: descriptor.suite_id.clone(),
            test: descriptor.id.clone(),
            compiled: NativeCompiledEntry {
                function: portable_function,
                compiled_function: function,
                body,
            },
        });
    }
    tests.sort_by(|left, right| (&left.suite, &left.test).cmp(&(&right.suite, &right.test)));
    let binding = NativePairedBuildBinding {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        program_revision: plan.program_revision.clone(),
        selected_product: selected
            .product()
            .selected()
            .expect("verified selected constructor")
            .clone(),
        application: application.clone(),
        tests,
    };
    binding
        .validate()
        .map_err(|error| invalid(format!("paired binding: {error}")))?;
    let application_header = NativePairedProductHeader {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        role: NativePairedProductRole::Application,
        binding: binding.clone(),
        linked_entries: vec![application.clone()],
    };
    let test_header = NativePairedProductHeader {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        role: NativePairedProductRole::Test,
        binding: binding.clone(),
        linked_entries: binding
            .tests
            .iter()
            .map(|entry| (entry.compiled.compiled_function, entry.compiled.clone()))
            .collect::<BTreeMap<_, _>>()
            .into_values()
            .collect(),
    };
    application_header
        .validate()
        .map_err(|error| invalid(format!("application header: {error}")))?;
    test_header
        .validate()
        .map_err(|error| invalid(format!("test header: {error}")))?;

    let mut application_retained = BTreeSet::new();
    application_retained.insert(application.compiled_function);
    let mut test_retained = application_retained.clone();
    test_retained.extend(
        binding
            .tests
            .iter()
            .map(|entry| entry.compiled.compiled_function),
    );
    let application_input = frozen
        .clone()
        .retain_functions(&application_retained)
        .map_err(|error| invalid(format!("application retention: {error}")))?;
    let test_input = frozen
        .retain_functions(&test_retained)
        .map_err(|error| invalid(format!("test retention: {error}")))?
        .with_additional_entrypoints(
            &binding
                .tests
                .iter()
                .map(|entry| entry.compiled.compiled_function)
                .collect(),
        )
        .map_err(|error| invalid(format!("test entrypoints: {error}")))?;
    let application_object =
        emit_role(&application_input, selected, &options, &application_header)?;
    let test_object = emit_role(&test_input, selected, &options, &test_header)?;
    Ok(PairedNativeObjects {
        binding,
        application_header,
        test_header,
        application: application_object,
        tests: test_object,
    })
}

fn emit_role(
    input: &runmat_core::NativeCompilationInput,
    selected: &SelectedNativeProduct<'_>,
    options: &NativeObjectOptions,
    header: &NativePairedProductHeader,
) -> AotResult<RelocatableNativeObject> {
    let assembly = input
        .lower(selected.target().clone())
        .map_err(|error| invalid(format!("paired role lowering: {error}")))?;
    let data = input
        .aot_object_data_for_selected(&assembly, object_data_options(options)?, selected)
        .map_err(|error| invalid(format!("paired role object data: {error}")))?;
    runmat_native_codegen::aot::emit_selected_paired_relocatable_object(
        selected,
        &assembly,
        header,
        data,
        options.runtime_binding_mode,
        options.retained_builtin_bindings.clone(),
    )
    .map_err(|error| invalid(format!("paired role object: {error}")))
}

fn invalid(message: impl Into<String>) -> AotError {
    AotError::contract("aot.compile.paired", message.into())
}
