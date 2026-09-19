use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use runmat_config::runtime::RunMatRuntimeConfig;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::BundleCallable;
use runmat_package::ProjectSymbolDefinition;
use runmat_types::{
    OperationParameterArity, OperationTypePattern, ShapeFact, ValueFact, ValueKindFact,
};

use crate::commands::session::create_session;

mod facts;

pub(super) use facts::compile_function_facts;

struct CompiledCallable {
    callable: BundleCallable,
    function: runmat_mir::analysis::FunctionAnalysis,
    parameters: Vec<SourceParameter>,
    executable: runmat_execution::ExecutableUnitEnvelope,
}

#[derive(Clone, Debug)]
struct SourceParameter {
    name: String,
    arity: OperationParameterArity,
}

async fn compile_project_definitions(
    requested: Vec<(String, String, ProjectSymbolDefinition)>,
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<(ProgramRevision, BTreeMap<String, CompiledCallable>)> {
    let project = frozen.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project.graph_digest.bytes()),
        Digest::from_bytes(*project.source_revision.bytes()),
        runmat_core::program_environment(crate::diagnostics::parser_compat(config.language.compat)),
    )?;
    let mut session = create_session(
        false,
        false,
        config,
        "create FEA callable compilation session",
    )?;
    session.install_project_handoff(runmat_package::FrozenProjectHandoff::new(frozen.clone()))?;
    let mut compiled = BTreeMap::new();
    for (key, qualified_function, definition) in requested {
        let source_id = definition.source_id.as_ref().with_context(|| {
            format!("FEA function `{qualified_function}` has no frozen source identity")
        })?;
        let source_bytes = runmat_filesystem::read_async(&definition.source_path)
            .await
            .with_context(|| {
                format!(
                    "read frozen FEA function source {}",
                    definition.source_path.display()
                )
            })?;
        let source_digest = Digest::sha256(&source_bytes);
        if source_digest.bytes() != source_id.content_digest.bytes() {
            bail!(
                "FEA function `{qualified_function}` changed after its project source graph was frozen"
            );
        }
        let source_text = String::from_utf8(source_bytes).with_context(|| {
            format!(
                "FEA function source {} is not UTF-8 MATLAB text",
                definition.source_path.display()
            )
        })?;
        let parameters = source_parameters(
            &source_text,
            entrypoint_name(&qualified_function)?,
            crate::diagnostics::parser_compat(config.language.compat),
        )?;
        let owner_identity = source_id.package_instance.to_string();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::frozen(
                    owner_identity.clone(),
                    source_id.relative_path.as_str(),
                    definition.source_path.to_string_lossy(),
                    source_text,
                ),
                Some(revision.clone()),
            )
            .await
            .with_context(|| format!("compile FEA function `{qualified_function}`"))?;
        let entrypoint = entrypoint_name(&qualified_function)?;
        let executable = unit
            .portable_envelope_for(Some(entrypoint))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("freeze FEA function `{qualified_function}`"))?;
        let function = unit
            .function_analysis(executable.manifest.identity.entrypoint_function)
            .cloned()
            .with_context(|| {
                format!("FEA function `{qualified_function}` has no static function facts")
            })?;
        compiled.insert(
            key,
            CompiledCallable {
                callable: BundleCallable {
                    owner_identity,
                    qualified_name: qualified_function,
                    source_digest,
                },
                function,
                parameters,
                executable,
            },
        );
    }
    Ok((revision, compiled))
}

pub(super) async fn compile_geometry_generator(
    document_path: &std::path::Path,
    source_root: &std::path::Path,
    definition: &runmat_geometry_core::GeometryDefinition,
    bindings: &std::collections::BTreeMap<String, runmat_geometry_core::GeometryAuthoringValue>,
    frozen: &runmat_package::FrozenProject,
    config: &RunMatRuntimeConfig,
) -> Result<runmat_analysis_execution::PreparedGeometryGenerator> {
    let runmat_geometry_core::GeometryDefinitionSource::Generated(generated) = &definition.source
    else {
        bail!("geometry generator compilation requires a generated geometry definition");
    };
    let module = runmat_analysis_language::LogicalSourcePath::new(generated.module_path.clone())?;
    let symbol = runmat_analysis_language::SourceSymbol::new(generated.function.clone())?;
    let expected_path = facts::expected_source_path(source_root, &module)?;
    let symbols = runmat_package::source_symbols_from_frozen(frozen, document_path);
    let project_definition =
        facts::exact_definition(&symbols.definitions, &expected_path, &symbol)?;
    let key = format!("{}::{symbol}", module.as_str());
    let (revision, mut compiled) = compile_project_definitions(
        vec![(
            key.clone(),
            project_definition.qualified_name.clone(),
            project_definition.clone(),
        )],
        frozen,
        config,
    )
    .await?;
    let compiled = compiled
        .remove(&key)
        .context("frozen geometry generator was not compiled")?;
    let request =
        runmat_analysis_execution::resolve_geometry_generator_request(definition, bindings)?;
    let admitted = runmat_analysis_execution::admit_named_geometry_generator(
        request,
        revision,
        compiled.callable,
        compiled.executable.manifest.identity.source_digest,
        &compiled.function,
        &std::collections::BTreeSet::new(),
    )?;
    runmat_analysis_execution::prepare_geometry_generator(admitted, compiled.executable)
        .map_err(anyhow::Error::new)
}

fn entrypoint_name(qualified_function: &str) -> Result<&str> {
    qualified_function
        .rsplit('.')
        .next()
        .filter(|name| !name.is_empty())
        .context("FEA function has an empty function name")
}

fn source_parameters(
    source: &str,
    entrypoint: &str,
    compat: runmat_parser::CompatMode,
) -> Result<Vec<SourceParameter>> {
    let statements =
        runmat_parser::parse_with_options(source, runmat_parser::ParserOptions::new(compat))
            .map_err(|error| {
                anyhow::anyhow!(
                    "cannot parse imported function `{entrypoint}`: {}",
                    error.message
                )
            })?;
    let parameters = statements
        .body
        .iter()
        .find_map(|statement| match statement {
            runmat_parser::Stmt::Function {
                name,
                params,
                argument_validations,
                ..
            } if name == entrypoint => Some(
                params
                    .iter()
                    .map(|name| SourceParameter {
                        name: name.clone(),
                        arity: if name == "varargin" {
                            OperationParameterArity::Variadic
                        } else if argument_validations.iter().any(|validation| {
                            validation.name == *name && validation.default_value.is_some()
                        }) {
                            OperationParameterArity::Optional
                        } else {
                            OperationParameterArity::Required
                        },
                    })
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        });
    parameters.with_context(|| format!("imported source does not define function `{entrypoint}`"))
}

fn operation_pattern(fact: &ValueFact) -> OperationTypePattern {
    match (&fact.kind, &fact.shape) {
        (ValueKindFact::Numeric(_), ShapeFact::Scalar) => OperationTypePattern::NumericScalar,
        (ValueKindFact::Numeric(_), _) => OperationTypePattern::NumericArray,
        (ValueKindFact::Logical, _) => OperationTypePattern::LogicalArray,
        (ValueKindFact::Character | ValueKindFact::String, ShapeFact::Scalar) => {
            OperationTypePattern::StringScalar
        }
        (ValueKindFact::Object(object), shape) if shape.element_count() == Some(1) => match object
            .runtime_class
            .as_ref()
            .and_then(runmat_types::QualifiedName::display_name)
            .as_deref()
        {
            Some(runmat_analysis_execution::GENERATED_GEOMETRY_RUNTIME_CLASS) => {
                OperationTypePattern::Analysis(runmat_types::AnalysisValueType::Geometry)
            }
            Some("geometry.Feature") => {
                OperationTypePattern::Analysis(runmat_types::AnalysisValueType::GeometryFeature)
            }
            Some("geometry.RegionOutput") => {
                OperationTypePattern::Analysis(runmat_types::AnalysisValueType::Region)
            }
            _ => OperationTypePattern::Any,
        },
        _ => OperationTypePattern::Any,
    }
}
