//! Exact geometry host composition shared by desktop and browser sidecars.
//! Hosts supply platform identity and bounded storage; geometry owns semantics.

use crate::GeometryProviderProfile;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GeometryMaterializationLimits {
    pub inventory: ObjectInventoryLimits,
}

#[cfg(feature = "occt-native")]
pub fn occt_geometry_provider_profile() -> Option<GeometryProviderProfile> {
    use runmat_geometry_ops::GeometryFeatureProvider;

    let kernel_abi = runmat_geometry_io::native_exact_kernel_abi()?;
    let provider = runmat_geometry_io::feature::OcctExactFeatureProvider::new(
        runmat_geometry_io::ExactCadImportOptions::default(),
        runmat_geometry_io::GeometryImportContext::new(),
    );
    Some(GeometryProviderProfile {
        provider_id: "occt".into(),
        provider_version: format!("runmat-geometry-io/{}", env!("CARGO_PKG_VERSION")),
        kernel_abi,
        capabilities: provider.capabilities(),
    })
}

#[cfg(not(feature = "occt-native"))]
pub fn occt_geometry_provider_profile() -> Option<GeometryProviderProfile> {
    None
}

pub fn execute_geometry_materialization_program_request<
    S: runmat_execution_artifact::cache::CacheImport + runmat_execution_artifact::cache::CacheExport,
>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    limits: GeometryMaterializationLimits,
    platform: runmat_meshing_core::PlatformBuildIdentity,
) -> ProgramExecutionResponse {
    #[cfg(feature = "occt-native")]
    {
        execute_occt(request, store, limits, platform).unwrap_or_else(failure)
    }
    #[cfg(not(feature = "occt-native"))]
    {
        let _ = (request, store, limits, platform);
        failure("geometry materialization requires an exact-CAD native worker")
    }
}

#[cfg(feature = "occt-native")]
fn execute_occt<S>(
    program: &ProgramExecutionRequest,
    store: &mut S,
    limits: GeometryMaterializationLimits,
    platform: runmat_meshing_core::PlatformBuildIdentity,
) -> Result<ProgramExecutionResponse, String>
where
    S: runmat_execution_artifact::cache::CacheImport
        + runmat_execution_artifact::cache::CacheExport,
{
    use crate::{
        decode_generated_geometry_product, decode_geometry_materialization_program,
        prepare_geometry_materialization_output, GeometryMaterializationOutput,
        GeometryMaterializationSource,
    };
    use runmat_execution::value::ValuePayload;
    use runmat_execution::Digest;
    use runmat_execution_artifact::ArtifactAccess;
    use runmat_geometry_core::{GeometryDefinitionSource, GeometrySourceFormat};
    use runmat_geometry_io::{
        detect_geometry_format, import_exact_cad, ExactCadImportOptions, GeometryFormat,
        GeometryImportContext, ImportedExactCad,
    };
    use runmat_geometry_ops::{
        materialize_geometry_definition, resolve_geometry_regions_with_generated_outputs,
        GeometryDependency, GeometryFeatureProvider, GeometryMaterializationHost,
        GeometryRegionEvidence, ResolvedGeneratedGeometryRequest,
    };
    use runmat_meshing_execution::{
        import_exact_geometry_input, prepare_exact_geometry_objects, prepare_exact_geometry_result,
    };

    if limits.inventory.max_objects == 0
        || limits.inventory.max_object_bytes == 0
        || limits.inventory.max_total_bytes == 0
    {
        return Err("geometry materialization inventory limits must be non-zero".into());
    }
    let descriptor = match program.arguments.first() {
        Some(ValuePayload::Object(reference)) => reference.as_ref(),
        _ => return Err("geometry workload has no externalized descriptor".into()),
    };
    let access = ArtifactAccess {
        authorization_scope: descriptor.authorization_scope.clone(),
        encryption_context: descriptor.encryption_context,
    };
    let (request, source_roots) = decode_geometry_materialization_program(program, store, &access)
        .map_err(|error| error.to_string())?;
    let inventory = runmat_execution_artifact::object::ObjectInventoryLimits {
        max_objects: limits.inventory.max_objects,
        max_object_bytes: limits
            .inventory
            .max_object_bytes
            .min(request.resources.maximum_object_bytes),
        max_total_bytes: limits
            .inventory
            .max_total_bytes
            .min(request.resources.maximum_artifact_bytes),
    };
    let actual_profile = occt_geometry_provider_profile()
        .ok_or_else(|| "exact-CAD worker profile is unavailable".to_string())?;
    if request.provider != actual_profile {
        return Err("geometry provider profile differs from the admitted worker profile".into());
    }
    let context = GeometryImportContext::new();
    let provider = runmat_geometry_io::feature::OcctExactFeatureProvider::new(
        request.import_options.clone(),
        context.clone(),
    );
    if provider.capabilities() != request.provider.capabilities {
        return Err("geometry provider capabilities changed after admission".into());
    }

    struct Host {
        imported: Option<ImportedExactCad>,
        generated: Option<(
            ResolvedGeneratedGeometryRequest,
            runmat_geometry_core::GeneratedGeometryProduct,
        )>,
    }
    impl GeometryMaterializationHost<ImportedExactCad> for Host {
        type Error = String;

        fn import_exact(
            &self,
            _source: &runmat_geometry_core::ImportedGeometryDefinition,
        ) -> Result<ImportedExactCad, Self::Error> {
            self.imported
                .clone()
                .ok_or_else(|| "imported geometry bytes were not admitted".into())
        }

        fn generate(
            &self,
            request: &ResolvedGeneratedGeometryRequest,
        ) -> Result<runmat_geometry_core::GeneratedGeometryProduct, Self::Error> {
            let (expected, product) = self
                .generated
                .as_ref()
                .ok_or_else(|| "generated geometry callable result was not admitted".to_string())?;
            if request != expected {
                return Err(
                    "generated geometry arguments differ from their callable receipt".into(),
                );
            }
            Ok(product.clone())
        }
    }

    let mut generated = None;
    let imported = match (&request.source, &request.definition.source) {
        (
            GeometryMaterializationSource::Imported {
                source_name,
                source_digest,
                encoded_length,
                ..
            },
            GeometryDefinitionSource::Imported(declared),
        ) => {
            let mut bytes = Vec::with_capacity(usize::try_from(*encoded_length).map_err(|_| {
                "imported geometry length exceeds the native address space".to_string()
            })?);
            for root in &source_roots {
                let chunk = store
                    .read_verified(root.logical_digest)
                    .map_err(|error| error.to_string())?
                    .ok_or_else(|| {
                        format!("missing imported geometry chunk {}", root.logical_digest)
                    })?;
                if chunk.len() as u64 != root.encoded_length {
                    return Err("imported geometry chunk length differs from its reference".into());
                }
                bytes.extend_from_slice(&chunk);
            }
            if bytes.len() as u64 != *encoded_length || Digest::sha256(&bytes) != *source_digest {
                return Err("imported geometry source bytes differ from their manifest".into());
            }
            let detected = detect_geometry_format(source_name, &bytes);
            let expected = geometry_format(declared.format)?;
            if detected != GeometryFormat::Unknown && detected != expected {
                return Err("imported geometry format differs from its declaration".into());
            }
            if expected == GeometryFormat::Step {
                let is_ap242 = runmat_geometry_io::ap242::has_ap242_schema_bytes(&bytes);
                let actual = is_ap242
                    .then(|| {
                        let text = std::str::from_utf8(&bytes)
                            .map_err(|_| "qualified AP242 source must be UTF-8".to_string())?;
                        runmat_geometry_io::ap242::inspect_ap242(text)
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?;
                match (&declared.exchange, actual) {
                    (Some(expected), Some(actual)) if expected == &actual => {}
                    (None, None) => {}
                    (Some(_), Some(_)) => {
                        return Err("AP242 exchange facts differ from the admitted source".into())
                    }
                    (Some(_), None) => {
                        return Err(
                            "declared AP242 exchange facts have no AP242 source profile".into()
                        )
                    }
                    (None, Some(_)) => {
                        return Err("AP242 source is missing admitted exchange facts".into())
                    }
                }
            }
            let geometry = import_exact_cad(
                source_name,
                &bytes,
                expected,
                &request.import_options,
                &context,
            )
            .map_err(|error| error.to_string())?;
            let closure = geometry
                .build_closure()
                .map_err(|error| error.to_string())?;
            if geometry.source_digest != declared.content_digest
                || closure.manifest.topology.digest != declared.topology_digest
            {
                return Err("imported exact geometry differs from its declared identity".into());
            }
            Some(geometry)
        }
        (GeometryMaterializationSource::Constructed, GeometryDefinitionSource::Constructed(_)) => {
            None
        }
        (
            GeometryMaterializationSource::Derived { document, root, .. },
            GeometryDefinitionSource::Derived(_),
        ) => {
            let root = source_roots
                .iter()
                .find(|candidate| {
                    candidate.logical_digest.bytes() == root.digest.bytes()
                        && candidate.encoded_length == root.encoded_length
                })
                .ok_or_else(|| "derived geometry source root is missing".to_string())?;
            let prepared = import_exact_geometry_input(
                store,
                document.as_ref().clone(),
                root,
                access.clone(),
                inventory,
            )
            .map_err(|error| error.to_string())?;
            let expected = prepared
                .input_objects()
                .iter()
                .map(|reference| (reference.logical_digest, reference.encoded_length))
                .collect::<std::collections::BTreeSet<_>>();
            let actual = source_roots
                .iter()
                .map(|reference| (reference.logical_digest, reference.encoded_length))
                .collect::<std::collections::BTreeSet<_>>();
            if expected != actual {
                return Err("derived geometry closure inventory is incomplete or contains unrelated objects".into());
            }
            let representation = prepared
                .geometry_objects()
                .kernel_representation()
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "derived exact geometry has no kernel representation".to_string())?;
            let mut options = exact_options_for_document(&request.import_options, document);
            options.source_units = Some(document.source.representation_units.unit());
            let geometry = import_exact_cad(
                "derived-source.brep",
                representation,
                GeometryFormat::Brep,
                &options,
                &context,
            )
            .map_err(|error| error.to_string())?;
            Some(geometry)
        }
        (
            GeometryMaterializationSource::Generated { receipt },
            GeometryDefinitionSource::Generated(_),
        ) => {
            let root = source_roots
                .first()
                .filter(|root| {
                    root.logical_digest == receipt.product_digest
                        && root.encoded_length == receipt.product_encoded_length
                })
                .ok_or_else(|| "generated geometry product root is missing".to_string())?;
            let bytes = store
                .read_verified(root.logical_digest)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| "generated geometry product is missing".to_string())?;
            if bytes.len() as u64 != root.encoded_length
                || Digest::sha256(&bytes) != root.logical_digest
            {
                return Err("generated geometry product differs from its receipt".into());
            }
            let product =
                decode_generated_geometry_product(&bytes).map_err(|error| error.to_string())?;
            generated = Some((receipt.admission.request.clone(), product));
            None
        }
        _ => return Err("geometry request source and definition do not converge".into()),
    };
    let dependency = match (&request.source, imported.as_ref()) {
        (GeometryMaterializationSource::Derived { definition, .. }, Some(geometry)) => {
            Some(GeometryDependency {
                reference: definition,
                geometry,
            })
        }
        _ => None,
    };
    let host = Host {
        imported: imported.clone(),
        generated,
    };
    let materialized = materialize_geometry_definition(
        &request.definition,
        &request.bindings,
        dependency,
        &provider,
        &host,
        request.plan_limits,
    )
    .map_err(|error| error.to_string())?;
    let geometry = materialized.geometry;
    let document = geometry
        .geometry_document()
        .map_err(|error| error.to_string())?;
    let region_evidence = ImportedRegionEvidence {
        imported: &geometry,
    };
    let regions = resolve_geometry_regions_with_generated_outputs(
        &request.definition,
        &geometry.topology,
        &materialized.feature_history,
        &materialized.generated_region_outputs,
        &region_evidence,
        geometry
            .healing_report
            .as_ref()
            .map(|report| &report.revision_map),
    )
    .map_err(|error| error.to_string())?;
    let source_face_ids = geometry.source_face_ids.clone();
    let exact_objects = prepare_exact_geometry_objects(
        document,
        geometry.topology,
        geometry.evaluators,
        Some(geometry.representation),
        geometry.healing_report,
        inventory,
    )
    .map_err(|error| error.to_string())?;
    for object in &exact_objects.objects {
        store
            .write_verified(object)
            .map_err(|error| error.to_string())?;
    }
    let exact_result = prepare_exact_geometry_result(exact_objects, access.clone(), inventory)
        .map_err(|error| error.to_string())?;
    if platform.exact_kernel_abi.as_deref() != Some(actual_profile.kernel_abi.as_str()) {
        return Err("geometry host platform does not identify the admitted exact kernel".into());
    }
    let output = GeometryMaterializationOutput {
        schema_version: 1,
        request_identity: request.identity().map_err(|error| error.to_string())?,
        definition: materialized.definition,
        materialization_kind: materialized.kind,
        provider: actual_profile,
        platform,
        document: exact_result.geometry_objects().document.clone(),
        geometry_root: exact_result.root_input().clone(),
        source_face_ids,
        regions,
        feature_history: materialized.feature_history,
        feature_resources: materialized.feature_resources,
    };
    output
        .validate_against(&request)
        .map_err(|error| error.to_string())?;
    let prepared_output = prepare_geometry_materialization_output(output, &access)
        .map_err(|error| error.to_string())?;
    prepared_output
        .write_to(store)
        .map_err(|error| error.to_string())?;
    let mut result_objects = exact_result.input_objects().to_vec();
    result_objects.push(prepared_output.root.clone());
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(prepared_output.root))],
        result_objects,
    };
    response
        .validate_against(program)
        .map_err(|error| error.to_string())?;
    fn geometry_format(format: GeometrySourceFormat) -> Result<GeometryFormat, String> {
        match format {
            GeometrySourceFormat::Step => Ok(GeometryFormat::Step),
            GeometrySourceFormat::Iges => Ok(GeometryFormat::Iges),
            GeometrySourceFormat::Brep | GeometrySourceFormat::NativeCad => {
                Ok(GeometryFormat::Brep)
            }
            _ => Err("geometry materialization requires an exact CAD source format".into()),
        }
    }

    fn exact_options_for_document(
        base: &ExactCadImportOptions,
        document: &runmat_geometry_core::GeometryDocument,
    ) -> ExactCadImportOptions {
        let mut options = base.clone();
        options.analysis.revision = document.revision.clone();
        options.analysis.absolute_tolerance_floor_m = runmat_types::physical::Length::new(
            document.tolerance.absolute_floor_m.canonical_value(),
        );
        options.analysis.model_relative_tolerance = document.tolerance.model_relative_term;
        options.analysis.requested_deviation_m = runmat_types::physical::Length::new(
            document.tolerance.requested_deviation_m.canonical_value(),
        );
        options.analysis.maximum_healing_displacement_m = runmat_types::physical::Length::new(
            document
                .tolerance
                .maximum_healing_displacement_m
                .canonical_value(),
        );
        options.analysis.healing = document.healing.clone();
        options
    }

    struct ImportedRegionEvidence<'a> {
        /// The exact import, whose authored CAD names and presentation layers select regions.
        imported: &'a runmat_geometry_io::ImportedExactCad,
    }

    impl GeometryRegionEvidence for ImportedRegionEvidence<'_> {
        type Error = String;

        fn by_name(
            &self,
            role: runmat_geometry_core::GeometryRegionRole,
            name: &str,
        ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, Self::Error> {
            self.imported.entities_named(role, name)
        }
        fn by_tag(
            &self,
            role: runmat_geometry_core::GeometryRegionRole,
            tag: &str,
        ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, Self::Error> {
            self.imported.entities_tagged(role, tag)
        }
        fn generated_output(
            &self,
            _role: runmat_geometry_core::GeometryRegionRole,
            _output: &str,
        ) -> Result<Vec<runmat_geometry_core::PersistentEntityId>, Self::Error> {
            Err("generated-output evidence requires a frozen generator receipt".into())
        }
    }
    Ok(response)
}

fn failure(message: impl Into<String>) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}
