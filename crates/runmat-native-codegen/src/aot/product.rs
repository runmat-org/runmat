use runmat_execution::host::NativeProductMode;
use runmat_execution::{Digest, ExecutableIdentity};
use runmat_types::{FunctionId, ProgramSourceId};

use crate::NativeTarget;

pub const NATIVE_OBJECT_SCHEMA_VERSION: u16 = 20;
pub const AOT_PROGRAM_MANIFEST_SCHEMA_VERSION: u16 = 13;
const MAX_AOT_PROGRAM_FUNCTIONS: usize = 100_000;
const MAX_AOT_PROGRAM_MANIFEST_BYTES: usize = 64 * 1024 * 1024;
pub const AOT_ENTRY_SYMBOL: &str = "runmat_aot_entry";
pub const AOT_RUNTIME_MAIN_SYMBOL: &str = "runmat_aot_main";
pub const AOT_RUNTIME_TEST_MAIN_SYMBOL: &str = "runmat_aot_test_main";
pub const AOT_PAIRED_PRODUCT_SYMBOL: &str = "runmat_aot_paired_product";
pub const AOT_SOURCE_SERVICE_FACTORY_SYMBOL: &str = "runmat_aot_source_evaluation_component_v2";
pub const AOT_CLOCK_SERVICE_FACTORY_SYMBOL: &str = "runmat_aot_clock_component_v1";
pub const AOT_LOCAL_SERVICE_FACTORY_SYMBOL: &str = "runmat_aot_local_component_v1";
pub const AOT_NETWORK_SERVICE_FACTORY_SYMBOL: &str = "runmat_aot_network_component_v1";
pub const AOT_UI_SERVICE_FACTORY_SYMBOL: &str = "runmat_aot_ui_component_v2";
pub const AOT_HOST_CALLBACK_FACTORY_SYMBOL: &str = "runmat_aot_host_callback_component_v2";
pub const AOT_HOSTED_DYNAMIC_FACTORY_SYMBOL: &str = "runmat_aot_hosted_dynamic_component_v1";
pub const AOT_NATIVE_IR_SYMBOL: &str = "runmat_aot_native_ir";
pub const AOT_PROGRAM_SYMBOL: &str = "runmat_aot_program";
pub const AOT_RESUME_POINTS_SYMBOL: &str = "runmat_aot_resume_points";
pub const AOT_NATIVE_INTERFACES_SYMBOL: &str = "runmat_aot_native_interfaces";
pub const AOT_MEX_ARTIFACTS_SYMBOL: &str = "runmat_aot_mex_artifacts";
pub const AOT_JAVA_ARTIFACTS_SYMBOL: &str = "runmat_aot_java_artifacts";
pub const AOT_PYTHON_ARTIFACTS_SYMBOL: &str = "runmat_aot_python_artifacts";

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AotRuntimeBindingMode {
    Dynamic,
    ClosedWorld,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotBuiltinBinding {
    pub name: String,
    pub variant: String,
    pub native_symbol: String,
}

impl AotBuiltinBinding {
    fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        if !valid_identity(&self.name, 512)
            || !valid_identity(&self.variant, 256)
            || self.native_symbol
                != runmat_builtins::native_binding_symbol(&self.name, &self.variant)
        {
            return Err(crate::NativeCodegenError::new(
                "native.aot.builtin_binding",
                "AOT builtin binding identity or native symbol is invalid",
            ));
        }
        Ok(())
    }
}

/// Frontend- and VM-independent callable inventory embedded in a native AOT
/// object. The standalone host needs names and source ownership for exact
/// runtime dispatch, but never bytecode or interpreter frame metadata.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotProgramManifest {
    pub schema_version: u16,
    pub product: NativeProductMode,
    pub executable: ExecutableIdentity,
    /// Compiler/native identity selected for this product. The executable
    /// identity independently retains its immutable portable function ordinal.
    pub entrypoint: FunctionId,
    pub native_ir_digest: Digest,
    /// The bounded, canonical Config-owned policy captured with this program.
    pub foreign_policy: Option<runmat_execution_artifact::LogicalObject>,
    /// Exact application-invocation policy. Hosted ambient clocks use None;
    /// deterministic clocks require explicit inputs captured in linked bytes.
    pub clock_inputs: Option<runmat_execution::host::NativeClockInputs>,
    /// Invocation-local environment, random-stream, and virtual filesystem
    /// inputs. These bytes bind data, not linked service implementation rights.
    pub local_service_inputs: Option<runmat_execution::host::NativeLocalServiceInputs>,
    pub source_map: runmat_execution::ExecutableSourceMapV2,
    pub functions: Vec<AotProgramFunction>,
    pub runtime_binding_mode: AotRuntimeBindingMode,
    /// Canonical MIR reachability contains an optional provider placement
    /// edge. This is intent, not target backend evidence or a runtime grant.
    pub provider_reachable: bool,
    pub builtin_bindings: Vec<AotBuiltinBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AotProgramFunction {
    pub function: FunctionId,
    pub portable_function: runmat_types::ProgramFunctionId,
    pub source: ProgramSourceId,
    pub name: String,
    pub role: runmat_types::FunctionSourceRole,
}

#[derive(serde::Deserialize)]
struct AotProgramManifestAdmission {
    schema_version: u16,
}

impl AotProgramManifest {
    /// Authenticate every local compiled coordinate against the portable
    /// semantic identity retained by the same verified Native IR body.
    pub fn validate_function_inventory(
        &self,
        assembly: &crate::NativeAssembly,
    ) -> Result<(), crate::NativeCodegenError> {
        let mut native = assembly
            .functions
            .iter()
            .map(|function| {
                (
                    function.id,
                    function.portable_id,
                    function.source,
                    function.name.as_str(),
                    function.source_role,
                )
            })
            .collect::<Vec<_>>();
        native.sort_by_key(|function| function.0);
        if !function_inventory_matches(&native, &self.functions) {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.function_inventory",
                "AOT program function inventory differs from verified Native IR",
            ));
        }
        Ok(())
    }

    pub fn from_assembly_for_selected(
        selected: &super::SelectedNativeProduct<'_>,
        assembly: &crate::NativeAssembly,
        native_ir_digest: Digest,
        foreign_policy: Option<runmat_execution_artifact::LogicalObject>,
        source_map: runmat_execution::ExecutableSourceMapV2,
        runtime_binding_mode: AotRuntimeBindingMode,
        builtin_bindings: Vec<AotBuiltinBinding>,
    ) -> Result<Self, crate::NativeCodegenError> {
        selected.validate_assembly(assembly)?;
        let mut manifest = Self::from_assembly(
            assembly,
            native_ir_digest,
            foreign_policy,
            source_map,
            runtime_binding_mode,
            builtin_bindings,
        )?;
        manifest.product = selected.product().clone();
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_assembly(
        assembly: &crate::NativeAssembly,
        native_ir_digest: Digest,
        foreign_policy: Option<runmat_execution_artifact::LogicalObject>,
        source_map: runmat_execution::ExecutableSourceMapV2,
        runtime_binding_mode: AotRuntimeBindingMode,
        mut builtin_bindings: Vec<AotBuiltinBinding>,
    ) -> Result<Self, crate::NativeCodegenError> {
        assembly.verify()?;
        let interop = &assembly.requirements.interop;
        if foreign_policy.is_none()
            && (!interop.foreign_types.is_empty()
                || !interop.adapters.is_empty()
                || !interop.adapter_contracts.is_empty())
        {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.foreign_policy",
                "foreign AOT program requires a captured policy",
            ));
        }
        let mut functions = assembly
            .functions
            .iter()
            .map(|function| AotProgramFunction {
                function: function.id,
                portable_function: function.portable_id,
                source: function.source,
                name: function.name.clone(),
                role: function.source_role,
            })
            .collect::<Vec<_>>();
        functions.sort_by_key(|function| function.function);
        builtin_bindings
            .sort_by(|left, right| (&left.name, &left.variant).cmp(&(&right.name, &right.variant)));
        let manifest = Self {
            schema_version: AOT_PROGRAM_MANIFEST_SCHEMA_VERSION,
            product: NativeProductMode::HostUnselected,
            executable: assembly.executable_identity.clone(),
            entrypoint: assembly.selected_entrypoint,
            native_ir_digest,
            foreign_policy,
            clock_inputs: None,
            local_service_inputs: None,
            source_map,
            functions,
            runtime_binding_mode,
            provider_reachable: false,
            builtin_bindings,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        if self.schema_version != AOT_PROGRAM_MANIFEST_SCHEMA_VERSION {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.schema",
                format!(
                    "AOT program manifest schema actual {}, expected {}",
                    self.schema_version, AOT_PROGRAM_MANIFEST_SCHEMA_VERSION
                ),
            ));
        }
        self.product.validate().map_err(|error| {
            crate::NativeCodegenError::new("native.aot.program.product", error.to_string())
        })?;
        runmat_execution::host::validate_native_clock_inputs(
            self.product.selected(),
            self.clock_inputs.as_ref(),
        )
        .map_err(|error| {
            crate::NativeCodegenError::new("native.aot.program.clock_inputs", error.to_string())
        })?;
        runmat_execution::host::validate_native_local_inputs(
            self.product.selected(),
            self.local_service_inputs.as_ref(),
        )
        .map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program.local_service_inputs",
                error.to_string(),
            )
        })?;
        self.executable.program.validate().map_err(|error| {
            crate::NativeCodegenError::new("native.aot.program", error.to_string())
        })?;
        if self.product.selected().is_some_and(|binding| {
            binding.runtime != *self.executable.program.runtime_fingerprint()
                || binding.catalog != *self.executable.program.catalog_fingerprint()
        }) {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.product",
                "selected product differs from program environment",
            ));
        }
        if let Some(policy) = &self.foreign_policy {
            runmat_execution_artifact::object::validate_foreign_policy_object(policy).map_err(
                |error| {
                    crate::NativeCodegenError::new(
                        "native.aot.program.foreign_policy",
                        error.to_string(),
                    )
                },
            )?;
        }
        self.source_map.validate().map_err(|error| {
            crate::NativeCodegenError::new("native.aot.program.source_map", error.to_string())
        })?;
        if !valid_identity(&self.executable.root_package, 256)
            || !valid_identity(&self.executable.entrypoint, 512)
            || self.functions.is_empty()
            || self.functions.len() > MAX_AOT_PROGRAM_FUNCTIONS
            || self
                .functions
                .windows(2)
                .any(|pair| pair[0].function >= pair[1].function)
            || !self
                .functions
                .iter()
                .any(|function| function.function == self.entrypoint)
            || self
                .functions
                .iter()
                .find(|function| function.function == self.entrypoint)
                .is_none_or(|function| {
                    Some(function.portable_function) != self.executable.entrypoint_function
                })
            || self
                .functions
                .iter()
                .map(|function| function.portable_function)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.functions.len()
            || self.functions.iter().any(|function| {
                function.name.is_empty()
                    || function.name.len() > 512
                    || function.name.chars().any(char::is_control)
                    || usize::try_from(function.source.0)
                        .ok()
                        .is_none_or(|source| source >= self.source_map.entries.len())
            })
            || self
                .builtin_bindings
                .iter()
                .any(|binding| binding.validate().is_err())
            || self
                .builtin_bindings
                .windows(2)
                .any(|pair| (&pair[0].name, &pair[0].variant) >= (&pair[1].name, &pair[1].variant))
            || (self.runtime_binding_mode == AotRuntimeBindingMode::Dynamic
                && !self.builtin_bindings.is_empty())
        {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program",
                "AOT program manifest is not canonical",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, crate::NativeCodegenError> {
        self.validate()?;
        let value = serde_json::to_value(self).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to encode AOT program manifest: {error}"),
            )
        })?;
        serde_json::to_vec(&value).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to encode AOT program manifest: {error}"),
            )
        })
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, crate::NativeCodegenError> {
        if bytes.len() > MAX_AOT_PROGRAM_MANIFEST_BYTES {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.bounds",
                format!(
                    "AOT program manifest exceeds its {MAX_AOT_PROGRAM_MANIFEST_BYTES}-byte bound"
                ),
            ));
        }
        let admission: AotProgramManifestAdmission =
            serde_json::from_slice(bytes).map_err(|error| {
                crate::NativeCodegenError::new(
                    "native.aot.program",
                    format!("failed to decode AOT program manifest: {error}"),
                )
            })?;
        if admission.schema_version != AOT_PROGRAM_MANIFEST_SCHEMA_VERSION {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program.schema",
                format!(
                    "AOT program manifest schema actual {}, expected {}",
                    admission.schema_version, AOT_PROGRAM_MANIFEST_SCHEMA_VERSION
                ),
            ));
        }
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to decode AOT program manifest: {error}"),
            )
        })?;
        let manifest: Self = serde_json::from_value(value).map_err(|error| {
            crate::NativeCodegenError::new(
                "native.aot.program",
                format!("failed to decode AOT program manifest: {error}"),
            )
        })?;
        manifest.validate()?;
        if manifest.canonical_bytes()? != bytes {
            return Err(crate::NativeCodegenError::new(
                "native.aot.program",
                "AOT program manifest encoding is not canonical",
            ));
        }
        Ok(manifest)
    }

    /// Only a unique public primary can be selected without a caller/source
    /// context. Local, nested, method, and private names require their own
    /// exact lexical or source-scoped selection authority.
    pub fn resolve_name(&self, name: &str) -> Option<FunctionId> {
        resolve_unique_public_name(&self.functions, &self.source_map, name)
    }
}

fn function_inventory_matches(
    native: &[(
        FunctionId,
        runmat_types::ProgramFunctionId,
        ProgramSourceId,
        &str,
        runmat_types::FunctionSourceRole,
    )],
    manifest: &[AotProgramFunction],
) -> bool {
    native.len() == manifest.len()
        && native.iter().zip(manifest).all(|(native, manifest)| {
            native.0 == manifest.function
                && native.1 == manifest.portable_function
                && native.2 == manifest.source
                && native.3 == manifest.name
                && native.4 == manifest.role
        })
}

fn resolve_unique_public_name(
    functions: &[AotProgramFunction],
    source_map: &runmat_execution::ExecutableSourceMapV2,
    name: &str,
) -> Option<FunctionId> {
    let mut matches = functions.iter().filter(|function| {
        if function.name != name || function.role != runmat_types::FunctionSourceRole::Primary {
            return false;
        }
        let Some(source) = usize::try_from(function.source.0)
            .ok()
            .and_then(|ordinal| source_map.entries.get(ordinal))
        else {
            return false;
        };
        std::path::Path::new(&source.relative_path)
            .parent()
            .and_then(std::path::Path::file_name)
            != Some(std::ffi::OsStr::new("private"))
    });
    let selected = matches.next()?.function;
    matches.next().is_none().then_some(selected)
}

#[cfg(test)]
mod admission_tests {
    use super::{AotBuiltinBinding, AotProgramManifest};

    #[test]
    fn public_name_requires_unique_primary_and_never_leaks_private_or_nested() {
        use runmat_execution::{Digest, ExecutableSourceMapV2, ExecutableSourceRecordV2};
        use runmat_types::{FunctionId, FunctionSourceRole, ProgramSourceId};

        let source_map = ExecutableSourceMapV2 {
            entries: ["a/helper.m", "b/helper.m", "a/private/helper.m"]
                .into_iter()
                .enumerate()
                .map(|(ordinal, path)| ExecutableSourceRecordV2 {
                    source_id: ProgramSourceId(u32::try_from(ordinal).expect("fixture ordinal")),
                    owner_identity: "package".into(),
                    relative_path: path.into(),
                    content_revision: Digest::sha256(b"source"),
                    display_name: path.into(),
                    text: "source".into(),
                })
                .collect(),
        };
        let function = |ordinal, source, role| super::AotProgramFunction {
            function: FunctionId(ordinal as usize),
            portable_function: runmat_types::ProgramFunctionId(ordinal),
            source: ProgramSourceId(source),
            name: "helper".into(),
            role,
        };
        let mut functions = vec![
            function(0, 2, FunctionSourceRole::Primary),
            function(1, 0, FunctionSourceRole::Nested),
            function(2, 0, FunctionSourceRole::Primary),
        ];
        assert_eq!(
            super::resolve_unique_public_name(&functions, &source_map, "helper"),
            Some(FunctionId(2))
        );
        functions.push(function(3, 1, FunctionSourceRole::Primary));
        assert_eq!(
            super::resolve_unique_public_name(&functions, &source_map, "helper"),
            None
        );
    }

    #[test]
    fn local_inventory_rejects_swapped_nonentry_portable_identities() {
        use runmat_types::{FunctionId, FunctionSourceRole, ProgramFunctionId, ProgramSourceId};

        let native = [
            (
                FunctionId(10),
                ProgramFunctionId(1),
                ProgramSourceId(0),
                "first",
                FunctionSourceRole::Local,
            ),
            (
                FunctionId(20),
                ProgramFunctionId(2),
                ProgramSourceId(0),
                "second",
                FunctionSourceRole::Local,
            ),
        ];
        let mut manifest = vec![
            super::AotProgramFunction {
                function: FunctionId(10),
                portable_function: ProgramFunctionId(1),
                source: ProgramSourceId(0),
                name: "first".into(),
                role: FunctionSourceRole::Local,
            },
            super::AotProgramFunction {
                function: FunctionId(20),
                portable_function: ProgramFunctionId(2),
                source: ProgramSourceId(0),
                name: "second".into(),
                role: FunctionSourceRole::Local,
            },
        ];
        assert!(super::function_inventory_matches(&native, &manifest));
        let first = manifest[0].portable_function;
        let second = manifest[1].portable_function;
        manifest[0].portable_function = second;
        manifest[1].portable_function = first;
        assert!(
            !super::function_inventory_matches(&native, &manifest),
            "same local/name/source inventory cannot authorize swapped portable origins"
        );
    }

    #[test]
    fn stale_program_manifest_is_rejected_before_changed_function_shape() {
        let error = AotProgramManifest::from_canonical_bytes(
            br#"{"schema_version":1,"functions":"changed representation"}"#,
        )
        .unwrap_err();
        assert_eq!(error.code, "native.aot.program.schema");
        assert!(error.message.contains(&format!(
            "actual 1, expected {}",
            super::AOT_PROGRAM_MANIFEST_SCHEMA_VERSION
        )));
    }

    #[test]
    fn untagged_v6_is_rejected_before_legacy_shape_decode() {
        let error = AotProgramManifest::from_canonical_bytes(
            br#"{"schema_version":6,"functions":"changed representation"}"#,
        )
        .unwrap_err();
        assert_eq!(error.code, "native.aot.program.schema");
    }

    #[test]
    fn schema_five_program_manifest_is_rejected_before_selected_entrypoint_decode() {
        let error = AotProgramManifest::from_canonical_bytes(
            br#"{"schema_version":5,"functions":"old selected-entrypoint semantics"}"#,
        )
        .unwrap_err();
        assert_eq!(error.code, "native.aot.program.schema");
        assert!(error.message.contains(&format!(
            "actual 5, expected {}",
            super::AOT_PROGRAM_MANIFEST_SCHEMA_VERSION
        )));
    }

    #[test]
    fn obsolete_builtin_signature_is_rejected_before_resolution() {
        let binding = AotBuiltinBinding {
            name: "sin".into(),
            variant: "default".into(),
            native_symbol: "runmat_builtin_binding_v2_73696e_64656661756c74".into(),
        };
        let error = binding
            .validate()
            .expect_err("old pointer signature must be inadmissible");
        assert_eq!(error.code, "native.aot.builtin_binding");
        let current = AotBuiltinBinding {
            native_symbol: runmat_builtins::native_binding_symbol(&binding.name, &binding.variant),
            ..binding
        };
        current.validate().expect("current signature is admissible");
    }
}

fn valid_identity(value: &str, maximum: usize) -> bool {
    !value.is_empty() && value.len() <= maximum && !value.chars().any(char::is_control)
}

pub use runmat_execution::host::{NativeObjectFormat, NativeOptimization};

pub fn native_object_format_for_target(
    target: &NativeTarget,
) -> Result<NativeObjectFormat, crate::NativeCodegenError> {
    Ok(target.native_parts()?.0.target.object_format)
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectFunction {
    pub function: FunctionId,
    pub symbol: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObjectData {
    pub symbol: String,
    pub bytes: Vec<u8>,
    pub alignment: u64,
}

pub fn embedded_blob(
    symbol: &str,
    bytes: Vec<u8>,
    alignment: u64,
) -> Result<NativeObjectData, crate::NativeCodegenError> {
    if !valid_data_symbol(symbol) || bytes.is_empty() {
        return Err(crate::NativeCodegenError::new(
            "native.object.data",
            "embedded AOT blob has an invalid symbol or empty payload",
        ));
    }
    Ok(NativeObjectData {
        symbol: symbol.to_string(),
        bytes,
        alignment,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectDataDescriptor {
    pub symbol: String,
    pub digest: Digest,
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeObjectManifest {
    pub schema_version: u16,
    pub product: NativeProductMode,
    pub target: NativeTarget,
    pub object_format: NativeObjectFormat,
    pub executable_cache_key: Digest,
    pub native_cache_key: Digest,
    pub runtime_fingerprint: Digest,
    pub catalog_fingerprint: Digest,
    pub optimization: NativeOptimization,
    pub runtime_binding_mode: AotRuntimeBindingMode,
    /// Exact requirements of verified Native IR. Optional provider
    /// reachability is carried separately by the program manifest.
    pub required_capabilities: runmat_types::CapabilitySet,
    pub foreign_policy: Option<runmat_execution::ForeignPolicyRef>,
    pub object_digest: Digest,
    pub object_bytes: u64,
    pub entrypoint: FunctionId,
    pub selected_entrypoint_body: Digest,
    pub functions: Vec<NativeObjectFunction>,
    pub data: Vec<NativeObjectDataDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelocatableNativeObject {
    pub manifest: NativeObjectManifest,
    pub bytes: Vec<u8>,
}

impl RelocatableNativeObject {
    pub fn validate(&self) -> Result<(), crate::NativeCodegenError> {
        if self.manifest.schema_version != NATIVE_OBJECT_SCHEMA_VERSION {
            return Err(crate::NativeCodegenError::new(
                "native.object.schema",
                "native object manifest schema is unsupported",
            ));
        }
        self.manifest.target.validate()?;
        super::selected::validate_native_product(
            &self.manifest.product,
            &self.manifest.target,
            self.manifest.runtime_fingerprint,
            self.manifest.catalog_fingerprint,
            self.manifest.optimization,
        )?;
        if let Some(policy) = self.manifest.foreign_policy {
            runmat_execution_artifact::foreign_policy_descriptor(policy).map_err(|error| {
                crate::NativeCodegenError::new("native.object.foreign_policy", error.to_string())
            })?;
        }
        if self.manifest.object_format != native_object_format_for_target(&self.manifest.target)? {
            return Err(crate::NativeCodegenError::new(
                "native.object.format",
                "native object format does not match its target",
            ));
        }
        let object_bytes = u64::try_from(self.bytes.len()).map_err(|_| {
            crate::NativeCodegenError::new(
                "native.object.size",
                "native object size exceeds the portable manifest limit",
            )
        })?;
        if self
            .manifest
            .runtime_fingerprint
            .bytes()
            .iter()
            .all(|byte| *byte == 0)
            || self
                .manifest
                .catalog_fingerprint
                .bytes()
                .iter()
                .all(|byte| *byte == 0)
        {
            return Err(crate::NativeCodegenError::new(
                "native.object.environment",
                "native object runtime and catalog identities must be present",
            ));
        }
        if self.bytes.is_empty()
            || self.manifest.object_bytes != object_bytes
            || self.manifest.object_digest != Digest::sha256(&self.bytes)
            || self.manifest.functions.is_empty()
            || !self
                .manifest
                .functions
                .iter()
                .any(|function| function.function == self.manifest.entrypoint)
            || self
                .manifest
                .functions
                .windows(2)
                .any(|pair| pair[0].function >= pair[1].function)
            || self.manifest.functions.iter().any(|function| {
                function.symbol != function_symbol(function.function, self.manifest.entrypoint)
            })
            || self
                .manifest
                .data
                .windows(2)
                .any(|pair| pair[0].symbol >= pair[1].symbol)
            || self.manifest.data.iter().any(|data| {
                !valid_data_symbol(&data.symbol)
                    || data.bytes == 0
                    || !matches!(data.alignment, 1 | 2 | 4 | 8 | 16)
            })
        {
            return Err(crate::NativeCodegenError::new(
                "native.object.identity",
                "native object does not match its canonical manifest",
            ));
        }
        Ok(())
    }
}

pub(super) fn valid_data_symbol(symbol: &str) -> bool {
    symbol.starts_with("runmat_aot_")
        && symbol.len() <= 128
        && symbol
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub(super) fn function_symbol(function: FunctionId, entrypoint: FunctionId) -> String {
    if function == entrypoint {
        AOT_ENTRY_SYMBOL.to_string()
    } else {
        format!("runmat_native_f{}", function.0)
    }
}
