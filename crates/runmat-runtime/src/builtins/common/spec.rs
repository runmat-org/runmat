use std::fmt;

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_registry {
    #![allow(dead_code)]
    use super::{BuiltinFusionSpec, BuiltinGpuSpec, FusionSpecRegistration, GpuSpecRegistration};
    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::sync::Mutex;

    static GPU_SPECS: Lazy<Mutex<Vec<GpuSpecRegistration>>> = Lazy::new(|| Mutex::new(Vec::new()));
    static FUSION_SPECS: Lazy<Mutex<Vec<FusionSpecRegistration>>> =
        Lazy::new(|| Mutex::new(Vec::new()));
    static RESIDENCY_POLICIES: Lazy<Mutex<HashMap<String, super::ResidencyPolicy>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

    pub(crate) fn submit_gpu_spec(registration: GpuSpecRegistration) {
        GPU_SPECS
            .lock()
            .expect("gpu spec registry poisoned")
            .push(registration);
        RESIDENCY_POLICIES
            .lock()
            .expect("gpu spec registry poisoned")
            .insert(
                registration.spec.name.to_ascii_lowercase(),
                registration.spec.residency,
            );
    }

    pub(crate) fn submit_fusion_spec(registration: FusionSpecRegistration) {
        FUSION_SPECS
            .lock()
            .expect("fusion spec registry poisoned")
            .push(registration);
    }

    pub(crate) fn gpu_specs() -> std::vec::IntoIter<&'static BuiltinGpuSpec> {
        GPU_SPECS
            .lock()
            .expect("gpu spec registry poisoned")
            .clone()
            .into_iter()
            .map(|registration| registration.spec)
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub(crate) fn gpu_spec_registrations() -> std::vec::IntoIter<GpuSpecRegistration> {
        GPU_SPECS
            .lock()
            .expect("gpu spec registry poisoned")
            .clone()
            .into_iter()
    }

    pub(crate) fn residency_policy(name: &str) -> Option<super::ResidencyPolicy> {
        RESIDENCY_POLICIES
            .lock()
            .expect("gpu spec registry poisoned")
            .get(&name.to_ascii_lowercase())
            .copied()
    }

    pub(crate) fn fusion_specs() -> std::vec::IntoIter<&'static BuiltinFusionSpec> {
        FUSION_SPECS
            .lock()
            .expect("fusion spec registry poisoned")
            .clone()
            .into_iter()
            .map(|registration| registration.spec)
            .collect::<Vec<_>>()
            .into_iter()
    }

    pub(crate) fn fusion_spec_registrations() -> std::vec::IntoIter<FusionSpecRegistration> {
        FUSION_SPECS
            .lock()
            .expect("fusion spec registry poisoned")
            .clone()
            .into_iter()
    }
}

/// Supported scalar precisions that GPU kernels may target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    F64,
    I32,
    Bool,
}

/// High-level GPU operation kind for builtin categorisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOpKind {
    Elementwise,
    Reduction,
    MatMul,
    Transpose,
    PlotRender,
    Custom(&'static str),
}

/// Broadcast semantics supported by the builtin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastSemantics {
    Matlab,
    ScalarOnly,
    None,
}

/// Hook names that providers may implement for specialised kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderHook {
    Unary {
        name: &'static str,
    },
    Binary {
        name: &'static str,
        commutative: bool,
    },
    Reduction {
        name: &'static str,
    },
    Custom(&'static str),
}

/// Strategy used when embedding constants in fused kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstantStrategy {
    InlineLiteral,
    UniformBuffer,
    WorkgroupMemory,
}

/// Residency policy for builtin outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyPolicy {
    InheritInputs,
    NewHandle,
    GatherImmediately,
}

/// How reductions should treat NaN values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionNaN {
    Include,
    Omit,
}

/// Shape requirements for fused kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeRequirements {
    BroadcastCompatible,
    Exact(&'static [usize]),
    Any,
}

/// Context provided to fusion expression builders.
pub struct FusionExprContext<'a> {
    pub scalar_ty: ScalarType,
    pub inputs: &'a [&'a str],
    pub constants: &'a [&'a str],
}

/// Builder used to generate WGSL expressions.
pub type FusionExprBuilder = fn(&FusionExprContext) -> Result<String, FusionError>;

/// Description of a fusion kernel template.
#[derive(Clone)]
pub struct FusionKernelTemplate {
    pub scalar_precisions: &'static [ScalarType],
    pub wgsl_body: FusionExprBuilder,
}

/// Possible errors emitted by a fusion builder.
#[derive(Debug)]
pub enum FusionError {
    MissingInput(usize),
    UnsupportedPrecision(ScalarType),
    Message(&'static str),
}

/// GPU metadata registered alongside builtin functions.
#[derive(Debug, Clone, Copy)]
pub struct BuiltinGpuSpec {
    pub name: &'static str,
    pub op_kind: GpuOpKind,
    pub supported_precisions: &'static [ScalarType],
    pub broadcast: BroadcastSemantics,
    pub provider_hooks: &'static [ProviderHook],
    pub constant_strategy: ConstantStrategy,
    pub residency: ResidencyPolicy,
    pub nan_mode: ReductionNaN,
    /// If set, reductions with reduce_len greater than this should prefer a two-pass kernel.
    pub two_pass_threshold: Option<usize>,
    /// Optional workgroup size hint for generated kernels.
    pub workgroup_size: Option<u32>,
    /// Whether the provider hook (if used) supports device-side omitnan handling.
    pub accepts_nan_mode: bool,
    pub notes: &'static str,
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionError::MissingInput(idx) => write!(f, "missing input {}", idx),
            FusionError::UnsupportedPrecision(ty) => write!(f, "unsupported precision {:?}", ty),
            FusionError::Message(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Fusion metadata registered alongside builtin functions.
#[derive(Clone)]
pub struct BuiltinFusionSpec {
    pub name: &'static str,
    pub shape: ShapeRequirements,
    pub constant_strategy: ConstantStrategy,
    pub elementwise: Option<FusionKernelTemplate>,
    pub reduction: Option<FusionKernelTemplate>,
    pub emits_nan: bool,
    pub notes: &'static str,
}

/// Inventory wrapper for GPU specs.
pub struct GpuSpecInventory {
    pub spec: &'static BuiltinGpuSpec,
    pub builtin_path: &'static str,
    pub declaration: &'static str,
    pub source_file: &'static str,
    pub module_path: &'static str,
}

/// Inventory wrapper for fusion specs.
pub struct FusionSpecInventory {
    pub spec: &'static BuiltinFusionSpec,
    pub builtin_path: &'static str,
    pub declaration: &'static str,
    pub source_file: &'static str,
    pub module_path: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct GpuSpecRegistration {
    pub spec: &'static BuiltinGpuSpec,
    pub builtin_path: &'static str,
    pub declaration: &'static str,
    pub source_file: &'static str,
    pub module_path: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct FusionSpecRegistration {
    pub spec: &'static BuiltinFusionSpec,
    pub builtin_path: &'static str,
    pub declaration: &'static str,
    pub source_file: &'static str,
    pub module_path: &'static str,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(GpuSpecInventory);
#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(FusionSpecInventory);

/// Iterate all registered GPU specs.
#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_gpu_specs() -> impl Iterator<Item = &'static BuiltinGpuSpec> {
    inventory::iter::<GpuSpecInventory>().map(|entry| entry.spec)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn builtin_gpu_spec_registrations() -> impl Iterator<Item = GpuSpecRegistration> {
    inventory::iter::<GpuSpecInventory>().map(|entry| GpuSpecRegistration {
        spec: entry.spec,
        builtin_path: entry.builtin_path,
        declaration: entry.declaration,
        source_file: entry.source_file,
        module_path: entry.module_path,
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn builtin_gpu_spec_registrations() -> std::vec::IntoIter<GpuSpecRegistration> {
    wasm_registry::gpu_spec_registrations()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_gpu_specs() -> std::vec::IntoIter<&'static BuiltinGpuSpec> {
    wasm_registry::gpu_specs()
}

/// Iterate all registered fusion specs.
#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_fusion_specs() -> impl Iterator<Item = &'static BuiltinFusionSpec> {
    inventory::iter::<FusionSpecInventory>().map(|entry| entry.spec)
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_fusion_specs() -> std::vec::IntoIter<&'static BuiltinFusionSpec> {
    wasm_registry::fusion_specs()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn builtin_fusion_spec_registrations() -> impl Iterator<Item = FusionSpecRegistration> {
    inventory::iter::<FusionSpecInventory>().map(|entry| FusionSpecRegistration {
        spec: entry.spec,
        builtin_path: entry.builtin_path,
        declaration: entry.declaration,
        source_file: entry.source_file,
        module_path: entry.module_path,
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn builtin_fusion_spec_registrations() -> std::vec::IntoIter<FusionSpecRegistration> {
    wasm_registry::fusion_spec_registrations()
}

#[cfg(not(target_arch = "wasm32"))]
static RESIDENCY_POLICY_MAP: OnceCell<HashMap<String, ResidencyPolicy>> = OnceCell::new();

#[cfg(not(target_arch = "wasm32"))]
fn build_residency_policy_map() -> HashMap<String, ResidencyPolicy> {
    let mut map = HashMap::new();
    for spec in builtin_gpu_specs() {
        map.insert(spec.name.to_ascii_lowercase(), spec.residency);
    }
    map
}

/// Return the declared residency policy for a builtin's GPU implementation.
///
/// This is used at the runtime/auto-offload boundary to decide whether GPU-resident
/// arguments must be gathered to host (`GatherImmediately`) or can remain device-resident.
pub fn builtin_residency_policy(name: &str) -> Option<ResidencyPolicy> {
    #[cfg(target_arch = "wasm32")]
    {
        wasm_registry::residency_policy(name)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let map = RESIDENCY_POLICY_MAP.get_or_init(build_residency_policy_map);
        map.get(&name.to_ascii_lowercase()).copied()
    }
}

impl fmt::Debug for BuiltinFusionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinFusionSpec")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("emits_nan", &self.emits_nan)
            .finish()
    }
}
