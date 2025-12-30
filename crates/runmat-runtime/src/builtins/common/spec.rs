use std::fmt;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm_registry {
    #![allow(dead_code)]
    use super::{BuiltinFusionSpec, BuiltinGpuSpec};
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    static GPU_SPECS: Lazy<Mutex<Vec<&'static BuiltinGpuSpec>>> =
        Lazy::new(|| Mutex::new(Vec::new()));
    static FUSION_SPECS: Lazy<Mutex<Vec<&'static BuiltinFusionSpec>>> =
        Lazy::new(|| Mutex::new(Vec::new()));

    pub(crate) fn submit_gpu_spec(spec: &'static BuiltinGpuSpec) {
        GPU_SPECS
            .lock()
            .expect("gpu spec registry poisoned")
            .push(spec);
    }

    pub(crate) fn submit_fusion_spec(spec: &'static BuiltinFusionSpec) {
        FUSION_SPECS
            .lock()
            .expect("fusion spec registry poisoned")
            .push(spec);
    }

    pub(crate) fn gpu_specs() -> std::vec::IntoIter<&'static BuiltinGpuSpec> {
        GPU_SPECS
            .lock()
            .expect("gpu spec registry poisoned")
            .clone()
            .into_iter()
    }

    pub(crate) fn fusion_specs() -> std::vec::IntoIter<&'static BuiltinFusionSpec> {
        FUSION_SPECS
            .lock()
            .expect("fusion spec registry poisoned")
            .clone()
            .into_iter()
    }

    use super::DocTextInventory;
    static DOC_TEXTS: Lazy<Mutex<Vec<&'static DocTextInventory>>> =
        Lazy::new(|| Mutex::new(Vec::new()));

    pub(crate) fn submit_doc_text(entry: &'static DocTextInventory) {
        DOC_TEXTS
            .lock()
            .expect("doc text registry poisoned")
            .push(entry);
    }

    pub(crate) fn doc_texts() -> std::vec::IntoIter<&'static DocTextInventory> {
        DOC_TEXTS
            .lock()
            .expect("doc text registry poisoned")
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
}

/// Inventory wrapper for fusion specs.
pub struct FusionSpecInventory {
    pub spec: &'static BuiltinFusionSpec,
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

impl fmt::Debug for BuiltinFusionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinFusionSpec")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("emits_nan", &self.emits_nan)
            .finish()
    }
}

// Documentation text inventory (only populated when doc_export feature is enabled)
pub struct DocTextInventory {
    pub name: &'static str,
    pub text: &'static str,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "doc_export"))]
inventory::collect!(DocTextInventory);

#[cfg(all(not(target_arch = "wasm32"), feature = "doc_export"))]
pub fn builtin_doc_texts() -> impl Iterator<Item = &'static DocTextInventory> {
    inventory::iter::<DocTextInventory>()
}

#[cfg(all(target_arch = "wasm32", feature = "doc_export"))]
pub fn builtin_doc_texts() -> std::vec::IntoIter<&'static DocTextInventory> {
    wasm_registry::doc_texts()
}

#[cfg(not(feature = "doc_export"))]
pub fn builtin_doc_texts() -> std::iter::Empty<&'static DocTextInventory> {
    std::iter::empty()
}
