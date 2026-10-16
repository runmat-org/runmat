use std::fmt;

mod admission;
mod unary_math;
pub use unary_math::{
    unary_math_expression, write_unary_math_expression, write_unary_math_first_expression,
    write_unary_math_scalar_expression,
};
mod canonical;
mod host_resources;
pub use host_resources::{
    FusionHostOutputDomain, FusionHostResourceBound, FusionHostResourceCallback,
    FusionHostResourceInput, FusionHostResourceQuery,
};
mod epilogue;
mod reduction;
mod reduction_sign;
mod structural;
pub use reduction::*;
pub use reduction_sign::FusionReductionSignProjection;
mod registration_index;
mod selected;
pub use admission::*;
pub use canonical::{
    canonical_fusion_declaration, canonical_fusion_spec, CanonicalFusionDeclaration,
};
pub use epilogue::{fusion_epilogue_operation, FusionEpilogueOperation};
pub use selected::{
    selected_builtin_binding, selected_fusion_binding, SelectedFusionBinding,
    SelectedFusionIdentity,
};
pub use structural::FusionStructuralOperation;

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
    static FUSION_INDEX: Lazy<Mutex<Vec<FusionSpecRegistration>>> =
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
        let mut index = FUSION_INDEX.lock().expect("fusion spec registry poisoned");
        super::registration_index::insert(&mut index, registration);
    }

    pub(super) fn with_fusion_index<R>(visit: impl FnOnce(&[FusionSpecRegistration]) -> R) -> R {
        let index = FUSION_INDEX.lock().expect("fusion spec registry poisoned");
        visit(&index)
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

pub use runmat_builtins::ReductionNaN;

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
    pub inputs: &'a [FusionExprInput<'a>],
    pub constants: &'a [&'a str],
}

/// One operand supplied to the canonical declaration writer. A measured
/// operand carries only its exact future source length; it cannot be emitted.
#[derive(Clone, Copy)]
pub struct FusionExprInput<'a> {
    text: Option<&'a str>,
    bytes: usize,
}

impl<'a> FusionExprInput<'a> {
    pub fn text(text: &'a str) -> Self {
        Self {
            text: Some(text),
            bytes: text.len(),
        }
    }

    pub fn measured_bytes(bytes: usize) -> Self {
        Self { text: None, bytes }
    }

    pub fn byte_len(self) -> usize {
        self.bytes
    }

    fn is_measured(self) -> bool {
        self.text.is_none()
    }
}

impl std::fmt::Display for FusionExprInput<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.text {
            Some(text) => formatter.write_str(text),
            None => {
                let mut bytes = self.bytes;
                // Format-based declarations use the same writer in both
                // passes. The count sink discards these stack-backed bytes;
                // no operand source String is constructed for admission.
                const FILL: &str =
                    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
                while bytes >= FILL.len() {
                    formatter.write_str(FILL)?;
                    bytes -= FILL.len();
                }
                formatter.write_str(&FILL[..bytes])
            }
        }
    }
}

/// One expression implementation owns both its checked byte count and its
/// emitted bytes for every registered fusion template.
#[derive(Clone, Copy)]
pub struct FusionExprBuilder {
    write: fn(&FusionExprContext, &mut FusionExprWriter<'_>) -> Result<(), FusionError>,
}

/// Infallible String output, preceded by checked source-byte accounting.
/// The counting pass never constructs an intermediate expression String.
pub struct FusionExprWriter<'a> {
    target: Option<&'a mut String>,
    bytes: usize,
}

impl FusionExprWriter<'_> {
    pub fn counting() -> Self {
        Self {
            target: None,
            bytes: 0,
        }
    }

    pub fn emitting(target: &mut String) -> FusionExprWriter<'_> {
        FusionExprWriter {
            target: Some(target),
            bytes: 0,
        }
    }

    pub fn push(&mut self, fragment: impl fmt::Display) -> Result<(), FusionError> {
        self.write_formatted(format_args!("{fragment}"))
    }

    fn push_str_fragment(&mut self, fragment: &str) -> Result<(), FusionError> {
        self.bytes = self
            .bytes
            .checked_add(fragment.len())
            .ok_or(FusionError::ExpressionTooLarge)?;
        if let Some(target) = self.target.as_mut() {
            target.push_str(fragment);
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn write_formatted(&mut self, arguments: fmt::Arguments<'_>) -> Result<(), FusionError> {
        fmt::write(self, arguments).map_err(|_| FusionError::ExpressionTooLarge)
    }
}

impl fmt::Write for FusionExprWriter<'_> {
    fn write_str(&mut self, fragment: &str) -> fmt::Result {
        self.push_str_fragment(fragment).map_err(|_| fmt::Error)
    }
}

/// A pure real scalar operator's source is composed once for both admission
/// and emission. Domain-sensitive power remains on its separate typed route.
pub fn write_real_binary_expression(
    operation: runmat_types::OperatorKind,
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
) -> Result<(), FusionError> {
    let lhs = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
    let rhs = context.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
    let infix = match operation {
        runmat_types::OperatorKind::Add => " + ",
        runmat_types::OperatorKind::Subtract => " - ",
        runmat_types::OperatorKind::ElementwiseMultiply => " * ",
        runmat_types::OperatorKind::ElementwiseDivide => " / ",
        runmat_types::OperatorKind::ElementwiseLeftDivide => " / ",
        _ => {
            return Err(FusionError::Message(
                "operator has no counted binary expression",
            ))
        }
    };
    writer.push("(")?;
    if operation == runmat_types::OperatorKind::ElementwiseLeftDivide {
        writer.push(rhs)?;
        writer.push(infix)?;
        writer.push(lhs)?;
    } else {
        writer.push(lhs)?;
        writer.push(infix)?;
        writer.push(rhs)?;
    }
    writer.push(")")
}

pub fn write_real_cast_expression(
    class: runmat_types::NumericClass,
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
) -> Result<(), FusionError> {
    if context.inputs.len() != 1 {
        return Err(FusionError::Message(
            "unary fusion requires exactly one data operand",
        ));
    }
    let input = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
    match (class, context.scalar_ty) {
        (runmat_types::NumericClass::Single, ScalarType::F32) => writer.push(input),
        (runmat_types::NumericClass::Single, ScalarType::F64) => {
            writer.push("f64(f32(")?;
            writer.push(input)?;
            writer.push("))")
        }
        (runmat_types::NumericClass::Double, ScalarType::F64) => {
            writer.push("(")?;
            writer.push(input)?;
            writer.push(")")
        }
        _ => Err(FusionError::UnsupportedPrecision(context.scalar_ty)),
    }
}

pub fn write_real_power_expression(
    context: &FusionExprContext<'_>,
    writer: &mut FusionExprWriter<'_>,
) -> Result<(), FusionError> {
    let base = context.inputs.first().ok_or(FusionError::MissingInput(0))?;
    let exponent = context.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
    let helper = FusionMathHelper::RealPower.entrypoint(context.scalar_ty)?;
    writer.push(helper)?;
    writer.push("(")?;
    writer.push(base)?;
    writer.push(", ")?;
    writer.push(exponent)?;
    writer.push(")")
}

impl FusionExprBuilder {
    pub const fn new(
        write: fn(&FusionExprContext, &mut FusionExprWriter<'_>) -> Result<(), FusionError>,
    ) -> Self {
        Self { write }
    }

    pub fn measured_bytes(self, context: &FusionExprContext<'_>) -> Result<usize, FusionError> {
        let mut sink = FusionExprWriter::counting();
        (self.write)(context, &mut sink)?;
        Ok(sink.bytes())
    }

    /// Count before producer allocation from indexed operand lengths. Current
    /// scalar declarations admit at most two operands; a wider declaration
    /// has no finite source certificate from this API.
    pub fn measured_bytes_from_lengths(
        self,
        scalar_ty: ScalarType,
        lengths: &[usize],
    ) -> Result<usize, FusionError> {
        if lengths.len() > 2 {
            return Err(FusionError::Message(
                "fusion expression has no finite operand-length geometry",
            ));
        }
        let mut inputs = [FusionExprInput::measured_bytes(0); 2];
        for (input, length) in inputs.iter_mut().zip(lengths.iter().copied()) {
            *input = FusionExprInput::measured_bytes(length);
        }
        self.measured_bytes(&FusionExprContext {
            scalar_ty,
            inputs: &inputs[..lengths.len()],
            constants: &[],
        })
    }

    pub fn render(self, context: &FusionExprContext<'_>) -> Result<String, FusionError> {
        if context.inputs.iter().any(|input| input.is_measured()) {
            return Err(FusionError::Message(
                "measured fusion operand cannot be published as source",
            ));
        }
        let expected = self.measured_bytes(context)?;
        let mut output = String::new();
        output
            .try_reserve_exact(expected)
            .map_err(|_| FusionError::ExpressionTooLarge)?;
        let mut sink = FusionExprWriter::emitting(&mut output);
        (self.write)(context, &mut sink)?;
        if sink.bytes() != expected {
            return Err(FusionError::ExpressionTooLarge);
        }
        Ok(output)
    }
}

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
    ExpressionTooLarge,
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
            FusionError::ExpressionTooLarge => {
                f.write_str("fusion expression byte count overflows")
            }
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
    pub semantics: Option<&'static FusionSemanticDeclaration>,
    pub structural: Option<FusionStructuralOperation>,
    pub epilogue: Option<FusionEpilogueOperation>,
    pub reduction: Option<&'static FusionReductionDeclaration>,
    pub host_resources: Option<FusionHostResourceCallback>,
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
    pub semantics: Option<&'static FusionSemanticDeclaration>,
    pub structural: Option<FusionStructuralOperation>,
    pub epilogue: Option<FusionEpilogueOperation>,
    pub reduction: Option<&'static FusionReductionDeclaration>,
    pub host_resources: Option<FusionHostResourceCallback>,
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
    #[cfg(test)]
    registration_index::INVENTORY_TRAVERSALS.with(|count| count.set(count.get() + 1));
    inventory::iter::<FusionSpecInventory>().map(|entry| FusionSpecRegistration {
        spec: entry.spec,
        semantics: entry.semantics,
        structural: entry.structural,
        epilogue: entry.epilogue,
        reduction: entry.reduction,
        host_resources: entry.host_resources,
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
