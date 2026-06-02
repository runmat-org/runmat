use runmat_hir::{
    BindingId, BindingStorage, EntrypointId, FunctionAbi, FunctionId, HirAssembly,
    WorkspaceExportPolicy, WorkspaceVisibility,
};
use runmat_mir::{MirAssembly, MirLocalId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VmSlotId(pub usize);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmAssemblyLayout {
    pub functions: HashMap<FunctionId, VmFunctionLayout>,
    pub entrypoints: HashMap<EntrypointId, VmEntrypointLayout>,
    pub storage_bindings: HashMap<BindingId, VmStorageBinding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmFunctionLayout {
    pub function: FunctionId,
    pub display_name: String,
    pub frame_abi: VmFrameAbi,
    pub binding_slots: HashMap<BindingId, VmSlotId>,
    pub mir_local_slots: HashMap<MirLocalId, VmSlotId>,
    pub captures: Vec<VmCaptureSlot>,
    pub local_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmFrameAbi {
    pub fixed_inputs: Vec<VmSlotId>,
    pub varargin: Option<VmSlotId>,
    pub fixed_outputs: Vec<VmSlotId>,
    pub varargout: Option<VmSlotId>,
    pub implicit_nargin: Option<VmSlotId>,
    pub implicit_nargout: Option<VmSlotId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmCaptureSlot {
    pub binding: BindingId,
    pub from_function: FunctionId,
    pub slot: VmSlotId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmEntrypointLayout {
    pub entrypoint: EntrypointId,
    pub target: FunctionId,
    pub workspace_export: WorkspaceExportPolicy,
    pub exports: Vec<VmWorkspaceExport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmWorkspaceExport {
    pub binding: BindingId,
    pub name: String,
    pub slot: VmSlotId,
    pub visibility: WorkspaceVisibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmStorageBinding {
    pub binding: BindingId,
    pub name: String,
    pub storage: BindingStorage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    MissingFunction(FunctionId),
    MissingMirBody(FunctionId),
    MissingBinding(BindingId),
    MissingBindingSlot {
        function: FunctionId,
        binding: BindingId,
    },
}

pub fn derive_layout(
    hir: &HirAssembly,
    mir: &MirAssembly,
) -> Result<VmAssemblyLayout, LayoutError> {
    let mut functions = HashMap::new();
    for function in &hir.functions {
        let body = mir
            .bodies
            .get(&function.id)
            .ok_or(LayoutError::MissingMirBody(function.id))?;
        functions.insert(function.id, derive_function_layout(hir, function, body)?);
    }

    let mut entrypoints = HashMap::new();
    for entrypoint in &hir.entrypoints {
        let function_layout = functions
            .get(&entrypoint.target)
            .ok_or(LayoutError::MissingFunction(entrypoint.target))?;
        let exports = match entrypoint.policy.workspace_export {
            WorkspaceExportPolicy::ExportTopLevelBindings | WorkspaceExportPolicy::HostDefined => {
                workspace_exports_for_function(hir, function_layout)?
            }
            WorkspaceExportPolicy::ReturnFunctionOutputs => Vec::new(),
        };
        entrypoints.insert(
            entrypoint.id,
            VmEntrypointLayout {
                entrypoint: entrypoint.id,
                target: entrypoint.target,
                workspace_export: entrypoint.policy.workspace_export.clone(),
                exports,
            },
        );
    }

    let storage_bindings = hir
        .bindings
        .iter()
        .filter(|binding| binding.storage != BindingStorage::Lexical)
        .map(|binding| {
            (
                binding.id,
                VmStorageBinding {
                    binding: binding.id,
                    name: binding.name.0.clone(),
                    storage: binding.storage.clone(),
                },
            )
        })
        .collect();

    Ok(VmAssemblyLayout {
        functions,
        entrypoints,
        storage_bindings,
    })
}

fn derive_function_layout(
    hir: &HirAssembly,
    function: &runmat_hir::HirFunction,
    body: &runmat_mir::MirBody,
) -> Result<VmFunctionLayout, LayoutError> {
    let mut binding_slots = HashMap::new();
    let mut next_slot = 0usize;

    let frame_abi = allocate_frame_abi(&function.abi, &mut binding_slots, &mut next_slot);

    for capture in &function.captures {
        allocate_binding_slot(capture.binding, &mut binding_slots, &mut next_slot);
    }
    for binding in &function.locals {
        allocate_binding_slot(*binding, &mut binding_slots, &mut next_slot);
    }
    for local in &body.locals {
        if let Some(binding) = local.binding {
            allocate_binding_slot(binding, &mut binding_slots, &mut next_slot);
        }
    }

    let mut mir_local_slots = HashMap::new();
    for local in &body.locals {
        let slot = if let Some(binding) = local.binding {
            *binding_slots
                .get(&binding)
                .ok_or(LayoutError::MissingBindingSlot {
                    function: function.id,
                    binding,
                })?
        } else {
            let slot = VmSlotId(next_slot);
            next_slot += 1;
            slot
        };
        mir_local_slots.insert(local.id, slot);
    }

    let captures = function
        .captures
        .iter()
        .map(|capture| {
            let slot =
                *binding_slots
                    .get(&capture.binding)
                    .ok_or(LayoutError::MissingBindingSlot {
                        function: function.id,
                        binding: capture.binding,
                    })?;
            Ok(VmCaptureSlot {
                binding: capture.binding,
                from_function: capture.from_function,
                slot,
            })
        })
        .collect::<Result<_, LayoutError>>()?;

    for binding in binding_slots.keys() {
        if hir.bindings.get(binding.0).map(|b| b.id) != Some(*binding) {
            return Err(LayoutError::MissingBinding(*binding));
        }
    }

    Ok(VmFunctionLayout {
        function: function.id,
        display_name: function.name.0.clone(),
        frame_abi,
        binding_slots,
        mir_local_slots,
        captures,
        local_count: next_slot,
    })
}

fn allocate_frame_abi(
    abi: &FunctionAbi,
    binding_slots: &mut HashMap<BindingId, VmSlotId>,
    next_slot: &mut usize,
) -> VmFrameAbi {
    VmFrameAbi {
        fixed_inputs: abi
            .fixed_inputs
            .iter()
            .map(|binding| allocate_binding_slot(*binding, binding_slots, next_slot))
            .collect(),
        varargin: abi
            .varargin
            .map(|binding| allocate_binding_slot(binding, binding_slots, next_slot)),
        fixed_outputs: abi
            .fixed_outputs
            .iter()
            .map(|binding| allocate_binding_slot(*binding, binding_slots, next_slot))
            .collect(),
        varargout: abi
            .varargout
            .map(|binding| allocate_binding_slot(binding, binding_slots, next_slot)),
        implicit_nargin: abi
            .implicit_nargin
            .map(|binding| allocate_binding_slot(binding, binding_slots, next_slot)),
        implicit_nargout: abi
            .implicit_nargout
            .map(|binding| allocate_binding_slot(binding, binding_slots, next_slot)),
    }
}

fn allocate_binding_slot(
    binding: BindingId,
    binding_slots: &mut HashMap<BindingId, VmSlotId>,
    next_slot: &mut usize,
) -> VmSlotId {
    if let Some(slot) = binding_slots.get(&binding) {
        return *slot;
    }
    let slot = VmSlotId(*next_slot);
    *next_slot += 1;
    binding_slots.insert(binding, slot);
    slot
}

fn workspace_exports_for_function(
    hir: &HirAssembly,
    function_layout: &VmFunctionLayout,
) -> Result<Vec<VmWorkspaceExport>, LayoutError> {
    let mut exports = Vec::new();
    for (binding, slot) in &function_layout.binding_slots {
        let hir_binding = hir
            .bindings
            .get(binding.0)
            .ok_or(LayoutError::MissingBinding(*binding))?;
        match hir_binding.workspace_visibility {
            WorkspaceVisibility::TopLevel
            | WorkspaceVisibility::ModuleVisible
            | WorkspaceVisibility::ImplicitAns => exports.push(VmWorkspaceExport {
                binding: *binding,
                name: hir_binding.name.0.clone(),
                slot: *slot,
                visibility: hir_binding.workspace_visibility.clone(),
            }),
            WorkspaceVisibility::Hidden => {}
        }
    }
    exports.sort_by_key(|export| export.slot);
    Ok(exports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::{
        BindingName, BindingOwner, BindingRole, CapturedBinding, EntrypointOrigin,
        EntrypointPolicy, FunctionKind, FunctionModifiers, FunctionName, HirBinding, HirBlock,
        HirEntrypoint, HirFunction, ModuleId, Span,
    };
    use runmat_mir::{MirBody, MirLocal, MirLocalKind};

    #[test]
    fn layout_reuses_shared_input_output_binding_slot() {
        let function = FunctionId(0);
        let binding = BindingId(0);
        let assembly = HirAssembly {
            functions: vec![HirFunction {
                id: function,
                module: ModuleId(0),
                parent: None,
                enclosing_class: None,
                name: FunctionName("f".into()),
                kind: FunctionKind::Named,
                params: vec![binding],
                outputs: vec![binding],
                abi: FunctionAbi {
                    fixed_inputs: vec![binding],
                    varargin: None,
                    fixed_outputs: vec![binding],
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                argument_validations: Vec::new(),
                locals: Vec::new(),
                captures: Vec::new(),
                modifiers: FunctionModifiers::default(),
                body: HirBlock { statements: vec![] },
                span: Span::default(),
            }],
            bindings: vec![HirBinding {
                id: binding,
                owner: BindingOwner::Function(function),
                name: BindingName("x".into()),
                role: BindingRole::Parameter,
                storage: BindingStorage::Lexical,
                workspace_visibility: WorkspaceVisibility::Hidden,
                declared_span: Span::default(),
            }],
            ..HirAssembly::default()
        };
        let mir = MirAssembly {
            bodies: HashMap::from([(
                function,
                MirBody {
                    function,
                    abi: assembly.functions[0].abi.clone(),
                    locals: vec![MirLocal {
                        id: MirLocalId(0),
                        binding: Some(binding),
                        kind: MirLocalKind::Parameter,
                        span: Span::default(),
                    }],
                    blocks: vec![],
                },
            )]),
        };

        let layout = derive_layout(&assembly, &mir).expect("layout");
        let function_layout = &layout.functions[&function];
        assert_eq!(function_layout.frame_abi.fixed_inputs, vec![VmSlotId(0)]);
        assert_eq!(function_layout.frame_abi.fixed_outputs, vec![VmSlotId(0)]);
        assert_eq!(function_layout.binding_slots[&binding], VmSlotId(0));
        assert_eq!(function_layout.mir_local_slots[&MirLocalId(0)], VmSlotId(0));
        assert_eq!(function_layout.local_count, 1);
    }

    #[test]
    fn entrypoint_exports_workspace_visible_bindings() {
        let function = FunctionId(0);
        let hidden = BindingId(0);
        let visible = BindingId(1);
        let entrypoint = EntrypointId(0);
        let assembly = HirAssembly {
            functions: vec![HirFunction {
                id: function,
                module: ModuleId(0),
                parent: None,
                enclosing_class: None,
                name: FunctionName("entry".into()),
                kind: FunctionKind::SyntheticEntrypoint,
                params: Vec::new(),
                outputs: Vec::new(),
                abi: FunctionAbi {
                    fixed_inputs: Vec::new(),
                    varargin: None,
                    fixed_outputs: Vec::new(),
                    varargout: None,
                    implicit_nargin: None,
                    implicit_nargout: None,
                },
                argument_validations: Vec::new(),
                locals: vec![hidden, visible],
                captures: vec![CapturedBinding {
                    binding: hidden,
                    from_function: function,
                }],
                modifiers: FunctionModifiers::default(),
                body: HirBlock { statements: vec![] },
                span: Span::default(),
            }],
            bindings: vec![
                HirBinding {
                    id: hidden,
                    owner: BindingOwner::Function(function),
                    name: BindingName("tmp".into()),
                    role: BindingRole::Local,
                    storage: BindingStorage::Lexical,
                    workspace_visibility: WorkspaceVisibility::Hidden,
                    declared_span: Span::default(),
                },
                HirBinding {
                    id: visible,
                    owner: BindingOwner::Function(function),
                    name: BindingName("x".into()),
                    role: BindingRole::Local,
                    storage: BindingStorage::Lexical,
                    workspace_visibility: WorkspaceVisibility::TopLevel,
                    declared_span: Span::default(),
                },
            ],
            entrypoints: vec![HirEntrypoint {
                id: entrypoint,
                name: None,
                target: function,
                origin: EntrypointOrigin::HostSynthetic,
                policy: EntrypointPolicy {
                    workspace_export: WorkspaceExportPolicy::ExportTopLevelBindings,
                    top_level_await: false,
                },
            }],
            ..HirAssembly::default()
        };
        let mir = MirAssembly {
            bodies: HashMap::from([(
                function,
                MirBody {
                    function,
                    abi: assembly.functions[0].abi.clone(),
                    locals: vec![
                        MirLocal {
                            id: MirLocalId(0),
                            binding: Some(hidden),
                            kind: MirLocalKind::Binding,
                            span: Span::default(),
                        },
                        MirLocal {
                            id: MirLocalId(1),
                            binding: Some(visible),
                            kind: MirLocalKind::Binding,
                            span: Span::default(),
                        },
                    ],
                    blocks: vec![],
                },
            )]),
        };

        let layout = derive_layout(&assembly, &mir).expect("layout");
        let exports = &layout.entrypoints[&entrypoint].exports;
        assert_eq!(exports.len(), 1);
        assert_eq!(exports[0].binding, visible);
        assert_eq!(exports[0].name, "x");
    }
}
