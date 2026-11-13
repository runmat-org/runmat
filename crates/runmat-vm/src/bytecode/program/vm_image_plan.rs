//! Exact selected multi-store image region installed beside its bytecode.
//! The decline right has no host-route or completed-call authority.

use runmat_accelerate::graph::AccelGraph;
use runmat_execution::Digest;
use runmat_runtime::context::{
    RuntimeContext, VmFusionCompletionRight, VmFusionDeclineRight, VmFusionInstalledRegion,
};
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use super::super::multi_store::{CapturedImageInputs, SelectedImageRegion, SelectedImageRegionSet};
use super::vm_fusion_plan::{vm_fusion_executable_identity, vm_fusion_semantic_identity};
use super::{Bytecode, FunctionBytecode, FunctionRegistry};

pub(super) type SelectedImageViews =
    HashMap<runmat_hir::FunctionId, Option<(Rc<SelectedImageRegionSet>, Digest)>>;

/// Function execution views are finalized from one immutable registry
/// generation. Entrypoint and caller-supplied Bytecode views remain unsealed
/// and take the checked source-key path below.
#[derive(Clone, Default)]
pub(crate) enum SealedFunctionImage {
    #[default]
    Unsealed,
    Absent {
        registry: std::rc::Weak<super::FunctionRegistryData>,
        generation: u64,
    },
    Selected {
        regions: Rc<SelectedImageRegionSet>,
        executable: Digest,
        registry: std::rc::Weak<super::FunctionRegistryData>,
        generation: u64,
    },
}

impl std::fmt::Debug for SealedFunctionImage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsealed => formatter.write_str("Unsealed"),
            Self::Absent { .. } => formatter.write_str("Absent"),
            Self::Selected { .. } => formatter.write_str("Selected(..)"),
        }
    }
}

impl FunctionRegistry {
    /// Rebuild selected descriptors once after a registry generation has
    /// reached its compiler, artifact, or session publication boundary.
    /// Registry edits invalidate these derived views before re-publication.
    pub fn finalize_selected_image_views(&self) {
        if self.static_overlay.is_some() {
            return;
        }
        for body in self.data.functions.values() {
            if body.fusion_stage_projection.is_some() {
                let _ = body.execution_bytecode(self);
            }
        }
    }

    /// Only the exact body in this copy-on-write registry can reuse the view.
    /// DerefMut clears the entire derived inventory before any authoring edit.
    pub(super) fn finalize_selected_image_for(
        &self,
        body: &FunctionBytecode,
        view: &Bytecode,
    ) -> SealedFunctionImage {
        if !self
            .data
            .functions
            .get(&body.function)
            .is_some_and(|selected| std::ptr::eq(selected, body))
            || self.static_overlay.is_some()
            || view.active_function != Some(body.function)
            || view.fusion_metadata.stage_projection.is_none()
        {
            return SealedFunctionImage::Unsealed;
        }
        if let Some(selected) = self.data.selected_image_views.borrow().get(&body.function) {
            return selected.as_ref().map_or_else(
                || SealedFunctionImage::Absent {
                    registry: std::rc::Rc::downgrade(&self.data),
                    generation: self.data.selected_image_generation,
                },
                |(selected, executable)| SealedFunctionImage::Selected {
                    regions: Rc::clone(selected),
                    executable: *executable,
                    registry: std::rc::Rc::downgrade(&self.data),
                    generation: self.data.selected_image_generation,
                },
            );
        }
        if body.validate_fusion_stage_projection().is_err() {
            return SealedFunctionImage::Unsealed;
        }
        let selected = match view.selected_image_regions() {
            Some(regions) => {
                let Some(executable) = vm_fusion_executable_identity(view) else {
                    return SealedFunctionImage::Unsealed;
                };
                Some((Rc::new(regions), executable))
            }
            None => None,
        };
        self.data
            .selected_image_views
            .borrow_mut()
            .insert(body.function, selected.clone());
        selected.map_or_else(
            || SealedFunctionImage::Absent {
                registry: std::rc::Rc::downgrade(&self.data),
                generation: self.data.selected_image_generation,
            },
            |(regions, executable)| SealedFunctionImage::Selected {
                regions,
                executable,
                registry: std::rc::Rc::downgrade(&self.data),
                generation: self.data.selected_image_generation,
            },
        )
    }
}

pub(crate) struct VmImagePlanInstallation {
    selected: Rc<SelectedImageRegionSet>,
    installed: Vec<VmFusionInstalledRegion>,
}

/// This cache is only a derived acceleration of the exact selected bytecode.
/// Its key includes serialized compiler facts and the local-only selected
/// function/private view. A clone or artifact hydration never imports its
/// answer without rechecking those facts.
#[derive(Clone)]
pub(crate) struct SelectedImageCacheEntry {
    identity: Digest,
    selected: Option<Rc<SelectedImageRegionSet>>,
}

/// A clone may share the cache cell, but a later edit to either executable
/// view changes its checked identity and cannot reuse the other's answer.
#[derive(Default)]
pub(crate) struct SelectedImageCache(RefCell<Option<SelectedImageCacheEntry>>);

impl PartialEq for SelectedImageCache {
    fn eq(&self, _other: &Self) -> bool {
        // Derived acceleration is never part of selected-source identity.
        true
    }
}

impl std::fmt::Debug for SelectedImageCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SelectedImageCache(..)")
    }
}

impl std::fmt::Debug for SelectedImageCacheEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectedImageCacheEntry")
            .field("identity", &self.identity)
            .field(
                "regions",
                &self.selected.as_ref().map(|value| value.regions.len()),
            )
            .finish()
    }
}

struct OrderedEntries<'a, K, V>(&'a BTreeMap<K, V>);

impl<K: Serialize, V: Serialize> Serialize for OrderedEntries<'_, K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries = serializer.serialize_seq(Some(self.0.len()))?;
        for entry in self.0 {
            entries.serialize_element(&entry)?;
        }
        entries.end()
    }
}

fn selected_image_owner_identity(bytecode: &Bytecode) -> Option<Digest> {
    bytecode.fusion_metadata.stage_projection.as_ref()?;
    // Only immutable selected-source facts belong in this key. In particular,
    // the VM changes initially_unassigned_slots for an invocation; hashing the
    // whole Bytecode would rebuild the graph on every function call. The
    // Selected-body identity and absence of an entrypoint body matter, but
    // serializing the entire registry on every invocation would defeat the
    // once-per-executable graph finalization. Retaining its Rc in this cache
    // would also cycle through a function's stage projection. Hash only the
    // selected body and whether an entrypoint body exists. The fields stream
    // into the digest without materializing a second bytecode payload.
    let selected_body = bytecode
        .active_function
        .and_then(|function| bytecode.function_registry.get(function));
    let entrypoint_has_body = bytecode.layout.as_ref().is_some_and(|layout| {
        layout
            .entrypoints
            .values()
            .any(|entry| bytecode.function_registry.get(entry.target).is_some())
    });
    Digest::sha256_serialized(&(
        "runmat.vm-image-selected-view.v2",
        (
            &bytecode.instructions,
            &bytecode.catch_scopes,
            &bytecode.instr_spans,
            OrderedEntries(&bytecode.resume_points),
            bytecode.source_id,
            bytecode.var_count,
            &bytecode.var_types,
            &bytecode.var_names,
        ),
        (
            &bytecode.layout,
            &bytecode.fusion_metadata,
            selected_body,
            entrypoint_has_body,
            bytecode.active_function,
            &bytecode.active_class_method_owner,
            &bytecode.function_private_layout,
        ),
    ))
    .ok()
}

pub(crate) struct VmImageSelected<'a> {
    pub(crate) graph: &'a AccelGraph,
    pub(crate) region: &'a SelectedImageRegion,
    pub(crate) decline: VmFusionDeclineRight,
    pub(crate) completion: Option<VmFusionCompletionRight>,
}

impl Bytecode {
    /// A copied function view can change invocation-local unassigned slots,
    /// but any edit to the source/layout/registry owner forfeits its cached
    /// selected image. This check borrows the immutable definition and never
    /// serializes or reconstructs a graph on an ordinary call.
    fn matches_finalized_function(
        &self,
        owner: &std::rc::Weak<super::FunctionRegistryData>,
        generation: u64,
    ) -> bool {
        let Some(registry) = owner.upgrade() else {
            return false;
        };
        if !std::rc::Rc::ptr_eq(&registry, &self.function_registry.data)
            || registry.selected_image_generation != generation
            || self.function_registry.static_overlay.is_some()
            || self.accel_graph.is_some()
            || !self.fusion_groups.is_empty()
            || !self.var_types.is_empty()
            || !self.bound_functions.is_empty()
            || self.async_metadata.mir_spawn_site_count != 0
            || !self.async_metadata.mir_spawn_sites.is_empty()
            || self.async_metadata.mir_await_site_count != 0
            || !self.async_metadata.mir_await_sites.is_empty()
        {
            return false;
        }
        let Some(function) = self.active_function else {
            return false;
        };
        let Some(body) = registry.functions.get(&function) else {
            return false;
        };
        let Some(projection) = body.fusion_stage_projection.as_ref() else {
            return false;
        };
        let Some(actual_projection) = self.fusion_metadata.stage_projection.as_ref() else {
            return false;
        };
        if !Arc::ptr_eq(actual_projection, projection)
            || self.instructions.len() != body.instructions.len()
            || !self
                .instructions
                .iter()
                .zip(&body.instructions)
                .all(|(actual, expected)| match (actual, expected) {
                    (
                        super::super::Instr::LoadConst(actual),
                        super::super::Instr::LoadConst(expected),
                    ) => actual.to_bits() == expected.to_bits(),
                    (
                        super::super::Instr::LoadComplex(actual_re, actual_im),
                        super::super::Instr::LoadComplex(expected_re, expected_im),
                    ) => {
                        actual_re.to_bits() == expected_re.to_bits()
                            && actual_im.to_bits() == expected_im.to_bits()
                    }
                    _ => actual == expected,
                })
            || self.catch_scopes != body.catch_scopes
            || self.instr_spans != body.instr_spans
            || self.call_arg_spans != body.call_arg_spans
            || self.coverage_sites != body.coverage_sites
            || self.resume_points != body.resume_points
            || self.source_id != body.source_id
            || self.var_count != body.var_count
            || self.var_names != body.var_names
            || self.active_class_method_owner != body.class_method_owner
            || self.regions != body.regions
            || self.parfor_regions != body.parfor_regions
            || self.spmd_regions != body.spmd_regions
            || self.distributed_values != body.distributed_values
            || self.collective_contracts != body.collective_contracts
        {
            return false;
        }
        let Some(expected_layout) = projection.selected_function_layout.as_ref() else {
            return false;
        };
        let Some(actual_layout) = self.layout.as_ref() else {
            return false;
        };
        if actual_layout.functions.len() != 1
            || actual_layout.functions.get(&function) != Some(expected_layout)
            || !actual_layout.entrypoints.is_empty()
            || !actual_layout.storage_bindings.is_empty()
        {
            return false;
        }
        let Some(scaffold) = self.function_registry.admitted_fusion_scaffold(body) else {
            return false;
        };
        self.function_private_layout == scaffold.private_layout
            && self.fusion_metadata.mir_fusion_signal_count == scaffold.signal_count
            && self.fusion_metadata.mir_fusion_candidate_group_count == scaffold.candidates.len()
            && self.fusion_metadata.mir_fusion_candidate_groups == scaffold.candidates
            && self.fusion_metadata.instruction_window_count == scaffold.windows.len()
            && self.fusion_metadata.instruction_windows == scaffold.windows
    }

    /// Finalize the immutable graph/store/publication descriptor for this
    /// exact executable view. Source edits, remapping, and decoded artifacts
    /// can only reuse a matching key; runtime rights are minted separately.
    pub(crate) fn sealed_selected_image_regions(&self) -> Option<Rc<SelectedImageRegionSet>> {
        let identity = selected_image_owner_identity(self)?;
        if let Ok(cache) = self.fusion_metadata.selected_image.0.try_borrow() {
            if let Some(entry) = cache.as_ref() {
                if entry.identity == identity {
                    return entry.selected.clone();
                }
            }
        }
        let selected = self.selected_image_regions().map(Rc::new);
        if let Ok(mut cache) = self.fusion_metadata.selected_image.0.try_borrow_mut() {
            *cache = Some(SelectedImageCacheEntry {
                identity,
                selected: selected.clone(),
            });
        }
        selected
    }

    pub(crate) fn prepare_vm_image_plan_installation(
        &self,
        runtime: &RuntimeContext,
    ) -> Option<VmImagePlanInstallation> {
        let (selected, executable) = match &self.sealed_function_image {
            SealedFunctionImage::Selected {
                regions,
                executable,
                registry,
                generation,
            } if self.matches_finalized_function(registry, *generation) => {
                (Rc::clone(regions), *executable)
            }
            SealedFunctionImage::Absent {
                registry,
                generation,
            } if self.matches_finalized_function(registry, *generation) => return None,
            _ => (
                self.sealed_selected_image_regions()?,
                vm_fusion_executable_identity(self)?,
            ),
        };
        let installed = selected
            .regions
            .iter()
            .map(|region| {
                let semantic = vm_fusion_semantic_identity(executable, &region.plan)?;
                VmFusionInstalledRegion::install(
                    runtime,
                    semantic,
                    region.plan.group.id,
                    region.plan.group.span.start,
                    region.plan.group.span.end,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        (!installed.is_empty()).then_some(VmImagePlanInstallation {
            selected,
            installed,
        })
    }
}

impl VmImagePlanInstallation {
    pub(crate) fn region_starting_at(
        &self,
        pc: usize,
        runtime: &RuntimeContext,
    ) -> Option<VmImageSelected<'_>> {
        let index = self
            .selected
            .regions
            .iter()
            .position(|region| region.publication.pcs().start == pc)?;
        let region = self.selected.regions.get(index)?;
        let installed = self.installed.get(index)?;
        if !installed.matches_group(
            region.plan.group.id,
            region.plan.group.span.start,
            region.plan.group.span.end,
        ) {
            return None;
        }
        Some(VmImageSelected {
            graph: &self.selected.graph,
            region,
            decline: installed.decline_right(runtime)?,
            completion: installed.right(runtime),
        })
    }

    pub(crate) fn capture_inputs(
        &self,
        region: &SelectedImageRegion,
        vars: &crate::interpreter::vars::InterpreterVars,
        context: &super::ExecutionContext,
        runtime: &RuntimeContext,
    ) -> Result<
        Option<CapturedImageInputs>,
        runmat_runtime::context::host_allocation::HostAllocationError,
    > {
        self.selected.capture_inputs(region, vars, context, runtime)
    }
}
