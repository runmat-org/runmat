use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::Digest;
use runmat_hir::FunctionId;
use runmat_runtime::build_runtime_error;
use runmat_runtime::context::RuntimeContext;
use runmat_vm::{
    Bytecode, BytecodeSourceRef, FunctionRegistry, SelectedExecutableAttachmentError,
    SelectedExecutableProduct, SelectedUnitInteropReceipt,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::foreign_artifacts::{selected_unit_interop_receipt, SelectedForeignPreparation};
use crate::session::source_evaluation::{static_capture, SelectedStaticTarget};
use crate::{ExecutableUnit, RunError};

fn source_error(error: impl ToString) -> RunError {
    RunError::Runtime(
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:ExecutableSourceIdentity")
            .build(),
    )
}

fn attachment_error(error: SelectedExecutableAttachmentError) -> RunError {
    RunError::Runtime(
        build_runtime_error(error.to_string())
            .with_identifier("RunMat:ExecutableSourceIdentity")
            .with_source(error)
            .build(),
    )
}

/// Canonical definition publication validates the exact executable body,
/// immutable origin, definition owner, and source content before replacing
/// its request-local source ID with SourcePool's durable one. Record that
/// validated pair as an alias of the same portable ordinal without issuing a
/// second portable binding or changing normal source lookup.
pub(super) fn attach_published_aliases(
    unit: &ExecutableUnit,
    runtime: &RuntimeContext,
    current: &HashSet<FunctionId>,
    executable: &FunctionRegistry,
    publication: &mut FunctionRegistry,
) -> Result<(), RunError> {
    let program = unit
        .revision()
        .program_revision
        .identity_digest()
        .map_err(source_error)?;
    let selected = current
        .iter()
        .copied()
        .filter(|id| {
            executable
                .get_current(*id)
                .and_then(|body| body.binding_origin)
                .is_some_and(|origin| origin.program == program)
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(());
    }
    let owner = selected
        .iter()
        .find_map(|id| {
            executable
                .get_current(*id)
                .and_then(|body| body.selected_source_owner.as_ref())
        })
        .ok_or_else(|| source_error("selected executable has no admitted source owner"))?;
    if selected.iter().any(|id| {
        executable
            .get_current(*id)
            .and_then(|body| body.selected_source_owner.as_ref())
            .is_none_or(|candidate| !candidate.same_issuance(owner))
    }) {
        return Err(source_error(
            "selected executable bodies do not share the admitted source owner",
        ));
    }
    let published = selected
        .into_iter()
        .filter(|id| publication.get_current(*id).is_some())
        .map(|id| {
            let issued = executable
                .get_current(id)
                .and_then(|body| body.source_id)
                .and_then(BytecodeSourceRef::runtime)
                .ok_or("selected executable body has no issued source")?;
            let durable = publication
                .get_current(id)
                .and_then(|body| body.source_id)
                .and_then(BytecodeSourceRef::runtime)
                .ok_or("published definition has no durable source")?;
            Ok((id, issued, durable))
        })
        .collect::<Result<Vec<_>, &'static str>>()
        .map_err(source_error)?;
    let owner = owner
        .with_durable_aliases(
            runtime,
            published
                .iter()
                .map(|(_, issued, durable)| (*issued, *durable)),
        )
        .map_err(source_error)?;
    publication
        .attach_selected_program_source_owner(
            published.iter().map(|(id, _, _)| *id),
            program,
            &owner,
        )
        .map_err(source_error)
}

/// One exact code/source carrier is shared by the executing, durable, and
/// retained bytecode views of this compilation. Its local source port holds
/// only a weak evaluator plus the issuing invocation's immutable capability;
/// it cannot keep its own definition owner alive after the last handle.
pub(super) struct SelectedProductPublication<'a> {
    pub unit: &'a ExecutableUnit,
    pub current: &'a HashSet<FunctionId>,
    pub bytecode: &'a mut Bytecode,
    pub executing: &'a mut FunctionRegistry,
    pub durable: &'a mut FunctionRegistry,
}

/// The canonical bytes and their checked foreign/static/source capability
/// are admitted together before any registry receives the product.
pub(super) struct SelectedProductAdmission<'a> {
    pub bytes: Arc<[u8]>,
    pub static_targets: &'a [SelectedStaticTarget],
    pub foreign_basis: Option<Digest>,
    pub dynamic_port: runmat_vm::DynamicEvalAdmissionPort,
    #[cfg(not(target_arch = "wasm32"))]
    pub prepared: Option<&'a SelectedForeignPreparation>,
}

pub(super) fn attach_selected_executable_product(
    publication: SelectedProductPublication<'_>,
    admission: SelectedProductAdmission<'_>,
) -> Result<Option<std::rc::Rc<SelectedUnitInteropReceipt>>, RunError> {
    let SelectedProductPublication {
        unit,
        current,
        bytecode,
        executing,
        durable,
    } = publication;
    let SelectedProductAdmission {
        bytes,
        static_targets,
        foreign_basis,
        dynamic_port,
        #[cfg(not(target_arch = "wasm32"))]
        prepared,
    } = admission;
    let program = unit
        .revision()
        .program_revision
        .identity_digest()
        .map_err(source_error)?;
    let ids = current
        .iter()
        .copied()
        .filter(|id| {
            executing
                .get_current(*id)
                .and_then(|body| body.binding_origin)
                .is_some_and(|origin| origin.program == program)
        })
        .collect::<Vec<_>>();
    let source_owner = ids
        .iter()
        .find_map(|id| {
            durable
                .get_current(*id)
                .and_then(|body| body.selected_source_owner.as_ref())
        })
        .or_else(|| {
            ids.iter().find_map(|id| {
                executing
                    .get_current(*id)
                    .and_then(|body| body.selected_source_owner.as_ref())
            })
        })
        .ok_or_else(|| source_error("selected code has no issued source owner"))?
        .clone();
    #[cfg(not(target_arch = "wasm32"))]
    let receipt = selected_unit_interop_receipt(Arc::clone(&bytes), &source_owner, prepared)
        .map_err(source_error)?;
    #[cfg(target_arch = "wasm32")]
    let receipt = None;
    let product = SelectedExecutableProduct::admit_with_basis_receipt_and_dynamic_admission(
        bytes,
        source_owner,
        foreign_basis,
        receipt.clone(),
        Some(dynamic_port),
    )
    .map_err(source_error)?;
    let capture = static_capture(Rc::clone(&product), static_targets).map_err(source_error)?;
    executing
        .attach_current_selected_executable_product(ids.iter().copied(), product.clone())
        .map_err(attachment_error)?;
    executing
        .attach_current_selected_static_capture(ids.iter().copied(), &capture)
        .map_err(source_error)?;
    durable
        .attach_published_selected_executable_product(
            ids.iter()
                .copied()
                .filter(|id| durable.get_current(*id).is_some())
                .collect::<Vec<_>>(),
            executing,
            product.clone(),
        )
        .map_err(attachment_error)?;
    durable
        .attach_current_selected_static_capture(
            ids.iter()
                .copied()
                .filter(|id| durable.get_current(*id).is_some())
                .collect::<Vec<_>>(),
            &capture,
        )
        .map_err(source_error)?;
    bytecode
        .function_registry
        .attach_current_selected_executable_product(ids.iter().copied(), product)
        .map_err(attachment_error)?;
    bytecode
        .function_registry
        .attach_current_selected_static_capture(ids.iter().copied(), &capture)
        .map_err(source_error)?;
    for id in ids {
        if let (Some(selected), Some(bound)) = (
            bytecode.function_registry.get_current(id),
            bytecode.bound_functions.get_mut(&id),
        ) {
            if Arc::ptr_eq(&selected.definition_owner, &bound.definition_owner)
                && selected.binding_origin == bound.binding_origin
            {
                bound.selected_executable_product = selected.selected_executable_product.clone();
                bound.selected_static_capture = selected.selected_static_capture.clone();
            }
        }
    }
    Ok(receipt)
}
