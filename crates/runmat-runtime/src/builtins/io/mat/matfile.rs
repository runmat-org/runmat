//! MATLAB-compatible `matfile` compatibility object.
use runmat_types::MemberAccess;

use std::collections::HashMap;
use std::path::PathBuf;

use runmat_builtins::{BuiltinErrorDescriptor, MatFileOptionName};
use runmat_macros::runtime_builtin;
use runmat_value::{ObjectInstance, Value};

use super::authority::{issue_matfile, resolve_matfile, MatFsPath};
use super::cooperation::MatCodecCooperation;
use super::encoding::{encode_workspace_to_mat_bytes_for_owner, MatEncodingOwner};
use super::load::read_mat_file_for_builtin_on_with_resource_liveness;
use crate::context::host_effect::{
    HostEffectResourceLivenessLease, HostEffectWorkload, MatAssignmentSuffix, MatDecodeOutput,
    MatDecodeSelection, MatEncodeMode, MatEncodeOutput, MatHandleControl,
};
use crate::execution::RootedValue;
use crate::{
    build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError,
    OBJECT_SUBSASGN_METHOD, OBJECT_SUBSREF_METHOD,
};

const BUILTIN_NAME: &str = "matfile";
pub(super) use runmat_builtins::MATFILE_CLASS;
const MATFILE_SUBSREF: &str = "matlab.io.MatFile.subsref";
const MATFILE_SUBSASGN: &str = "matlab.io.MatFile.subsasgn";
mod admission;
pub(super) const PROPERTIES_FIELD: &str = "Properties";
pub(super) const SOURCE_FIELD: &str = "Source";
pub(super) const WRITABLE_FIELD: &str = "Writable";

static MATFILE_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(MATFILE_CLASS);

use runmat_builtins::{
    MATFILE_ERROR_INVALID_ARGUMENT, MATFILE_ERROR_INVALID_OPTION, MATFILE_ERROR_IO,
    MATFILE_ERROR_SELECTION, MATFILE_ERROR_UNSUPPORTED,
};

#[runtime_builtin(
    name = "matfile",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    ensure_matfile_class_registered();
    let controls = prepare_writable_options(rest)?;
    let path = normalise_path(&filename)?;
    let public_path = path.to_string_lossy().into_owned();
    let source = std::sync::Arc::new(MatFsPath::capture(path).map_err(|err| {
        matfile_error_with_source(
            &MATFILE_ERROR_IO,
            format!("matfile: failed to resolve path: {err}"),
            err,
        )
    })?);
    let control_facts = controls.identity_controls();
    let workload = HostEffectWorkload::MatHandle {
        source: source.path(),
        controls: &control_facts,
    };
    let receipt = super::admission::select(runtime.as_ref(), &source, BUILTIN_NAME, workload)?;
    let result = async {
        let mut cooperation = MatCodecCooperation::new(
            runtime.as_ref(),
            Some(&source),
            BUILTIN_NAME,
            &MATFILE_ERROR_IO,
        )
        .with_resource_liveness(
            receipt
                .as_ref()
                .map(|selected| selected.resource_liveness_lease()),
        );
        let writable = controls.resolve(&mut cooperation).await?;
        cooperation.completed(0).await?;
        let reference = issue_matfile(
            runtime.as_ref(),
            source.clone(),
            public_path.clone(),
            writable,
        )?;
        matfile_object(runtime.as_ref(), &public_path, writable, reference)
    }
    .await;
    super::admission::finish(receipt, &result);
    result
}

#[runtime_builtin(
    name = "matlab.io.MatFile.subsref",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsref(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
) -> BuiltinResult<RootedValue> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let first = path.steps().first().expect("parsed nonempty MAT path");
    let payload = first.selector_value()?;
    if first.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSREF}: only whole-variable member access is supported"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSREF)?;
    reject_internal_field(&field)?;
    let object = into_matfile_object(obj.clone(), MATFILE_SUBSREF)?;
    let owner = resolve_matfile(runtime.as_ref(), &object)?;
    let suffix = admission::suffix_digest(&path)
        .map(MatAssignmentSuffix::Exact)
        .unwrap_or(MatAssignmentSuffix::Unavailable);
    let workload = HostEffectWorkload::MatDecode {
        source: owner.source.path(),
        selection: MatDecodeSelection::MemberPath {
            member: &field,
            suffix,
        },
        output: MatDecodeOutput::SingleMember,
    };
    let receipt =
        super::admission::select(runtime.as_ref(), &owner.source, MATFILE_SUBSREF, workload)?;
    let liveness = receipt
        .as_ref()
        .map(|selected| selected.resource_liveness_lease());
    let read_runtime = runtime.clone();
    let result = crate::object::protocol::execute_owned_subsref_with_context(
        runtime.as_ref(),
        obj,
        path,
        move |obj, step| async move {
            check_resource_liveness(liveness.as_ref())?;
            let value =
                matfile_read_step(read_runtime.as_ref(), obj, step, liveness.clone()).await?;
            check_resource_liveness(liveness.as_ref())?;
            Ok(value)
        },
        None,
    )
    .await;
    let result = match result {
        Ok(value) => receipt
            .as_ref()
            .map_or(Ok(()), |selected| selected.checkpoint())
            .map(|()| value),
        Err(error) => Err(error),
    };
    super::admission::finish(receipt, &result);
    result
}

mod steps;
use steps::{matfile_read_step, matfile_write_step};

#[runtime_builtin(
    name = "matlab.io.MatFile.subsasgn",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::mat::matfile"
)]
pub async fn matfile_subsasgn(
    runtime: Option<crate::context::RuntimeContext>,
    obj: Value,
    subscript: Value,
    rhs: Value,
) -> BuiltinResult<RootedValue> {
    ensure_matfile_class_registered();
    let path = crate::object::indexing::parse_standard_substruct(&subscript)?;
    let first = path.steps().first().expect("parsed nonempty MAT path");
    let payload = first.selector_value()?;
    if first.kind() != crate::object::indexing::ObjectIndexKind::Member {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: partial MAT-file assignment is not supported yet"),
        ));
    }
    let field = member_name(&payload, MATFILE_SUBSASGN)?;
    if field == PROPERTIES_FIELD || field == SOURCE_FIELD || field == WRITABLE_FIELD {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            format!("{MATFILE_SUBSASGN}: '{field}' is read-only"),
        ));
    }
    reject_internal_field(&field)?;
    validate_variable_name(&field, MATFILE_SUBSASGN)?;
    let object = into_matfile_object(obj.clone(), MATFILE_SUBSASGN)?;
    let owner = resolve_matfile(runtime.as_ref(), &object)?;
    if !owner.writable {
        return Err(matfile_error_with(
            &MATFILE_ERROR_UNSUPPORTED,
            "matfile: object is not writable; construct it with matfile(filename, 'Writable', true)",
        ));
    }
    let suffix = admission::suffix_digest(&path)
        .map(MatAssignmentSuffix::Exact)
        .unwrap_or(MatAssignmentSuffix::Unavailable);
    let workload = HostEffectWorkload::MatEncode {
        destination: owner.source.path(),
        mode: MatEncodeMode::AssignMember {
            member: &field,
            suffix,
        },
        output: MatEncodeOutput::UpdatedObject,
    };
    let receipt =
        super::admission::select(runtime.as_ref(), &owner.source, MATFILE_SUBSASGN, workload)?;
    let liveness = receipt
        .as_ref()
        .map(|selected| selected.resource_liveness_lease());
    let read_runtime = runtime.clone();
    let write_runtime = runtime.clone();
    let read_liveness = liveness.clone();
    let result = crate::object::protocol::execute_owned_subsasgn_with_context(
        runtime.as_ref(),
        obj,
        path,
        vec![rhs],
        move |obj, step| async move {
            check_resource_liveness(read_liveness.as_ref())?;
            let result =
                matfile_read_step(read_runtime.as_ref(), obj, step, read_liveness.clone()).await?;
            check_resource_liveness(read_liveness.as_ref())?;
            Ok(result)
        },
        move |obj, step, mut values| async move {
            let rhs = values
                .pop()
                .ok_or_else(|| matfile_error("matfile assignment value is missing"))?;
            matfile_write_step(write_runtime.as_ref(), obj, step, rhs, liveness).await
        },
        None,
    )
    .await;
    let result = match result {
        Ok(value) => receipt
            .as_ref()
            .map_or(Ok(()), |selected| selected.checkpoint())
            .map(|()| value),
        Err(error) => Err(error),
    };
    super::admission::finish(receipt, &result);
    result
}

fn check_resource_liveness(
    lease: Option<&std::sync::Arc<dyn HostEffectResourceLivenessLease>>,
) -> BuiltinResult<()> {
    if lease.is_some_and(|lease| !lease.is_current()) {
        Err(crate::context::host_effect::stale_resource())
    } else {
        Ok(())
    }
}

pub fn ensure_matfile_class_registered() {
    MATFILE_CLASS_REGISTERED.ensure(|| {
        let mut properties = HashMap::new();
        properties.insert(
            PROPERTIES_FIELD.into(),
            crate::class_registry::RuntimeProperty {
                name: PROPERTIES_FIELD.into(),
                is_static: false,
                is_constant: false,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        );

        let mut methods = HashMap::new();
        methods.insert(
            OBJECT_SUBSREF_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSREF_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSREF.into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );
        methods.insert(
            OBJECT_SUBSASGN_METHOD.into(),
            crate::class_registry::RuntimeMethod {
                name: OBJECT_SUBSASGN_METHOD.into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: MemberAccess::Public,
                function_name: MATFILE_SUBSASGN.into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        );

        crate::class_registry::register_builtin_class(
            MATFILE_CLASS,
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: MATFILE_CLASS.into(),
                parent: None,
                properties,
                methods,
            },
        );
    });
}

mod receiver;
pub(super) use receiver::matfile_error;
use receiver::{
    into_matfile_object, matfile_object, member_name, normalise_path, prepare_writable_options,
    reject_internal_field, validate_variable_name,
};

fn matfile_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

pub(in crate::builtins::io::mat) fn matfile_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
#[path = "matfile/tests.rs"]
mod tests;
