use std::collections::BTreeSet;

use super::super::*;
use crate::{NativeCodegenError, NativeCodegenResult};

pub(super) fn verify(function: &NativeFunction) -> NativeCodegenResult<()> {
    let error = |code, message| NativeCodegenError::new(code, message).at_function(function.id);

    super::ensure_sorted_unique_by(
        "native.ir.binding_locals",
        &function.binding_locals,
        |entry| entry.binding,
    )
    .map_err(|error| error.at_function(function.id))?;
    if function.binding_locals.len()
        != function
            .locals
            .iter()
            .filter(|local| local.binding.is_some())
            .count()
        || function.binding_locals.iter().any(|entry| {
            function
                .local(entry.local)
                .is_none_or(|local| local.binding != Some(entry.binding))
        })
    {
        return Err(error(
            "native.ir.binding_locals",
            "binding lookup must exactly index every bound Native local",
        ));
    }

    let mut capture_bindings = BTreeSet::new();
    if function
        .closure_captures
        .iter()
        .any(|capture| !capture_bindings.insert(capture.binding))
    {
        return Err(error(
            "native.ir.closure_captures",
            "closure capture bindings must be unique and compiler ordered",
        ));
    }
    if function.closure_captures.iter().any(|capture| {
        function
            .local(capture.local)
            .is_none_or(|local| local.binding != Some(capture.binding))
            || match capture.disposition {
                NativeClosureCaptureDisposition::ByValue => function
                    .local(capture.local)
                    .is_none_or(|local| local.kind != NativeLocalKind::Capture),
                NativeClosureCaptureDisposition::Lexical => {
                    !function.lexical_bindings.iter().any(|binding| {
                        binding.binding == capture.binding
                            && binding.local == capture.local
                            && binding.kind == NativeLexicalBindingKind::Inherited
                    })
                }
                NativeClosureCaptureDisposition::TransitSnapshot => {
                    function
                        .local(capture.local)
                        .is_none_or(|local| local.kind != NativeLocalKind::Capture)
                        || !function.lexical_bindings.iter().any(|binding| {
                            binding.binding == capture.binding
                                && binding.local == capture.local
                                && binding.kind == NativeLexicalBindingKind::Transit
                        })
                }
                NativeClosureCaptureDisposition::InheritedStorage => {
                    !function.inherited_storage_bindings.iter().any(|binding| {
                        binding.binding == capture.binding && binding.local == capture.local
                    })
                }
            }
    }) || function.locals.iter().any(|local| {
        local.kind == NativeLocalKind::Capture
            && local.binding.is_none_or(|binding| {
                !function
                    .closure_captures
                    .iter()
                    .any(|capture| capture.binding == binding && capture.local == local.id)
            })
    }) || function.lexical_bindings.iter().any(|shared| {
        matches!(
            shared.kind,
            NativeLexicalBindingKind::Inherited | NativeLexicalBindingKind::Transit
        ) && !function.closure_captures.iter().any(|capture| {
            capture.binding == shared.binding
                && capture.local == shared.local
                && match shared.kind {
                    NativeLexicalBindingKind::Inherited => {
                        capture.disposition == NativeClosureCaptureDisposition::Lexical
                    }
                    NativeLexicalBindingKind::Transit => {
                        capture.disposition == NativeClosureCaptureDisposition::TransitSnapshot
                    }
                    NativeLexicalBindingKind::Owned => false,
                }
        })
    }) || function.inherited_storage_bindings.iter().any(|storage| {
        !function.closure_captures.iter().any(|capture| {
            capture.binding == storage.binding
                && capture.local == storage.local
                && capture.disposition == NativeClosureCaptureDisposition::InheritedStorage
        })
    }) {
        return Err(error(
            "native.ir.closure_captures",
            "closure capture ABI disagrees with its unique frame authority",
        ));
    }
    Ok(())
}
