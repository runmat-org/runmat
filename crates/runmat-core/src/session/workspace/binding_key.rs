//! Exact retained backing for one compiler workspace key.
//!
//! The public ABI remains `WorkspaceBindingKey`. This owner copies each nested
//! String/Vec with an exact capacity after a current-operation claim is issued;
//! moving a key between admitted table buckets never copies its deep backing.

use crate::abi::{GlobalScopeKey, SourceIdentity, WorkspaceBindingKey};
use runmat_hir::{BindingName, DefPath, DefPathSegment, PackageName, QualifiedName, SymbolName};
use runmat_value::{
    ExactHostAllocationClaim, ExactHostAllocationPermit, ExactHostNumericAllocationError,
};
use std::alloc::{alloc, Layout};
use uuid::Uuid;

#[derive(Debug)]
pub(in crate::session) struct RetainedBindingKey {
    key: WorkspaceBindingKey,
    // Declaration order ensures every nested key allocation retires first.
    _claim: Option<ExactHostAllocationClaim>,
}

/// Compiler origin facts retained by a publication observer while source
/// stores execute. This is separate from each published key's own claim.
#[derive(Debug)]
pub(super) struct RetainedPublicationSource {
    pub(super) source: Option<SourceIdentity>,
    pub(super) def_path: DefPath,
    _claim: Option<ExactHostAllocationClaim>,
}

impl RetainedPublicationSource {
    pub(super) fn requested_bytes(
        source: Option<&SourceIdentity>,
        source_name: &str,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let mut bytes = 0;
        if let Some(SourceIdentity::PathAndContentHash { path, hash }) = source {
            add_string(&mut bytes, path)?;
            add_string(&mut bytes, hash)?;
        }
        let stem = source_stem(source_name);
        add_string(&mut bytes, "workspace")?;
        add(
            &mut bytes,
            Layout::array::<SymbolName>(1)
                .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?,
        )?;
        add_string(&mut bytes, stem)?;
        add(
            &mut bytes,
            Layout::array::<DefPathSegment>(1)
                .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?,
        )?;
        add_string(&mut bytes, stem)?;
        Ok(bytes)
    }

    pub(super) fn prepare(
        source: Option<&SourceIdentity>,
        source_name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes(source, source_name)?)?;
        Self::build(source, source_name, Some(claim))
    }

    pub(super) fn unbounded(
        source: Option<&SourceIdentity>,
        source_name: &str,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        Self::build(source, source_name, None)
    }

    fn build(
        source: Option<&SourceIdentity>,
        source_name: &str,
        claim: Option<ExactHostAllocationClaim>,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let source = match source {
            Some(SourceIdentity::Interactive { session }) => {
                Some(SourceIdentity::Interactive { session: *session })
            }
            Some(SourceIdentity::PathAndContentHash { path, hash }) => {
                Some(SourceIdentity::PathAndContentHash {
                    path: exact_string(path)?,
                    hash: exact_string(hash)?,
                })
            }
            None => None,
        };
        let stem = source_stem(source_name);
        let mut module = exact_vec(1)?;
        module.push(SymbolName(exact_string(stem)?));
        let mut item = exact_vec(1)?;
        item.push(DefPathSegment::Function(SymbolName(exact_string(stem)?)));
        Ok(Self {
            source,
            def_path: DefPath {
                package: PackageName(exact_string("workspace")?),
                module: QualifiedName(module),
                item,
            },
            _claim: claim,
        })
    }
}

fn source_stem(source_name: &str) -> &str {
    std::path::Path::new(source_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.trim().is_empty())
        .unwrap_or("entrypoint")
}

impl RetainedBindingKey {
    pub(super) fn as_key(&self) -> &WorkspaceBindingKey {
        &self.key
    }

    pub(super) fn interactive_requested_bytes(
        name: &str,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let mut bytes = 0;
        add_string(&mut bytes, name)?;
        Ok(bytes)
    }

    pub(super) fn source_requested_bytes(
        source: Option<&SourceIdentity>,
        def_path: &DefPath,
        name: &str,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let Some(source) = source else {
            return Self::interactive_requested_bytes(name);
        };
        let mut bytes = 0;
        if let SourceIdentity::PathAndContentHash { path, hash } = source {
            add_string(&mut bytes, path)?;
            add_string(&mut bytes, hash)?;
        }
        add_def_path(&mut bytes, def_path)?;
        add_string(&mut bytes, name)?;
        Ok(bytes)
    }

    pub(super) fn prepare_interactive(
        session: Uuid,
        name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::interactive_requested_bytes(name)?)?;
        Ok(Self {
            key: WorkspaceBindingKey::Interactive {
                session,
                name: BindingName(exact_string(name)?),
            },
            _claim: Some(claim),
        })
    }

    pub(super) fn prepare_session_global(
        session: Uuid,
        name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::interactive_requested_bytes(name)?)?;
        Ok(Self {
            key: WorkspaceBindingKey::Global {
                scope: GlobalScopeKey::Session(session),
                name: BindingName(exact_string(name)?),
            },
            _claim: Some(claim),
        })
    }

    pub(super) fn session_global_retained_bytes(
        target: &runmat_runtime::workspace::GlobalBindingTarget,
    ) -> Result<u64, runmat_runtime::context::host_allocation::HostAllocationError> {
        use runmat_runtime::context::host_allocation::{HostAllocationError, HostAllocationPlan};
        match target {
            runmat_runtime::workspace::GlobalBindingTarget::Named(name) => {
                let bytes = Self::interactive_requested_bytes(name.as_str())
                    .map_err(HostAllocationError::Numeric)?;
                HostAllocationPlan::retained_exact_output_bytes(
                    u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow)?,
                    1,
                )
            }
            runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(_) => Ok(0),
        }
    }

    pub(super) fn prepare_session_global_target(
        session: Uuid,
        target: &runmat_runtime::workspace::GlobalBindingTarget,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        match target {
            runmat_runtime::workspace::GlobalBindingTarget::Named(name) => {
                Self::prepare_session_global(session, name.as_str(), permit)
            }
            runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(slot) => {
                // The surrounding admitted binding owns these inline fields;
                // this key has no independent string backing to claim.
                Ok(Self {
                    key: WorkspaceBindingKey::GlobalSlot {
                        scope: GlobalScopeKey::Session(session),
                        slot: *slot,
                    },
                    _claim: None,
                })
            }
        }
    }

    pub(super) fn prepare_source(
        source: Option<&SourceIdentity>,
        def_path: &DefPath,
        session: Uuid,
        name: &str,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let Some(source) = source else {
            return Self::prepare_interactive(session, name, permit);
        };
        let claim = permit.reserve_exact_claim(Self::source_requested_bytes(
            Some(source),
            def_path,
            name,
        )?)?;
        let source = match source {
            SourceIdentity::Interactive { session } => {
                SourceIdentity::Interactive { session: *session }
            }
            SourceIdentity::PathAndContentHash { path, hash } => {
                SourceIdentity::PathAndContentHash {
                    path: exact_string(path)?,
                    hash: exact_string(hash)?,
                }
            }
        };
        Ok(Self {
            key: WorkspaceBindingKey::SourceBinding {
                source,
                def_path: copy_def_path(def_path)?,
                binding: BindingName(exact_string(name)?),
            },
            _claim: Some(claim),
        })
    }

    pub(super) fn requested_bytes(
        key: &WorkspaceBindingKey,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let mut bytes = 0usize;
        match key {
            WorkspaceBindingKey::Interactive { name, .. } => add_string(&mut bytes, &name.0)?,
            WorkspaceBindingKey::SourceBinding {
                source,
                def_path,
                binding,
            } => {
                match source {
                    SourceIdentity::Interactive { .. } => {}
                    SourceIdentity::PathAndContentHash { path, hash } => {
                        add_string(&mut bytes, path)?;
                        add_string(&mut bytes, hash)?;
                    }
                }
                add_def_path(&mut bytes, def_path)?;
                add_string(&mut bytes, &binding.0)?;
            }
            WorkspaceBindingKey::Global { scope, name } => {
                if let GlobalScopeKey::Package(package) = scope {
                    add_string(&mut bytes, package)?;
                }
                add_string(&mut bytes, &name.0)?;
            }
            WorkspaceBindingKey::GlobalSlot { scope, .. } => {
                if let GlobalScopeKey::Package(package) = scope {
                    add_string(&mut bytes, package)?;
                }
            }
            WorkspaceBindingKey::Persistent { function, name } => {
                add_def_path(&mut bytes, function)?;
                add_string(&mut bytes, &name.0)?;
            }
        }
        Ok(bytes)
    }

    pub(super) fn prepare(
        key: &WorkspaceBindingKey,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, ExactHostNumericAllocationError> {
        let claim = permit.reserve_exact_claim(Self::requested_bytes(key)?)?;
        let key = copy_key(key)?;
        Ok(Self {
            key,
            _claim: Some(claim),
        })
    }

    pub(super) fn unbounded(key: WorkspaceBindingKey) -> Self {
        Self { key, _claim: None }
    }

    pub(in crate::session) fn into_parts(
        self,
    ) -> (WorkspaceBindingKey, Option<ExactHostAllocationClaim>) {
        (self.key, self._claim)
    }
}

fn add(bytes: &mut usize, layout: Layout) -> Result<(), ExactHostNumericAllocationError> {
    *bytes = bytes
        .checked_add(layout.size())
        .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
    Ok(())
}

fn add_string(bytes: &mut usize, value: &str) -> Result<(), ExactHostNumericAllocationError> {
    add(
        bytes,
        Layout::array::<u8>(value.len())
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?,
    )
}

fn add_def_path(bytes: &mut usize, path: &DefPath) -> Result<(), ExactHostNumericAllocationError> {
    add_string(bytes, &path.package.0)?;
    add(
        bytes,
        Layout::array::<SymbolName>(path.module.0.len())
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?,
    )?;
    for symbol in &path.module.0 {
        add_string(bytes, &symbol.0)?;
    }
    add(
        bytes,
        Layout::array::<DefPathSegment>(path.item.len())
            .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?,
    )?;
    for segment in &path.item {
        match segment {
            DefPathSegment::Function(symbol)
            | DefPathSegment::Class(symbol)
            | DefPathSegment::Method(symbol) => add_string(bytes, &symbol.0)?,
            DefPathSegment::ScriptSection { title, .. } => add_string(bytes, title)?,
        }
    }
    Ok(())
}

fn exact_string(value: &str) -> Result<String, ExactHostNumericAllocationError> {
    if value.is_empty() {
        return Ok(String::new());
    }
    let layout = Layout::array::<u8>(value.len())
        .map_err(|_| ExactHostNumericAllocationError::LayoutOverflow)?;
    // SAFETY: a successful allocation is exactly `value.len()` bytes. The
    // source slice is live and nonoverlapping; String uses the global allocator.
    let pointer = unsafe { alloc(layout) };
    if pointer.is_null() {
        return Err(ExactHostNumericAllocationError::AllocationFailed);
    }
    unsafe {
        pointer.copy_from_nonoverlapping(value.as_ptr(), value.len());
        Ok(String::from_raw_parts(pointer, value.len(), value.len()))
    }
}

fn exact_vec<T>(len: usize) -> Result<Vec<T>, ExactHostNumericAllocationError> {
    runmat_value::exact_capacity_vec(len)
}

fn copy_def_path(path: &DefPath) -> Result<DefPath, ExactHostNumericAllocationError> {
    let mut module = exact_vec(path.module.0.len())?;
    for symbol in &path.module.0 {
        module.push(SymbolName(exact_string(&symbol.0)?));
    }
    let mut item = exact_vec(path.item.len())?;
    for segment in &path.item {
        item.push(match segment {
            DefPathSegment::Function(symbol) => {
                DefPathSegment::Function(SymbolName(exact_string(&symbol.0)?))
            }
            DefPathSegment::Class(symbol) => {
                DefPathSegment::Class(SymbolName(exact_string(&symbol.0)?))
            }
            DefPathSegment::Method(symbol) => {
                DefPathSegment::Method(SymbolName(exact_string(&symbol.0)?))
            }
            DefPathSegment::ScriptSection { ordinal, title } => DefPathSegment::ScriptSection {
                ordinal: *ordinal,
                title: exact_string(title)?,
            },
        });
    }
    Ok(DefPath {
        package: PackageName(exact_string(&path.package.0)?),
        module: QualifiedName(module),
        item,
    })
}

fn copy_key(
    key: &WorkspaceBindingKey,
) -> Result<WorkspaceBindingKey, ExactHostNumericAllocationError> {
    Ok(match key {
        WorkspaceBindingKey::Interactive { session, name } => WorkspaceBindingKey::Interactive {
            session: *session,
            name: BindingName(exact_string(&name.0)?),
        },
        WorkspaceBindingKey::SourceBinding {
            source,
            def_path,
            binding,
        } => WorkspaceBindingKey::SourceBinding {
            source: match source {
                SourceIdentity::Interactive { session } => {
                    SourceIdentity::Interactive { session: *session }
                }
                SourceIdentity::PathAndContentHash { path, hash } => {
                    SourceIdentity::PathAndContentHash {
                        path: exact_string(path)?,
                        hash: exact_string(hash)?,
                    }
                }
            },
            def_path: copy_def_path(def_path)?,
            binding: BindingName(exact_string(&binding.0)?),
        },
        WorkspaceBindingKey::Global { scope, name } => WorkspaceBindingKey::Global {
            scope: match scope {
                GlobalScopeKey::Session(session) => GlobalScopeKey::Session(*session),
                GlobalScopeKey::Package(package) => GlobalScopeKey::Package(exact_string(package)?),
            },
            name: BindingName(exact_string(&name.0)?),
        },
        WorkspaceBindingKey::GlobalSlot { scope, slot } => WorkspaceBindingKey::GlobalSlot {
            scope: match scope {
                GlobalScopeKey::Session(session) => GlobalScopeKey::Session(*session),
                GlobalScopeKey::Package(package) => GlobalScopeKey::Package(exact_string(package)?),
            },
            slot: *slot,
        },
        WorkspaceBindingKey::Persistent { function, name } => WorkspaceBindingKey::Persistent {
            function: copy_def_path(function)?,
            name: BindingName(exact_string(&name.0)?),
        },
    })
}
