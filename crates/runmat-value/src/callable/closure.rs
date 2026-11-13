use std::ops::Deref;

use crate::{LexicalEnvironment, OutputListStorage, Utf8Text, Value};

use super::CallableInstanceId;

/// Semantic callable data. This is read-only through a constructed Closure;
/// changing its target or captures requires a new physical instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosureData {
    pub function_name: Utf8Text,
    pub binding: Option<crate::BoundFunctionBinding>,
    /// Compiler/source-owned classification; display spelling is never authority.
    pub source_role: runmat_types::FunctionSourceRole,
    /// Compiler-registered origin retained with the callable across sessions.
    pub source_file: Option<Utf8Text>,
    pub captures: OutputListStorage,
    /// A live ancestor link for nested functions or anonymous frames carrying
    /// a descendant's shared binding. Anonymous captures remain by value.
    pub lexical_environment: Option<LexicalEnvironment>,
}

#[derive(Debug, Clone)]
pub struct Closure {
    data: crate::immutable_metadata::ImmutableMetadata<ClosureData>,
    instance: CallableInstanceId,
}

#[derive(Debug)]
pub enum ClosureHeaderError {
    CapturesNotShared,
    Allocation(crate::ExactHostNumericAllocationError),
}

impl Closure {
    /// Constructs only the original immutable header. Captures must already
    /// have their canonical shared owner; this never detaches or copies them.
    pub fn try_from_data_with_exact_header(
        data: ClosureData,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<Self, ClosureHeaderError> {
        if !data.captures.is_shallow_clone() {
            return Err(ClosureHeaderError::CapturesNotShared);
        }
        let data = crate::immutable_metadata::ImmutableMetadata::try_new_exact(data, permit)
            .map_err(ClosureHeaderError::Allocation)?;
        Ok(Self {
            data,
            instance: CallableInstanceId::mint(),
        })
    }

    pub fn new(
        function_name: String,
        binding: Option<crate::BoundFunctionBinding>,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
        captures: Vec<Value>,
        lexical_environment: Option<LexicalEnvironment>,
    ) -> Self {
        Self::from_data(ClosureData {
            function_name: function_name.into(),
            binding,
            source_role,
            source_file: source_file.map(Utf8Text::new),
            captures: OutputListStorage::shared_unbounded(captures),
            lexical_environment,
        })
    }

    pub fn from_data(mut data: ClosureData) -> Self {
        if !data.captures.is_shallow_clone() {
            let owned = data
                .captures
                .try_into_owned()
                .expect("owned capture storage must transfer uniquely");
            let values = owned
                .try_into_unbounded_vec()
                .expect("ordinary captures must be unclaimed");
            data.captures = OutputListStorage::shared_unbounded(values);
        }
        Self {
            data: crate::immutable_metadata::ImmutableMetadata::new(data),
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn into_data(self) -> ClosureData {
        self.data.into_inner()
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }

    /// A resolved binding changes the target. Mint a new instance so an
    /// unbound clone cannot retain the same ID with different semantic data.
    pub fn bind_resolved_unbounded(
        self,
        binding: crate::BoundFunctionBinding,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
    ) -> Self {
        assert!(
            self.binding.is_none(),
            "only an unresolved closure may be refined"
        );
        let data = self.resolved_data(binding, source_role, source_file.map(Utf8Text::new));
        Self::from_data(data)
    }

    fn resolved_data(
        &self,
        binding: crate::BoundFunctionBinding,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<Utf8Text>,
    ) -> ClosureData {
        assert!(
            self.binding.is_none(),
            "only an unresolved closure may be refined"
        );
        let mut data = self.data.get().clone();
        data.binding = Some(binding);
        if data.source_role == runmat_types::FunctionSourceRole::Unknown {
            data.source_role = source_role;
        }
        if data.source_file.is_none() {
            data.source_file = source_file;
        }
        data
    }

    pub fn resolved_header_bytes() -> usize {
        crate::immutable_metadata::ImmutableMetadata::<ClosureData>::requested_header_bytes()
    }

    /// Refine immutable callable metadata under the caller's real allocation
    /// permit. Text/source owners and captures retain their independent claims.
    /// No shared generation is mutated and no child metadata is copied.
    pub fn try_bind_resolved(
        &self,
        binding: crate::BoundFunctionBinding,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<Utf8Text>,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<Self, crate::ExactHostNumericAllocationError> {
        let data = self.resolved_data(binding, source_role, source_file);
        let data = crate::immutable_metadata::ImmutableMetadata::try_new_exact(data, permit)?;
        Ok(Self {
            data,
            instance: CallableInstanceId::mint(),
        })
    }

    pub fn bound_function(&self) -> Option<usize> {
        self.binding.as_ref().map(|binding| binding.function)
    }

    pub(crate) fn metadata_quote(
        &self,
    ) -> Result<(usize, usize), crate::ExactHostNumericAllocationError> {
        let (mut bytes, mut claims) =
            super::metadata::quote(&self.function_name, self.source_file.as_ref())?;
        if let Some(header) = self.data.unclaimed_bytes(|_| Ok(0))? {
            bytes = bytes
                .checked_add(header)
                .ok_or(crate::ExactHostNumericAllocationError::LayoutOverflow)?;
            claims += 1;
        }
        Ok((bytes, claims))
    }
    pub(crate) fn try_promote_metadata(
        &self,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<(), crate::ExactHostNumericAllocationError> {
        super::metadata::promote(&self.function_name, self.source_file.as_ref(), permit)?;
        self.data.promote(permit, |_| Ok(0))
    }
}

impl Deref for Closure {
    type Target = ClosureData;

    fn deref(&self) -> &Self::Target {
        self.data.get()
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.data.get() == other.data.get()
    }
}
