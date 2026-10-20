use std::ops::Deref;

use super::{BoundFunctionAuthority, CallableInstanceId};

#[derive(Clone, Debug, PartialEq)]
pub struct BoundFunctionHandleData {
    pub name: crate::Utf8Text,
    pub function: usize,
    pub authority: BoundFunctionAuthority,
    pub source_role: runmat_types::FunctionSourceRole,
    pub source_file: Option<crate::Utf8Text>,
}

/// Bound execution authority is independent of this handle's physical
/// instance. A cloned handle retains its instance; a newly issued one does not.
#[derive(Clone, Debug)]
pub struct BoundFunctionHandle {
    data: crate::immutable_metadata::ImmutableMetadata<BoundFunctionHandleData>,
    instance: CallableInstanceId,
}

impl BoundFunctionHandle {
    pub fn new(
        name: String,
        function: usize,
        authority: BoundFunctionAuthority,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
    ) -> Self {
        Self::from_data(BoundFunctionHandleData {
            name: name.into(),
            function,
            authority,
            source_role,
            source_file: source_file.map(crate::Utf8Text::new),
        })
    }

    pub fn from_data(data: BoundFunctionHandleData) -> Self {
        Self {
            data: crate::immutable_metadata::ImmutableMetadata::new(data),
            instance: CallableInstanceId::mint(),
        }
    }

    pub fn into_data(self) -> BoundFunctionHandleData {
        self.data.into_inner()
    }

    pub fn requested_header_bytes() -> usize {
        crate::immutable_metadata::ImmutableMetadata::<BoundFunctionHandleData>::requested_header_bytes()
    }

    /// Admits only the original header; child text and authority retain their
    /// own existing owners and claims.
    pub fn try_from_data_with_exact_header(
        data: BoundFunctionHandleData,
        permit: &dyn crate::ExactHostAllocationPermit,
    ) -> Result<Self, crate::ExactHostNumericAllocationError> {
        Ok(Self {
            data: crate::immutable_metadata::ImmutableMetadata::try_new_exact(data, permit)?,
            instance: CallableInstanceId::mint(),
        })
    }

    pub fn instance_id(&self) -> CallableInstanceId {
        self.instance
    }

    pub(crate) fn metadata_quote(
        &self,
    ) -> Result<(usize, usize), crate::ExactHostNumericAllocationError> {
        let (mut bytes, mut claims) =
            super::metadata::quote(&self.name, self.source_file.as_ref())?;
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
        super::metadata::promote(&self.name, self.source_file.as_ref(), permit)?;
        self.data.promote(permit, |_| Ok(0))
    }
}

impl Deref for BoundFunctionHandle {
    type Target = BoundFunctionHandleData;

    fn deref(&self) -> &Self::Target {
        self.data.get()
    }
}

impl PartialEq for BoundFunctionHandle {
    fn eq(&self, other: &Self) -> bool {
        self.data.get() == other.data.get()
    }
}
