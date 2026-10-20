use runmat_gc_api::GcHandle;
use runmat_types::MemberAccess;

use crate::{
    ExactHostAllocationPermit, ExactHostNumericAllocationError, SharedClassIdentity, Utf8Text,
};

// -------- Class registry (scaffolding) --------
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicPropertyDef {
    pub name: Utf8Text,
    pub defining_class: SharedClassIdentity,
    pub metadata_handle: Option<GcHandle>,
    pub get_access: MemberAccess,
    pub set_access: MemberAccess,
    pub dependent: bool,
    pub hidden: bool,
    pub transient: bool,
    pub non_copyable: bool,
    pub abort_set: bool,
    pub set_observable: bool,
    pub get_observable: bool,
    pub description: Utf8Text,
}

#[derive(Debug)]
pub struct DynamicPropertyAdmissionError {
    definition: DynamicPropertyDef,
    source: ExactHostNumericAllocationError,
}

impl DynamicPropertyAdmissionError {
    pub fn source(&self) -> ExactHostNumericAllocationError {
        self.source
    }
    pub fn into_definition(self) -> DynamicPropertyDef {
        self.definition
    }
}

impl DynamicPropertyDef {
    pub fn new(name: String, defining_class: impl Into<SharedClassIdentity>) -> Self {
        Self {
            name: name.into(),
            defining_class: defining_class.into(),
            metadata_handle: None,
            get_access: MemberAccess::Public,
            set_access: MemberAccess::Public,
            dependent: false,
            hidden: false,
            transient: false,
            non_copyable: false,
            abort_set: false,
            set_observable: false,
            get_observable: false,
            description: Utf8Text::default(),
        }
    }

    pub fn try_admit_exact(
        self,
        permit: &dyn ExactHostAllocationPermit,
    ) -> Result<Self, DynamicPropertyAdmissionError> {
        let source = self
            .name
            .clone()
            .try_admit_exact(permit)
            .map(|_| ())
            .map_err(|error| error.source())
            .and_then(|()| {
                self.defining_class
                    .clone()
                    .try_admit_exact(permit)
                    .map(|_| ())
                    .map_err(|error| error.source())
            })
            .and_then(|()| {
                self.description
                    .clone()
                    .try_admit_exact(permit)
                    .map(|_| ())
                    .map_err(|error| error.source())
            });
        match source {
            Ok(()) => Ok(self),
            Err(source) => Err(DynamicPropertyAdmissionError {
                definition: self,
                source,
            }),
        }
    }
}
