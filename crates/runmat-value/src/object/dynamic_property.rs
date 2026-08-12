use runmat_gc_api::GcHandle;

// -------- Class registry (scaffolding) --------
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Access {
    Public,
    Private,
    Protected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicPropertyDef {
    pub name: String,
    pub defining_class: String,
    pub metadata_handle: Option<GcHandle>,
    pub get_access: Access,
    pub set_access: Access,
    pub dependent: bool,
    pub hidden: bool,
    pub transient: bool,
    pub non_copyable: bool,
    pub abort_set: bool,
    pub set_observable: bool,
    pub get_observable: bool,
    pub description: String,
}

impl DynamicPropertyDef {
    pub fn new(name: String, defining_class: String) -> Self {
        Self {
            name,
            defining_class,
            metadata_handle: None,
            get_access: Access::Public,
            set_access: Access::Public,
            dependent: false,
            hidden: false,
            transient: false,
            non_copyable: false,
            abort_set: false,
            set_observable: false,
            get_observable: false,
            description: String::new(),
        }
    }
}
