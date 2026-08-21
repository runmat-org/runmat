pub const RUNMAT_TYPES_SCHEMA: SchemaIdentity = SchemaIdentity {
    name: "runmat-types",
    major: 1,
    minor: 0,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaIdentity {
    pub name: &'static str,
    pub major: u16,
    pub minor: u16,
}
