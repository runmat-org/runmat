pub use inventory;

pub struct Builtin {
    pub name: &'static str,
    pub doc: &'static str,
    pub func: *const (),
}

inventory::collect!(Builtin);

pub fn builtins() -> inventory::iter<Builtin> {
    inventory::iter::<Builtin>
}
