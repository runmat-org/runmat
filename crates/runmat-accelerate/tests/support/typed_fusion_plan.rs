use runmat_accelerate::fusion::FusionOp;

#[path = "fusion_plan.rs"]
pub mod builder;
pub use builder::plan;

pub fn cast(name: &str, input: u32, output: u32) -> FusionOp {
    FusionOp::Builtin {
        origin: Default::default(),
        name: name.into(),
        inputs: vec![input],
        output: Some(output),
    }
}
