//! Source-level indexed growth through lowering, bytecode, and the shared
//! assignment owner. Runtime storage tests separately cover typed re-striding.

use futures::executor::block_on;
use runmat_value::{NumericDType, Value};

#[test]
fn numeric_row_growth_preserves_columns_zero_padding_and_existing_alias() {
    for (source, rows, expected) in [
        (
            "alias=zeros(1,3); for k=1:3; b(k)=k^2; end;",
            1,
            vec![1.0, 4.0, 9.0],
        ),
        (
            "b=zeros(1,3); alias=b; b(2,:)=[1 2 3];",
            2,
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        ),
        (
            "b=zeros(1,3); alias=b; b(end+1,:)=[1 2 3];",
            2,
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        ),
        (
            "b=zeros(1,3); alias=b; for k=2:3; b(end+1,:)=[1 2 3]; end;",
            3,
            vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 3.0, 3.0],
        ),
    ] {
        let ast = runmat_parser::parse(source).expect("parse row growth");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower row growth");
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
        let entrypoint = hir.assembly.entrypoints[0].id;
        let bytecode = super::compile(&hir.assembly, &mir, entrypoint).expect("compile row growth");
        let layout = bytecode.layout.as_ref().expect("compiled layout");
        let exports = &layout.entrypoints[&entrypoint].exports;
        let slot = |name| {
            exports
                .iter()
                .find(|export| export.name == name)
                .expect("source binding export")
                .slot
                .0
        };
        let values = block_on(crate::interpret(&bytecode)).expect(source);
        let Value::Tensor(grown) = &values[slot("b")] else {
            panic!("row growth must retain a numeric array: {source}");
        };
        assert_eq!(grown.shape(), &[rows, 3], "{source}");
        assert_eq!(grown.numeric_dtype(), NumericDType::F64, "{source}");
        assert_eq!(grown.materialize_f64(), expected, "{source}");
        let Value::Tensor(alias) = &values[slot("alias")] else {
            panic!("existing alias must retain a numeric array: {source}");
        };
        assert_eq!(alias.shape(), &[1, 3], "{source}");
        assert_eq!(alias.materialize_f64(), vec![0.0; 3], "{source}");
    }
}
