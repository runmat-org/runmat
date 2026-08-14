#[path = "support/mod.rs"]
mod test_helpers;

use runmat_builtins::{NumericDType, NumericScalar, Value};
use test_helpers::execute_source;

#[test]
fn compiled_null_accepts_every_integer_class_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!("z = null({constructor}([1 2;2 4]));");
        let values = execute_source(&source).expect("compiled integer null-space semantics");
        assert!(
            values.iter().any(|value| {
                matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F64 && tensor.shape == vec![2, 1])
            }),
            "{constructor}"
        );
    }
}

#[test]
fn compiled_optimization_and_ode_integer_extensions_execute_in_runmat_mode() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "xdata=uint64([0 1 2 3]); ydata=uint16([1 3 5 7]); p=lsqcurvefit(@(p,x) p(1).*x+p(2),int16([0;0]),xdata,ydata); [t15,y15]=ode15s(@(t,y) -y,uint16([0 1]),int8(1)); [t23,y23]=ode23(@(t,y) -y,uint16([0 1]),int8(1)); [t45,y45]=ode45(@(t,y) -y,uint16([0 1]),int8(1)); oo=optimoptions('fsolve','MaxIter',uint16(5)); os=optimset('MaxIter',uint64(17)); q=quad(@sin,int16(0),uint16(1),uint8(1));",
    )
    .expect("compiled optimization and ODE integer extensions");
}

#[test]
fn matlab_mode_rejects_each_runmat_only_integer_family_before_computation() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "x=lsqcurvefit(@(p,x) p,uint16(0),0,0);",
            "RunMat:compatibility:LsqcurvefitIntegerX0Extension",
        ),
        (
            "z=null(uint16([1 2;2 4]));",
            "RunMat:compatibility:NullIntegerMatrixExtension",
        ),
        (
            "y=ode15s(@(t,y) -y,uint16([0 1]),1);",
            "RunMat:compatibility:Ode15sIntegerTspanExtension",
        ),
        (
            "y=ode23(@(t,y) -y,uint16([0 1]),1);",
            "RunMat:compatibility:Ode23IntegerTspanExtension",
        ),
        (
            "y=ode45(@(t,y) -y,uint16([0 1]),1);",
            "RunMat:compatibility:Ode45IntegerTspanExtension",
        ),
        (
            "o=optimoptions('fsolve','MaxIter',uint16(5));",
            "RunMat:compatibility:OptimoptionsIntegerOptionExtension",
        ),
        (
            "o=optimset('MaxIter',uint16(5));",
            "RunMat:compatibility:OptimsetIntegerOptionExtension",
        ),
        (
            "q=quad(@sin,uint16(0),1);",
            "RunMat:compatibility:QuadIntegerBoundExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict compatibility gate");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_wide_integer_floating_boundaries_reject_without_rounding() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for source in [
        "w=uint64(9007199254740992)+uint64(1); x=lsqcurvefit(@(p,x) p,w,0,0);",
        "w=uint64(9007199254740992)+uint64(1); z=null(w);",
        "w=uint64(9007199254740992)+uint64(1); y=ode45(@(t,y) -y,[0 w],1);",
        "w=uint64(9007199254740992)+uint64(1); o=optimoptions('fsolve','TolX',w);",
        "w=uint64(9007199254740992)+uint64(1); q=quad(@sin,0,w);",
    ] {
        let error = execute_source(source).expect_err("lossy binary64 boundary must reject");
        assert!(
            error.message().contains("exactly representable"),
            "{source}"
        );
    }
}

#[test]
fn documented_integer_controls_and_callback_payloads_remain_exact_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let values = execute_source(
        "z=null([1 2;2 4],uint16(1)); p=uint64(9007199254740992)+uint64(1); q=quad(@(x,p) x,0,1,[],uint64(0),p); o=optimset('Display','off');",
    )
    .expect("documented integer controls");
    assert!(values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor) if matches!(tensor.numeric_value_at(0), Some(NumericScalar::F64(_))))
    }));
}
