//! Both supervised callers retain the selected initializer and runtime
//! across a suspended callback, then publish only the finished network.
use super::*;

#[test]
fn supervised_initializers_suspend_and_resume_once_per_caller() {
    for name in ["trainNetwork", "trainnet"] {
        reset(Outcome::Initializer);
        let (runtime, service) = scoped_runtime();
        let weak = Rc::downgrade(&service);
        let id = runtime.local_identity();
        let layers = super::super::training_layers(name == "trainNetwork");
        let options = super::super::training_options();
        let mut owned_args = runmat_runtime::execution::RootedValuesBuilder::with_capacity(5);
        let args = vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
        ];
        for arg in args {
            owned_args.push(arg).expect("root training input");
        }
        owned_args.push_owned(layers);
        if name == "trainnet" {
            owned_args
                .push(Value::String("mse".into()))
                .expect("root loss name");
        }
        owned_args.push_owned(options);
        let args = owned_args.finish().expect("root training arguments");
        let mut future = invoke(name, runtime.clone(), args.values());
        assert!(
            poll_once(&mut future).is_pending(),
            "{name} initializer suspended"
        );
        assert_eq!(calls(), vec![(id, Some(1))], "{name} selected one callback");
        let Poll::Ready(Ok(outputs)) = poll_once(&mut future) else {
            panic!("{name} should publish its trained network after callback completion");
        };
        assert_eq!(
            outputs.kind(),
            runmat_value::ValueSequenceKind::CommaSeparated
        );
        assert!(matches!(outputs.as_slice(), [Value::Object(_)]));
        assert_eq!(
            calls(),
            vec![(id, Some(1))],
            "{name} did not replay callback"
        );
        assert_eq!(POLLS.with(Cell::get), 2);
        drop(outputs);
        drop(runtime);
        drop(service);
        drop(future);
        assert!(
            weak.upgrade().is_none(),
            "{name} retained session after completion"
        );
    }
}
