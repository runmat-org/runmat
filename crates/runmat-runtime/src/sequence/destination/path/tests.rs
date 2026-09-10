use super::*;
use crate::object::indexing::ObjectIndexComponent;
use futures::executor::block_on;

#[test]
fn colon_cannot_define_an_undefined_member_destination_shape() {
    let mut builder = SequenceDestinationBuilder::new(&Value::Num(0.0));
    block_on(builder.push_step(
        AssignmentStepSpec::Parentheses {
            selectors: vec![ObjectIndexComponent::Colon],
        },
        None,
    ))
    .expect("colon can be prepared against the placeholder scalar");
    let error = block_on(builder.finish(SequenceEndpointSpec::Member("value".into())))
        .expect_err("colon cannot define a new structure-array shape");
    assert_eq!(
        error.identifier(),
        Some("RunMat:UndefinedColonSequenceDestination")
    );
}
