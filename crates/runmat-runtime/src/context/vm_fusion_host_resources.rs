//! Allocation bound for the existing canonical Single Add -> min host path.
//! The caller must separately prove the selected bindings, shape, class, and
//! actual host branches; this byte calculation grants no route authority.

/// Scratch allocated by the checked host producer and terminal operation,
/// excluding separately accounted source readbacks and their owner workspace.
/// All producer outputs are retained in the bound because the VM may retain
/// private intermediates through the terminal instruction. Each shape is
/// obtained from the checked producer fact for that exact operation output.
pub fn checked_single_add_min_host_scratch(
    input_shape: &[usize],
    output_shape: &[usize],
    add_output_shapes: &[Vec<usize>],
    add_operand_shapes: &[[Vec<usize>; 2]],
) -> Option<u64> {
    if input_shape.is_empty() || output_shape.is_empty() || add_output_shapes.is_empty() {
        return None;
    }
    if add_output_shapes.len() != add_operand_shapes.len() {
        return None;
    }
    let input_elements = element_count(input_shape)?;
    let output_elements = element_count(output_shape)?;
    let word = u64::try_from(std::mem::size_of::<usize>()).ok()?;
    let producer_outputs = add_output_shapes
        .iter()
        .try_fold(0_u64, |retained, shape| {
            let rank = u64::try_from(shape.len()).ok()?;
            let bytes = element_count(shape)?
                .checked_mul(4)?
                .checked_add(rank.checked_mul(word)?)?;
            retained.checked_add(bytes)
        })?;
    // The canonical Plus binding clones its two Value arguments. Tensor's
    // HostNumericBuffer clone is COW, but plus_real_real consumes each with
    // into_numeric_storage while the private/argument owners are live. Both
    // native F32 payloads can therefore be copied before output allocation.
    // Two argument-shape clones coexist with BroadcastPlan's seven rank-sized
    // vectors and its output-shape clone. Retain the maximum one active Add.
    let active_add = add_output_shapes.iter().zip(add_operand_shapes).try_fold(
        0_u64,
        |peak, (output, operands)| {
            let plan =
                crate::builtins::common::broadcast::BroadcastPlan::new(&operands[0], &operands[1])
                    .ok()?;
            if plan.output_shape() != output.as_slice() {
                return None;
            }
            let copies = element_count(&operands[0])?
                .checked_mul(4)?
                .checked_add(element_count(&operands[1])?.checked_mul(4)?)?;
            let rank = u64::try_from(output.len()).ok()?;
            let metadata = rank.checked_mul(word)?.checked_mul(10)?;
            Some(peak.max(copies.checked_add(metadata)?))
        },
    )?;
    let terminal = crate::builtins::math::reduction::min::checked_single_host_scratch(
        input_elements,
        output_elements,
        u64::try_from(input_shape.len().max(output_shape.len()).max(2)).ok()?,
    )?;
    // The Min binding likewise clones its Tensor before reduce_real_tensor
    // consumes into_numeric_storage, then materializes a separate F64 input.
    // The F32 COW copy can overlap both the retained producer and F64 input.
    producer_outputs
        .checked_add(active_add)?
        .checked_add(terminal)?
        .checked_add(input_elements.checked_mul(4)?)
}

fn element_count(shape: &[usize]) -> Option<u64> {
    shape.iter().try_fold(1_u64, |count, &extent| {
        count.checked_mul(u64::try_from(extent).ok()?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_single_add_min_accounts_all_producers_and_terminal() {
        let inputs = [vec![3, 4], vec![1, 4]];
        let one = checked_single_add_min_host_scratch(
            &[3, 4],
            &[1, 4],
            &[vec![3, 4]],
            std::slice::from_ref(&inputs),
        )
        .unwrap();
        let two = checked_single_add_min_host_scratch(
            &[3, 4],
            &[1, 4],
            &[vec![3, 4], vec![3, 4]],
            &[inputs.clone(), inputs],
        )
        .unwrap();
        assert_eq!(
            two - one,
            3 * 4 * 4 + 2 * std::mem::size_of::<usize>() as u64
        );
        assert!(one > 3 * 4 * 8 + 4 * 8);
        assert!(checked_single_add_min_host_scratch(
            &[usize::MAX, usize::MAX, 2],
            &[1],
            &[vec![1]],
            &[[vec![1], vec![1]]]
        )
        .is_none());
        assert!(
            checked_single_add_min_host_scratch(&[3, 4], &[1, 4], &[vec![3, 4]], &[]).is_none()
        );
        assert!(checked_single_add_min_host_scratch(
            &[3, 4],
            &[1, 4],
            &[vec![3, 4]],
            &[[vec![2, 4], vec![1, 4]]]
        )
        .is_none());
    }
}
