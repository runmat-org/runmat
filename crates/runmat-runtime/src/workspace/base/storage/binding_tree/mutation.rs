//! Offstate AVL path copying. Every constructed node admits its own exact
//! header before allocation; no tree mutation is visible until root install.

use std::cmp::Ordering;

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use crate::workspace::retained::WorkspaceName;
use runmat_execution::resource::ExactShared;

use super::node::{Link, Node};
use super::{BindingTreeValue, SortedBindingTreeError};

pub(super) fn insert(
    old: &Link,
    name: WorkspaceName,
    binding: BindingTreeValue,
    context: Option<&RuntimeContext>,
) -> Result<(Link, bool), HostAllocationError> {
    let Some(old) = old else {
        return Ok((Some(Node::make(name, binding, None, None, context)?), true));
    };
    let old = old.get();
    let (next, added) = match name.as_str().cmp(old.name.as_str()) {
        Ordering::Less => {
            let (left, added) = insert(&old.left, name, binding, context)?;
            let next = Node::make(
                old.name.clone(),
                old.binding.clone(),
                left,
                old.right.clone(),
                context,
            )?;
            (rebalance(next, context)?, added)
        }
        Ordering::Greater => {
            let (right, added) = insert(&old.right, name, binding, context)?;
            let next = Node::make(
                old.name.clone(),
                old.binding.clone(),
                old.left.clone(),
                right,
                context,
            )?;
            (rebalance(next, context)?, added)
        }
        Ordering::Equal => (
            Node::make(
                old.name.clone(),
                binding,
                old.left.clone(),
                old.right.clone(),
                context,
            )?,
            false,
        ),
    };
    Ok((Some(next), added))
}

pub(super) fn remove(
    old: &Link,
    name: &str,
    context: Option<&RuntimeContext>,
) -> Result<(Link, bool), HostAllocationError> {
    let Some(old) = old else {
        return Ok((None, false));
    };
    let node = old.get();
    match name.cmp(node.name.as_str()) {
        Ordering::Less => {
            let (left, removed) = remove(&node.left, name, context)?;
            if !removed {
                return Ok((Some(old.clone()), false));
            }
            let next = Node::make(
                node.name.clone(),
                node.binding.clone(),
                left,
                node.right.clone(),
                context,
            )?;
            Ok((Some(rebalance(next, context)?), true))
        }
        Ordering::Greater => {
            let (right, removed) = remove(&node.right, name, context)?;
            if !removed {
                return Ok((Some(old.clone()), false));
            }
            let next = Node::make(
                node.name.clone(),
                node.binding.clone(),
                node.left.clone(),
                right,
                context,
            )?;
            Ok((Some(rebalance(next, context)?), true))
        }
        Ordering::Equal => match (&node.left, &node.right) {
            (None, None) => Ok((None, true)),
            (Some(left), None) => Ok((Some(left.clone()), true)),
            (None, Some(right)) => Ok((Some(right.clone()), true)),
            (Some(left), Some(right)) => {
                let (successor, binding, right) = pop_min(right, context)?;
                let next = Node::make(successor, binding, Some(left.clone()), right, context)?;
                Ok((Some(rebalance(next, context)?), true))
            }
        },
    }
}

fn pop_min(
    old: &ExactShared<Node>,
    context: Option<&RuntimeContext>,
) -> Result<(WorkspaceName, BindingTreeValue, Link), HostAllocationError> {
    let node = old.get();
    let Some(left) = &node.left else {
        return Ok((node.name.clone(), node.binding.clone(), node.right.clone()));
    };
    let (name, binding, next_left) = pop_min(left, context)?;
    let next = Node::make(
        node.name.clone(),
        node.binding.clone(),
        next_left,
        node.right.clone(),
        context,
    )?;
    Ok((name, binding, Some(rebalance(next, context)?)))
}

fn rebalance(
    root: ExactShared<Node>,
    context: Option<&RuntimeContext>,
) -> Result<ExactShared<Node>, HostAllocationError> {
    let node = root.get();
    let left_height = Node::link_height(&node.left);
    let right_height = Node::link_height(&node.right);
    if left_height.saturating_sub(right_height) > 1 {
        let left = node
            .left
            .as_ref()
            .expect("left-heavy AVL node has a left child");
        let left = left.get();
        if Node::link_height(&left.right) > Node::link_height(&left.left) {
            let rotated = rotate_left(
                node.left.as_ref().expect("left-heavy AVL node").clone(),
                context,
            )?;
            let root = Node::make(
                node.name.clone(),
                node.binding.clone(),
                Some(rotated),
                node.right.clone(),
                context,
            )?;
            return rotate_right(root, context);
        }
        return rotate_right(root, context);
    }
    if right_height.saturating_sub(left_height) > 1 {
        let right = node.right.as_ref().expect("right-heavy AVL node");
        let right = right.get();
        if Node::link_height(&right.left) > Node::link_height(&right.right) {
            let rotated = rotate_right(
                node.right.as_ref().expect("right-heavy AVL node").clone(),
                context,
            )?;
            let root = Node::make(
                node.name.clone(),
                node.binding.clone(),
                node.left.clone(),
                Some(rotated),
                context,
            )?;
            return rotate_left(root, context);
        }
        return rotate_left(root, context);
    }
    Ok(root)
}

fn rotate_left(
    root: ExactShared<Node>,
    context: Option<&RuntimeContext>,
) -> Result<ExactShared<Node>, HostAllocationError> {
    let node = root.get();
    let pivot = node
        .right
        .as_ref()
        .expect("left rotation has a right child");
    let pivot_node = pivot.get();
    let next_left = Node::make(
        node.name.clone(),
        node.binding.clone(),
        node.left.clone(),
        pivot_node.left.clone(),
        context,
    )?;
    Node::make(
        pivot_node.name.clone(),
        pivot_node.binding.clone(),
        Some(next_left),
        pivot_node.right.clone(),
        context,
    )
}

fn rotate_right(
    root: ExactShared<Node>,
    context: Option<&RuntimeContext>,
) -> Result<ExactShared<Node>, HostAllocationError> {
    let node = root.get();
    let pivot = node.left.as_ref().expect("right rotation has a left child");
    let pivot_node = pivot.get();
    let next_right = Node::make(
        node.name.clone(),
        node.binding.clone(),
        pivot_node.right.clone(),
        node.right.clone(),
        context,
    )?;
    Node::make(
        pivot_node.name.clone(),
        pivot_node.binding.clone(),
        pivot_node.left.clone(),
        Some(next_right),
        context,
    )
}

pub(super) fn build_sorted(
    len: usize,
    bindings: &mut impl Iterator<Item = (WorkspaceName, BindingTreeValue)>,
    previous: &mut Option<WorkspaceName>,
    make: &mut impl FnMut(
        WorkspaceName,
        BindingTreeValue,
        Link,
        Link,
    ) -> Result<ExactShared<Node>, HostAllocationError>,
) -> Result<Link, SortedBindingTreeError> {
    if len == 0 {
        return Ok(None);
    }
    let left_len = len / 2;
    let left = build_sorted(left_len, bindings, previous, make)?;
    let (name, binding) = bindings
        .next()
        .ok_or(SortedBindingTreeError::MissingInput)?;
    if previous
        .as_ref()
        .is_some_and(|previous| previous.as_str() >= name.as_str())
    {
        return Err(SortedBindingTreeError::NonAscendingNames);
    }
    *previous = Some(name.clone());
    let right = build_sorted(len - left_len - 1, bindings, previous, make)?;
    Ok(Some(make(name, binding, left, right)?))
}
