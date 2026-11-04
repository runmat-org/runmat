//! The positioned cursor layer of the lossless tree (rowan's "red" tree): nodes and tokens
//! that know their offset and parent. Built lazily over a [`GreenNode`]; cheap to create and
//! to drop. Not `Send`: build one per thread from the shared green tree.

use super::green::{GreenElement, GreenNode, GreenToken};
use super::kind::{NodeKind, TokenKind};
use runmat_types::Span;
use std::fmt;
use std::rc::Rc;

#[derive(Clone)]
pub struct SyntaxNode(Rc<NodeData>);

struct NodeData {
    green: GreenNode,
    parent: Option<SyntaxNode>,
    index: usize,
    offset: usize,
}

#[derive(Clone)]
pub struct SyntaxToken {
    green: GreenToken,
    parent: SyntaxNode,
    index: usize,
    offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyntaxElement {
    Node(SyntaxNode),
    Token(SyntaxToken),
}

impl SyntaxElement {
    pub fn span(&self) -> Span {
        match self {
            SyntaxElement::Node(node) => node.span(),
            SyntaxElement::Token(token) => token.span(),
        }
    }

    pub fn as_node(&self) -> Option<&SyntaxNode> {
        match self {
            SyntaxElement::Node(node) => Some(node),
            SyntaxElement::Token(_) => None,
        }
    }

    pub fn as_token(&self) -> Option<&SyntaxToken> {
        match self {
            SyntaxElement::Token(token) => Some(token),
            SyntaxElement::Node(_) => None,
        }
    }
}

impl PartialEq for SyntaxNode {
    fn eq(&self, other: &Self) -> bool {
        self.0.offset == other.0.offset && self.0.green == other.0.green
    }
}

impl Eq for SyntaxNode {}

impl PartialEq for SyntaxToken {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset && self.green == other.green
    }
}

impl Eq for SyntaxToken {}

impl SyntaxNode {
    pub fn new_root(green: GreenNode) -> Self {
        Self(Rc::new(NodeData {
            green,
            parent: None,
            index: 0,
            offset: 0,
        }))
    }

    pub fn kind(&self) -> NodeKind {
        self.0.green.kind()
    }

    pub fn green(&self) -> &GreenNode {
        &self.0.green
    }

    pub fn span(&self) -> Span {
        Span {
            start: self.0.offset,
            end: self.0.offset + self.0.green.text_len(),
        }
    }

    /// The node's span without leading and trailing trivia.
    pub fn trimmed_span(&self) -> Span {
        let mut significant = self
            .tokens()
            .filter(|token| !token.kind().is_trivia() && token.kind() != TokenKind::EndOfFile);
        match significant.next() {
            None => Span {
                start: self.0.offset,
                end: self.0.offset,
            },
            Some(first) => {
                let last = significant.last().unwrap_or_else(|| first.clone());
                Span {
                    start: first.span().start,
                    end: last.span().end,
                }
            }
        }
    }

    pub fn text(&self) -> String {
        self.0.green.text()
    }

    pub fn parent(&self) -> Option<SyntaxNode> {
        self.0.parent.clone()
    }

    /// The position of this node among its parent's children (nodes and tokens).
    pub fn index(&self) -> usize {
        self.0.index
    }

    pub fn ancestors(&self) -> impl Iterator<Item = SyntaxNode> {
        std::iter::successors(Some(self.clone()), SyntaxNode::parent)
    }

    pub fn children_with_tokens(&self) -> impl Iterator<Item = SyntaxElement> + '_ {
        let mut offset = self.0.offset;
        self.0
            .green
            .children()
            .iter()
            .enumerate()
            .map(move |(index, child)| {
                let start = offset;
                offset += child.text_len();
                match child {
                    GreenElement::Node(green) => {
                        SyntaxElement::Node(SyntaxNode(Rc::new(NodeData {
                            green: green.clone(),
                            parent: Some(self.clone()),
                            index,
                            offset: start,
                        })))
                    }
                    GreenElement::Token(green) => SyntaxElement::Token(SyntaxToken {
                        green: green.clone(),
                        parent: self.clone(),
                        index,
                        offset: start,
                    }),
                }
            })
    }

    pub fn children(&self) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.children_with_tokens()
            .filter_map(|element| match element {
                SyntaxElement::Node(node) => Some(node),
                SyntaxElement::Token(_) => None,
            })
    }

    pub fn child(&self, kind: NodeKind) -> Option<SyntaxNode> {
        self.children().find(|child| child.kind() == kind)
    }

    pub fn children_of(&self, kind: NodeKind) -> impl Iterator<Item = SyntaxNode> + '_ {
        self.children().filter(move |child| child.kind() == kind)
    }

    /// The direct child tokens (not those of child nodes).
    pub fn child_tokens(&self) -> impl Iterator<Item = SyntaxToken> + '_ {
        self.children_with_tokens()
            .filter_map(|element| match element {
                SyntaxElement::Token(token) => Some(token),
                SyntaxElement::Node(_) => None,
            })
    }

    /// The first direct child token that is not trivia.
    pub fn first_child_token(&self) -> Option<SyntaxToken> {
        self.child_tokens().find(|token| !token.kind().is_trivia())
    }

    /// Every node below and including this one, in source order.
    pub fn descendants(&self) -> Vec<SyntaxNode> {
        let mut out = Vec::new();
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            let children: Vec<SyntaxNode> = node.children().collect();
            out.push(node);
            stack.extend(children.into_iter().rev());
        }
        out
    }

    /// Every leaf token below this node, in source order, trivia included.
    pub fn tokens(&self) -> impl Iterator<Item = SyntaxToken> {
        let mut stack: Vec<SyntaxElement> = vec![SyntaxElement::Node(self.clone())];
        std::iter::from_fn(move || loop {
            match stack.pop()? {
                SyntaxElement::Token(token) => return Some(token),
                SyntaxElement::Node(node) => {
                    let children: Vec<SyntaxElement> = node.children_with_tokens().collect();
                    stack.extend(children.into_iter().rev());
                }
            }
        })
    }

    /// The significant (non-trivia, non-EOF) tokens below this node.
    pub fn significant_tokens(&self) -> impl Iterator<Item = SyntaxToken> {
        self.tokens()
            .filter(|token| !token.kind().is_trivia() && token.kind() != TokenKind::EndOfFile)
    }

    /// The smallest node whose span covers `span`.
    pub fn covering_node(&self, span: Span) -> SyntaxNode {
        let mut node = self.clone();
        loop {
            let next = node.children().find(|child| {
                let own = child.span();
                own.start <= span.start && span.end <= own.end
            });
            match next {
                Some(child) => node = child,
                None => return node,
            }
        }
    }

    /// The node of `kind` whose trimmed span is exactly `span`, if one exists.
    pub fn find(&self, kind: NodeKind, span: Span) -> Option<SyntaxNode> {
        self.descendants()
            .into_iter()
            .find(|node| node.kind() == kind && node.trimmed_span() == span)
    }

    /// The previous sibling element (node or token), if any.
    pub fn prev_sibling_or_token(&self) -> Option<SyntaxElement> {
        let parent = self.parent()?;
        let index = self.index().checked_sub(1)?;
        let element = parent.children_with_tokens().nth(index);
        element
    }

    pub fn next_sibling_or_token(&self) -> Option<SyntaxElement> {
        let parent = self.parent()?;
        let element = parent.children_with_tokens().nth(self.index() + 1);
        element
    }

    pub fn next_sibling(&self) -> Option<SyntaxNode> {
        let parent = self.parent()?;
        let siblings: Vec<SyntaxNode> = parent
            .children()
            .filter(|node| node.index() > self.index())
            .collect();
        siblings.into_iter().next()
    }

    pub fn prev_sibling(&self) -> Option<SyntaxNode> {
        let parent = self.parent()?;
        let siblings: Vec<SyntaxNode> = parent
            .children()
            .filter(|node| node.index() < self.index())
            .collect();
        siblings.into_iter().last()
    }
}

impl SyntaxToken {
    pub fn kind(&self) -> TokenKind {
        self.green.kind()
    }

    pub fn text(&self) -> &str {
        self.green.text()
    }

    pub fn span(&self) -> Span {
        Span {
            start: self.offset,
            end: self.offset + self.green.text_len(),
        }
    }

    pub fn parent(&self) -> SyntaxNode {
        self.parent.clone()
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

impl fmt::Debug for SyntaxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}@{}..{}",
            self.kind(),
            self.span().start,
            self.span().end
        )
    }
}

impl fmt::Debug for SyntaxToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}@{}..{} {:?}",
            self.kind(),
            self.span().start,
            self.span().end,
            self.text()
        )
    }
}
