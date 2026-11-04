//! The immutable, position-free layer of the lossless tree (rowan's "green" tree).
//!
//! A green node knows its kind, its children and its text length, never its offset or its
//! parent, so an unchanged subtree is shared between versions of a file. It is `Send + Sync`
//! and cheap to clone.

use super::kind::{NodeKind, TokenKind};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GreenToken(Arc<GreenTokenData>);

#[derive(PartialEq, Eq, Hash)]
struct GreenTokenData {
    kind: TokenKind,
    text: Box<str>,
}

impl GreenToken {
    pub fn new(kind: TokenKind, text: &str) -> Self {
        Self(Arc::new(GreenTokenData {
            kind,
            text: text.into(),
        }))
    }

    pub fn kind(&self) -> TokenKind {
        self.0.kind
    }

    pub fn text(&self) -> &str {
        &self.0.text
    }

    pub fn text_len(&self) -> usize {
        self.0.text.len()
    }
}

impl fmt::Debug for GreenToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} {:?}", self.kind(), self.text())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GreenNode(Arc<GreenNodeData>);

#[derive(PartialEq, Eq, Hash)]
struct GreenNodeData {
    kind: NodeKind,
    text_len: usize,
    children: Box<[GreenElement]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GreenElement {
    Node(GreenNode),
    Token(GreenToken),
}

impl GreenElement {
    pub fn text_len(&self) -> usize {
        match self {
            GreenElement::Node(node) => node.text_len(),
            GreenElement::Token(token) => token.text_len(),
        }
    }
}

impl GreenNode {
    pub fn new(kind: NodeKind, children: Vec<GreenElement>) -> Self {
        let text_len = children.iter().map(GreenElement::text_len).sum();
        Self(Arc::new(GreenNodeData {
            kind,
            text_len,
            children: children.into_boxed_slice(),
        }))
    }

    pub fn kind(&self) -> NodeKind {
        self.0.kind
    }

    pub fn text_len(&self) -> usize {
        self.0.text_len
    }

    pub fn children(&self) -> &[GreenElement] {
        &self.0.children
    }

    /// The node's source text: every byte of every token below it, in order.
    pub fn text(&self) -> String {
        let mut out = String::with_capacity(self.text_len());
        self.write_text(&mut out);
        out
    }

    fn write_text(&self, out: &mut String) {
        for child in self.children() {
            match child {
                GreenElement::Node(node) => node.write_text(out),
                GreenElement::Token(token) => out.push_str(token.text()),
            }
        }
    }
}

impl fmt::Debug for GreenNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn go(node: &GreenNode, depth: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(
                f,
                "{:indent$}{:?}@{}",
                "",
                node.kind(),
                node.text_len(),
                indent = depth * 2
            )?;
            for child in node.children() {
                match child {
                    GreenElement::Node(child) => go(child, depth + 1, f)?,
                    GreenElement::Token(token) => {
                        writeln!(f, "{:indent$}{token:?}", "", indent = (depth + 1) * 2)?
                    }
                }
            }
            Ok(())
        }
        go(self, 0, f)
    }
}
