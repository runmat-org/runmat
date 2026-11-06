//! Typed views over the lossless tree for the design dialects (rowan-style AST).
//!
//! Each view wraps a [`SyntaxNode`] and reads its parts from the tree, so a front end always
//! sees the text as written and every piece has its span. The `.fea` dialect keeps its own
//! AST ([`crate::FeaSyntaxModule`]) for its consumers; its tree is the same structure.

use crate::lexical::FeaTokenKind;
use crate::parser::decode_string;
use crate::tree::{NodeKind, SyntaxNode, SyntaxToken};
use runmat_types::Span;

macro_rules! view {
    ($name:ident, $($kind:ident)|+) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name(SyntaxNode);

        impl $name {
            pub fn cast(node: SyntaxNode) -> Option<Self> {
                matches!(node.kind(), $(NodeKind::$kind)|+).then(|| Self(node))
            }

            pub fn syntax(&self) -> &SyntaxNode {
                &self.0
            }

            /// The span without surrounding trivia.
            pub fn span(&self) -> Span {
                self.0.trimmed_span()
            }
        }
    };
}

view!(SourceFile, SourceFile);
view!(VersionHeader, LanguageHeader | StdHeader);
view!(UseDeclaration, UseDeclaration);
view!(Import, Import);
view!(Declaration, Declaration);
view!(Block, Block);
view!(Name, Name);
view!(Clause, Clause);

/// The text of a name node (a quoted record key is decoded).
fn name_text(node: &SyntaxNode) -> String {
    let token = node.first_child_token();
    match token {
        Some(token) if token.kind() == FeaTokenKind::String => decode_string(token.text()),
        Some(token) => token.text().to_string(),
        None => String::new(),
    }
}

/// Significant tokens directly under `node`, in order.
fn direct_tokens(node: &SyntaxNode) -> Vec<SyntaxToken> {
    node.child_tokens()
        .filter(|token| !token.kind().is_trivia() && token.kind() != FeaTokenKind::EndOfFile)
        .collect()
}

/// The text of significant tokens, joined without trivia (`kg / m^3` → `kg/m^3`).
pub fn compact_text(node: &SyntaxNode) -> String {
    node.significant_tokens()
        .map(|token| token.text().to_string())
        .collect()
}

impl Name {
    pub fn text(&self) -> String {
        name_text(&self.0)
    }
}

impl SourceFile {
    pub fn headers(&self) -> Vec<VersionHeader> {
        self.0.children().filter_map(VersionHeader::cast).collect()
    }

    pub fn items(&self) -> Vec<Item> {
        self.0.children().filter_map(Item::cast).collect()
    }

    pub fn version(&self, std: bool) -> Option<String> {
        self.headers()
            .into_iter()
            .find(|header| header.is_std() == std)
            .and_then(|header| header.version())
    }
}

impl VersionHeader {
    pub fn is_std(&self) -> bool {
        self.0.kind() == NodeKind::StdHeader
    }

    pub fn version(&self) -> Option<String> {
        let node = self.0.child(NodeKind::Version)?;
        let token = node.first_child_token()?;
        Some(if token.kind() == FeaTokenKind::String {
            decode_string(token.text())
        } else {
            token.text().to_string()
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Use(UseDeclaration),
    Import(Import),
    Declaration(Declaration),
    Error(SyntaxNode),
}

impl Item {
    pub fn cast(node: SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            NodeKind::UseDeclaration => Item::Use(UseDeclaration(node)),
            NodeKind::Import => Item::Import(Import(node)),
            NodeKind::Declaration => Item::Declaration(Declaration(node)),
            NodeKind::Error => Item::Error(node),
            _ => return None,
        })
    }
}

/// Where a `use` reads from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UseSource {
    /// A quoted folder or file path.
    Path(String),
    /// A package or folder path (`std.materials`).
    Package(Vec<String>),
}

impl UseDeclaration {
    fn names(&self) -> Vec<Name> {
        self.0.children().filter_map(Name::cast).collect()
    }

    /// The symbol form's kind and name: `use part bracket from …`, or `use part bracket as left`
    /// for a design in the same folder.
    pub fn symbol(&self) -> Option<(Name, Name)> {
        let names = self.names();
        if self.0.child(NodeKind::Path).is_some() && names.len() < 2 {
            return None;
        }
        let tokens = direct_tokens(&self.0);
        let has_from = tokens
            .iter()
            .any(|token| token.kind() == FeaTokenKind::Keyword(crate::lexical::FeaKeyword::From));
        if has_from {
            return (names.len() >= 2).then(|| (names[0].clone(), names[1].clone()));
        }
        if self.0.child(NodeKind::Path).is_some() || self.0.child(NodeKind::StringLiteral).is_some()
        {
            return None;
        }
        // No source: the kind and name are the two names before `as`.
        let alias_at = tokens
            .iter()
            .find(|token| token.text() == "as")
            .map_or(usize::MAX, |token| token.span().start);
        let before: Vec<Name> = names
            .into_iter()
            .filter(|name| name.span().start < alias_at)
            .collect();
        (before.len() == 2).then(|| (before[0].clone(), before[1].clone()))
    }

    /// Whether the `use` names a design with no source: one in the same folder.
    pub fn is_same_folder(&self) -> bool {
        self.symbol().is_some() && self.source().is_none()
    }

    pub fn source(&self) -> Option<UseSource> {
        if let Some(string) = self.0.child(NodeKind::StringLiteral) {
            let token = string.first_child_token()?;
            return Some(UseSource::Path(decode_string(token.text())));
        }
        let path = self.0.child(NodeKind::Path)?;
        Some(UseSource::Package(path_segments(&path)))
    }

    pub fn pin(&self) -> Option<String> {
        let node = self.0.child(NodeKind::Version)?;
        Some(decode_string(node.first_child_token()?.text()))
    }

    pub fn alias(&self) -> Option<Name> {
        let tokens = direct_tokens(&self.0);
        let as_token = tokens.iter().find(|token| token.text() == "as")?;
        self.names()
            .into_iter()
            .find(|name| name.span().start > as_token.span().start)
    }

    pub fn overrides(&self) -> Option<Block> {
        self.0.child(NodeKind::Block).map(Block)
    }
}

impl Import {
    pub fn kind(&self) -> Option<String> {
        direct_tokens(&self.0)
            .get(1)
            .map(|token| token.text().to_string())
    }

    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn path(&self) -> Option<String> {
        let node = self.0.child(NodeKind::StringLiteral)?;
        Some(decode_string(node.first_child_token()?.text()))
    }
}

/// The header pieces shared by declarations and block statements.
pub trait Headed {
    fn node(&self) -> &SyntaxNode;

    /// The heading word (`part`, `sketch`, `region`).
    fn keyword(&self) -> String {
        direct_tokens(self.node())
            .first()
            .map(|token| token.text().to_string())
            .unwrap_or_default()
    }

    fn keyword_span(&self) -> Span {
        direct_tokens(self.node())
            .first()
            .map(SyntaxToken::span)
            .unwrap_or_default()
    }

    fn name(&self) -> Option<Name> {
        self.node().child(NodeKind::Name).map(Name)
    }

    /// The tokens between the name and the `{`.
    fn header(&self) -> Option<SyntaxNode> {
        self.node().child(NodeKind::Header)
    }

    /// The type after `:` in the header, as written without trivia.
    fn type_text(&self) -> Option<String> {
        let header = self.header()?;
        header.child(NodeKind::Type).map(|node| compact_text(&node))
    }

    fn clauses(&self) -> Vec<Clause> {
        self.header()
            .map(|header| header.children().filter_map(Clause::cast).collect())
            .unwrap_or_default()
    }

    fn block(&self) -> Option<Block> {
        self.node().child(NodeKind::Block).map(Block)
    }
}

impl Headed for Declaration {
    fn node(&self) -> &SyntaxNode {
        &self.0
    }
}

impl Clause {
    pub fn word(&self) -> String {
        direct_tokens(&self.0)
            .first()
            .map(|token| token.text().to_string())
            .unwrap_or_default()
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }
}

impl Block {
    pub fn statements(&self) -> Vec<Statement> {
        self.0.children().filter_map(Statement::cast).collect()
    }
}

view!(ValueStatement, ValueStatement);
view!(Assignment, Assignment);
view!(Constrain, Constrain);
view!(When, When);
view!(Publish, Publish);
view!(Check, Check);
view!(Link, Link);
view!(Solved, Solved);
view!(BlockStatement, BlockStatement);
view!(Command, Command);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    Value(ValueStatement),
    Assignment(Assignment),
    Constrain(Constrain),
    When(When),
    Use(UseDeclaration),
    Publish(Publish),
    Check(Check),
    Link(Link),
    Solved(Solved),
    Block(BlockStatement),
    Command(Command),
    /// A `.fea` statement shape (see [`crate::FeaStatementKind`]).
    Fea(SyntaxNode),
}

impl Statement {
    pub fn cast(node: SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            NodeKind::ValueStatement => Statement::Value(ValueStatement(node)),
            NodeKind::Assignment => Statement::Assignment(Assignment(node)),
            NodeKind::Constrain => Statement::Constrain(Constrain(node)),
            NodeKind::When => Statement::When(When(node)),
            NodeKind::UseDeclaration => Statement::Use(UseDeclaration(node)),
            NodeKind::Publish => Statement::Publish(Publish(node)),
            NodeKind::Check => Statement::Check(Check(node)),
            NodeKind::Link => Statement::Link(Link(node)),
            NodeKind::Solved => Statement::Solved(Solved(node)),
            NodeKind::BlockStatement => Statement::Block(BlockStatement(node)),
            NodeKind::Command => Statement::Command(Command(node)),
            kind if kind.is_statement() => Statement::Fea(node),
            _ => return None,
        })
    }

    pub fn syntax(&self) -> &SyntaxNode {
        match self {
            Statement::Value(s) => s.syntax(),
            Statement::Assignment(s) => s.syntax(),
            Statement::Constrain(s) => s.syntax(),
            Statement::When(s) => s.syntax(),
            Statement::Use(s) => s.syntax(),
            Statement::Publish(s) => s.syntax(),
            Statement::Check(s) => s.syntax(),
            Statement::Link(s) => s.syntax(),
            Statement::Solved(s) => s.syntax(),
            Statement::Block(s) => s.syntax(),
            Statement::Command(s) => s.syntax(),
            Statement::Fea(node) => node,
        }
    }

    pub fn span(&self) -> Span {
        self.syntax().trimmed_span()
    }
}

impl ValueStatement {
    /// `let`, `input`, `property` or a registry word (`feature`, `plane`, `component`).
    pub fn keyword(&self) -> String {
        direct_tokens(&self.0)
            .first()
            .map(|token| token.text().to_string())
            .unwrap_or_default()
    }

    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn type_text(&self) -> Option<String> {
        self.0.child(NodeKind::Type).map(|node| compact_text(&node))
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }
}

/// The relation of an assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    Equal,
    LessEqual,
    GreaterEqual,
}

impl Assignment {
    pub fn target(&self) -> Vec<String> {
        self.0
            .child(NodeKind::Path)
            .map(|path| path_segments(&path))
            .unwrap_or_default()
    }

    pub fn target_span(&self) -> Span {
        self.0
            .child(NodeKind::Path)
            .map_or_else(|| self.span(), |path| path.trimmed_span())
    }

    pub fn relation(&self) -> Relation {
        for token in direct_tokens(&self.0) {
            match token.kind() {
                FeaTokenKind::LessEqual => return Relation::LessEqual,
                FeaTokenKind::GreaterEqual => return Relation::GreaterEqual,
                FeaTokenKind::Equal => return Relation::Equal,
                _ => {}
            }
        }
        Relation::Equal
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }
}

impl Constrain {
    pub fn expression(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }

    pub fn equals(&self) -> Option<Expr> {
        self.0.children().filter_map(Expr::cast).nth(1)
    }
}

impl When {
    pub fn condition(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }

    pub fn block(&self) -> Option<Block> {
        self.0.child(NodeKind::Block).map(Block)
    }
}

impl Publish {
    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }
}

impl Check {
    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }

    pub fn block(&self) -> Option<Block> {
        self.0.child(NodeKind::Block).map(Block)
    }
}

impl Link {
    pub fn kind(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn from(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }

    pub fn to(&self) -> Option<Expr> {
        self.0.children().filter_map(Expr::cast).nth(1)
    }
}

impl Solved {
    pub fn block(&self) -> Option<Block> {
        self.0.child(NodeKind::Block).map(Block)
    }
}

impl Headed for BlockStatement {
    fn node(&self) -> &SyntaxNode {
        &self.0
    }
}

impl Command {
    pub fn word(&self) -> String {
        direct_tokens(&self.0)
            .first()
            .map(|token| token.text().to_string())
            .unwrap_or_default()
    }

    pub fn argument(&self) -> Option<Expr> {
        self.0.children().find_map(Expr::cast)
    }
}

// ------------------------------------------------------------------------------- expressions

view!(Number, NumberLiteral);
view!(Str, StringLiteral);
view!(Reference, Reference);
view!(List, List);
view!(Record, Record);
view!(RecordEntry, RecordEntry);
view!(Group, Group);
view!(Unary, Unary);
view!(Binary, Binary);
view!(Call, Call);
view!(NamedArgument, NamedArgument);
view!(PrefixApplication, PrefixApplication);
view!(Quantity, Quantity);
view!(Toleranced, Toleranced);
view!(Coerce, Coerce);
view!(Pinned, Pinned);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Number(Number),
    String(Str),
    Reference(Reference),
    List(List),
    Record(Record),
    Group(Group),
    Unary(Unary),
    Binary(Binary),
    Call(Call),
    PrefixApplication(PrefixApplication),
    Quantity(Quantity),
    Toleranced(Toleranced),
    Coerce(Coerce),
    Pinned(Pinned),
}

impl Expr {
    pub fn cast(node: SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            NodeKind::NumberLiteral => Expr::Number(Number(node)),
            NodeKind::StringLiteral => Expr::String(Str(node)),
            NodeKind::Reference => Expr::Reference(Reference(node)),
            NodeKind::List => Expr::List(List(node)),
            NodeKind::Record => Expr::Record(Record(node)),
            NodeKind::Group => Expr::Group(Group(node)),
            NodeKind::Unary => Expr::Unary(Unary(node)),
            NodeKind::Binary => Expr::Binary(Binary(node)),
            NodeKind::Call => Expr::Call(Call(node)),
            NodeKind::PrefixApplication => Expr::PrefixApplication(PrefixApplication(node)),
            NodeKind::Quantity => Expr::Quantity(Quantity(node)),
            NodeKind::Toleranced => Expr::Toleranced(Toleranced(node)),
            NodeKind::Coerce => Expr::Coerce(Coerce(node)),
            NodeKind::Pinned => Expr::Pinned(Pinned(node)),
            _ => return None,
        })
    }

    pub fn syntax(&self) -> &SyntaxNode {
        match self {
            Expr::Number(e) => e.syntax(),
            Expr::String(e) => e.syntax(),
            Expr::Reference(e) => e.syntax(),
            Expr::List(e) => e.syntax(),
            Expr::Record(e) => e.syntax(),
            Expr::Group(e) => e.syntax(),
            Expr::Unary(e) => e.syntax(),
            Expr::Binary(e) => e.syntax(),
            Expr::Call(e) => e.syntax(),
            Expr::PrefixApplication(e) => e.syntax(),
            Expr::Quantity(e) => e.syntax(),
            Expr::Toleranced(e) => e.syntax(),
            Expr::Coerce(e) => e.syntax(),
            Expr::Pinned(e) => e.syntax(),
        }
    }

    pub fn span(&self) -> Span {
        self.syntax().trimmed_span()
    }
}

fn exprs(node: &SyntaxNode) -> Vec<Expr> {
    node.children().filter_map(Expr::cast).collect()
}

fn path_segments(node: &SyntaxNode) -> Vec<String> {
    node.significant_tokens()
        .filter(|token| token.kind() != FeaTokenKind::Dot)
        .map(|token| token.text().to_string())
        .collect()
}

impl Number {
    pub fn text(&self) -> String {
        compact_text(&self.0)
    }
}

impl Str {
    pub fn value(&self) -> String {
        self.0
            .first_child_token()
            .map(|token| decode_string(token.text()))
            .unwrap_or_default()
    }
}

impl Reference {
    pub fn segments(&self) -> Vec<String> {
        path_segments(&self.0)
    }
}

impl List {
    pub fn items(&self) -> Vec<Expr> {
        exprs(&self.0)
    }
}

impl Record {
    pub fn entries(&self) -> Vec<RecordEntry> {
        self.0.children().filter_map(RecordEntry::cast).collect()
    }
}

impl RecordEntry {
    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn value(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }
}

impl Group {
    pub fn inner(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }
}

impl Unary {
    pub fn is_negative(&self) -> bool {
        direct_tokens(&self.0)
            .first()
            .is_some_and(|token| token.kind() == FeaTokenKind::Minus)
    }

    pub fn operand(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }
}

impl Binary {
    pub fn operator(&self) -> Option<FeaTokenKind> {
        direct_tokens(&self.0).first().map(SyntaxToken::kind)
    }

    pub fn lhs(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn rhs(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().nth(1)
    }
}

/// One call argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Argument {
    Positional(Expr),
    Named(NamedArgument),
}

impl Call {
    pub fn callee(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn arguments(&self) -> Vec<Argument> {
        let mut out = Vec::new();
        let mut first = true;
        for child in self.0.children() {
            if let Some(named) = NamedArgument::cast(child.clone()) {
                out.push(Argument::Named(named));
            } else if let Some(expr) = Expr::cast(child) {
                if first {
                    first = false;
                    continue;
                }
                out.push(Argument::Positional(expr));
            }
        }
        out
    }
}

impl NamedArgument {
    pub fn name(&self) -> Option<Name> {
        self.0.child(NodeKind::Name).map(Name)
    }

    pub fn value(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }
}

impl PrefixApplication {
    pub fn callee(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn argument(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().nth(1)
    }
}

impl Quantity {
    pub fn value(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn unit(&self) -> String {
        self.0
            .child(NodeKind::Unit)
            .map(|unit| compact_text(&unit))
            .unwrap_or_default()
    }
}

impl Toleranced {
    pub fn nominal(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn tolerance(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().nth(1)
    }
}

impl Coerce {
    pub fn expr(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn unit(&self) -> String {
        self.0
            .child(NodeKind::Unit)
            .map(|unit| compact_text(&unit))
            .unwrap_or_default()
    }
}

impl Pinned {
    pub fn expr(&self) -> Option<Expr> {
        exprs(&self.0).into_iter().next()
    }

    pub fn revision(&self) -> Option<String> {
        let node = self.0.child(NodeKind::Version)?;
        Some(decode_string(node.first_child_token()?.text()))
    }
}
