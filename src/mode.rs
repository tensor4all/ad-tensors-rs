/// Opaque identifier of a reverse-mode graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Opaque identifier of a tape instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeId(pub u64);

/// Plain numeric value wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct Primal<T> {
    pub value: T,
}

/// Forward-mode dual value wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct Dual<T> {
    pub primal: T,
    pub tangent: T,
}

/// Reverse-mode tracked value wrapper.
#[derive(Debug, Clone, PartialEq)]
pub struct Tracked<T> {
    pub primal: T,
    pub node: NodeId,
    pub tape: TapeId,
    pub tangent: Option<T>,
}
