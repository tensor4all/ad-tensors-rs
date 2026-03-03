use thiserror::Error;

/// Crate-wide error type for API skeleton operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A required thread-local global context is missing.
    #[error("missing global context for type `{type_name}`")]
    MissingGlobalContext { type_name: &'static str },

    /// Stored context could not be downcast to the expected type.
    #[error("global context type mismatch: expected `{expected}`")]
    ContextTypeMismatch { expected: &'static str },

    /// Placeholder for unimplemented POC kernel behavior.
    #[error("`{op}` is not implemented in this POC")]
    NotImplemented { op: &'static str },
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, Error>;
