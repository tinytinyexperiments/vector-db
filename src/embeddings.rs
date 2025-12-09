use std::sync::Arc;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("embedding backend not configured")]
    NotConfigured,

    #[error("other error: {0}")]
    Other(String),
}

/// Simple abstraction over "something that can turn text into vectors".
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

/// Placeholder implementation that you can later replace with:
/// - ONNX Runtime via `ort`
/// - external embedding API (OpenAI, etc.)
pub struct DummyEmbedder {
    pub dim: usize,
}

impl Embedder for DummyEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // For now, just return zero vectors with the right dimension.
        Ok(texts
            .iter()
            .map(|_| vec![0.0_f32; self.dim])
            .collect())
    }
}

pub type SharedEmbedder = Arc<dyn Embedder>;


