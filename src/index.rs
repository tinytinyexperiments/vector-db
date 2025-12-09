use hnsw_rs::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("hnsw error: {0}")]
    Hnsw(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub dim: usize,
    /// Maximum number of elements the index is expected to hold.
    pub max_elements: usize,
    /// HNSW parameter: number of neighbors in layers.
    pub m: usize,
    /// HNSW parameter: construction/search effort.
    pub ef_construction: usize,
    pub ef_search: usize,
}

/// Thin wrapper around `hnsw_rs::Hnsw` to keep the rest of the codebase decoupled
/// from the concrete ANN implementation.
pub struct HnswIndex {
    dim: usize,
    hnsw: Hnsw<f32, DistL2>,
}

impl HnswIndex {
    pub fn new(cfg: &IndexConfig) -> Result<Self, IndexError> {
        // Hnsw::new(m, max_nb_connection, ef_construction, nb_layer, seed)
        // The exact signature may evolve; this wrapper keeps the rest of the app stable.
        let max_nb_connection = cfg.max_elements;
        let nb_layer = 16;
        let seed = 42;

        let hnsw = Hnsw::<f32, DistL2>::new(
            cfg.m,
            max_nb_connection,
            cfg.ef_construction,
            nb_layer,
            seed,
        )
        .map_err(|e| IndexError::Hnsw(format!("{e:?}")))?;

        Ok(Self {
            dim: cfg.dim,
            hnsw,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.hnsw
            .insert((id, vector))
            .map_err(|e| IndexError::Hnsw(format!("{e:?}")))?;
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let results = self
            .hnsw
            .search(query, k)
            .map_err(|e| IndexError::Hnsw(format!("{e:?}")))?;

        let neighbors = results
            .iter()
            .map(|neigh| (neigh.d_id, neigh.distance))
            .collect();

        Ok(neighbors)
    }
}


