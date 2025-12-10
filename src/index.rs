use hnsw_rs::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IndexError {
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
    /// HNSW parameter: construction effort.
    pub ef_construction: usize,
    /// HNSW parameter: search effort.
    pub ef_search: usize,
}

/// Thin wrapper around `hnsw_rs::Hnsw` to keep the rest of the codebase decoupled
/// from the concrete ANN implementation.
pub struct HnswIndex {
    dim: usize,
    ef_search: usize,
    hnsw: Hnsw<f32, DistL2>,
}

impl HnswIndex {
    pub fn new(cfg: &IndexConfig) -> Result<Self, IndexError> {
        // hnsw_rs 0.1.x signature:
        // Hnsw::new(max_nb_connection, max_elements, max_layer, ef_construction, dist_fn)
        let max_layer = 16;
        let hnsw = Hnsw::<f32, DistL2>::new(
            cfg.m,
            cfg.max_elements,
            max_layer,
            cfg.ef_construction,
            DistL2 {},
        );

        Ok(Self {
            dim: cfg.dim,
            ef_search: cfg.ef_search,
            hnsw,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.hnsw.get_nb_point()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn insert(&self, id: usize, vector: Vec<f32>) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        // HNSW insert takes (&Vec<T>, external_id)
        self.hnsw.insert((&vector, id));
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        // HNSW search signature: search(&[T], knbn, ef_arg) -> Vec<Neighbour>
        let results = self.hnsw.search(query, k, self.ef_search);

        let neighbors = results
            .into_iter()
            .map(|neigh| (neigh.d_id as usize, neigh.distance))
            .collect();

        Ok(neighbors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search_returns_self_as_nearest() {
        let cfg = IndexConfig {
            dim: 4,
            max_elements: 16,
            m: 8,
            ef_construction: 16,
            ef_search: 16,
        };

        let index = HnswIndex::new(&cfg).expect("index created");

        index
            .insert(1, vec![1.0, 0.0, 0.0, 0.0])
            .expect("insert should succeed");

        let neighbors = index
            .search(&[1.0, 0.0, 0.0, 0.0], 1)
            .expect("search should succeed");

        assert!(!neighbors.is_empty());
        assert_eq!(neighbors[0].0, 1);
    }

    #[test]
    fn dim_mismatch_on_insert_errors() {
        let cfg = IndexConfig {
            dim: 4,
            max_elements: 16,
            m: 8,
            ef_construction: 16,
            ef_search: 16,
        };

        let index = HnswIndex::new(&cfg).expect("index created");

        let err = index.insert(1, vec![1.0, 0.0, 0.0]).unwrap_err();
        matches!(err, IndexError::DimMismatch { expected: 4, got: 3 });
    }
}


