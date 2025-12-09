use std::path::PathBuf;

use serde::Serialize;

use crate::embeddings::SharedEmbedder;
use crate::health::{basic_index_health, HealthReport};
use crate::index::{HnswIndex, IndexConfig, IndexError};
use crate::storage::{SqliteVectorStore, StorageConfig, StorageError};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub dim: usize,
    pub storage_path: PathBuf,
    pub hnsw_max_elements: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("index error: {0}")]
    Index(#[from] IndexError),

    #[error("storage error: {0}")]
    Storage(#[from] StorageError),
}

pub struct SelfHealingVectorDb {
    dim: usize,
    index: HnswIndex,
    store: SqliteVectorStore,
    _embedder: Option<SharedEmbedder>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

impl SelfHealingVectorDb {
    pub fn new(cfg: EngineConfig, embedder: Option<SharedEmbedder>) -> Result<Self, EngineError> {
        let store = SqliteVectorStore::new(&StorageConfig {
            path: cfg.storage_path.clone(),
            dim: cfg.dim,
        })?;

        let mut index = HnswIndex::new(&IndexConfig {
            dim: cfg.dim,
            max_elements: cfg.hnsw_max_elements,
            m: cfg.hnsw_m,
            ef_construction: cfg.hnsw_ef_construction,
            ef_search: cfg.hnsw_ef_search,
        })?;

        // Bootstrap from storage (self-healing on startup)
        let (ids, vecs) = store.load_all()?;
        for (i, chunk) in vecs.chunks(cfg.dim).enumerate() {
            let id = ids[i] as usize;
            index.insert(id, chunk.to_vec())?;
        }

        Ok(Self {
            dim: cfg.dim,
            index,
            store,
            _embedder: embedder,
        })
    }

    pub fn add_vectors(&mut self, ids: &[i64], vectors: &[f32]) -> Result<(), EngineError> {
        self.store.add(ids, vectors)?;

        for (i, chunk) in vectors.chunks(self.dim).enumerate() {
            let id = ids[i] as usize;
            self.index.insert(id, chunk.to_vec())?;
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, EngineError> {
        let neighbors = self.index.search(query, k)?;
        Ok(neighbors
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect())
    }

    pub fn health(&self) -> HealthReport {
        basic_index_health(&self.index)
    }
}


