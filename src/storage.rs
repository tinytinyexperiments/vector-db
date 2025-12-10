use std::path::PathBuf;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub path: PathBuf,
    pub dim: usize,
}

pub struct SqliteVectorStore {
    conn: Mutex<Connection>,
    dim: usize,
}

impl SqliteVectorStore {
    pub fn new(cfg: &StorageConfig) -> Result<Self, StorageError> {
        if let Some(parent) = cfg.path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(&cfg.path)?;
        let store = Self {
            conn: Mutex::new(conn),
            dim: cfg.dim,
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL
            );",
            [],
        )?;
        Ok(())
    }

    pub fn add(&self, ids: &[i64], vectors: &[f32]) -> Result<(), StorageError> {
        let n = vectors.len() / self.dim;
        if n != ids.len() {
            return Err(StorageError::Sqlite(rusqlite::Error::InvalidQuery));
        }

        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO vectors (id, vector) VALUES (?1, ?2);",
            )?;

            for (i, chunk) in vectors.chunks(self.dim).enumerate() {
                stmt.execute(params![ids[i], bytemuck::cast_slice(chunk)])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_all(&self) -> Result<(Vec<i64>, Vec<f32>), StorageError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, vector FROM vectors;")?;
        let mut rows = stmt.query([])?;

        let mut ids = Vec::new();
        let mut all_vecs = Vec::new();

        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let floats: &[f32] = bytemuck::cast_slice(&blob);
            if floats.len() != self.dim {
                // skip malformed rows for now
                continue;
            }
            ids.push(id);
            all_vecs.extend_from_slice(floats);
        }

        Ok((ids, all_vecs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn add_and_load_roundtrips_vectors() {
        let dim = 4;
        let tmp_dir = tempdir().expect("tempdir");
        let path = tmp_dir.path().join("vectors.sqlite");

        let store = SqliteVectorStore::new(&StorageConfig { path, dim })
            .expect("store created");

        let ids = vec![1_i64, 2_i64];
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // id 1
            0.0, 1.0, 0.0, 0.0, // id 2
        ];

        store.add(&ids, &vectors).expect("add should succeed");

        let (loaded_ids, loaded_vecs) = store.load_all().expect("load_all");
        assert_eq!(loaded_ids.len(), 2);
        assert_eq!(loaded_vecs.len(), vectors.len());
    }
}


