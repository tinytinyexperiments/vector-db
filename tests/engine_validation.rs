use std::path::PathBuf;

use self_healing_vector_db::{EngineConfig, SelfHealingVectorDb};
use tempfile::tempdir;

#[test]
fn add_vectors_with_wrong_dim_fails_and_does_not_change_health() {
    let dim = 384;
    let tmp_dir = tempdir().expect("tempdir");
    let db_path: PathBuf = tmp_dir.path().join("vectors.sqlite");

    let cfg = EngineConfig {
        dim,
        storage_path: db_path.clone(),
        hnsw_max_elements: 10_000,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 64,
    };

    let mut engine =
        SelfHealingVectorDb::new(cfg, None).expect("engine created");
    let health_before = engine.health();
    assert_eq!(health_before.size, 0);

    // Build a flat vector buffer with the wrong dimension (dim - 1 per vector).
    let wrong_dim = dim - 1;
    let mut v = vec![0.0_f32; wrong_dim];
    v[0] = 1.0;
    let mut flat_wrong = Vec::new();
    flat_wrong.extend_from_slice(&v);

    let ids = vec![1_i64];
    let res = engine.add_vectors(&ids, &flat_wrong);
    assert!(
        res.is_err(),
        "adding vectors with wrong dimension should error"
    );

    let health_after = engine.health();
    assert_eq!(
        health_after.size, 0,
        "index size should remain 0 after failed add"
    );
}


