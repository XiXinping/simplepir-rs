mod matrix;
mod regev;
use matrix::{a_matrix_mul_db, mat_vec_mul, packed_mat_vec_mul};
pub use matrix::{Matrix, Vector};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_distr::Uniform;
use regev::{encrypt, gen_a_matrix, gen_secret_key};
use std::ops::RangeInclusive;

/// A square matrix that contains `u64` data records.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Database {
    pub data: Matrix,
    pub modulus: u64,
}

impl Database {
    /// Creates a new Database from an existing square Matrix. Panics if the matrix is not square.
    pub fn from_matrix(data: Matrix, modulus: u64) -> Database {
        assert_eq!(data.nrows(), data.ncols());
        Database { data, modulus }
    }
    /// Creates a new Database of size `side_len` × `side_len` populated by random data. The data is
    /// sampled from a uniform distribution over the specified `range`.
    pub fn new_random(side_len: usize, range: RangeInclusive<u64>) -> Database {
        Database {
            data: Matrix::new_random(side_len, side_len, range.clone(), None),
            modulus: range.end() + 1,
        }
    }
    /// Creates a new Database of size `side_len`×`side_len` populated by random data generated
    /// using a `seed. The data is samples from a uniform distribution over the specified `range`.
    pub fn new_random_seed(side_len: usize, range: RangeInclusive<u64>, seed: u64) -> Database {
        Database {
            data: Matrix::new_random(side_len, side_len, range.clone(), Some(seed)),
            modulus: range.end() + 1,
        }
    }
    /// Creates a new Database from a `Vec<u64>` of data and resizes it into a square matrix. Panics
    /// if the number of entries cannot be evenly resized into a square matrix.
    pub fn from_vector(data: Vec<u64>, modulus: u64) -> Database {
        let db_side_len = (data.len() as f32).sqrt().ceil() as usize;
        Database {
            data: Matrix::from_vec(data, db_side_len, db_side_len),
            modulus,
        }
    }
    /// Creates a new Database populated entirely by zeros.
    pub fn zeros(side_len: usize, modulus: Option<u64>) -> Database {
        let modulus = if let Some(num) = modulus { num } else { 1 };
        Database {
            data: Matrix::zeros(side_len, side_len),
            modulus,
        }
    }

    /// Gets the length of one side of the square Matrix within the Database.
    pub fn side_len(&self) -> usize {
        self.data.nrows()
    }

    /// Get a record at an index. The index is as if the square Matrix was resized into a vector
    /// according to row-major order.
    pub fn get(&self, index: usize) -> Option<u64> {
        let row_index = index / self.data.nrows();
        let col_index = index % self.data.ncols();
        self.data.get(row_index, col_index)
    }

    /// Compresses the database by packing three records into one 64-bit integer. The compression takes
    /// place along each row, meaning there'll be one third the number of columns in the new
    /// database compared to the old one.
    pub fn compress(&self, mod_power: u32) -> CompressedDatabase {
        assert!(mod_power < 64 / 3);

        let mask = 2_u64.pow(mod_power) - 1;
        let data: Vec<u64> = self
            .data
            .data
            .iter()
            .map(move |row| {
                (0..row.len().div_ceil(3)).map(move |i| {
                    row.get(i * 3).unwrap_or(&0) & mask
                        | (row.get(i * 3 + 1).unwrap_or(&0) & mask) << mod_power
                        | (row.get(i * 3 + 2).unwrap_or(&0) & mask) << (mod_power * 2)
                })
            })
            .flatten()
            .collect();
        CompressedDatabase {
            data: Matrix::from_vec(data, self.data.nrows(), self.data.ncols().div_ceil(3)),
            nrows: self.data.nrows(),
            ncols: self.data.ncols().div_ceil(3),
            mod_power,
        }
    }
}

/// A compressed version of a regular Database with 3 records packed into one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct CompressedDatabase {
    data: Matrix,
    nrows: usize,
    ncols: usize,
    mod_power: u32,
}

// A struct that contains information about the client's query, including the row and column index,
// the a-matrix of the database, the side length of the database, the client's secret key, and the
// key's length.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
pub struct ClientState {
    row_index: usize,
    column_index: usize,
    a_matrix_seed: u64,
    db_side_len: usize,
    secret_key: Vector,
    secret_dimension: usize,
}

impl ClientState {
    fn new(
        row_index: usize,
        column_index: usize,
        a_matrix_seed: u64,
        db_side_len: usize,
        secret_key: &Vector,
    ) -> ClientState {
        ClientState {
            row_index,
            column_index,
            a_matrix_seed,
            db_side_len,
            secret_key: secret_key.clone(),
            secret_dimension: secret_key.len(),
        }
    }
}

/// Outputs one hint for the server and one hint for the client. The server hint is the seed to
/// generate the a-matrix for the query, which since it stays constant, can be generated ahead of
/// time. The client hint is the matrix multiplication of this a-matrix with the data in the
/// database. This also stays constant and can be generated ahead of time to save on computation.
pub fn setup(database: &Database, secret_dimension: usize, seed: Option<u64>) -> (u64, Matrix) {
    let mut rng = if let Some(num) = seed {
        ChaCha20Rng::seed_from_u64(num)
    } else {
        ChaCha20Rng::from_entropy()
    };
    let server_hint = Uniform::from(0..=u64::MAX).sample(&mut rng);
    let data = database.data.add_scalar(u64::MAX * (database.modulus / 2));
    let a_matrix = gen_a_matrix(database.side_len(), secret_dimension, Some(server_hint));
    let client_hint = a_matrix_mul_db(&a_matrix, &data);
    (server_hint, client_hint)
}
/// Takes an index in the length-N database and outputs a vector with all 0s except for a 1 at the
/// column index
pub fn query(
    index: usize,
    db_side_len: usize,
    secret_dimension: usize,
    a_matrix_seed: u64,
    plain_mod: u64,
) -> (ClientState, Vector) {
    let secret_key = gen_secret_key(secret_dimension, None);
    let a_matrix = gen_a_matrix(db_side_len, secret_dimension, Some(a_matrix_seed));
    let row_index = index % db_side_len;
    let column_index = index / db_side_len;
    let mut query_vector = Vector::zeros(db_side_len);
    query_vector.data[row_index] = 1;
    let client_state = ClientState::new(
        row_index,
        column_index,
        a_matrix_seed,
        db_side_len,
        &secret_key,
    );
    (
        client_state,
        encrypt(&secret_key, &a_matrix, &query_vector, plain_mod).1,
    )
}

/// Computes the matrix-vector product of the packed database and the encrypted query. The output is an
/// encrypted vector that can be decrypted to reveal the records along the column indicated in the
/// query.
pub fn answer(database: &CompressedDatabase, query_cipher: &Vector) -> Vector {
    packed_mat_vec_mul(&query_cipher, &database.data, database.mod_power)
}
/// Computes the matrix-vector product of the **non-packed** database and the encrypted query. The
/// output is an encrypted vector that can be decrypted to reveal the records along the column
/// indicated in the query.
pub fn answer_uncompressed(database: &Database, query_cipher: &Vector) -> Vector {
    mat_vec_mul(&query_cipher, &database.data)
}

/// Takes the encrypted vector of records along the column specified in the query, decrypts it using
/// the secret key, and returns the record at the row and column that was specified in the query.
pub fn recover(
    client_state: &ClientState,
    client_hint: &Matrix,
    answer_cipher: &Vector,
    query_cipher: &Vector,
    plaintext_mod: u64,
) -> u64 {
    let ciphertext_mod = 2u128.pow(64);
    let q_over_p = (ciphertext_mod / plaintext_mod as u128) as u64;

    let secret_key = &client_state.secret_key;
    let column_index = client_state.column_index;

    let ratio = plaintext_mod / 2;
    let noised = answer_cipher.get_unchecked(column_index)
        - ratio * query_cipher.sum()
        - Vector::from_vec(client_hint.row_unchecked(column_index)).dot(secret_key);
    let denoised = (noised + q_over_p / 2) / q_over_p;

    (denoised - ratio).rem_euclid(plaintext_mod)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        const SEED: u64 = 42;

        let secret_dimension = 2048;
        let db_side_len = 40;
        let mod_power = 3;
        let plain_mod = 2_u64.pow(mod_power);
        let index = 0;

        let database = Database::new_random_seed(db_side_len, 0..=plain_mod - 1, SEED);
        let compressed_db = database.compress(mod_power);
        let (server_hint, client_hint) = setup(&database, secret_dimension, Some(42));
        let (client_state, query_cipher) =
            query(index, db_side_len, secret_dimension, server_hint, plain_mod);
        let answer_cipher = answer(&compressed_db, &query_cipher);
        let record = recover(
            &client_state,
            &client_hint,
            &answer_cipher,
            &query_cipher,
            plain_mod,
        );
        assert_eq!(record, database.get(index).unwrap())
    }

    #[test]
    fn test2() {
        const SEED: u64 = 42;

        let secret_dimension = 2048;
        let db_side_len = 1000;
        let mod_power = 17;
        let plain_mod = 2u64.pow(mod_power);

        let database = Database::new_random_seed(db_side_len, 0..=plain_mod - 1, SEED);
        let compressed_database = database.compress(mod_power);
        let (server_hint, client_hint) = setup(&database, secret_dimension, Some(42));
        for index in 0..100 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&compressed_database, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }

    #[test]
    fn test3() {
        const SEED: u64 = 42;

        let secret_dimension = 10;
        let db_side_len = 1000;
        let plain_mod = 2u64.pow(3);

        let database = Database::new_random_seed(db_side_len, 0..=plain_mod - 1, SEED);
        let (server_hint, client_hint) = setup(&database, secret_dimension, Some(42));
        for index in 0..100 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer_uncompressed(&database, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }

    #[test]
    fn test4() {
        const SEED: u64 = 42;

        let secret_dimension = 1000;
        let db_side_len = 38;
        let mod_power = 17;
        let plain_mod = 2u64.pow(mod_power);

        let database = Database::new_random_seed(db_side_len, 0..=plain_mod - 1, SEED);
        let database_compressed = database.compress(mod_power);
        let (server_hint, client_hint) = setup(&database, secret_dimension, Some(42));
        for index in 0..1444 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&database_compressed, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }
}
