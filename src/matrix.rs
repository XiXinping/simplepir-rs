use rand::distributions::Uniform;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use std::iter::zip;
use std::ops::RangeInclusive;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Matrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<Vec<u64>>,
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Vector {
    pub len: usize,
    pub data: Vec<u64>,
}

impl Matrix {
    pub fn from_data(data: Vec<Vec<u64>>) -> Matrix {
        let nrows = data.len();
        let ncols = data[0].len();
        for row in &data {
            assert_eq!(row.len(), ncols);
        }
        Matrix { nrows, ncols, data }
    }
    pub fn from_vec(vector: Vec<u64>, nrows: usize, ncols: usize) -> Matrix {
        assert_eq!(vector.len(), nrows * ncols);
        let mut result = Vec::<Vec<u64>>::with_capacity(nrows * ncols);
        for row in vector.chunks_exact(ncols) {
            result.push(row.to_owned());
        }
        Matrix {
            nrows,
            ncols,
            data: result,
        }
    }

    pub fn new_random(
        nrows: usize,
        ncols: usize,
        range: RangeInclusive<u64>,
        seed: Option<u64>,
    ) -> Matrix {
        let mut rng = if let Some(num) = seed {
            ChaCha20Rng::seed_from_u64(num)
        } else {
            ChaCha20Rng::from_entropy()
        };
        let distribution = Uniform::from(range);
        let mut result = Vec::<Vec<u64>>::with_capacity(nrows * ncols);
        for _ in 0..nrows {
            let mut row = Vec::<u64>::with_capacity(ncols);
            for _ in 0..ncols {
                row.push(distribution.sample(&mut rng));
            }
            result.push(row);
        }
        Matrix::from_data(result)
    }

    // Create a new matrix with each element set to zero
    pub fn zeros(nrows: usize, ncols: usize) -> Matrix {
        let mut matrix = Vec::<Vec<u64>>::with_capacity(nrows * ncols);
        let mut vector = Vec::<u64>::with_capacity(ncols);
        vector.resize(ncols, 0);
        matrix.resize(nrows, vector);

        Matrix::from_data(matrix)
    }

    // Perform element-wise addition
    pub fn add(&self, other_matrix: &Matrix) -> Matrix {
        assert_eq!(self.nrows(), other_matrix.nrows());
        assert_eq!(self.ncols(), other_matrix.ncols());
        Matrix::from_data(
            self.data
                .iter()
                .zip(other_matrix.data.iter())
                .map(|(row, other_row)| {
                    row.iter()
                        .zip(other_row.iter())
                        .map(|(item, other_item)| item + other_item)
                        .collect::<Vec<u64>>()
                })
                .collect::<Vec<Vec<u64>>>(),
        )
    }

    // Add the scalar to each element
    pub fn add_scalar(&self, scalar: u64) -> Matrix {
        Matrix::from_data(
            self.data
                .iter()
                .map(|row| row.iter().map(|item| item + scalar).collect::<Vec<u64>>())
                .collect::<Vec<Vec<u64>>>(),
        )
    }

    // Multiply each element by the scalar
    pub fn mul_scalar(&self, scalar: u64) -> Matrix {
        Matrix::from_data(
            self.data
                .iter()
                .map(|row| row.iter().map(|item| item * scalar).collect::<Vec<u64>>())
                .collect::<Vec<Vec<u64>>>(),
        )
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nrows * self.ncols
    }
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn get(&self, row_index: usize, col_index: usize) -> Option<u64> {
        if let Some(item) = self.data.get(row_index)?.get(col_index) {
            Some(*item)
        } else {
            None
        }
    }
    #[inline]
    pub fn get_unchecked(&self, row_index: usize, col_index: usize) -> u64 {
        unsafe { *self.data.get_unchecked(row_index).get_unchecked(col_index) }
    }

    pub fn row(&self, row_index: usize) -> Option<Vec<u64>> {
        if let Some(row) = self.data.get(row_index) {
            Some(row.clone())
        } else {
            None
        }
    }
    pub fn row_unchecked(&self, row_index: usize) -> Vec<u64> {
        unsafe { self.data.get_unchecked(row_index).clone() }
    }
}

impl Vector {
    pub fn from_vec(vector: Vec<u64>) -> Vector {
        Vector {
            len: vector.len(),
            data: vector,
        }
    }
    pub fn new_random(len: usize, range: RangeInclusive<u64>, seed: Option<u64>) -> Vector {
        let mut rng = if let Some(num) = seed {
            ChaCha20Rng::seed_from_u64(num)
        } else {
            ChaCha20Rng::from_entropy()
        };
        let distribution = Uniform::from(range);
        let mut result = Vec::<u64>::with_capacity(len);
        for _ in 0..len {
            result.push(distribution.sample(&mut rng));
        }
        Vector::from_vec(result)
    }

    pub fn zeros(len: usize) -> Vector {
        let mut vector = Vec::<u64>::with_capacity(len);
        vector.resize(len, 0);
        Vector::from_vec(vector)
    }
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, index: usize) -> Option<u64> {
        if let Some(item) = self.data.get(index) {
            Some(*item)
        } else {
            None
        }
    }

    pub fn get_unchecked(&self, index: usize) -> u64 {
        unsafe { *self.data.get_unchecked(index) }
    }

    // performs element-wise addition
    pub fn add(&self, other_vector: &Vector) -> Vector {
        assert_eq!(self.len(), other_vector.len());
        Vector::from_vec(
            self.data
                .iter()
                .zip(other_vector.data.iter())
                .map(|(item, other_item)| item + other_item)
                .collect::<Vec<u64>>(),
        )
    }

    pub fn add_scalar(&self, scalar: u64) -> Vector {
        Vector::from_vec(
            self.data
                .iter()
                .map(|item| item + scalar)
                .collect::<Vec<u64>>(),
        )
    }

    pub fn mul_scalar(&self, scalar: u64) -> Vector {
        Vector::from_vec(
            self.data
                .iter()
                .map(|item| item * scalar)
                .collect::<Vec<u64>>(),
        )
    }
    pub fn dot(&self, other_vector: &Vector) -> u64 {
        self.data
            .iter()
            .zip(other_vector.data.iter())
            .map(|(item, other_item)| item * other_item)
            .sum()
    }
    pub fn sum(&self) -> u64 {
        self.data.iter().sum()
    }
}

pub fn mat_vec_mul(vector: &Vector, matrix: &Matrix) -> Vector {
    let mut result = Vec::<u64>::with_capacity(matrix.nrows);
    for row in &matrix.data {
        let mut row_sum = 0;
        for (i, item) in row.iter().enumerate() {
            row_sum += vector.data[i] * item;
        }
        result.push(row_sum);
    }
    Vector::from_vec(result)
}

// Adds one vector to another in place.
fn vec_add_mut(v1: &mut Vec<u64>, v2: &Vec<u64>) {
    assert_eq!(v1.len(), v2.len());
    *v1 = v1
        .iter()
        .zip(v2.iter())
        .map(|(v1_item, v2_item)| v1_item + v2_item)
        .collect();
}

// Multiplies each element in a vector by a scalar.
fn vec_mul_scalar(vector: &Vec<u64>, scalar: u64) -> Vec<u64> {
    vector.iter().map(|item| item * scalar).collect()
}

pub fn a_matrix_mul_db(a_matrix: &Matrix, db: &Matrix) -> Matrix {
    assert_eq!(a_matrix.nrows(), db.ncols());
    let mut result = Vec::<Vec<u64>>::with_capacity(a_matrix.nrows() * a_matrix.ncols());
    for db_row in &db.data {
        let mut row_sum = Vec::<u64>::with_capacity(a_matrix.ncols());
        row_sum.resize(a_matrix.ncols(), 0);
        for (db_item, a_matrix_row) in zip(db_row, &a_matrix.data) {
            vec_add_mut(&mut row_sum, &vec_mul_scalar(a_matrix_row, *db_item));
        }
        result.push(row_sum);
    }
    Matrix::from_data(result)
}

pub fn packed_mat_vec_mul(vector: &Vector, packed_matrix: &Matrix, mod_power: u32) -> Vector {
    let basis = mod_power as u64;
    let mask = 2_u64.pow(mod_power) - 1;
    // println!("Basis: {basis} Mask: {mask}");
    assert!(mod_power < 64 / 3);
    assert_eq!(vector.len().div_ceil(3), packed_matrix.ncols);
    let rows = packed_matrix.nrows();
    let cols = packed_matrix.ncols();

    let mut result = vec![0u64; rows];

    for i in (0..rows / 4 * 4).step_by(4) {
        let mut row1_sum = 0;
        let mut row2_sum = 0;
        let mut row3_sum = 0;
        let mut row4_sum = 0;
        for j in 0..cols - 1 {
            let vec1 = vector.get_unchecked(j * 3);
            let vec2 = vector.get_unchecked(j * 3 + 1);
            let vec3 = vector.get_unchecked(j * 3 + 2);

            let db1 = packed_matrix.get_unchecked(i, j);
            let db2 = packed_matrix.get_unchecked(i + 1, j);
            let db3 = packed_matrix.get_unchecked(i + 2, j);
            let db4 = packed_matrix.get_unchecked(i + 3, j);

            let mut val1 = db1 & mask;
            let mut val2 = db2 & mask;
            let mut val3 = db3 & mask;
            let mut val4 = db4 & mask;

            row1_sum += val1 * vec1;
            row2_sum += val2 * vec1;
            row3_sum += val3 * vec1;
            row4_sum += val4 * vec1;

            val1 = db1 >> basis & mask;
            val2 = db2 >> basis & mask;
            val3 = db3 >> basis & mask;
            val4 = db4 >> basis & mask;

            row1_sum += val1 * vec2;
            row2_sum += val2 * vec2;
            row3_sum += val3 * vec2;
            row4_sum += val4 * vec2;

            val1 = db1 >> basis * 2 & mask;
            val2 = db2 >> basis * 2 & mask;
            val3 = db3 >> basis * 2 & mask;
            val4 = db4 >> basis * 2 & mask;

            row1_sum += val1 * vec3;
            row2_sum += val2 * vec3;
            row3_sum += val3 * vec3;
            row4_sum += val4 * vec3;
        }
        let index = cols - 1;
        let db1 = packed_matrix.get_unchecked(i, index);
        let db2 = packed_matrix.get_unchecked(i + 1, index);
        let db3 = packed_matrix.get_unchecked(i + 2, index);
        let db4 = packed_matrix.get_unchecked(i + 3, index);

        let vec1 = vector.get_unchecked(index * 3);
        let vec2 = vector.get(index * 3 + 1).unwrap_or(0);
        let vec3 = vector.get(index * 3 + 2).unwrap_or(0);

        row1_sum +=
            (db1 & mask) * vec1 + (db1 >> basis & mask) * vec2 + (db1 >> basis * 2 & mask) * vec3;
        row2_sum +=
            (db2 & mask) * vec1 + (db2 >> basis & mask) * vec2 + (db2 >> basis * 2 & mask) * vec3;
        row3_sum +=
            (db3 & mask) * vec1 + (db3 >> basis & mask) * vec2 + (db3 >> basis * 2 & mask) * vec3;
        row4_sum +=
            (db4 & mask) * vec1 + (db4 >> basis & mask) * vec2 + (db4 >> basis * 2 & mask) * vec3;

        result[i] = row1_sum;
        result[i + 1] = row2_sum;
        result[i + 2] = row3_sum;
        result[i + 3] = row4_sum;
    }

    for row_index in (rows / 4 * 4)..rows {
        let mut row_sum = 0;
        for j in 0..cols - 1 {
            let vec1 = vector.get_unchecked(j * 3);
            let vec2 = vector.get_unchecked(j * 3 + 1);
            let vec3 = vector.get_unchecked(j * 3 + 2);

            let db = packed_matrix.get(row_index, j).unwrap_or(0);
            let mut val = db & mask;

            // println!("DB Entry: {db}");
            // println!("First Index: {val}");
            row_sum += val * vec1;

            val = db >> basis & mask;
            // println!("Second Index: {val}");

            row_sum += val * vec2;

            val = db >> basis * 2 & mask;
            // println!("Third Index: {val}");

            row_sum += val * vec3;
        }
        let col_index = cols - 1;
        let db = packed_matrix.get(row_index, col_index).unwrap_or(0);
        row_sum += (db & mask) * vector.get(col_index * 3).unwrap_or(0)
            + (db >> basis & mask) * vector.get(col_index * 3 + 1).unwrap_or(0)
            + (db >> basis * 2 & mask) * vector.get(col_index * 3 + 2).unwrap_or(0);

        result[row_index] = row_sum;
    }
    Vector::from_vec(result)
}
