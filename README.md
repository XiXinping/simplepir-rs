# simplepir-rs
An implementation of SimplePIR in Rust. 

## Background
[Private information retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval)
(PIR) is a protocol for accessing information from a database hosted on a
server without the server knowing anything about the information that was
accessed, including the index of the information within the database. SimplePIR
is a fast and efficient implementation of PIR that provides √N communication
costs and linear computation costs. To learn more about SimplePIR check out
[this paper](https://eprint.iacr.org/2022/949) by Alexandra Henzinger, Matthew
M. Hong, Henry Corrigan-Gibbs, Sarah Meiklejohn, and Vinod Vaikuntanathan.

## Getting Started
We'll start by specifying some basic parameters for the SimplePIR scheme. For
good security and performance, a secret-key dimension (the length of the
encryption key) of 2048 is recommended. We'll also specify the plaintext
modulus, which tells us the range of numbers that can be accurately accessed
and decrypted. In this example, we'll use 2^17.
```rust
use simplepir::{Matrix, Database};
let secret_dimension = 2048;
let mod_power = 17;
let plaintext_mod = 2_u64.pow(mod_power);
```
We'll then create a simple database to store our data. Databases can be created
at random or from an existing Matrix.
```rust
let matrix = Matrix::from_data(
    vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ]
);
let db = Database::from_matrix(matrix, 17);
```
To increase performance and decrease memory consumption, the database can be
compressed by packing three data records (numbers) into a single record.
```rust
let compressed_db = db.compress();
```
Now for the fun parts!

There are four main functions of the SimplePIR protocol:


The first function is run during the "offline" phase.

### `setup()`
Takes the database as input and outputs a hint for the client and for the
server. This is called by the **server**. It's very computationally heavy, but
massively speeds up the "online" portion of the protocol.

```rust
let (server_hint, client_hint) = setup(&compressed_db, secret_dimension, None);
```

The next three functions are run during the "online" phase.

### `query()`
Takes an index into the database and outputs an encrypted query. This is called
by the **client**.


```rust
let index = 0;
let (client_state, query_cipher) = query(index, 4, secret_dimension, server_hint, plaintext_mod);

```
The `client_state` variable just stores some basic information about the
client's query. 


### `answer()`
Takes the matrix-vector product between the encrypted query and the entire
database and outputs an encrypted answer vector. This is called by **server**
and is the most computationally intense part of the online phase.

```rust
let answer_cipher = answer(&compressed_db, &query_cipher);
```

### `recover()`
Takes the encrypted answer vector, decrypts it, and returns the desired record.

```rust
let record = recover(&client_state, &client_hint, &answer_cipher, &query_cipher, plaintext_mod);
```

Now if we did everything right, this assert shouldn't fail!
```rust
assert_eq!(database.get(index).unwrap(), record);
```


## But is it fast?
Yes.

