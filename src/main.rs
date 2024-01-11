use std::arch::x86_64::*;

fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32], rows_a: usize, cols_a: usize, cols_b: usize) {
    assert_eq!(a.len(), rows_a * cols_a);
    assert_eq!(b.len(), cols_a * cols_b);
    assert_eq!(c.len(), rows_a * cols_b);

    let mut i = 0;
    while i < rows_a {
        let mut j = 0;
        while j < cols_b {
            let mut k = 0;
            while k < cols_a {
                unsafe {
                    let a_ptr = a.as_ptr().add(i * cols_a + k);
                    let b_ptr = b.as_ptr().add(k * cols_b + j);
                    let c_ptr = c.as_mut_ptr().add(i * cols_b + j);

                    let a_val = *a_ptr;
                    let a_broadcast = _mm256_set1_ps(a_val);

                    let b_vec = _mm256_loadu_ps(b_ptr);
                    let c_vec = _mm256_loadu_ps(c_ptr);

                    let result = _mm256_add_ps(_mm256_mul_ps(a_broadcast, b_vec), c_vec);

                    _mm256_storeu_ps(c_ptr, result);
                }
                k += 1;
            }
            j += 16;
        }
        i += 6;
    }
}

fn main() {
    let rows_a = 6;
    let cols_a = 4;
    let cols_b = 8;

    let a = vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0,
                 17.0, 18.0, 19.0, 20.0,
                 21.0, 22.0, 23.0, 24.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0];
    let mut c = vec![0.0; rows_a * cols_b];

    matmul_simd(&a, &b, &mut c, rows_a, cols_a, cols_b);

    println!("{:?}", c);
}
