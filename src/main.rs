use std::arch::x86_64::*;

fn kernel(a: &[f32], b: &[f32], c: &mut [f32], x: usize, y: usize, l: usize, r: usize, n: usize, m: usize) {

    unsafe {
        let mut t: [[__m256; 2]; 6] = [[_mm256_setzero_ps(); 2]; 6];
        for k in l..r {
            let b0 = _mm256_load_ps(&b[k * n + y]);
            let b1 = _mm256_load_ps(&b[k * n + y + 8]);
            for i in 0..6 {
                let alpha = _mm256_broadcast_ss(&a[(x + i) * m + k]);

                t[i][0] = _mm256_fmadd_ps(alpha, b0, t[i][0]);
                t[i][1] = _mm256_fmadd_ps(alpha, b1, t[i][1]);
            }
        }
    

        for i in 0..6 {
            for j in 0..2 {
                let mut temp = [0.0; 8];
                _mm256_store_ps(temp.as_mut_ptr(), t[i][j]);
                for k in 0..8 {
                    c[((x + i) * n + y) / 8 + j] += temp[k];
                }
            }
        }
    }
}

fn matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize, m: usize) {
    let nx = (n + 5) / 6 * 6;
    let ny = (m + 15) / 16 * 16;
    
    let mut a_padded = vec![0.0; nx * ny];
    let mut b_padded = vec![0.0; nx * ny];
    let mut c_padded = vec![0.0; nx * ny];

    for i in 0..n {
        a_padded[i * ny..(i + 1) * ny].copy_from_slice(&a[i * m..(i + 1) * m]);
    }

    for i in 0..m {
        b_padded[i * ny..(i + 1) * ny].copy_from_slice(&b[i * n..(i + 1) * n]);
    }

    for x in (0..nx).step_by(6) {
        for y in (0..ny).step_by(16) {
            kernel(&a_padded, &b_padded, &mut c_padded, x, y, 0, m, ny, n);
        }
    }

    for i in 0..n {
        c[i * m..(i + 1) * m].copy_from_slice(&c_padded[i * ny..(i + 1) * ny]);
    }
}

// some changes