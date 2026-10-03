//! Radix-2 complex FFT — a direct port of the JS `makeFFT` in the
//! library's `src/dsp.js` (precomputed bit-reversal + per-stage twiddles,
//! f64 internally, identical values to the JS/Math pipeline).

pub struct Fft {
    n: usize,
    rev: Vec<u32>,
    stages: Vec<(Vec<f64>, Vec<f64>)>, // (cos, sin) per stage
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let mut rev = vec![0u32; n];
        let mut j = 0u32;
        for i in 1..n {
            let mut bit = (n >> 1) as u32;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            rev[i] = j;
        }
        let mut stages = Vec::new();
        let mut len = 2;
        while len <= n {
            let half = len >> 1;
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            let c: Vec<f64> = (0..half).map(|k| (ang * k as f64).cos()).collect();
            let s: Vec<f64> = (0..half).map(|k| (ang * k as f64).sin()).collect();
            stages.push((c, s));
            len <<= 1;
        }
        Fft { n, rev, stages }
    }

    pub fn run(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        for i in 1..n {
            let j = self.rev[i] as usize;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut st = 0;
        let mut len = 2;
        while len <= n {
            let half = len >> 1;
            let (tc, ts) = &self.stages[st];
            let mut i = 0;
            while i < n {
                for k in 0..half {
                    let (c, s) = (tc[k], ts[k]);
                    let (ur, ui) = (re[i + k], im[i + k]);
                    let (xr, xi) = (re[i + k + half], im[i + k + half]);
                    re[i + k] = ur + xr * c - xi * s;
                    im[i + k] = ui + xr * s + xi * c;
                    re[i + k + half] = ur - xr * c + xi * s;
                    im[i + k + half] = ui - xr * s - xi * c;
                }
                i += len;
            }
            st += 1;
            len <<= 1;
        }
    }
}
