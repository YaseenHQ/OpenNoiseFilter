use open_noise_filter::{DenoiseState, Tier, H};
fn main() {
    for tier in [Tier::Tiny, Tier::Base, Tier::Small] {
        let mut st = DenoiseState::new(tier);
        let inp = [0.1f32; H];
        let mut out = [0f32; H];
        st.process_frame(&inp, &mut out);
        let t0 = std::time::Instant::now();
        let n = 200;
        for _ in 0..n {
            std::hint::black_box(st.process_frame(&inp, &mut out));
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{:?}: {:.3} ms/frame (RTF {:.3})", tier, ms, ms / 10.667);
    }
}
