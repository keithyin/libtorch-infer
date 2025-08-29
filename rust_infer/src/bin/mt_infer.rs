use rust_infer::{self, torch_stream_infer_wrapper::ModuleInferCtx};
use std::{thread, time::Instant};
fn main() {
    thread::scope(|thread_scope| {
        for _ in 0..3 {
            thread_scope.spawn(|| {
            let model_path =
                "/root/projects/libtorch-infer/models/self-attn-newpe-nomaskcheck-nocausal/model";
            let mut ctx = ModuleInferCtx::new(model_path, 3, 256, 200, 61, 2, true);

            let mut final_res = 0.0;
            let now = Instant::now();
            for _ in 0..8 {
                ctx.feature_mut().unwrap().fill(1.0);
                ctx.length_mut().unwrap().fill(200);
                ctx.do_infer();
                let res = ctx.output().unwrap();
                final_res += res[[0, 0, 1]];
            }
            println!("result:{}, timeelapsed:{}", final_res, now.elapsed().as_secs_f32());

        });
        }
    });
}
