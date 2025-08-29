use std::ffi::CString;

use ndarray::{ArrayView3, ArrayViewMut1, ArrayViewMut3};

use crate::torch_stream_infer_ffi;

pub struct ModuleInferCtx {
    inner: torch_stream_infer_ffi::ModuleInferCtx,
    batch: i32,
    timestemp: i32,
    feat_size: i32,
    output_size: i32,
}

impl ModuleInferCtx {
    pub fn new(
        model_path: &str,
        device_id: i32,
        batch: i32,
        tt: i32,
        feat_len: i32,
        output_size: i32,
        pin_memory: bool,
    ) -> Self {
        let model_path = CString::new(model_path).unwrap();

        let inner = unsafe {
            torch_stream_infer_ffi::build_module_infer_ctx(
                model_path.as_ptr(),
                device_id,
                batch,
                tt,
                feat_len,
                output_size,
                pin_memory,
            )
        };
        Self {
            inner,
            batch,
            timestemp: tt,
            feat_size: feat_len,
            output_size: output_size,
        }
    }

    pub fn feature_mut(&mut self) -> Option<ArrayViewMut3<'_, f32>> {
        if self.inner.batch_feature.is_null() {
            return None;
        }

        unsafe {
            Some(ArrayViewMut3::from_shape_ptr(
                (
                    self.batch as usize,
                    self.timestemp as usize,
                    self.feat_size as usize,
                ),
                self.inner.batch_feature,
            ))
        }
    }

    pub fn length_mut(&mut self) -> Option<ArrayViewMut1<'_, i64>> {
        if self.inner.batch_lengths.is_null() {
            return None;
        }
        unsafe {
            Some(ArrayViewMut1::from_shape_ptr(
                self.batch as usize,
                self.inner.batch_lengths,
            ))
        }
    }

    pub fn output(&self) -> Option<ArrayView3<'_, f32>> {
        if self.inner.output.is_null() {
            return None;
        }
        unsafe {
            Some(ArrayView3::from_shape_ptr(
                (
                    self.batch as usize,
                    self.timestemp as usize,
                    self.output_size as usize,
                ),
                self.inner.output,
            ))
        }
    }

    pub fn do_infer(&mut self) {
        unsafe { torch_stream_infer_ffi::do_infer(&mut self.inner) };
    }
}

impl Drop for ModuleInferCtx {
    fn drop(&mut self) {
        unsafe { torch_stream_infer_ffi::free_module_infer_ctx(&mut self.inner) };
    }
}

#[cfg(test)]
mod test {
    use crate::torch_stream_infer_wrapper::ModuleInferCtx;

    #[test]
    fn test_infer() {
        let model_path =
            "/root/projects/libtorch-infer/models/self-attn-newpe-nomaskcheck-nocausal/model";
        let mut ctx = ModuleInferCtx::new(model_path, 3, 256, 200, 61, 2, true);
        println!("here");

        ctx.feature_mut().unwrap().fill(1.0);
        println!("feature:{}", ctx.feature_mut().unwrap()[[0, 0, 0]]);
        ctx.length_mut().unwrap().fill(200);
        println!("length:{}", ctx.length_mut().unwrap()[0]);

        ctx.do_infer();
        println!("done infer");
        let res = ctx.output().unwrap();
        println!("{}", res[[0, 0, 0]]);
        println!("{}", res[[0, 0, 1]]);
    }
}
