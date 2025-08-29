use std::{env, process::Command};

fn main() {
    // let build_option = "RelWithDebInfo";
    let build_option = "Release";

    let out_path_str = env::var("OUT_DIR").unwrap();
    let origin_source_file_dir = "libtorch_infer_c";
    let current_dir = env::current_dir().unwrap().to_str().unwrap().to_string();
    Command::new("sh")
        .arg("-c")
        .arg(&format!(
            "cp -r {} {}",
            origin_source_file_dir, out_path_str
        ))
        .status()
        .unwrap();

    Command::new("sh")
        .arg("-c")
        .arg(&format!(
            "/usr/bin/cmake -DCMAKE_BUILD_TYPE:STRING={} -S./ -B./build -G 'Unix Makefiles'",
            build_option
        ))
        .current_dir(&format!("{}/{}", out_path_str, origin_source_file_dir))
        .status()
        .unwrap();

    let _ = Command::new("sh")
        .arg("-c")
        .arg(&format!(
            "/usr/bin/cmake --build build/ --config {} --target all -j40 --",
            build_option
        ))
        .current_dir(&format!("{}/{}", out_path_str, origin_source_file_dir))
        .status();

    Command::new("sh")
        .arg("-c")
        .arg(&format!(
            "/usr/bin/cmake --build build/ --config {} --target all --",
            build_option
        ))
        .current_dir(&format!("{}/{}", out_path_str, origin_source_file_dir))
        .status()
        .unwrap();

    let libdir = format!("{}/{}/build", out_path_str, origin_source_file_dir);
    println!(
        "cargo:rerun-if-changed={}",
        &format!("{}/{}", current_dir, origin_source_file_dir)
    );

    let lib_dir = "/data/libs/libtorch2.3/lib/";
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_dir);

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");

    println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");

    println!("cargo:rustc-link-search=native={}", libdir);
    println!("cargo:rustc-link-lib=static=torch_stream_infer_ffi");

    // torch_library
    // let torch_dynlibs = "torch".split(";").collect::<Vec<_>>();

    // let torch_abslibs = "/data/libs/libtorch2.3/lib/libc10.so;/usr/lib/x86_64-linux-gnu/libcuda.so;/usr/local/cuda/lib64/libnvrtc.so;/usr/local/cuda/lib64/libnvToolsExt.so;/usr/local/cuda/lib64/libcudart.so;/data/libs/libtorch2.3/lib/libc10_cuda.so".split(";").collect::<Vec<_>>();
    // for dynlib in torch_abslibs {
    //     println!("cargo:rustc-link-search=native={}", dynlib);
    // }

    // let torch_dynlibs = vec![
    //     "torch",
    //     "c10",
    //     "cuda",
    //     "nvrtc",
    //     "nvToolsExt",
    //     "cudart",
    //     "c10_cuda",
    // ];
    // for dynlib in torch_dynlibs {
    //     println!("cargo:rustc-link-lib=dylib={}", dynlib);
    // }
    // println!("cargo:rustc-link-lib=static=kineto");

    // // libtorch cpu dependencies
    // let libtorch_cpu_deps = vec![
    //     // "linux-vdso.so.1",
    //     "/lib/x86_64-linux-gnu/librt.so.1",
    //     "/lib/x86_64-linux-gnu/libgcc_s.so.1",
    //     "/lib/x86_64-linux-gnu/libdl.so.2",
    //     "/lib/x86_64-linux-gnu/libpthread.so.0",
    //     "/lib/x86_64-linux-gnu/libm.so.6",
    //     "/root/miniconda3/lib/python3.12/site-packages/torch/lib/libgomp-a34b3233.so.1",
    //     "/lib/x86_64-linux-gnu/libc.so.6",
    //     "/lib64/ld-linux-x86-64.so.2",
    // ];
    // for dep in libtorch_cpu_deps {
    //     println!("cargo:rustc-link-arg=-Wl,{}", dep);
    // }

    // println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rustc-link-search=native=/data/libs/libtorch2.3/lib/");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=nvToolsExt");
    println!("cargo:rustc-link-lib=kineto");
    println!("cargo:rustc-link-lib=stdc++");
}
