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
    println!("cargo:rustc-link-search=native={}", libdir);
    println!(
        "cargo:rustc-link-search=native={}",
        "/data/libs/libtorch2.3/lib/"
    );
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");

    println!("cargo:rustc-link-lib=static=torch_stream_infer_ffi");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let torch_dynlibs = vec!["c10", "cuda", "nvrtc", "nvToolsExt", "cudart", "c10_cuda"];
    // let torch_static_libs = vec!["kineto"];

    for dynlib in torch_dynlibs {
        println!("cargo:rustc-link-lib=dylib={}", dynlib);
    }

    // for static_lib in torch_static_libs {
    // println!("cargo:rustc-link-lib=static=torch_stream_infer_ffi");

    // }
}
