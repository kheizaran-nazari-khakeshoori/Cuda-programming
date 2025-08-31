# CUDA Matrix Multiplication (CPU vs GPU)

This project compares **matrix multiplication performance** between:

- **CPU with OpenMP parallelization**
- **GPU using Thrust**
- **GPU using cuBLAS**

---

## 🚀 Features
- Parallel CPU implementation with OpenMP
- GPU implementation with Thrust
- GPU optimized implementation with cuBLAS
- Timing comparison between CPU and GPU approaches

---

## 📂 Project Structure
```
src/matrix_multiply.cu   # Main CUDA code
report/report.pdf        # Report with benchmarks (to be added later)
```

---

## 🛠️ Requirements
- CUDA Toolkit (nvcc, cuBLAS)
- OpenMP support
- C++17 or later

---

## ▶️ How to Run
Compile with:

```bash
nvcc -Xcompiler -fopenmp src/matrix_multiply.cu -lcublas -o matrix_multiply
./matrix_multiply
```

---

## 📊 Sample Output
```
CPU time (OpenMP Parallelized): X.XXXX seconds
GPU Thrust time: Y.YYYY seconds
GPU cuBLAS time: Z.ZZZZ seconds
```

---

## 📄 Report
A detailed performance analysis with graphs will be uploaded soon.
