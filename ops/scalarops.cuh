#pragma once
#include "../tensor/tensor.cuh"


Tensor add(const Tensor& A,const Tensor& B);
Tensor sum(const Tensor& A,int dim,bool keep_dim = true);
Tensor sum(const Tensor& A); // 全局求和

Tensor mean(const Tensor& A);
Tensor mean(const Tensor& A,int dim);

Tensor sub(const Tensor& A, const Tensor& B);
Tensor& sub_(Tensor& A, const Tensor& B);