#pragma once
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
#include <iostream>



enum class DType{
    Float32,
    // Float16,   // half
    Int32,
    // BFloat16,  // nv_bfloat16
};

inline std::ostream& operator<<(std::ostream& os, DType dtype) {
    switch (dtype) {
        case DType::Float32: return os << "Float32";
        // case DType::Float16: return os << "Float16";
        case DType::Int32: return os << "Int32";
        // case DType::BFloat16: return os << "BFloat16";
        default: return os << "Unknown";
    }
}


inline size_t dtype_size(DType type){
    switch (type)
    {
    // case DType::Float16: return 2;
    case DType::Float32: return 4;
    case DType::Int32: return 4;
    // case DType::BFloat16: return 2;
    
    default:
        return 0;
    }
}


#define DISPATCH_DTYPE(dtype,scalar_t , ...)\
switch (dtype)\
{\
    case DType::Float32:{\
        using scalar_t = float;\
        __VA_ARGS__;\
        break;\
    }\
    case DType::Int32:{\
        using scalar_t = int;\
        __VA_ARGS__;\
        break;\
    }\
    default:\
        assert(false);\
}\

    // case DType::BFloat16:{\
    //     using scalar_t = __nv_bfloat16;\
    //     __VA_ARGS__;\
    //     break;\
    // }\
    // case DType::Float16:{\
    //     using scalar_t = half;\
    //     __VA_ARGS__;\
    //     break;\
    // }

template <typename ElementType>
struct alignas(16) Packed128{
    Packed128() = default;
    __device__ explicit Packed128 (int4 bits){
        static_assert(sizeof(bits)==sizeof(payload),"Size mismatch");
        memcpy(&payload,&bits,sizeof(bits));
    }

    __device__ static Packed128 constant(ElementType value){
        Packed128 result;
        for(int k=0;k < size;k++){
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros(){
        return constant(0.0f);
    }
    __device__ static Packed128 ones(){
        return constant(1.0f);
    }

    __device__ ElementType& operator [](int index){
        return payload[index]; 
    }
    __device__ const ElementType& operator [](int index)const {
        return payload[index]; 
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload),"Size mismatch");
        memcpy(&bits,&payload,sizeof(bits));
        return bits;
    }
    

    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};


template<typename ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address){
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
template<typename ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address){
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
template<typename ElementType>
__device__ void store128(ElementType* target,Packed128<ElementType> value){
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
template<typename ElementType>
__device__ void store128cs(ElementType* target,Packed128<ElementType> value){
    __stcs(reinterpret_cast<int4*>(target),value.get_bits());
}
template<typename ElementType>
__device__ void store128cg(ElementType* target,Packed128<ElementType> value){
    __stcg(reinterpret_cast<int4*>(target),value.get_bits());
}

// #define floatX half

typedef Packed128<float> f128;
// typedef Packed128<floatX> x128;


template<typename scalar_t>
using pack128 = Packed128<scalar_t>;


#define DISPATCH_DTYPE_AND_PACK(dtype,scalar_t,vec_t,...)\
switch(dtype){\
    case DType::Float32:{\
        using scalar_t = float;\
        using vec_t = pack128<scalar_t>;\
        __VA_ARGS__\
        break;\
    }\
    case DType::Int32:{\
        using scalar_t = int;\
        using vec_t = pack128<scalar_t>;\
        __VA_ARGS__\
        break;\
    }\
    default:assert(false);\
} 

    // case DType::Float16:{\
    //     using scalar_t = half;\
    //     using vec_t = pack128<scalar_t>;\
    //     __VA_ARGS__\
    //     break;\
    // }
     // case DType::BFloat16:{\
    //     using scalar_t = __nv_bfloat16;\
    //     using vec_t = pack128<scalar_t>;\
    //     __VA_ARGS__\
    //     break;\
    // }