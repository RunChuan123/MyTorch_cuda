#pragma once
#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>

class Tensor;
class TensorStorage;


enum class OpType{
    None,
    Add,
    Mul,
    MatMul,
    Relu,
    Sub,
    Div,
    Pow,
    Sum,
    SumDim,
    Mean,
    MeanDim,
};

inline std::ostream& operator<<(std::ostream& os,OpType op){
    switch (op){
    
    case OpType::Add :return os << " Add";
    case OpType::Mul :return os << " Mul";
    case OpType::MatMul :return os << " MatMul";
    case OpType::Relu :return os << " Relu";
    case OpType::Sub :return os << " Sub";
    case OpType::Div :return os << " Div";
    case OpType::Pow :return os << " Pow";
    case OpType::Sum :return os << " Sum";
    case OpType::SumDim :return os << " SumDim";
    case OpType::Mean :return os << " Mean";
    case OpType::MeanDim :return os << " MeanDim";
    default: return os << "";
    }
}   


// 每当进行一次运算backwardfn对应的forwardfn
// 就会生成一个新的tensor，绑定到tensornode上
class TensorNode{
public:
    // 依赖的张量 前驱节点
    std::vector<Tensor*> parents;
    int version = 0;
    OpType op = OpType::None;
    // 梯度函数 
    std::function<void()> backward_fn;

    bool is_leaf= false;
    Tensor* owner = nullptr;  // 当前节点对应的Tensor

    std::unordered_map<std::string,std::vector<int>> int_args;
    std::unordered_map<std::string,size_t > sz_args;
    std::unordered_map<std::string,float> float_args;
    std::unordered_map<std::string,bool> bool_args;

    std::unordered_map<std::string,std::weak_ptr<TensorStorage>> tensor_args;

    TensorNode() = default;
    TensorNode(Tensor* t) : owner(t) {}
};


